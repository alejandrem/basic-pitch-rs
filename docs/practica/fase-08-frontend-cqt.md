# Fase 4: implementación del frontend CQT

**Objetivo:** reproducir la representación acústica que la red neuronal espera recibir.

Esta es una de las fases más delicadas del proyecto.

El modelo no recibe directamente muestras de audio ni un espectrograma STFT convencional. Recibe una representación CQT calculada mediante la implementación de `nnaudio.CQT` utilizada por Basic Pitch.

Por lo tanto, sustituirla por una CQT genérica o una STFT con parámetros parecidos no garantiza resultados equivalentes.

## 4.1. Comprender la transformación CQT

La transformación Constant-Q representa el contenido frecuencial del audio usando bins distribuidos logarítmicamente.

A diferencia de una STFT convencional, la resolución frecuencial está relacionada con la frecuencia de cada bin.

Para esta implementación se deberán reproducir:

- Frecuencia de muestreo.
- Frecuencia mínima.
- Número de bins.
- Bins por octava.
- Longitud del kernel.
- Ventana.
- Normalización.
- Padding.
- Hop length.
- Algoritmo de submuestreo.
- Orden de las operaciones.

Los parámetros finales se extraerán de la configuración real del modelo.

## 4.1.1. Corrección: la CQT produce 309 bins, no 264

Este es el error de mayor impacto potencial de todo el proyecto. La CQT
produce **309** bins de frecuencia; la reducción a 264 ocurre después, dentro de
Harmonic Stacking (Fase 5).

```python
MAX_N_SEMITONES = int(np.floor(12.0 * np.log2(0.5 * 22050 / 27.5)))   # 103
n_semitones      = min(ceil(12.0 * log2(8)) + 88, MAX_N_SEMITONES)      # 103
n_bins           = n_semitones * CONTOURS_BINS_PER_SEMITONE            # 309
```

```text
CQT               -> (batch, 172, 309)
NormalizedLog     -> (batch, 172, 309)
BatchNorm inicial -> (batch, 172, 309, 1)
HarmonicStacking  -> (batch, 172, 264, 8)   <-- truncado aquí
```

Los 264 bins de salida existen para que los armónicos superiores (hasta el
desplazamiento +101) dispongan de contenido. Implementar la CQT con 264 bins
directamente produce un modelo numéricamente incorrecto, con error silencioso
hasta las notas finales.

Confirmación independiente: el inicializador `model_1/cq_t2010v2_1/Sqrt` del
ONNX oficial tiene dimensiones `[309, 1, 1]`.

Los parámetros verificados de la CQT están recogidos en
`docs/teoria/fase-03-espe-tecnicas.md` (apartado 3.1.2).

## 4.2. Reproducir NNAudio CQT

La capa oficial utiliza una variante de CQT con procesamiento multirresolución y submuestreo.

La implementación deberá reproducir las operaciones de:

1. Cálculo del factor Q.
2. Determinación de las frecuencias de cada bin.
3. Construcción de kernels.
4. Generación de componentes reales e imaginarios.
5. Padding de la señal.
6. Convoluciones para calcular la transformada.
7. Obtención de la magnitud.
8. Submuestreo entre octavas cuando corresponda.
9. Ensamblado de la representación CQT final.

Los kernels CQT son parámetros deterministas generados a partir de una configuración; no deben confundirse con los pesos entrenados de la red neuronal.

Se podrán generar desde Rust o exportar como recursos calculados durante el desarrollo, siempre que la ejecución final no necesite Python.

### 4.2.1. Procedimiento recomendado: extraer los kernels del ONNX oficial

Reimplementar `create_cqt_kernels` y `scipy.signal.firwin2` en Rust es posible,
pero es la parte con mayor riesgo de perder precisión numérica. El artefacto
ONNX oficial local ya contiene los kernels horneados como constantes:

| Inicializador ONNX | Dimensiones | Contenido |
|---|---|---|
| `const_fold_opt__664` | `[36, 1, 1, 256]` | kernel CQT, parte real |
| `const_fold_opt__655` | `[36, 1, 1, 256]` | kernel CQT, parte imaginaria |
| `const_fold_opt__734` | `[1, 1, 1, 256]` | filtro lowpass `firwin2` |

La estrategia recomendada es:

1. En la Fase 1, extraer esos tres tensores y el factor de normalización
   `sqrt(lengths)` (`[309, 1, 1]`) a `models/basic-pitch/icassp-2022/cqt_kernels.safetensors`.
2. En Rust, implementar la **aritmética** de la CQT (padding reflect, conv1d
   sobre el kernel, submuestreo multioctava, magnitud, normalización) usando
   esos kernels como datos.
3. Verificar contra la salida de TensorFlow en la Fase 7.

Esto elimina la reimplementación de `firwin2` y del diseño de ventanas
scipy, y es coherente con el principio del plan de que "Python es solo una
herramienta de desarrollo": los kernels se generan una vez y se distribuyen
como asset.

Si se opta por la reimplementación completa en Rust, la equivalencia numérica
deberá verificarse explícitamente contra el ONNX antes de continuar.

La primera versión deberá priorizar la coincidencia numérica con la referencia. Las optimizaciones podrán implementarse posteriormente.

## 4.3. Implementación de NormalizedLog

Después del CQT se aplica una transformación logarítmica normalizada.

La implementación deberá reproducir el procedimiento original:

1. Obtener la potencia elevando la magnitud al cuadrado.
2. Agregar el valor de estabilidad numérica utilizado por la referencia.
3. Calcular el logaritmo en base 10.
4. Multiplicar por 10.
5. Obtener el mínimo de la representación correspondiente a cada muestra.
6. Restar ese mínimo.
7. Obtener el máximo del resultado desplazado.
8. Dividir entre ese máximo mediante una operación segura que evite divisiones inválidas.

La operación no debe sustituirse por una normalización estándar de imagen ni por una normalización de amplitud global.

La reducción debe efectuarse sobre los ejes correctos y conservar las dimensiones esperadas.

## 4.4. Batch Normalization inicial

Después de NormalizedLog se aplica Batch Normalization.

Se deberán recuperar del modelo los parámetros originales:

- `gamma`
- `beta`
- `moving_mean`
- `moving_variance`
- `epsilon`

La operación en Rust se ejecutará en modo de inferencia.

No se deberán calcular estadísticas nuevas a partir de cada audio ni actualizar
los parámetros almacenados.

### 4.4.1. Corrección: un único canal, estadísticas escalares, epsilon de Keras

La BatchNorm inicial se aplica **después** de `expand_dims`, sobre un tensor de
forma `(batch, 172, 309, 1)`:

```python
x = tf.expand_dims(x, -1)        # (batch, 172, 309, 1)
x = tfkl.BatchNormalization()(x)  # axis = -1, 1 solo canal
```

Consecuencias que deben respetarse:

- El eje de normalización es el **último** (los canales), y hay **un único
  canal**. Por tanto `gamma`, `beta`, `moving_mean` y `moving_variance` tienen
  longitud **1**: son escalares, no un vector por bin de frecuencia.
- La normalización se calcula sobre las dimensiones `(batch, time, freq)`, es
  decir, una única estadística global por muestra, **no** una por bin de
  frecuencia. Implementar un BN por bin de frecuencia daría un resultado
  completamente distinto.
- `epsilon` debe tomarse del modelo Keras: **`0.001`** (valor por defecto de
  `tf.keras.layers.BatchNormalization`). El atributo `epsilon` de
  `FusedBatchNormV3` en el ONNX oficial vale `1e-5` y **no** es el valor
  correcto para este modelo, porque el ONNX fusiona la normalización con
  operaciones distintas. La fuente canónica del epsilon es TensorFlow.

### 4.4.2. Las tres BatchNorm del modelo

| # | Posición | Canales | `gamma`/`beta`/`mean`/`var` | `epsilon` |
|---|---|---:|---|---:|
| 1 | Inicial, tras NormalizedLog | 1 | longitud 1 | 0.001 |
| 2 | Rama de contornos, tras conv (3,39) | 8 | longitud 8 | 0.001 |
| 3 | Rama de onsets, tras conv (5,5) | 32 | longitud 32 | 0.001 |

La rama de notas **no** tiene BatchNorm (solo ReLU).

## 4.5. Pruebas del frontend

Se compararán los resultados de Python y Rust para:

- CQT sin normalizar.
- NormalizedLog.
- Batch Normalization inicial.

Cada resultado se guardará con sus dimensiones, tipo de datos y métricas de error.

**Entregable de la fase:** implementación CQT y normalización funcional en Rust.

**Criterio de aceptación:** la representación que sale del frontend deberá ser suficientemente equivalente a la referencia antes de conectarla con la red neuronal.

## 4.6. Implementación en Rust

La aritmética vive en el crate `basic-pitch-cqt`:

| Fichero | Contenido |
|---|---|
| `transform.rs` | El cálculo de la CQT multirresolución. |
| `normalization.rs` | `NormalizedLog`. |
| `batch_norm.rs` | La BatchNorm inicial en modo inferencia. |

`transform()` devuelve `(batch, frames, n_bins)`, es decir `(1, 172, 309)`,
con el eje de frecuencia en **orden ascendente** (el bin 0 es la frecuencia
más grave).

Los puntos donde la implementación puede divergir de la referencia, y cómo
se han resuelto:

| Punto | Referencia | Rust |
|---|---|---|
| Relleno antes de la conv | `REFLECT` | `index_select` con índices reflejados |
| Relleno del submuestreo | ceros, `(256-1)//2` | ceros, `(256-1)//2` |
| Submuestreo | conv `VALID`, salto 2 | `conv1d` con salto 2 |
| Parte imaginaria | `-conv(kernel_imag)` | `-conv(kernel_imag)` |
| Normalización | `sqrt(lengths)` antes de la magnitud | idéntica |
| Ensamblado | concatenar y conservar los últimos 309 | `narrow` sobre el eje de frecuencia |

### 4.6.1. Una rareza de la referencia: el factor de normalización solo admite batch 1

En TensorFlow, la normalización se aplica con la forma `(309, 1, 1)`
sobre un tensor `(batch, 309, frames)`. TensorFlow alinea los ejes por la
derecha, así que la operación **solo es válida con `batch == 1`**
(para `batch > 1` exigiría `batch == 309`).

En Rust la operación se hace en NCHW, con la forma `(1, 309, 1)`, que
funciona con cualquier batch. Es un **superconjunto** del comportamiento de
la referencia: el resultado con batch 1 es idéntico.

## 4.7. Resultados de la validación

La validación se hace con `cargo run --example validar_frontend -p
basic-pitch-model`, que compara cada etapa contra los `.npy` de
`reference/reference_outputs/` (el oráculo de la Fase 0).

La **misma** entrada que vio TensorFlow: las 43.844 muestras de
`input_audio.npy`. Así el error medido es solo del frontend, sin
contaminar con las diferencias de decodificación y remuestreo entre
Symphonia y soundfile.

| Etapa | MAE | RMSE | Máx. absoluta | Máx. relativa |
|---|---:|---:|---:|---:|
| CQT | 1.76e-7 | 2.84e-7 | 5.72e-6 | 3.19e-7 |
| NormalizedLog | 1.01e-5 | 2.99e-5 | 1.38e-3 | 1.38e-3 |
| BatchNorm inicial | 2.51e-5 | 7.41e-5 | 3.42e-3 | 2.13e-3 |
| Harmonic Stacking | 1.78e-5 | 5.48e-5 | 3.42e-3 | 2.13e-3 |

### 4.7.1. La CQT coincide al nivel del epsilon de `float32`

El error máximo de la CQT es de 5.72e-6 **absolutos** sobre valores que
llegan a 17.96: un 3.2e-7 relativo, es decir, el epsilon de `float32`.

La causa es que la convolución de la CQT acumula **256 sumandos**, y
TensorFlow y Candle los acumulan en órdenes distintos. No es un
algoritmo diferente: es la misma suma en orden distinto.

### 4.7.2. Por qué el NormalizedLog amplifica el error (y por qué no es un bug)

El error del NormalizedLog (1.4e-3) es **mucho mayor** que el de la CQT
(5.7e-6). Es un efecto de la sensibilidad de la propia referencia, y se
puede explicar por completo:

`NormalizedLog` calcula `10 * log10(v² + 1e-10)` y divide por el rango
`(máx - mín)` de ese valor. En los bins donde la señal es casi
silenciosa, `v` es del orden de `5e-6`, su cuadrado (`2.6e-11`) es del
mismo orden que el estabilizador `1e-10`, y la derivada se dispara:

```text
d(salida)/d(v) = (1/max_offset) * (10/ln10) * 2v/(v² + 1e-10)
               = 2819          (con max_offset = 124.98)
```

El error del CQT es **absoluto** (no depende del valor del bin), así que
en esos bins silenciosos es enorme en términos relativos:

| Magnitud | Valor |
|---|---|
| Bin con el error máximo | frame 112, bin 80 |
| Valor del CQT ahí | 5.12e-6 |
| Error absoluto del CQT | 4.77e-7 (**9.3 %** relativo) |
| Sensibilidad `d(salida)/d(v)` | 2819 |
| Error **previsto** en la salida | 1.344e-3 |
| Error **observado** | 1.378e-3 |
| Cociente observado/previsto | **1.025** |

El modelo analítico explica el error casi exactamente. No es ruido sin
explicar: es la sensibilidad de la referencia ante una señal que roza el
silencio.

**Prueba adicional.** Aplicando `NormalizedLog` de Rust a la CQT **de la
referencia**, el error cae a 1.34e-7 (MAE 7.8e-9). Eso demuestra que la
implementación de `NormalizedLog` es correcta y que el 1.4e-3 viene
enteramente de la amplificación del ruido de la CQT.

### 4.7.3. Distribuciones del error

| Métrica | NormalizedLog |
|---|---:|
| Mediana | 2.74e-6 |
| Percentil 99 | 9.22e-5 |
| Máximo | 1.38e-3 |
| Valores con error > 1e-4 | 462 de 53.148 (0.9 %) |

El error grande es una **cola**: afecta a menos del 1 % de los valores, y
precisamente a los bins donde no hay señal.

### 4.7.4. Tolerancias fijadas

Las tolerancias de `crates/basic-pitch-model/tests/frontend_reference.rs`
son cuatro veces el error medido, y su justificación es la de este
apartado:

| Etapa | Tolerancia | Motivo |
|---|---:|---|
| CQT | 2e-5 | ruido de `float32` en la conv de 256 sumandos |
| NormalizedLog | 5e-3 | amplificación logarítmica en bins silenciosos |
| BatchNorm | 1e-2 | escala la entrada ya amplificada |
| Harmonic Stacking | 1e-2 | copia la entrada con más error acumulado |

El test de BatchNorm comprueba además que el **error medio** del
NormalizedLog se mantiene por debajo de 1e-4, que es la comprobación
estricta: si el error grande se generalizara, ese test lo detectaría.

### 4.7.5. El informe se guarda, no solo se imprime

El apartado 4.5 pide que «cada resultado se guardará con sus dimensiones,
tipo de datos y métricas de error». Imprimir por pantalla no es guardar:
el informe se perdía en el scroll del terminal.

`cargo run --example validar_frontend -p basic-pitch-model` escribe ahora
**`reference/validation_results.json`**, con una entrada por etapa:

```json
"CQT": {
  "shape_rust": [1, 172, 309],
  "shape_referencia": [1, 172, 309],
  "dtype": "float32",
  "mae": 1.762387e-7,
  "rmse": 2.834531e-7,
  "max_abs_diff": 5.722046e-6,
  "max_rel_diff": 3.186061e-7,
  "tolerancia": "2e-5"
}
```

Es la foto del estado de la validación: si alguien cambia el frontend, el
fichero deja de coincidir con lo que dice el código, y el test de
regresión lo detecta.

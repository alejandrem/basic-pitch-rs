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
HarmonicStacking  -> (batch, 172, 264, 8)   <-- truncado aqui
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

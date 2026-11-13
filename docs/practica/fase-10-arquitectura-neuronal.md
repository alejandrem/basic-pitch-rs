# Fase 6: implementación de la arquitectura neuronal en Candle

**Objetivo:** reproducir la red neuronal activa del modelo utilizando `candle-core` y `candle-nn`.

Esta fase deberá ejecutarse únicamente cuando los pesos y el frontend acústico ya hayan sido comprobados.

## 6.1. Convenciones de dimensiones

TensorFlow utiliza habitualmente una organización de datos NHWC:

```text
(batch, height, width, channels)
```

Para las convoluciones de Candle se deberá trabajar con la disposición que espera su API, normalmente NCHW:

```text
(batch, channels, height, width)
```

En este modelo:

- Height representa el eje temporal.
- Width representa el eje de frecuencia.
- Channels representa los canales de características.

Se deberán implementar las transposiciones necesarias en los límites entre módulos.

La conversión de dimensiones no debe alterar la interpretación de la información musical.

También se deberá prestar especial atención al aplanamiento de frecuencia y canales: la operación original conserva un orden específico al transformar la matriz en la salida de cada rama.

## 6.1.1. Corrección: `padding="same"` y strides por eje no existen en Candle

`candle_nn::Conv2dConfig` solo admite:

```rust
pub struct Conv2dConfig {
    pub padding: usize,   // un unico valor, simetrico, aplicado a AMBOS ejes
    pub stride: usize,    // un unico valor, aplicado a AMBOS ejes
    pub dilation: usize,
    pub groups: usize,
}
```

No hay `Padding::Same`, ni padding por eje, ni stride por eje. El modelo
original de TensorFlow usa `padding="same"` (asimétrico cuando el stride > 1)
y strides por eje `(1, 3)`. Por tanto **no se puede usar `candle_nn::Conv2d`
tal cual**.

### Padding SAME de TensorFlow: valores exactos

Para una entrada `in`, kernel `k` y stride `s`:

```text
out      = ceil(in / s)
pad_tot  = max((out - 1) * s + k - in, 0)
pad_ant  = pad_tot // 2
pad_post = pad_tot - pad_ant
```

Aplicado a este modelo:

| Capa | Kernel (t, f) | Stride (t, f) | in (t, f) | out (t, f) | pad (t, f) |
|---|---|---|---|---|---|
| contour conv | (3, 39) | (1, 1) | (172, 264) | (172, 264) | (1, 19) |
| contours-reduced | (5, 5) | (1, 1) | (172, 264) | (172, 264) | (2, 2) |
| notes conv1 | (7, 7) | **(1, 3)** | (172, 264) | (172, 88) | (3, 2) |
| notes conv2 | (7, 3) | (1, 1) | (172, 88) | (172, 88) | (3, 1) |
| onset conv1 | (5, 5) | **(1, 3)** | (172, 264) | (172, 88) | (2, 1) |
| onset conv2 | (3, 3) | (1, 1) | (172, 88) | (172, 88) | (1, 1) |

### Estrategia: padding manual + conv stride-1 + submuestreo

La estrategia aprobada es:

1. **Padding manual** del tensor de entrada con ceros, replicando exactamente
   `pad_ant` y `pad_post` por eje.
2. **Convolution con `padding = 0` y `stride = 1`** en ambos ejes.
3. **Submuestreo en el eje de frecuencia** tomando los índices `0, s, 2s, ...`
   cuando el stride de frecuencia sea 3.

La equivalencia es exacta, no aproximada. Para un stride `s` con el mismo
padding manual:

```text
y_salida[o] = sum_j w[j] * x_pad[o*s + j]
y_conv1[m]  = sum_j w[j] * x_pad[m + j]
=> y_salida[o] = y_conv1[o*s]
```

Es decir, la convolución con stride `s` es idéntica a la convolución con
stride 1 **tomando cada s-ésimo resultado**, siempre que el padding se aplique
antes y de la misma forma.

### Advertencia sobre la equivalencia numérica

Esta equivalencia es **algebraicamente exacta**, pero **no se promete
igualdad bit a bit** entre backends. TensorFlow, Candle y el motor de
convolución de Rust acumulan en órdenes distintos y pueden usar instrucciones
SIMD distintas, de modo que diferencias del orden de ~1e-6 en `float32` son
esperadas y aceptables.

La equivalencia **debe comprobarse numéricamente** en la Fase 7 con las
métricas allí definidas (MAE, RMSE, similitud coseno). No se debe declarar
"equivalente" sin esa medición, ni asumir igualdad exacta.

## 6.2. Estructura del modelo

Se recomienda definir una estructura principal:

```rust
pub struct BasicPitchModel {
    // Frontend acústico
    // Batch normalization inicial
    // Rama de contornos
    // Rama de notas
    // Rama de onsets
}
```

La construcción del modelo recibirá los pesos mediante `VarBuilder`.

Se deberán separar las estructuras internas de cada rama para mantener el código legible y permitir pruebas independientes.

## 6.3. Rama de contornos

La rama de contornos comienza con la salida de Harmonic Stacking.

### Capa 1: convolución de características

```text
Conv2D
Filtros: 8
Kernel: 3 × 39
Padding: same
```

Después:

```text
Batch Normalization
ReLU
```

Esta es la primera convolución activa de la rama de contornos.

**Distinción obligatoria:** existen dos capas con kernel `(5, 5)` en el código
original.

| Constante | Kernel | Filtros | Activación | Estado |
|---|---|---:|---|---|
| `CONTOUR_KERNEL_SIZE_1` | (5, 5) | 32 | ninguna | **COMENTADA - no implementar** |
| `CONTOUR_KERNEL_SIZE_3` | (5, 5) | 1 | sigmoid | **ACTIVA - es la capa 2 de abajo** |

Se implementa `CONTOUR_KERNEL_SIZE_3`; `CONTOUR_KERNEL_SIZE_1` queda fuera del
grafo activo.

### Capa 2: reducción de contornos

```text
Conv2D
Filtros: 1
Kernel: 5 × 5
Padding: same
Activación: sigmoid
```

El resultado se reorganizará para obtener una salida de contorno con 264 bins de frecuencia por instante.

La salida deberá conservarse porque también alimenta la rama de notas y posteriormente se utiliza para estimar pitch bend.

## 6.4. Rama de notas

La rama de notas recibe los contornos reducidos.

Primero se volverá a introducir la dimensión de canal necesaria para procesar la salida como un mapa bidimensional.

### Capa 1

```text
Conv2D
Filtros: 32
Kernel: 7 × 7
Stride: 1 × 3
Padding: same
```

Después se aplicará ReLU.

### Capa 2

```text
Conv2D
Filtros: 1
Kernel: 7 × 3
Padding: same
Activación: sigmoid
```

La reducción del eje frecuencial permitirá obtener las 88 salidas correspondientes a las notas musicales del modelo.

La salida final será:

```text
note: (batch, time, 88)
```

Internamente deberá conservarse el tensor previo que se utiliza en la concatenación con la rama de onsets.

## 6.5. Rama de onsets

Esta rama recibe directamente la representación obtenida después de Harmonic Stacking.

### Capa 1

```text
Conv2D
Filtros: 32
Kernel: 5 × 5
Stride: 1 × 3
Padding: same
```

Después:

```text
Batch Normalization
ReLU
```

### Concatenación

Se combinará la representación de la rama de onsets con el tensor de predicción previa de notas.

**Corrección obligatoria: el concatenado produce 33 canales, en un orden fijo.**

En `basic_pitch/models.py`:

```python
x_onset = tfkl.Concatenate(axis=3, name="concat")([x_notes_pre, x_onset])
```

- `x_notes_pre` es la salida **sigmoid de la conv de notas**, con forma
  `(batch, 172, 88, 1)`. **No** es la salida final `note` (que aplanada tiene
  88 valores).
- `x_onset` es la salida de la rama de onsets, `(batch, 172, 88, 32)`.
- El concatenado va sobre el **eje de canales** (eje 3 en NHWC) y el resultado
  es `(batch, 172, 88,  33)`.

El **orden es obligatorio**: primero `x_notes_pre` (canal 0), después `x_onset`
(canales 1..32). Invertirlo produce un modelo que difiere del original sin
error visible.

Confirmación independiente en el ONNX oficial: el kernel de la convolución final
de onsets es `const_fold_opt__680` con dimensiones `[1, 33, 3, 3]`.

La concatenación se realizará sobre el eje de canales, conservando las dimensiones temporales y frecuenciales.

Esta operación es especialmente sensible a errores de layout. Se deberán comprobar los tamaños y el orden de los datos antes de continuar.

### Capa final

```text
Conv2D
Filtros: 1
Kernel: 3 × 3
Padding: same
Activación: sigmoid
```

La salida final será:

```text
onset: (batch, time, 88)
```

## 6.6. Salidas del modelo

La función `forward` deberá entregar una estructura con las tres predicciones:

```rust
pub struct ModelOutput {
    pub note: Tensor,
    pub onset: Tensor,
    pub contour: Tensor,
}
```

Las tres salidas deberán mantener el orden y las dimensiones esperadas.

La predicción `note` será la representación utilizada como activación de notas en el algoritmo de detección de eventos.

No se deberá eliminar la salida de contornos aunque inicialmente solo se busque producir un MIDI básico.

## 6.7. Carga de parámetros

El módulo `weights.rs` se encargará de:

- Abrir el archivo Safetensors.
- Construir el `VarBuilder`.
- Comprobar la existencia de los tensores.
- Comprobar sus dimensiones.
- Construir las capas.
- Asociar los parámetros correctos con cada operación.
- Mantener las estadísticas de Batch Normalization.
- Detectar parámetros faltantes o incompatibles.

El programa deberá producir errores descriptivos si un tensor no coincide con el manifiesto.

## 6.8. Ejecución de inferencia

La red se ejecutará en modo de inferencia y sin cálculo de gradientes.

El primer objetivo será ejecutar correctamente una sola ventana en CPU con `float32`.

Después se podrá introducir el procesamiento por lotes de ventanas para mejorar el rendimiento.

**Entregable de la fase:** arquitectura neuronal completa en Candle, capaz de cargar los pesos y producir las tres salidas.

**Criterio de aceptación:** una ventana válida de audio deberá recorrer toda la red sin errores de dimensiones y generar salidas con formas correctas.

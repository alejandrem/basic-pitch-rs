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

No se implementará la capa `(5, 5)` comentada en el código original.

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

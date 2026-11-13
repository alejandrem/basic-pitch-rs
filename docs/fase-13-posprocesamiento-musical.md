# Fase 9: implementación del posprocesamiento musical

**Objetivo:** transformar las matrices de predicciones en notas musicales reales.

Esta etapa convierte la información neuronal en eventos que pueden representarse en MIDI.

No deberá implementarse como una simple búsqueda de valores superiores a `0.3`.

Basic Pitch utiliza un procedimiento de detección polifónica que considera las predicciones de notas, los comienzos de notas y, opcionalmente, información adicional para encontrar eventos.

## 9.1. Detección polifónica

Se implementará una versión en Rust del procedimiento equivalente a `output_to_notes_polyphonic`.

La entrada contendrá:

- Activaciones de notas.
- Activaciones de onsets.
- Umbral de onset.
- Umbral de activación de nota.
- Duración mínima.
- Restricciones de frecuencia.
- Configuración de inferencia de onsets.
- Configuración de Melodia Trick.
- Tolerancia de energía.

## 9.2. Umbrales

Se deberán conservar los parámetros predeterminados de la versión de referencia.

Los umbrales deberán poder configurarse desde una estructura de configuración, pero inicialmente utilizarán los mismos valores que el modelo original.

No se deberán asumir valores de `0.3` o `0.5` sin comprobarlos en la versión fijada.

## 9.3. Inferencia de onsets

Cuando esté habilitada, se deberá reproducir la lógica que permite detectar comienzos adicionales de notas a partir de cambios significativos en las activaciones de frames.

Esto evita depender exclusivamente de que la red produzca una activación de onset clara para cada nota.

## 9.4. Melodia Trick

Se deberá portar el procedimiento que busca notas adicionales utilizando la energía residual de las activaciones.

Deberá reproducirse el comportamiento de:

1. Identificar las activaciones restantes más altas.
2. Determinar su continuidad temporal.
3. Evaluar el umbral de frames.
4. Aplicar la tolerancia de energía.
5. Eliminar la energía correspondiente a la nota detectada.
6. Evitar confundir notas cercanas en frecuencia.

La implementación deberá conservar la lógica de eliminación de energía en las frecuencias vecinas.

## 9.5. Duración mínima

La duración mínima se convertirá desde milisegundos a frames utilizando los parámetros temporales del modelo.

No se deberá comparar directamente una duración en milisegundos con un índice de frame.

## 9.6. Restricciones de frecuencia

Se incorporará la posibilidad de limitar el rango de frecuencia de las notas detectadas.

Esto deberá hacerse utilizando la conversión correcta entre frecuencia, nota MIDI e índice de la matriz.

## 9.7. Eventos internos

Se recomienda definir un tipo de datos independiente del formato MIDI:

```rust
pub struct NoteEvent {
    pub start_time: f64,
    pub end_time: f64,
    pub pitch: u8,
    pub velocity: u8,
    pub pitch_bend: Option<Vec<i16>>,
}
```

El evento musical deberá poder utilizarse tanto para exportar MIDI como para pruebas, visualización y futuras funciones de edición.

**Entregable de la fase:** detector de notas polifónicas en Rust que reproduzca el posprocesamiento original.

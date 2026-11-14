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

## 9.8. Implementación en Rust

`crates/transcription/src/note_detection.rs` y `crates/midi-io/`.

### 9.8.1. El tipo de evento canónico

La fase pide un tipo de datos independiente del formato MIDI, y el
proyecto tiene **uno solo**:

```rust
pub struct NoteEvent {
    pub start_time: f64,
    pub end_time: f64,
    pub pitch: u8,
    pub velocity: u8,       // 0..=127
    pub amplitude: f32,     // 0..=1, la activacion original del modelo
    pub pitch_bend: Option<Vec<i16>>,
}
```

Vive en `midi-io`, no en `transcription`, y lo **reexporta** el crate de
transcripción. Es el mismo criterio que sigue el upstream: en `pretty_midi`,
el tipo de evento y el escritor de MIDI están juntos.

`amplitude` es un campo extra respecto al ejemplo de la fase: es el dato
**original** que produce el modelo, del que la velocidad es su proyección
(`round(127 * amplitud)`, como en la referencia). Guardar las dos permite
saber que una nota salió de una activación de 0.51 y no de 0.52.

Antes de este cambio había **dos** `NoteEvent` distintos (uno en
`transcription` con `amplitude` y `pitch_bends`, otro en `midi-io` con
`velocity` y `pitch_bends`), que podían divergir sin que nada lo
detectara.

### 9.8.2. La detección

`output_to_notes_polyphonic` reproduce el procedimiento del upstream:

1. **Restricción de frecuencia** (`constrain_frequency`): se ponen a cero
   las activaciones fuera del rango permitido, en notas y en onsets.
2. **Onsets inferidos** (`get_infered_onsets`): se combinan los onsets
   predichos con unos onsets deducidos de los saltos de la activación de
   nota. La diferencia entre frames consecutivos se calcula hacia atrás
   (con ceros al principio), se toma el **mínimo** de las diferencias, se
   recorta por debajo de cero y se reescala al máximo de los onsets
   predichos.
3. **Picos de onset**: un frame es un pico si es estrictamente mayor que
   sus dos vecinos (`scipy.signal.argrelmax` con `order=1`). Los picos
   que superan el umbral se recorren **de más reciente a más antiguo**.
4. **Búsqueda del final de la nota**: desde el onset, se avanza hasta que
   la activación cae por debajo del umbral durante `energy_tol` frames.
5. **Consumo de energía**: la nota consume la energía de su banda de
   frecuencia y de las vecinas, para que no se detecte dos veces.
6. **Melodia Trick**: mientras quede energía sin explicar por encima del
   umbral, se extrae la nota más fuerte que quede, con una pasada hacia
   delante y otra hacia atrás.

Las notas se detectan sobre la salida **note** y los onsets sobre la
salida **onset** (88 valores cada una). El **contour** (264 valores) no
interviene: solo se usa para el pitch bend (Fase 10) y para saber cuántos
frames hay.

### 9.8.3. Una diferencia con el upstream

El upstream divide por `max(frame_diff)` al reescalar los onsets
inferidos. Cuando ese máximo es cero (activaciones que nunca aumentan,
por ejemplo en silencio) eso produce `NaN`, que se propaga y hace que **no
se detecte ninguna nota**.

Aquí, cuando el máximo es cero, el resultado es cero: el mismo resultado
práctico (ninguna nota) sin propagar `NaN`.

### 9.8.4. El instrumento del MIDI

**Piano acústico** (`Acoustic Grand Piano`, programa GM **0**).

La referencia usa `Electric Piano 1` (GM 4). Aquí se usa el piano
acústico por decisión del proyecto.

Además, el MIDI **emite explícitamente** el cambio de programa. Sin él,
el reproductor usaría el programa por defecto del canal 0, que
justamente es `Acoustic Grand Piano`: o se emite el cambio, o el
resultado depende de lo que haga el reproductor.

### 9.8.5. Validación contra el upstream

El oráculo lo genera `tools/weight-export/reference_notes.py` con
`basic_pitch.inference.predict` **del upstream, sin modificar**.

| Audio | Notas de Rust | Notas del upstream | Desajustes |
|---|---:|---:|---|
| `vocadito_10` | 28 | 28 | 0 pitches, 0 frames de inicio, 0 de fin |
| `peso_del_silencio` (3 min) | 2158 | 2158 | 0 pitches, 0 frames de inicio, 0 de fin |

La amplitud difiere en 3.9e-6 y 2.6e-5 respectivamente: es el error de
la red (el de la CQT es de 5.7e-6 y se propaga), no un error de la
detección.

Sobre la canción completa del usuario: **el 87 % de sus notas originales**
se recupera (1716 de 1972, con 30 ms de tolerancia). El 20 % de notas
"extra" son en su mayoría armonías: el modelo oye los parciales de un
acorde de piano y los detecta como notas propias. Es el comportamiento
esperado de Basic Pitch.

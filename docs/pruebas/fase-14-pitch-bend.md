# Fase 10: estimación de pitch bend

**Objetivo:** aprovechar la salida de contornos para detectar variaciones de afinación dentro de las notas.

La salida de contornos no representa únicamente qué notas existen. También aporta información de frecuencia con mayor resolución que una matriz de 88 notas.

El modelo trabaja con tres bins de contorno por semitono.

## 10.1. Conversión de nota a índice de contorno

Se deberá implementar la conversión entre una nota MIDI y su posición correspondiente en la matriz de contornos.

La referencia utiliza la frecuencia base de 27.5 Hz y la escala logarítmica de frecuencias.

La conversión deberá ser consistente con:

- Frecuencia base.
- Número de bins por semitono.
- Número de semitonos.
- Rango de frecuencias disponible.

## 10.2. Estimación por nota

Para cada evento detectado:

1. Obtener el índice de contorno correspondiente a su nota MIDI.
2. Crear una región de búsqueda alrededor de ese índice.
3. Aplicar el peso gaussiano utilizado por la referencia.
4. Extraer la región temporal correspondiente a la duración de la nota.
5. Encontrar el bin de mayor activación por frame.
6. Convertir el desplazamiento obtenido en unidades de contorno.
7. Asociar la secuencia al evento.

Se deberán conservar los parámetros originales de tolerancia y desviación de la ventana gaussiana.

## 10.3. Conversión a MIDI

La información de contorno se transformará en eventos MIDI Pitch Bend.

Se deberá reproducir la escala original, los límites y el rango de variación.

También se tendrá que considerar el comportamiento de notas simultáneas.

El formato MIDI tradicional aplica pitch bend por canal, no de manera independiente a cada nota dentro del mismo canal. La implementación original resuelve este caso mediante una organización específica de instrumentos cuando se permiten pitch bends múltiples.

La versión Rust deberá documentar y reproducir el mismo comportamiento.

## 10.4. Pruebas

Se probarán:

- Notas sin vibrato.
- Notas con vibrato.
- Notas con deslizamiento de afinación.
- Notas con variaciones pequeñas.
- Notas con variaciones superiores al rango permitido.
- Varias notas simultáneas con pitch bend.
- Notas superpuestas.

**Entregable de la fase:** estimación de pitch bend integrada en los eventos musicales.

---

## 10.5. Implementación en Rust

### 10.5.1. Dónde vive cada parte

| Pieza | Dónde |
|---|---|
| El cálculo (la estimación del bend) | `crates/transcription/src/pitch_bend.rs` |
| La aplicación al evento musical | `crates/transcription/src/inference.rs` |
| La política del formato MIDI (canal, recorte, tiempos) | `crates/midi-io/src/writer.rs` |

La separación no es caprichosa: **el cálculo** es música y **la política del
formato** es MIDI. El upstream hace lo mismo: `get_pitch_bends` calcula, y
`note_events_to_midi` decide qué hacer con el resultado.

### 10.5.2. El cálculo

`PitchBendEstimator::estimate` reproduce `get_pitch_bends`:

1. El índice de contorno de la nota: `3 * (pitch_midi - 21)`, que es la
   simplificación exacta de `12 * 3 * log2(hz / 27.5)`.
2. La región de búsqueda: `max(bin - 25, 0)` .. `min(264, bin + 26)`.
3. La ventana gaussiana de 51 muestras con desviación 5, **recortada** por
   los dos lados:

   ```text
   gaussian[max(0, 25 - bin) : 51 - max(0, bin - (264 - 25 - 1))]
   ```

   El recorte no es un detalle: sin él, una nota muy aguda (cerca del
   final de la matriz de contornos) usaría la cola equivocada de la
   gaussiana y daría un bin desplazado.
4. Por cada frame de la nota, el bin de mayor activación ponderada. Los
   empates los gana el primero, como `np.argmax`.
5. El desplazamiento respecto al bin de la nota, en unidades de **1/3 de
   semitono**.

### 10.5.3. Las unidades, y por qué importan

Los bends se guardan en el evento musical en **unidades de 1/3 de
semitono**, que es la unidad en la que el upstream los produce
(`get_pitch_bends` devuelve "this is in units of 1/3 semitones").

La conversión a ticks de MIDI ocurre **al escribir el fichero**, que es
donde la hace el upstream (`note_events_to_midi`):

```text
ticks = round(shift * 4096 / 3)      recortados a [-8192, 8191]
```

El recorte limita los bends a unas dos semitonas.

### 10.5.4. El pitch bend es POR CANAL

El formato MIDI aplica el pitch bend a todo el canal, no nota a nota. Si
dos notas se solapan en el tiempo y quieren bends distintos, es
impossible: el último que se escribe manda.

El upstream lo resuelve **quitando los bends** de las notas que se
solapan (salvo que se pida `multiple_pitch_bends`, que asigna un canal a
cada nota).

**Y lo hace en el escritor, no en la detección.** Es un detalle
importante: la lista de notas que devuelve la detección conserva **todos**
sus bends, y es el MIDI el que los descarta. Por eso el recorte está en
`midi-io` y no en `transcription`.

El criterio es el del upstream: se ordenan las notas por
`(inicio, final, pitch)` y, para cada una, se marca sin bends a todas
las siguientes que empiezan antes de que acabe.

### 10.5.5. Los tiempos de los bends

Los bends se reparten uniformemente entre el inicio y el final de la
nota, como el `np.linspace(start_time, end_time, len(pitch_bend))` del
upstream. OJO: el **último** punto es exactamente el final, así que el
divisor es `count - 1` y no `count`. Con la division equivocada, los
tiempos se desplazaban unos 4,5 ms.

## 10.6. Validación

### 10.6.1. Los siete escenarios del apartado 10.4

`crates/transcription/tests/pitch_bend.rs` cubre los siete, contra el
upstream sin modificar:

| Escenario | Caso | Resultado |
|---|---|---|
| Notas sin vibrato | `vocadito_10` (un piano real) | 28 notas, 28 con bends, **0** de diferencia |
| Notas con vibrato | `vibrato` | 2 notas, 2 con bends, **0** |
| Notas con deslizamiento | `deslizamiento` | 5 notas, 5 con bends, **0** |
| Variaciones pequeñas | `vibrato_sutil` (5 cents) | 2 notas, 2 con bends, **0** |
| Variaciones por encima del rango | `vibrato_extremo` (120 cents) | 4 notas, 4 con bends, **0** |
| Varias notas simultáneas con bend | `acorde_vibrato` | 6 notas, 6 con bends, **0** |
| Notas superpuestas | `notas_superpuestas` | 3 notas, 3 con bends, **0** |

Se comparan el número de notas, el pitch, el inicio, el final, la
amplitud y la **secuencia completa de bends**, en unidades de 1/3 de
semitono.

### 10.6.2. El MIDI completo

Sobre la canción de 3 minutos del autor del proyecto (2158 notas):

```text
Rust     : 2158 notas,  54 bends
upstream : 2158 notas,  54 bends
bends identicos: True
notas identicas: True
```

Las 2158 notas (inicio, final, pitch y velocidad) y los 54 pitch bends
(tiempo y valor) coinciden **exactamente**.

# Estado de P3 y P4 (Fases 6 a 9)

Estado del proyecto tras las Fases 3 a 9: el frontend acústico, la red
neuronal, la reconstrucción temporal y la detección de notas.

---

## Qué está hecho

| Fase | Contenido | Validado contra |
|---|---|---|
| 3 · Procesamiento de audio | Decodificación (MP3/WAV/FLAC/OGG/M4A), mono por promedio, remuestreo `soxr` HQ, ventaneo | `soundfile`+`soxr_hq` del upstream: 2.7e-8 |
| 4 · Frontend CQT | CQT multirresolución, `NormalizedLog`, BatchNorm inicial | TensorFlow: CQT a 5.7e-6 (epsilon de `float32`) |
| 5 · Harmonic Stacking | Los 8 desplazamientos, con relleno asimétrico | TensorFlow: 3.4e-3 |
| 6 · Red neuronal | Las 6 capas activas, `padding="same"` emulado, concat de 33 canales | TensorFlow: 4.4e-5 / 1.7e-5 / 5.3e-5 |
| 7 · Validación | Los 7 puntos de inspección de la red, con su traza | TensorFlow, punto a punto |
| 8 · Duración arbitraria | Reconstrucción temporal, flujo por lotes | `unwrap_output` del upstream, de extremo a extremo |
| 9 · Posprocesamiento | Detección polifónica, onsets inferidos, Melodia Trick | `output_to_notes_polyphonic` del upstream: **0 desajustes** |

La cadena completa, de un archivo de audio a un MIDI:

```bash
cargo run --release -- transcribe cancion.mp3 -o cancion.mid
```

Medido con una canción de 3 minutos: **10 segundos** de proceso
(0.06× el tiempo del audio) y 2158 notas detectadas.

---

## Resultados de la validación

### La red (Fase 7)

| Punto | max abs | MAE |
|---|---:|---:|
| contour_conv | 1.92e-3 | 4.97e-5 |
| contour_reduced | 4.43e-5 | 1.33e-6 |
| contour | 4.43e-5 | 1.33e-6 |
| notes_features | 6.86e-5 | 2.06e-6 |
| note | 1.66e-5 | 5.86e-7 |
| onset_features | 2.04e-3 | 3.66e-5 |
| onset | 5.34e-5 | 2.65e-6 |

Los puntos pre-activación tienen más error absoluto porque su rango
dinámico es mucho mayor (el `contour_conv` va de -2.7 a 4.6); el
BatchNorm de la capa siguiente lo comprime.

### La reconstrucción (Fase 8)

Contra `basic_pitch.inference.unwrap_output` **del upstream**:

| Audio | Etapa | max abs | MAE |
|---|---|---:|---:|
| `vocadito_10` | note | 1.45e-5 | 2.27e-7 |
| `vocadito_10` | onset | 6.49e-5 | 1.21e-6 |
| `vocadito_10` | contour | 2.95e-5 | 4.81e-7 |

### La detección de notas (Fase 9)

| Audio | Notas de Rust | Notas del upstream | Desajustes |
|---|---:|---:|---|
| `vocadito_10` | 28 | 28 | 0 pitches, 0 frames de inicio, 0 de fin |
| `peso_del_silencio` | 2158 | 2158 | 0 pitches, 0 frames de inicio, 0 de fin |

La amplitud difiere en ~1e-5: es el error de la red, no de la detección.

---

## Reproducirlo

```bash
# La validacion del frontend.
cargo run --release --example validar_frontend -p basic-pitch-model

# La validacion de la red, con la traza.
cargo test -p basic-pitch-model --test model_trace

# La reconstruccion, con sus siete escenarios.
cargo test -p basic-pitch-transcription --test reconstruction

# La deteccion de notas, contra el upstream.
cargo test --release -p basic-pitch-transcription --test notes_reference

# Todo.
cargo test --workspace
cargo clippy --workspace --all-targets
cargo fmt --all -- --check
```

### Regenerar los oráculos

```bash
# La reconstruccion del upstream.
.venv/bin/python tools/weight-export/reference_reconstruction.py

# Las notas del upstream.
.venv/bin/python tools/weight-export/reference_notes.py <audio>...

# Un MIDI a audio, para transcribirlo.
.venv/bin/python tools/weight-export/render_midi.py entrada.mid salida.wav
```

---

## Estado del repositorio

```text
tests:   168 en verde
clippy:  0 avisos
fmt:     sin diferencias
```

---

## Decisiones que quedan fijadas

1. **El `NoteEvent` es un tipo solo**, en `midi-io`, reexportado por
   `transcription`. Antes había dos, que podían divergir.

2. **El MIDI usa piano acústico** (programa GM 0), que es lo que pedía el
   autor del proyecto y lo que usaba su MIDI de origen. La referencia
   usa `Electric Piano 1` (GM 4). El cambio de programa se emite
   **explícitamente**: sin él, el resultado dependería del reproductor.

3. **El pitch bend no se calcula todavía** (es la Fase 10). El
   `NoteEvent` ya tiene el campo `pitch_bend`, y la red ya expone el
   `contour` que hace falta para estimarlo.

4. **La comparación con el upstream es siempre nota a nota**, no por
   conjuntos: es lo que permite localizar un desajuste en vez de
   perderse en un "coinciden en un 95 %".

5. **El material personal del usuario no se versiona**: ni el MIDI
   original ni el audio renderizado (30 MB).

---

## Lo que queda

| Fase | Contenido |
|---|---|
| 10 · Pitch bend | Estimar el bend a partir del contour y escribirlo en el MIDI |
| 11 · MIDI | Pulir el escritor (pitch bend por canal, tempo, metadatos) |
| 12 · API | Biblioteca pública, configuración por comando, API de Rust |
| 13 · Tests | Suite de integración, regresión y CI |
| 14 · Rendimiento | Benchmarks, backends |
| 15 · Documentación | README definitivo, procedencia, licencias |

La Fase 10 es la siguiente natural: el `contour` ya está disponible y el
campo `pitch_bend` del evento ya existe.

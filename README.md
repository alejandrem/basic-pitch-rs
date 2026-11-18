# Basic Pitch RS

**Implementación nativa en Rust de Spotify Basic Pitch para la transcripción de audio a MIDI.**

> [!WARNING]
> Este proyecto se encuentra en desarrollo activo. La implementación del modelo y del procesamiento de audio está en proceso de desarrollo y validación frente a la implementación original.

## Descripción

Basic Pitch RS busca llevar las capacidades de [Basic Pitch](https://github.com/spotify/basic-pitch) al ecosistema de Rust.

El proyecto consiste en implementar directamente en Rust el procesamiento de audio, la arquitectura de la red neuronal, la inferencia y el procesamiento posterior necesario para convertir audio en archivos MIDI.

El objetivo es utilizar los pesos preentrenados originales del modelo, sin modificarlos, realizar ajustes (*fine-tuning*) ni volver a entrenarlo.

## Objetivos

- Implementar la arquitectura de la red neuronal de Basic Pitch en Rust.
- Ejecutar la inferencia mediante [Candle](https://github.com/huggingface/candle).
- Reproducir el procesamiento de audio de la implementación original.
- Convertir archivos de audio en transcripciones MIDI.
- Validar la equivalencia numérica con los resultados de referencia.
- Desarrollar un núcleo reutilizable en Rust que pueda servir como base para futuras aplicaciones.

## Arquitectura del proyecto

El proyecto está organizado como un espacio de trabajo de Rust (*Rust Workspace*), dividido en módulos independientes:

```text

basic-pitch-rs/
├── Cargo.toml
├── LICENSE
├── .gitignore
├── docs/
│   └── estado-p1.md
├── reference/
│   ├── reference_metadata.json
│   └── reference_outputs/
│       └── onset.npy
├── src/
│   └── main.rs
└── crates/
    ├── audio-io/
    │   ├── Cargo.toml
    │   └── src/
    │       └── lib.rs
    ├── audio-preprocess/
    │   ├── Cargo.toml
    │   └── src/
    │       ├── lib.rs
    │       ├── resample.rs
    │       └── windowing.rs
    ├── cqt/
    │   ├── Cargo.toml
    │   └── src/
    │       └── lib.rs
    ├── basic-pitch-model/
    │   ├── Cargo.toml
    │   └── src/
    │       ├── lib.rs
    │       ├── harmonic_stacking.rs
    │       ├── layers.rs
    │       └── model.rs
    ├── transcription/
    │   ├── Cargo.toml
    │   └── src/
    │       └── lib.rs
    └── midi-io/
        ├── Cargo.toml
        └── src/
            └── lib.rs
```

### Descripción de los módulos

| Módulo | Responsabilidad |
|---|---|
| `audio-io` | Decodificación y lectura de archivos de audio. |
| `audio-preprocess` | Conversión a mono, remuestreo y segmentación de audio. |
| `cqt` | Transformada Q Constante y operaciones del frontend. |
| `basic-pitch-model` | Arquitectura de la red neuronal e inferencia. |
| `transcription` | Coordinación del proceso de transcripción. |
| `midi-io` | Generación y exportación de archivos MIDI. |

## Tecnologías utilizadas

- **Rust:** lenguaje principal del proyecto.
- **Candle:** ejecución de inferencia de redes neuronales.
- **Safetensors:** almacenamiento de los pesos del modelo.
- **Symphonia:** decodificación de archivos de audio.
- **SOXR:** remuestreo de señales de audio.
- **Midly:** generación de archivos MIDI.

## Flujo de procesamiento

El proceso de transcripción busca reproducir las etapas de la implementación original de Basic Pitch:

```text
Archivo de audio
       │
       ▼
Decodificación de audio
       │
       ▼
Conversión a mono y remuestreo
       │
       ▼
Segmentación en ventanas
       │
       ▼
Transformada Q Constante (CQT)
       │
       ▼
Normalización y Harmonic Stacking
       │
       ▼
Red neuronal Basic Pitch
       │
       ├── Predicción de notas
       ├── Predicción de ataques (Onsets)
       └── Predicción de contornos (Contours)
       │
       ▼
Extracción de eventos musicales
       │
       ▼
Generación del archivo MIDI
       │
       ▼
Archivo MIDI de salida
```

## Uso

### Transcribir un archivo de audio a MIDI

```bash
cargo run --release -- transcribe cancion.mp3 -o cancion.mid
```

El comando acepta **WAV, MP3, FLAC, OGG y M4A** (los que decodifica
Symphonia). Si no se indica `-o`, el MIDI se escribe junto al audio de
entrada.

```text
$ basic-pitch transcribe amaris.mp3 -o amaris.mid
amaris.mp3 -> amaris.mid  (1842 notas, 163.71 s de audio)
```

El MIDI usa **piano acústico** (`Acoustic Grand Piano`, programa GM 0) y
suena a 120 BPM.

### Convertir un MIDI a audio (para oírlo)

Un archivo `.mid` **no es audio**: es una lista de notas. Para poder
oír una transcripción hay que renderizarla a audio, que es un proceso
inverso y no forma parte de la transcripción.

```bash
fluidsynth -F cancion.wav -r 44100 -g 0.7 \
    .venv/lib/python3.11/site-packages/pretty_midi/TimGM6mb.sf2 cancion.mid
```

También está disponible como script, que busca el soundfont por su sitio:

```bash
.venv/bin/python tools/weight-export/render_midi.py cancion.mid cancion.wav
```

### Los dos sentidos

```text
  audio real  --transcribe-->  MIDI  --render_midi-->  audio (para oír)
```

Lo segundo es un **renderizado**, no una transcripción: el MIDI no tiene
sonido hasta que un sintetizador lo convierte en audio. El soundfont
`TimGM6mb` que se usa es el mismo que trae `pretty_midi`, con el que el
upstream de Spotify evalúa sus resultados.

### Ajustar el resultado

```bash
# Sin el truco de Melodia: solo notas con onset claro, sin las armonias
# del timbre. La diferencia es pequena: las armonias las predice la red.
basic-pitch transcribe cancion.mp3 -o limpio.mid --no-melodia

# Subiendo el umbral de activacion de nota (0.3 por defecto) se
# eliminan las armonias mas debles, pero tambien algunas notas reales.
basic-pitch transcribe cancion.mp3 -o limpio.mid --frame-threshold 0.4
```

### Comprobar la fidelidad

El motor se compara **etapa a etapa** con la implementación original: las
once etapas del pipeline (las cuatro del frontend y las siete de la red),
con seis métricas (MAE, MSE, RMSE, máximo, diferencia relativa y
similitud coseno).

```bash
# El validador: mide las once etapas y escribe el informe.
cargo run --release --example validar_model -p basic-pitch-model
# -> reference/validation_results.json

# Los 13 casos de prueba de la Fase 7 (señal controlada contra el upstream).
cargo test --release -p basic-pitch-transcription --test casos -- --ignored
```

Las notas, nota a nota contra el upstream:

```bash
.venv/bin/python tools/weight-export/reference_notes.py cancion.mp3
cargo test --release -p basic-pitch-transcription --test notes_reference
```

## Compatibilidad con el modelo original

Uno de los principales objetivos es mantener la compatibilidad con la implementación original de Basic Pitch.

Se busca conservar:

- La arquitectura original de la red neuronal.
- Los pesos preentrenados originales.
- Los parámetros del procesamiento de audio.
- El comportamiento del frontend CQT.
- Las operaciones de Harmonic Stacking.
- El procesamiento posterior de las predicciones.
- La extracción de eventos musicales y generación MIDI.

Para comprobar la fidelidad de la implementación se realizan comparaciones numéricas entre los resultados de Rust y los resultados de referencia de la implementación original.

## API de biblioteca

El motor se puede usar como crate de Rust sin tocar el CLI:

```rust
use basic_pitch_transcription::Transcriber;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let transcriptor = Transcriber::load(Path::new("models/basic-pitch/icassp-2022"))?;
    let resultado = transcriptor.transcribe("cancion.mp3")?;

    println!("{} notas, {:.2} s", resultado.notes.len(), resultado.duration_seconds);

    // MIDI ya generado en memoria.
    std::fs::write("cancion.mid", &resultado.midi_data)?;
    Ok(())
}
```

`Transcriber::load` carga el modelo una sola vez y se reutiliza para varios
archivos. La API admite `load_with_device(...)` para compatibilidad futura;
la implementacion actual es CPU-only. Los errores son tipados
(`TranscriptionError`) y cubren audio, remuestreo, modelo, frontend, Candle,
reconstruccion, E/S y generacion MIDI.

## Rendimiento

Linea base medida con `cargo bench -p basic-pitch-transcription` sobre
`vocadito_10.wav` (9.1 s de audio), en CPU, modo release:

| Etapa | Tiempo |
|---|---:|
| Decodificacion | ~1.06 ms |
| Remuestreo | ~1.09 ms |
| Ventanas + frontend + red neuronal | ~584 ms |
| Pipeline completo | ~585 ms |

El cuello de botella es la red neuronal y el frontend acustico, no el audio.
La medicion es reproducible con `cargo bench`; los resultados detallados
quedan en `target/criterion/`.

## Procedencia del modelo

- Modelo: `basic-pitch` / `icassp-2022` (TensorFlow SavedModel).
- Repositorio de referencia: https://github.com/spotify/basic-pitch (commit `fa5997af0a8210982619003269994a1be25eddf3`, paquete 0.4.0).
- Artefacto ONNX: `basic_pitch/saved_models/icassp_2022/nmp.onnx`
  (sha256 `2c3c1d144bfa61ad236e92e169c13535c880469a12a047d4e73451f2c059a0ec`).
- Pesos en `models/basic-pitch/icassp-2022/` con SHA-256 publicados en
  `SHA256SUMS`. Las transformaciones aplicadas estan en `source_metadata.json`.
- Precision: `float32`, sin cuantizacion ni redondeo.

## Estado del desarrollo

El proyecto se desarrolla de manera incremental.

| Componente | Estado | Validado contra |
|---|---|---|
| Estructura del proyecto Rust | Completo | - |
| Decodificación de audio (5 formatos) | Completo | `soundfile` del upstream: 2.7e-8 |
| Preprocesamiento de audio | Completo | ídem |
| Frontend CQT | Completo | TensorFlow: 5.7e-6 (epsilon de `float32`) |
| Harmonic Stacking | Completo | TensorFlow: 3.4e-3 |
| Arquitectura de la red neuronal | Completo | TensorFlow: 1.7e-5 (nota), 5.3e-5 (onset) |
| Reconstrucción temporal | Completo | `unwrap_output` del upstream, de extremo a extremo |
| Detección polifónica de notas | Completo | `output_to_notes_polyphonic` del upstream: **0 desajustes** |
| Escritura de MIDI | Completo | - |
| Transcripción de extremo a extremo | Completo | - |
| Pitch bend | Completo | `get_pitch_bends` del upstream: 0 ticks de diferencia |
| API pública | Pendiente | - |

## Visión a futuro

Basic Pitch RS está pensado como la base de una aplicación completa de transcripción y edición musical.

Entre las posibles funcionalidades futuras se encuentran:

- Editor MIDI mediante un piano roll.
- Editor de partituras musicales.
- Reproducción de audio y MIDI.
- Exportación a WAV, MP3 y otros formatos.
- Aplicación de escritorio mediante Tauri.
- Integración futura de herramientas de generación musical mediante inteligencia artificial.

Estas funcionalidades forman parte de la visión a largo plazo y no necesariamente están disponibles en la implementación actual.

## Reconocimientos

Este proyecto toma como referencia el trabajo de investigación y desarrollo de [Spotify Basic Pitch](https://github.com/spotify/basic-pitch).

El reconocimiento del modelo original, su investigación y sus pesos preentrenados corresponde a sus respectivos autores.

## Licencia

Consulta el archivo [LICENSE](LICENSE) para conocer los términos de licencia de este repositorio.

El proyecto original Basic Pitch y los recursos del modelo conservan sus respectivas condiciones de licencia y atribución.
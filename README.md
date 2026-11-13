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

## Estado del desarrollo

El proyecto se desarrolla de manera incremental.

| Componente | Estado |
|---|---|
| Estructura del proyecto Rust | En desarrollo |
| Decodificación de audio | En desarrollo |
| Preprocesamiento de audio | En desarrollo |
| Frontend CQT | En desarrollo |
| Harmonic Stacking | En desarrollo |
| Arquitectura de la red neuronal | En desarrollo |
| Inferencia del modelo | Pendiente de validación |
| Procesamiento posterior de MIDI | Pendiente |
| Transcripción de extremo a extremo | Pendiente |

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
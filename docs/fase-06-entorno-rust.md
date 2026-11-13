 6. Fase 2: preparación del entorno Rust

**Objetivo:** preparar un espacio de trabajo modular y establecer las dependencias del proyecto.

## 6.1. Crear el workspace

Se recomienda organizar el proyecto mediante un Cargo Workspace:

```text
basic-pitch-rs/
│
├── Cargo.toml
├── Cargo.lock
├── README.md
├── LICENSE
├── NOTICE
│
├── crates/
│   │
│   ├── audio-io/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── decoder.rs
│   │       └── formats.rs
│   │
│   ├── audio-preprocess/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── mono.rs
│   │       ├── resample.rs
│   │       └── windowing.rs
│   │
│   ├── cqt/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── kernels.rs
│   │       ├── downsample.rs
│   │       ├── transform.rs
│   │       └── normalization.rs
│   │
│   ├── basic-pitch-model/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── model.rs
│   │       ├── contour.rs
│   │       ├── notes.rs
│   │       ├── onsets.rs
│   │       ├── harmonic_stacking.rs
│   │       └── weights.rs
│   │
│   ├── transcription/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── inference.rs
│   │       ├── stitching.rs
│   │       ├── note_detection.rs
│   │       └── pitch_bend.rs
│   │
│   └── midi-io/
│       ├── Cargo.toml
│       └── src/
│           ├── lib.rs
│           ├── events.rs
│           └── writer.rs
│
├── tools/
│   └── weight-export/
│       ├── inspect_model.py
│       ├── export_weights.py
│       └── validate_weights.py
│
├── models/
│   └── basic-pitch/
│       └── icassp-2022/
│
├── tests/
│   ├── fixtures/
│   ├── audio/
│   ├── model/
│   ├── transcription/
│   └── midi/
│
└── examples/
    └── transcribe.rs
```

Cada crate tendrá una responsabilidad específica. No se deberá colocar todo el procesamiento en un único archivo `main.rs`.

## 6.2. Dependencias principales

Se evaluarán e incorporarán las siguientes bibliotecas:

| Biblioteca | Función |
|---|---|
| `candle-core` | Operaciones con tensores |
| `candle-nn` | Capas y operaciones neuronales |
| `safetensors` | Herramientas de lectura y escritura del formato de pesos, cuando sea necesario |
| `symphonia` | Decodificación de formatos de audio |
| `rubato` | Remuestreo de audio |
| `midly` | Lectura y escritura MIDI |
| `serde` | Serialización de configuraciones |
| `serde_json` | Configuración y manifiestos JSON |
| `anyhow` | Manejo contextual de errores en herramientas y aplicaciones |
| `thiserror` | Definición de errores de las bibliotecas |
| `tracing` | Registro de eventos y diagnóstico |
| `tempfile` | Pruebas con archivos temporales |

Las versiones se fijarán en `Cargo.toml` y `Cargo.lock` después de comprobar la compatibilidad entre ellas.

No se deberá fijar una versión antigua de Candle simplemente porque aparezca en un ejemplo. Se seleccionará una versión estable cuyas APIs, dependencias y características de backend hayan sido verificadas.

## 6.3. Configuración del dispositivo

La primera versión utilizará CPU y tensores `float32`.

El objetivo es tener una referencia reproducible que no dependa de GPU ni de diferencias entre backends.

Posteriormente podrán evaluarse CUDA u otros backends compatibles con Candle.

**Entregable de la fase:** workspace compilable, dependencias fijadas y estructura modular creada.

---

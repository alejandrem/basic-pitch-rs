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
│       ├── validate_weights.py
│       └── tensor_provenance.py
│
├── reference/                      <-- artefactos de auditoria (Fase 0)
│   ├── model_inventory.json
│   ├── model_summary.txt
│   ├── model_config.json
│   ├── tensor_provenance.json
│   ├── reference_metadata.json
│   └── reference_outputs/
│
├── models/                         <-- DIRECTORIO CANONICO del modelo
│   └── basic-pitch/
│       └── icassp-2022/
│           ├── model.safetensors
│           ├── model_config.json
│           ├── tensor_inventory.json
│           ├── tensor_provenance.json
│           ├── source_metadata.json
│           ├── cqt_kernels.safetensors
│           └── SHA256SUMS
│
├── .venv/                          <-- entorno Python de referencia (uv)
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

**Regla de directorios canónicos.** Cada tipo de artefacto tiene exactamente
una ubicación:

| Artefacto | Ubicación única | Naturaleza |
|---|---|---|
| Modelo desplegable y sus assets | `models/basic-pitch/icassp-2022/` | distribución |
| Artefactos de auditoría y referencias | `reference/` | desarrollo |
| Scripts de extracción | `tools/weight-export/` | desarrollo |
| Entorno Python de referencia | `.venv/` | desarrollo (ignorada por git) |
| Código Rust | `crates/` | producción |

No se crearán rutas alternativas ni copias. En particular, `model-assets/` no
existe: el nombre canónico es `models/`.

Cada crate tendrá una responsabilidad específica. No se deberá colocar todo el procesamiento en un único archivo `main.rs`.

## 6.2. Dependencias principales

Se evaluarán e incorporarán las siguientes bibliotecas:

| Biblioteca | Función |
|---|---|
| `candle-core` | Operaciones con tensores |
| `candle-nn` | Capas y operaciones neuronales |
| `safetensors` | Herramientas de lectura y escritura del formato de pesos, cuando sea necesario |
| `symphonia` | Decodificación de formatos de audio |
| `soxr` (con FFI a `libsoxr`) | Remuestreo de audio, equivalente a `soxr_hq` de librosa |
| `midly` | Lectura y escritura MIDI |
| `serde` | Serialización de configuraciones |
| `serde_json` | Configuración y manifiestos JSON |
| `anyhow` | Manejo contextual de errores en herramientas y aplicaciones |
| `thiserror` | Definición de errores de las bibliotecas |
| `tracing` | Registro de eventos y diagnóstico |
| `tempfile` | Pruebas con archivos temporales |

Nota sobre el remuestreo: `rubato` queda **descartado** por no ser
bit-idéntico a `soxr_hq`, que es lo que usa la referencia (librosa 0.10.2).
Ver `docs/practica/fase-07-procesamiento-audio.md`, apartado 3.3.1.

Las versiones se fijarán en `Cargo.toml` y `Cargo.lock` después de comprobar la compatibilidad entre ellas.

No se deberá fijar una versión antigua de Candle simplemente porque aparezca en un ejemplo. Se seleccionará una versión estable cuyas APIs, dependencias y características de backend hayan sido verificadas.

## 6.2.1. Dependencias de sistema

El crate `soxr` enlaza `libsoxr` **dinamicamente**, resuelto en tiempo de
compilacion con `pkg-config`. Make falta la biblioteca de desarrollo:

```bash
# Debian / Ubuntu
sudo apt-get install libsoxr0-dev

# Fedora
sudo dnf install soxr-devel

# macOS
brew install soxr
```

Sin las cabeceras, la compilacion falla con un mensaje explicito del
`build.rs` de `libsoxr-sys`. El enlazado dinamico (y no estatico) es
deliberado: es el supuesto que evita las obligaciones de la LGPL.

El decodificador de audio (`symphonia`) es Rust puro y no necesita
librerias del sistema: en particular, este port **no** enlaza `libsndfile`
(a diferencia de la referencia en Python, que usa `soundfile`).

## 6.3. Configuración del dispositivo

La primera versión utilizará CPU y tensores `float32`.

El objetivo es tener una referencia reproducible que no dependa de GPU ni de diferencias entre backends.

Posteriormente podrán evaluarse CUDA u otros backends compatibles con Candle.

**Entregable de la fase:** workspace compilable, dependencias fijadas y estructura modular creada.

---

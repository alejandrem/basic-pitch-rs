<div align="center">

# 🎵 Basic Pitch RS

**Implementación nativa en Rust de Spotify Basic Pitch para la transcripción de audio a MIDI.**

[![CI](https://github.com/alejandrem/basic-pitch-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/alejandrem/basic-pitch-rs/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/alejandrem/basic-pitch-rs?include_prereleases)](https://github.com/alejandrem/basic-pitch-rs/releases)
[![Rust](https://img.shields.io/badge/rust-1.82%2B-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

> ⚠️ **Proyecto en desarrollo activo.** El motor acústico y neuronal está validado
> contra la implementación original; la API pública y el empaquetado siguen
> evolucionando.

</div>

---

## ✨ Descripción

Basic Pitch RS lleva las capacidades de
[Basic Pitch](https://github.com/spotify/basic-pitch) al ecosistema de Rust:
decodificación de audio, frontend CQT, red neuronal (Candle), reconstrucción
temporal, detección polifónica de notas, pitch bend y exportación MIDI,
todo sin Python en tiempo de ejecución.

El objetivo es usar los pesos preentrenados originales **sin modificarlos**,
ni hacer fine-tuning ni reentrenarlos.

---

## 🎯 Objetivos

- Implementar la arquitectura de la red neuronal de Basic Pitch en Rust.
- Ejecutar la inferencia con [Candle](https://github.com/huggingface/candle).
- Reproducir el procesamiento de audio de la implementación original.
- Convertir archivos de audio en transcripciones MIDI.
- Validar la equivalencia numérica etapa a etapa contra la referencia.
- Desarrollar un núcleo reutilizable que sirva de base para futuras aplicaciones.

---

## 🏗️ Arquitectura

```text
basic-pitch-rs/
├── Cargo.toml
├── src/                  # CLI
├── crates/
│   ├── audio-io/
│   ├── audio-preprocess/
│   ├── cqt/
│   ├── basic-pitch-model/
│   ├── transcription/
│   ├── midi-io/
│   ├── reference-npy/
│   └── validation/
├── models/basic-pitch/icassp-2022/
├── docs/
└── .github/workflows/ci.yml
```

| Módulo | Responsabilidad |
|---|---|
| `audio-io` | Decodificación de audio (WAV, MP3, FLAC, OGG, M4A). |
| `audio-preprocess` | Mono por promedio, remuestreo `soxr` HQ, ventaneo. |
| `cqt` | Transformada Q constante y operaciones del frontend. |
| `basic-pitch-model` | Arquitectura neuronal e inferencia. |
| `transcription` | Orquestación del pipeline completo. |
| `midi-io` | Eventos musicales y escritura MIDI. |

---

## 📦 Requisitos

- **Rust 1.82+**.
- **libsoxr**, como librería del sistema con cabeceras y `pkg-config`. El remuestreo
  usa `soxr` para reproducir el `soxr_hq` de librosa; sin la librería,
  `cargo build` falla al compilar `libsoxr-sys`.

| Sistema | Instalación |
|---|---|
| Arch / CachyOS | `sudo pacman -S soxr pkgconf` |
| Debian / Ubuntu | `sudo apt install libsoxr-dev pkg-config` |
| macOS (Homebrew) | `brew install libsoxr pkg-config` |

> `libsoxr` es LGPL-2.1+ y se enlaza de forma **dinámica**. Si algún día
> distribuyes un binario (por ejemplo, empaquetado con Tauri), tendrás que
> incluir la librería y cumplir los términos de la LGPL.

Python **no** hace falta para ejecutar el programa; solo para regenerar y
auditar los pesos (ver más abajo).

---

## 🚀 Uso rápido

### Transcribir audio a MIDI

```bash
cargo run --release -- transcribe cancion.mp3 -o cancion.mid
```

Si no se indica `-o`, el MIDI se escribe junto al audio de entrada.

```text
$ basic-pitch transcribe amaris.mp3 -o amaris.mid
amaris.mp3 -> amaris.mid  (1842 notas, 163.71 s de audio)
```

El MIDI usa **piano acústico** (`Acoustic Grand Piano`, programa GM 0) a 120 BPM.

### Opciones

```bash
# Sin el truco de Melodia: solo notas con onset claro.
basic-pitch transcribe cancion.mp3 -o limpio.mid --no-melodia

# Subir el umbral de activacion de nota (0.3 por defecto).
basic-pitch transcribe cancion.mp3 -o limpio.mid --frame-threshold 0.4
```

### Convertir MIDI a audio (para oirlo)

```bash
.venv/bin/python tools/weight-export/render_midi.py cancion.mid cancion.wav
```

---

## 📚 API de biblioteca

```rust
use basic_pitch_transcription::Transcriber;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let transcriptor = Transcriber::load(Path::new("models/basic-pitch/icassp-2022"))?;
    let resultado = transcriptor.transcribe("cancion.mp3")?;

    println!("{} notas, {:.2} s", resultado.notes.len(), resultado.duration_seconds);
    std::fs::write("cancion.mid", &resultado.midi_data)?;
    Ok(())
}
```

- `Transcriber::load` carga el modelo **una sola vez** y se reutiliza para varios archivos.
- `load_with_device(...)` mantiene la firma estable; la implementación actual es CPU-only y devuelve `TranscriptionError::UnsupportedDevice` si se pide otro dispositivo.
- `TranscriptionResult` expone `notes`, `midi_data` y `duration_seconds`.
- Errores tipados (`TranscriptionError`): audio, remuestreo, modelo, frontend, Candle, reconstrucción, E/S y MIDI.

---

## ⚡ Rendimiento

Línea base medida con `cargo bench -p basic-pitch-transcription` sobre
`vocadito_10.wav` (9.1 s de audio), CPU, modo release:

| Etapa | Tiempo (mediana) |
|---|---:|
| Decodificación | ~1.06 ms |
| Remuestreo | ~1.09 ms |
| Ventanas + frontend + red neuronal | ~584 ms |
| Pipeline completo | ~585 ms |

El cuello de botella es el frontend acústico y la red neuronal, no el audio.
Detalle completo en [`docs/baseline-rendimiento.md`](docs/baseline-rendimiento.md).

---

## 🔬 Validación y fidelidad

El motor se compara **etapa a etapa** con el upstream, con seis métricas
(MAE, MSE, RMSE, máximo, diferencia relativa y similitud coseno).

```bash
# Validador del modelo (11 etapas).
cargo run --release --example validar_model -p basic-pitch-model

# 13 casos de prueba de la Fase 7.
cargo test --release -p basic-pitch-transcription --test casos -- --ignored

# Notas nota a nota contra el upstream.
.venv/bin/python tools/weight-export/reference_notes.py cancion.mp3
cargo test --release -p basic-pitch-transcription --test notes_reference
```

---

## 🧬 Procedencia del modelo

- **Modelo:** `basic-pitch` / `icassp-2022` (TensorFlow SavedModel).
- **Referencia:** https://github.com/spotify/basic-pitch
  (commit `fa5997af0a8210982619003269994a1be25eddf3`, paquete 0.4.0).
- **Artefacto ONNX:** `basic_pitch/saved_models/icassp_2022/nmp.onnx`
  (`sha256: 2c3c1d144bfa61ad236e92e169c13535c880469a12a047d4e73451f2c059a0ec`).
- **Pesos:** `models/basic-pitch/icassp-2022/` con SHA-256 en `SHA256SUMS`.
- **Transformaciones:** ver `source_metadata.json` (kernels permutados, CQT extraído del ONNX, sin cuantización).
- **Precisión:** `float32`, sin cuantización ni redondeo.
- **BatchNorm:** estadísticas `moving_mean`/`moving_variance` provienen de TensorFlow (el ONNX las tiene fusionadas).

> **Los pesos de este repositorio no son un artefacto original del proyecto.**
> Son los pesos preentrenados de Spotify, convertidos de TensorFlow a
> Safetensors: los kernels se permutan de `[kh, kw, in, out]` a
> `[out, in, kh, kw]` sin modificar ningún valor. El modelo sigue siendo
> trabajo de Spotify y se rige por la licencia Apache-2.0 del upstream
> (ver `NOTICE`).

Para comprobar la integridad de los ficheros que hay en el repositorio:

```bash
(cd models/basic-pitch/icassp-2022 && sha256sum -c SHA256SUMS)
```

### Regenerar los pesos

Solo hace falta para auditar la conversión; el programa Rust no depende de Python.

```bash
# 1. Clon del upstream en el commit exacto, donde lo esperan los scripts
#    (la carpeta basic-pitch/ está ignorada por git).
git clone https://github.com/spotify/basic-pitch basic-pitch
git -C basic-pitch checkout fa5997af0a8210982619003269994a1be25eddf3

# 2. Entorno Python fijado (ver requirements-ref.txt).
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements-ref.txt

# 3. Exportar a models/basic-pitch/icassp-2022/ y validar contra TensorFlow y ONNX.
.venv/bin/python tools/weight-export/export_weights.py
.venv/bin/python tools/weight-export/validate_weights.py
```

Los tensores regenerados son idénticos bit a bit a los del repositorio (diferencia
máxima 0.0 en todos los tensores del modelo y del CQT), pero el fichero
`.safetensors` **no** es byte-idéntico: el orden de las claves del metadata
no es determinista entre ejecuciones. Por eso `sha256sum -c` solo sirve para
comprobar la integridad de los ficheros ya versionados, no para validar una
regeneración; para eso se comparan los tensores (`validate_weights.py`), no los
hashes.

---

## ⚠️ Limitaciones conocidas

- **Fidelidad, no calidad.** El proyecto valida que el resultado coincida con
  el de Spotify, no que transcriba bien: todavía no hay una evaluación contra
  un ground truth (por ejemplo, con `mir_eval`). El modelo original produce
  notas espurias ("armonías del timbre"), y este port las reproduce.
- **Deriva temporal heredada.** La conversión de frames a tiempo reproduce la
  del upstream, que numera las ventanas dividiendo entre 172 aunque sobreviven
  142 frames por ventana (B-06 en `docs/bugs-resueltos.md`). Los tiempos de
  las notas acumulan un desplazamiento que aún no se ha medido, y no existe
  un modo "corregido".
- **Memoria lineal con la duración.** Las predicciones reconstruidas se
  conservan hasta el final: del orden de 500 MiB por hora de audio, más copias
  temporales. Adecuado para canciones, no para audios de varias horas.
- **Solo CPU.**
- **Dependencia del sistema:** `libsoxr` (ver Requisitos).

---

## 🧭 Decisión de diseño: Candle en lugar de ONNX Runtime

Una alternativa más corta habría sido cargar el ONNX oficial con un runtime
como `ort`. Este proyecto reimplementa el frontend CQT y la red sobre Candle
porque eso es un objetivo explícito (ver Objetivos). El coste es real: un CQT
propio que mantener y numerosas trampas numéricas (34 bugs documentados, 17 de
ellos silenciosos), compensado con la validación etapa a etapa contra el
upstream. Si lo que necesitas es solo transcribir, el paquete original de
Spotify es la opción más directa.

---

## 🔮 Visión a futuro

> Ideas, no compromisos: cada una es un proyecto mayor que este port.

- Editor MIDI mediante piano roll.
- Editor de partituras musicales.
- Reproducción de audio y MIDI.
- Exportación a WAV, MP3 y otros formatos.
- Aplicación de escritorio mediante Tauri.
- Integración futura de herramientas de generación musical con IA.

---

## 🙏 Reconocimientos

Este proyecto toma como referencia el trabajo de investigación y desarrollo de
[Spotify Basic Pitch](https://github.com/spotify/basic-pitch). El reconocimiento
del modelo original, su investigación y sus pesos preentrenados corresponde a
sus respectivos autores.

## 📄 Licencia

Consulta el archivo [LICENSE](LICENSE) para conocer los términos de licencia
de este repositorio. El proyecto original Basic Pitch y los recursos del
modelo conservan sus respectivas condiciones de licencia y atribución.
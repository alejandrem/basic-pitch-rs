//! Fachada de `basic-pitch-rs`.
//!
//! Reexporta la API principal para que quien quiera usar el proyecto como
//! biblioteca dependa de este crate, en lugar de enlazar los crates internos
//! (`basic-pitch-transcription`, `basic-pitch-midi-io`, ...) uno a uno.
//!
//! ```text
//! let mut transcriptor = Transcriber::load(Path::new("models/basic-pitch/icassp-2022"))?;
//! transcriptor.config.note_detection.onset_threshold = 0.6;
//! let resultado = transcriptor.transcribe(Path::new("cancion.mp3"))?;
//! std::fs::write("cancion.mid", &resultado.midi_data)?;
//! ```
//!
//! El CLI (`basic-pitch transcribe`) es una capa fina sobre esta misma API.

pub use basic_pitch_transcription::inference::TranscriptionError;
pub use basic_pitch_transcription::{
    InferenceConfig, NoteDetectionConfig, NoteEvent, Transcriber, TranscriptionResult,
};

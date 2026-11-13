//! Motor de transcripcion: de un archivo de audio a eventos musicales.
//!
//! Pipeline completo:
//!
//! ```text
//! archivo -> decodificar -> mono -> remuestrar -> ventanas
//!         -> modelo (frontend acustico + red) -> predicciones
//!         -> reconstruir -> detectar notas -> pitch bend -> MIDI
//! ```

pub mod inference;
pub mod note_detection;
pub mod pitch_bend;
pub mod stitching;

pub use inference::{Transcriber, TranscriptionResult};
pub use note_detection::{NoteDetectionConfig, NoteEvent};
pub use pitch_bend::PitchBendEstimator;

/// Configuracion de la inferencia.
#[derive(Debug, Clone, PartialEq)]
pub struct InferenceConfig {
    /// Umbral de activacion de onset.
    pub onset_threshold: f64,
    /// Umbral de activacion de frame.
    pub frame_threshold: f64,
    /// Duracion minima de nota, en milisegundos.
    pub minimum_note_length_ms: f64,
    /// Si se permiten pitch bends simultaneos.
    pub multiple_pitch_bends: bool,
    /// Si se aplica el truco de Melodia.
    pub melodia_trick: bool,
    /// Tempo del MIDI de salida.
    pub midi_tempo: f64,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            onset_threshold: note_detection::DEFAULT_ONSET_THRESHOLD,
            frame_threshold: note_detection::DEFAULT_FRAME_THRESHOLD,
            minimum_note_length_ms: note_detection::DEFAULT_MINIMUM_NOTE_LENGTH_MS,
            multiple_pitch_bends: false,
            melodia_trick: true,
            midi_tempo: 120.0,
        }
    }
}

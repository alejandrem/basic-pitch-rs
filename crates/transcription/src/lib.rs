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
pub use note_detection::NoteDetectionConfig;
// El tipo canonico de evento musical vive en midi-io.
pub use basic_pitch_midi_io::NoteEvent;
pub use pitch_bend::PitchBendEstimator;

/// Configuracion de la inferencia.
#[derive(Debug, Clone, PartialEq)]
pub struct InferenceConfig {
    /// Umbrales y opciones de la deteccion de notas.
    pub note_detection: NoteDetectionConfig,
    /// Si se permiten pitch bends simultaneos.
    pub multiple_pitch_bends: bool,
    /// Tempo del MIDI de salida.
    pub midi_tempo: f64,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            note_detection: NoteDetectionConfig::default(),
            multiple_pitch_bends: false,
            midi_tempo: 120.0,
        }
    }
}

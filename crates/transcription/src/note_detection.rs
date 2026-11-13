//! Deteccion de notas musicales a partir de las predicciones del modelo.
//!
//! # Umbrales por defecto
//!
//! | Parametro | Valor | Origen |
//! |---|---|---|
//! | `onset_threshold` | 0.5 | `DEFAULT_ONSET_THRESHOLD` |
//! | `frame_threshold` | 0.3 | `DEFAULT_FRAME_THRESHOLD` |
//! | `minimum_note_length_ms` | 127.7 | `DEFAULT_MINIMUM_NOTE_LENGTH_MS` |
//! | `energy_tol` | 11 | `ENERGY_TOLERANCE` |
//! | `melodia_trick` | activado | `--no-melodia` lo desactiva |
//!
//! # Conversion de milisegundos a frames
//!
//! ```python
//! min_note_len = int(np.round(minimum_note_length / 1000 * (AUDIO_SAMPLE_RATE / FFT_HOP)))
//! ```
//!
//! Con los valores por defecto: `round(127.7 / 1000 * 86.13) = 11` frames.

/// Umbral de activacion de onset por defecto.
pub const DEFAULT_ONSET_THRESHOLD: f64 = 0.5;

/// Umbral de activacion de frame por defecto.
pub const DEFAULT_FRAME_THRESHOLD: f64 = 0.3;

/// Duracion minima de nota por defecto, en milisegundos.
pub const DEFAULT_MINIMUM_NOTE_LENGTH_MS: f64 = 127.7;

/// Tolerancia de energia por defecto, en frames.
pub const DEFAULT_ENERGY_TOL: usize = 11;

/// Desplazamiento entre el indice MIDI y el indice de la matriz de notas.
pub const MIDI_OFFSET: u8 = 21;

/// Numero de notas del modelo (88 teclas).
pub const N_NOTES: usize = 88;

/// Configuracion de la deteccion de notas.
#[derive(Debug, Clone, PartialEq)]
pub struct NoteDetectionConfig {
    /// Umbral de activacion de onset.
    pub onset_threshold: f64,
    /// Umbral de activacion de frame.
    pub frame_threshold: f64,
    /// Duracion minima de nota, en milisegundos.
    pub minimum_note_length_ms: f64,
    /// Tolerancia de energia, en frames.
    pub energy_tol: usize,
    /// Si se infieren onsets adicionales a partir de los cambios de frame.
    pub infer_onsets: bool,
    /// Si se aplica el truco de Melodia.
    pub melodia_trick: bool,
    /// Frecuencia minima permitida, en Hz.
    pub min_freq: Option<f64>,
    /// Frecuencia maxima permitida, en Hz.
    pub max_freq: Option<f64>,
}

impl Default for NoteDetectionConfig {
    fn default() -> Self {
        Self {
            onset_threshold: DEFAULT_ONSET_THRESHOLD,
            frame_threshold: DEFAULT_FRAME_THRESHOLD,
            minimum_note_length_ms: DEFAULT_MINIMUM_NOTE_LENGTH_MS,
            energy_tol: DEFAULT_ENERGY_TOL,
            infer_onsets: true,
            melodia_trick: true,
            min_freq: None,
            max_freq: None,
        }
    }
}

impl NoteDetectionConfig {
    /// Duracion minima de nota convertida a frames.
    ///
    /// Reproduce `int(np.round(ms / 1000 * (22050 / 256)))`.
    pub fn min_note_len_frames(&self) -> usize {
        let fps = basic_pitch_audio_preprocess::ANNOTATIONS_FPS as f64;
        (self.minimum_note_length_ms / 1000.0 * fps).round() as usize
    }
}

/// Un evento de nota detectado.
#[derive(Debug, Clone, PartialEq)]
pub struct NoteEvent {
    /// Tiempo de inicio, en segundos.
    pub start_time: f64,
    /// Tiempo de finalizacion, en segundos.
    pub end_time: f64,
    /// Nota MIDI (21 = A0, 108 = C8).
    pub pitch: u8,
    /// Amplitud, entre 0 y 1.
    pub amplitude: f32,
    /// Secuencia de pitch bend, en unidades de 1/3 de semitono.
    pub pitch_bends: Option<Vec<i16>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn la_duracion_minima_en_frames_es_11() {
        let config = NoteDetectionConfig::default();
        assert_eq!(config.min_note_len_frames(), 11);
    }

    #[test]
    fn los_umbrales_por_defecto_son_los_de_la_referencia() {
        let config = NoteDetectionConfig::default();
        assert_eq!(config.onset_threshold, 0.5);
        assert_eq!(config.frame_threshold, 0.3);
        assert_eq!(config.energy_tol, 11);
        assert!(config.melodia_trick);
        assert!(config.infer_onsets);
    }
}

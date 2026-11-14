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
//! Con los valores por defecto: `round(127.7 / 1000 * 86.1328125) = 11` frames.
//!
//! El fps es la division **flotante** `AUDIO_SAMPLE_RATE / FFT_HOP`, no el
//! entero `ANNOTATIONS_FPS` (= 86) de `constants.py`: con otros valores de
//! duracion minima los dos no coinciden.

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
    /// Reproduce `int(np.round(ms / 1000 * (AUDIO_SAMPLE_RATE / FFT_HOP)))`.
    ///
    /// # El fps es flotante, no entero
    ///
    /// El upstream usa `AUDIO_SAMPLE_RATE / FFT_HOP` = **86.1328125**, que es
    /// division flotante. No es lo mismo que `ANNOTATIONS_FPS`, que en
    /// `constants.py` es `AUDIO_SAMPLE_RATE // FFT_HOP` = 86 (division
    /// entera) y se usa para otras cosas.
    ///
    /// Con el valor por defecto (127.7 ms) ambos dan 11 frames, asi que el
    /// error pasava desapercibido. Pero no siempre coinciden: con 5.81 ms,
    /// el upstream da 1 frame y el valor entero daria 0.
    ///
    /// El redondeo es `round_ties_even`, que es lo que hace `np.round`.
    pub fn min_note_len_frames(&self) -> usize {
        let fps = basic_pitch_audio_preprocess::AUDIO_SAMPLE_RATE as f64
            / basic_pitch_audio_preprocess::FFT_HOP as f64;
        (self.minimum_note_length_ms / 1000.0 * fps).round_ties_even() as usize
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
    fn el_fps_es_el_flotante_del_upstream() {
        // Con 5.81 ms, la division flotante (86.1328125) da 0.50043 y
        // redondea a 1 frame; la entera (86) daria 0.49966 y 0 frames.
        let config = NoteDetectionConfig {
            minimum_note_length_ms: 5.81,
            ..Default::default()
        };
        assert_eq!(config.min_note_len_frames(), 1);

        // El valor por defecto sigue dando 11.
        assert_eq!(NoteDetectionConfig::default().min_note_len_frames(), 11);
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

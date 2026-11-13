//! Estimacion de pitch bend a partir de los contornos de frecuencia.
//!
//! # Conversion de nota MIDI a indice de contorno
//!
//! ```python
//! def midi_pitch_to_contour_bin(pitch_midi):
//!     pitch_hz = librosa.midi_to_hz(pitch_midi)
//!     return 12.0 * CONTOURS_BINS_PER_SEMITONE * np.log2(pitch_hz / ANNOTATIONS_BASE_FREQUENCY)
//! ```
//!
//! Con `CONTOURS_BINS_PER_SEMITONE = 3` y `ANNOTATIONS_BASE_FREQUENCY = 27.5`,
//! esto se simplifica exactamente a `3 * (pitch_midi - 21)`, porque
//! `27.5 Hz` es la nota MIDI 21 (A0). El indice resultante esta en el rango
//! `[0, 261]` para las notas MIDI `[21, 108]`.
//!
//! # Region de busqueda
//!
//! ```python
//! n_bins_tolerance = 25
//! window_length = n_bins_tolerance * 2 + 1          # 51
//! freq_gaussian = scipy.signal.windows.gaussian(window_length, std=5)
//! ```
//!
//! El desplazamiento se expresa en **unidades de 1/3 de semitono**, que es la
//! resolucion de la matriz de contornos.

/// Bins por semitono de la matriz de contornos.
pub const CONTOURS_BINS_PER_SEMITONE: usize = 3;

/// Frecuencia base del modelo, en Hz (nota MIDI 21, A0).
pub const ANNOTATIONS_BASE_FREQUENCY: f64 = 27.5;

/// Desplazamiento entre el indice MIDI y el indice de la matriz de notas.
pub const MIDI_OFFSET: u8 = 21;

/// Numero de bins de la matriz de contornos.
pub const N_FREQ_BINS_CONTOURS: usize = 264;

/// Tolerancia de busqueda, en bins.
pub const N_BINS_TOLERANCE: i64 = 25;

/// Desviacion estandar de la ventana gaussiana.
pub const GAUSSIAN_STD: f64 = 5.0;

/// Escala de conversion a ticks de MIDI.
pub const PITCH_BEND_SCALE: f64 = 4096.0;

/// Limite de los ticks de pitch bend.
pub const N_PITCH_BEND_TICKS: i64 = 8192;

/// Calcula el indice de contorno de una nota MIDI.
///
/// La referencia (`note_creation.py`) calcula:
///
/// ```python
/// pitch_hz = librosa.midi_to_hz(pitch_midi)          # 440 * 2^((p-69)/12)
/// return 12.0 * CONTOURS_BINS_PER_SEMITONE * np.log2(pitch_hz / ANNOTATIONS_BASE_FREQUENCY)
/// ```
///
/// Como `440 / 27.5 = 16 = 2^4`, se tiene
/// `log2(pitch_hz / 27.5) = 4 + (p - 69) / 12 = (p - 21) / 12`, de modo que la
/// formula se simplifica **exactamente** a:
///
/// ```text
/// bin = CONTOURS_BINS_PER_SEMITONE * (pitch_midi - 21)
/// ```
///
/// que es lo que se implementa aqui, sin perder la equivalencia con la
/// referencia.
pub fn midi_pitch_to_contour_bin(pitch_midi: u8) -> i64 {
    let semitones = pitch_midi as f64 - MIDI_OFFSET as f64;
    (CONTOURS_BINS_PER_SEMITONE as f64 * semitones).round() as i64
}

/// Convierte un desplazamiento de contorno (unidades de 1/3 de semitono) a
/// ticks de MIDI.
///
/// ```python
/// pitch_bend_midi_ticks = np.round(
///     np.array(pitch_bend) * PITCH_BEND_SCALE / CONTOURS_BINS_PER_SEMITONE
/// ).astype(int)
/// ```
///
/// El resultado se recorta a `[-8192, 8191]`, lo que limita los bends a
/// unas dos semitonas.
pub fn contour_shift_to_midi_ticks(shift: i64) -> i64 {
    let ticks =
        (shift as f64 * PITCH_BEND_SCALE / CONTOURS_BINS_PER_SEMITONE as f64).round() as i64;
    ticks.clamp(-N_PITCH_BEND_TICKS, N_PITCH_BEND_TICKS - 1)
}

/// Estimador de pitch bend.
#[derive(Debug, Clone)]
pub struct PitchBendEstimator {
    /// Tolerancia de busqueda, en bins.
    pub n_bins_tolerance: i64,
}

impl Default for PitchBendEstimator {
    fn default() -> Self {
        Self {
            n_bins_tolerance: N_BINS_TOLERANCE,
        }
    }
}

impl PitchBendEstimator {
    /// Estima el pitch bend de una nota a partir de la matriz de contornos.
    ///
    /// `contours` es la matriz `(n_frames, 264)`.
    /// `start_idx` y `end_idx` son indices de frame.
    /// `pitch_midi` es la nota del evento.
    ///
    /// Devuelve la secuencia de desplazamientos, en unidades de 1/3 de
    /// semitono, uno por frame del intervalo.
    pub fn estimate(
        &self,
        contours: &[Vec<f32>],
        start_idx: usize,
        end_idx: usize,
        pitch_midi: u8,
    ) -> Vec<i16> {
        if end_idx <= start_idx || contours.is_empty() {
            return Vec::new();
        }

        let freq_idx = midi_pitch_to_contour_bin(pitch_midi);
        let tolerance = self.n_bins_tolerance;

        let freq_start = (freq_idx - tolerance).max(0) as usize;
        let freq_end = ((freq_idx + tolerance + 1) as usize).min(N_FREQ_BINS_CONTOURS);
        if freq_start >= freq_end {
            return Vec::new();
        }

        // Ventana gaussiana de 51 muestras con desviacion 5.
        let window_length = (tolerance * 2 + 1) as usize;
        let gaussian = gaussian_window(window_length, GAUSSIAN_STD);

        // Recorte de la gaussiana cuando la nota esta en un borde.
        let left_cut = (tolerance - freq_idx).max(0) as usize;
        let right_cut = (freq_idx - (N_FREQ_BINS_CONTOURS as i64 - tolerance - 1)).max(0) as usize;
        let pb_shift = tolerance - (tolerance - freq_idx).max(0);

        let mut bends = Vec::new();
        for row in contours.iter().take(end_idx).skip(start_idx) {
            let mut best_bin = 0usize;
            let mut best_value = f32::NEG_INFINITY;
            let mut any = false;

            for (offset, bin) in (freq_start..freq_end).enumerate() {
                let gaussian_index = left_cut + offset;
                if gaussian_index >= gaussian.len() {
                    break;
                }
                let value = row.get(bin).copied().unwrap_or(0.0) * gaussian[gaussian_index];
                if !any || value > best_value {
                    best_value = value;
                    best_bin = offset;
                    any = true;
                }
            }

            let shift = best_bin as i64 - pb_shift;
            bends.push(shift as i16);
        }

        let _ = right_cut;
        bends
    }
}

/// Ventana gaussiana normalizada, como `scipy.signal.windows.gaussian`.
///
/// `scipy.signal.windows.gaussian(M, std)` usa
/// `exp(-0.5 * ((n - (M-1)/2) / std)^2)`, sin normalizar por la suma.
pub fn gaussian_window(length: usize, std: f64) -> Vec<f32> {
    let center = (length as f64 - 1.0) / 2.0;
    (0..length)
        .map(|i| {
            let x = (i as f64 - center) / std;
            (-0.5 * x * x).exp() as f32
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn el_indice_de_contorno_coincide_con_la_formula() {
        assert_eq!(midi_pitch_to_contour_bin(21), 0);
        assert_eq!(midi_pitch_to_contour_bin(22), 3);
        assert_eq!(midi_pitch_to_contour_bin(33), 36);
        assert_eq!(midi_pitch_to_contour_bin(108), 261);
    }

    #[test]
    fn el_indice_de_contorno_coincide_con_la_formula_de_frecuencia() {
        // Comprobacion independiente: se aplica la formula literal de la
        // referencia, `12 * 3 * log2(hz / 27.5)`, para todas las notas.
        for pitch in 21u8..=108 {
            let hz = 440.0 * 2f64.powf((pitch as f64 - 69.0) / 12.0);
            let reference = (12.0
                * CONTOURS_BINS_PER_SEMITONE as f64
                * (hz / ANNOTATIONS_BASE_FREQUENCY).log2())
            .round() as i64;
            assert_eq!(
                midi_pitch_to_contour_bin(pitch),
                reference,
                "nota MIDI {pitch}"
            );
        }
    }

    #[test]
    fn los_ticks_de_midi_se_recortan_a_dos_semitonos() {
        // 1/3 de semitono -> 4096/3 = 1365.33 ticks.
        assert_eq!(contour_shift_to_midi_ticks(1), 1365);
        assert_eq!(contour_shift_to_midi_ticks(0), 0);
        assert_eq!(contour_shift_to_midi_ticks(-1), -1365);
        // El limite superior es 8191.
        assert_eq!(contour_shift_to_midi_ticks(100), 8191);
        assert_eq!(contour_shift_to_midi_ticks(-100), -8192);
    }

    #[test]
    fn la_ventana_gaussiana_tiene_51_muestras_y_es_simetrica() {
        let window = gaussian_window(51, 5.0);
        assert_eq!(window.len(), 51);
        assert!((window[25] - 1.0).abs() < 1e-6);
        assert!((window[0] - window[50]).abs() < 1e-6);
        assert!((window[10] - window[40]).abs() < 1e-6);
    }

    #[test]
    fn una_nota_central_localiza_su_propio_bin() {
        let mut contours = vec![vec![0.0f32; 264]; 4];
        // La nota MIDI 60 (C4) esta en el bin 3 * (60 - 21) = 117.
        let bin = midi_pitch_to_contour_bin(60) as usize;
        for row in &mut contours {
            row[bin] = 1.0;
        }
        let estimator = PitchBendEstimator::default();
        let bends = estimator.estimate(&contours, 0, 4, 60);
        assert_eq!(bends.len(), 4);
        assert!(bends.iter().all(|b| b.abs() <= 1), "bends: {bends:?}");
    }
}

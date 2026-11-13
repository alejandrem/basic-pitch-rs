//! Reconstruccion temporal de las predicciones de multiples ventanas.
//!
//! # Comportamiento de compatibilidad (no corregir)
//!
//! La reconstruccion reproduce literalmente `basic_pitch/inference.py`:
//!
//! ```python
//! n_olap = int(0.5 * n_overlapping_frames)        # 15
//! output = output[:, n_olap:-n_olap, :]            # 172 -> 142 frames
//! unwrapped_output = output.reshape(n_windows * 142, n_freqs)
//!
//! n_expected_windows = audio_original_length / hop_size   # division float
//! n_frames_per_window = (AUDIO_WINDOW_LENGTH * ANNOTATIONS_FPS) - n_overlapping_frames
//! return unwrapped_output[: int(n_expected_windows * n_frames_per_window), :]
//! ```
//!
//! Y la conversion a tiempo (`basic_pitch/note_creation.py`):
//!
//! ```python
//! window_numbers = np.floor(np.arange(n_frames) / ANNOT_N_FRAMES)   # divide por 172
//! window_offset = (FFT_HOP / AUDIO_SAMPLE_RATE) * (
//!     ANNOT_N_FRAMES - (AUDIO_N_SAMPLES / FFT_HOP)
//! ) + MAGIC_ALIGNMENT_OFFSET
//! times = original_times - (window_offset * window_numbers)
//! ```
//!
//! **Inconsistencia que hay que conservar:** la numeracion de ventanas divide
//! entre `ANNOT_N_FRAMES` (172), pero solo sobreviven 142 frames por ventana.
//! El desplazamiento temporal se acumula mas lentamente de lo que "deberia".
//! Es un defecto del upstream, pero corregirlo cambiaria los tiempos de las
//! notas respecto a los de Spotify, que es justo lo que se quiere reproducir.

/// Numero de frames superpuestos entre ventanas consecutivas.
pub const N_OVERLAPPING_FRAMES: usize = 30;

/// Frames emitidos por ventana.
pub const ANNOT_N_FRAMES: usize = 172;

/// Frames conservados por ventana tras descartar los solapados.
pub const FRAMES_KEPT_PER_WINDOW: usize = ANNOT_N_FRAMES - N_OVERLAPPING_FRAMES; // 142

/// Frames descartados por delante y por detras en cada ventana.
pub const N_OLAP: usize = N_OVERLAPPING_FRAMES / 2; // 15

/// Desplazamiento de alineacion magico del upstream, en segundos.
pub const MAGIC_ALIGNMENT_OFFSET: f64 = 0.0018;

/// Calcula el numero de frames que sobreviven a la reconstruccion.
///
/// `audio_original_length` es la longitud del audio **antes** del relleno
/// inicial, en muestras. `hop_size` es el desplazamiento entre ventanas.
pub fn expected_frames(audio_original_length: usize, hop_size: usize) -> usize {
    if hop_size == 0 {
        return 0;
    }
    // La referencia usa division flotante, no `ceil`.
    let n_expected_windows = audio_original_length as f64 / hop_size as f64;
    (n_expected_windows * FRAMES_KEPT_PER_WINDOW as f64) as usize
}

/// Convierte un indice de frame a tiempo, en segundos.
///
/// Reproduce `model_frames_to_time` del upstream, incluida la division por
/// `ANNOT_N_FRAMES` (172) en lugar de por los frames conservados (142).
///
/// # Parametros
///
/// - `frame`: indice del frame dentro de la salida reconstruida.
/// - `sample_rate`: 22.050 Hz.
/// - `fft_hop`: 256 muestras.
/// - `audio_n_samples`: 43.844 muestras.
pub fn frame_to_time(
    frame: usize,
    sample_rate: u32,
    fft_hop: usize,
    audio_n_samples: usize,
) -> f64 {
    let original_time = frame as f64 * fft_hop as f64 / sample_rate as f64;
    let window_number = (frame / ANNOT_N_FRAMES) as f64;
    let window_offset = (fft_hop as f64 / sample_rate as f64)
        * (ANNOT_N_FRAMES as f64 - audio_n_samples as f64 / fft_hop as f64)
        + MAGIC_ALIGNMENT_OFFSET;
    original_time - window_offset * window_number
}

/// Convierte una secuencia de indices de frame a tiempos, en segundos.
pub fn frames_to_time(
    frames: &[usize],
    sample_rate: u32,
    fft_hop: usize,
    audio_n_samples: usize,
) -> Vec<f64> {
    frames
        .iter()
        .map(|f| frame_to_time(*f, sample_rate, fft_hop, audio_n_samples))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn se_descartan_15_frames_por_cada_lado() {
        assert_eq!(N_OLAP, 15);
        assert_eq!(FRAMES_KEPT_PER_WINDOW, 142);
        assert_eq!(ANNOT_N_FRAMES - 2 * N_OLAP, FRAMES_KEPT_PER_WINDOW);
    }

    #[test]
    fn los_frames_esperados_usan_division_flotante() {
        // audio_original_length = 100000, hop_size = 36164
        // n_expected_windows = 2.7649... -> int(2.7649 * 142) = 392
        let frames = expected_frames(100_000, 36_164);
        assert_eq!(frames, (100_000f64 / 36_164f64 * 142.0) as usize);
        assert_eq!(frames, 392);
    }

    #[test]
    fn un_audio_corto_produce_pocos_frames() {
        // 22050 muestras (1 segundo) -> 0.6096 ventanas -> 86 frames
        assert_eq!(expected_frames(22_050, 36_164), 86);
    }

    #[test]
    fn el_primer_frame_no_tiene_desplazamiento_de_ventana() {
        let t = frame_to_time(0, 22_050, 256, 43_844);
        assert_eq!(t, 0.0);
    }

    #[test]
    fn la_numeracion_de_ventanas_divide_entre_172() {
        // El frame 172 pertenece a la ventana 1; el frame 171 a la ventana 0.
        // La comprobacion es que el desplazamiento salta en el frame 172.
        let before = frame_to_time(171, 22_050, 256, 43_844);
        let after = frame_to_time(172, 22_050, 256, 43_844);
        let step = 256.0 / 22_050.0;
        // El salto normal es un frame; al cruzar el limite de ventana se
        // descuenta el offset de la ventana.
        let expected = before + step - window_offset();
        assert!((after - expected).abs() < 1e-12);
    }

    fn window_offset() -> f64 {
        (256.0 / 22_050.0) * (172.0 - 43_844.0 / 256.0) + MAGIC_ALIGNMENT_OFFSET
    }
}

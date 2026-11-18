//! Reconstruccion temporal de las predicciones de multiples ventanas.
//!
//! # Que hace
//!
//! El modelo se aplica a ventanas de dos segundos que se solapan. Las
//! predicciones de cada ventana **no** se pueden concatenar tal cual: hay
//! frames duplicados en los limites. La reconstruccion los quita y produce
//! una unica matriz de frames.
//!
//! # El comportamiento de compatibilidad (no corregir)
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
//!
//! # Nota sobre el layout
//!
//! Las salidas del modelo son `(batch, 172, n_freqs)`, con el tiempo antes
//! que la frecuencia: exactamente el layout que espera el upstream. Por eso
//! la reconstruccion no necesita ninguna conversion.

use candle_core::Tensor;

use basic_pitch_model::ModelOutput;

/// Numero de frames superpuestos entre ventanas consecutivas.
///
/// Viene del modulo de audio, que es quien produce las ventanas: asi la
/// reconstruccion y el ventaneo **no pueden** desincronizarse.
pub const N_OVERLAPPING_FRAMES: usize = basic_pitch_audio_preprocess::N_OVERLAPPING_FRAMES;

/// Frames emitidos por ventana.
pub const ANNOT_N_FRAMES: usize = basic_pitch_audio_preprocess::ANNOT_N_FRAMES;

/// Frames conservados por ventana tras descartar los solapados.
pub const FRAMES_KEPT_PER_WINDOW: usize = basic_pitch_audio_preprocess::FRAMES_KEPT_PER_WINDOW; // 142

/// Frames descartados por delante y por detras en cada ventana.
pub const N_OLAP: usize = N_OVERLAPPING_FRAMES / 2; // 15

/// Desplazamiento de alineacion magico del upstream, en segundos.
pub const MAGIC_ALIGNMENT_OFFSET: f64 = 0.0018;

/// Desplazamiento entre ventanas, en muestras.
pub const HOP_SIZE: usize = basic_pitch_audio_preprocess::HOP_SIZE;

/// Error en la reconstruccion temporal.
#[derive(Debug, thiserror::Error)]
pub enum StitchingError {
    /// La forma de una ventana no es la esperada.
    #[error("una ventana debe tener {ANNOT_N_FRAMES} frames, tiene {actual}")]
    UnexpectedFrames {
        /// Frames de la ventana recibida.
        actual: usize,
    },

    /// Se han recibido ventanas de distinta anchura.
    #[error("las ventanas tienen anchuras distintas: {anterior} y {actual}")]
    InconsistentWidth {
        /// Anchura de la ventana anterior.
        anterior: usize,
        /// Anchura de la ventana actual.
        actual: usize,
    },

    /// No se ha recibido ninguna ventana.
    #[error("no hay ventanas que reconstruir")]
    NoWindows,

    /// Error de Candle.
    #[error("error de Candle en la reconstruccion: {0}")]
    Candle(#[from] candle_core::Error),
}

/// Calcula el numero de frames que sobreviven a la reconstruccion.
///
/// `audio_original_length` es la longitud del audio **antes** del relleno
/// inicial, en muestras. `hop_size` es el desplazamiento entre ventanas.
///
/// La referencia usa division flotante, no `ceil`:
/// `int(audio_original_length / hop_size * 142)`.
pub fn expected_frames(audio_original_length: usize, hop_size: usize) -> usize {
    if hop_size == 0 {
        return 0;
    }
    // La referencia usa division flotante, no `ceil`.
    let n_expected_windows = audio_original_length as f64 / hop_size as f64;
    (n_expected_windows * FRAMES_KEPT_PER_WINDOW as f64) as usize
}

/// Reconstruye las predicciones de una matriz de ventanas.
///
/// # Argumentos
///
/// - `tensor`: `(n_ventanas, 172, n_freqs)`. Es el layout de las salidas
///   del modelo: el tiempo antes que la frecuencia.
/// - `audio_original_length`: longitud del audio **antes** del relleno
///   inicial, en muestras.
/// - `n_overlapping_frames`: frames superpuestos (30).
/// - `hop_size`: desplazamiento entre ventanas, en muestras (36.164).
///
/// # Salida
///
/// `(n_frames, n_freqs)`, con `n_frames = int(audio_original_length /
/// hop_size * 142)`.
///
/// Es la version "por lotes" de `Unwrapper`: reproduce exactamente el
/// `unwrap_output` del upstream.
pub fn unwrap_output(
    tensor: &Tensor,
    audio_original_length: usize,
    n_overlapping_frames: usize,
    hop_size: usize,
) -> Result<Tensor, StitchingError> {
    let (ventanas, frames, _freqs) = tensor.dims3()?;
    if frames != ANNOT_N_FRAMES {
        return Err(StitchingError::UnexpectedFrames { actual: frames });
    }

    let n_frames_finales = expected_frames(audio_original_length, hop_size);
    let mut reconstructor = Unwrapper::new(n_frames_finales, n_overlapping_frames);

    for v in 0..ventanas {
        reconstructor.push(&tensor.narrow(0, v, 1)?)?;
    }
    reconstructor.finish()
}

/// Reconstruccion incremental: se le van pasando las ventanas y se van
/// acumulando sus frames.
///
/// Es la misma operacion que `unwrap_output`, pero **no necesita tener todas
/// las ventanas en memoria**: la prediccion de cada ventana se puede soltar
/// en cuanto se ha incorporado. Eso es lo que pide el apartado 8.1 de la
/// fase de inferencia sobre audios de duracion arbitraria.
///
/// # Uso
///
/// ```ignore
/// let mut reconstructor = Unwrapper::new(expected_frames(len, hop), 30);
/// for ventana in ventanas {
///     reconstructor.push(&salida_de_la_ventana)?;
/// }
/// let matriz = reconstructor.finish()?;
/// ```
pub struct Unwrapper {
    /// Frames que se descartan por delante y por detras de cada ventana.
    n_olap: usize,
    /// Frames que aporta cada ventana.
    frames_por_ventana: usize,
    /// Numero de frames que sobreviven al final.
    n_frames_finales: usize,
    /// Tramos ya reconstruidos, uno por ventana.
    tramos: Vec<Tensor>,
    /// Anchura (frecuencias) de la salida.
    freqs: Option<usize>,
}

impl Unwrapper {
    /// Crea un reconstructor.
    ///
    /// - `n_frames_finales`: frames que sobreviven. Se calcula con
    ///   `expected_frames` **antes** de inferir, porque depende solo de la
    ///   longitud del audio.
    /// - `n_overlapping_frames`: frames superpuestos entre ventanas (30).
    pub fn new(n_frames_finales: usize, n_overlapping_frames: usize) -> Self {
        let n_olap = n_overlapping_frames / 2;
        Self {
            n_olap,
            frames_por_ventana: ANNOT_N_FRAMES - 2 * n_olap,
            n_frames_finales,
            tramos: Vec::new(),
            freqs: None,
        }
    }

    /// Incorpora una ventana.
    ///
    /// `ventana` es `(1, 172, n_freqs)`: el batch, el tiempo y la
    /// frecuencia. Es el layout de las salidas del modelo.
    pub fn push(&mut self, ventana: &Tensor) -> Result<(), StitchingError> {
        let (_, frames, freqs) = ventana.dims3()?;
        if frames != ANNOT_N_FRAMES {
            return Err(StitchingError::UnexpectedFrames { actual: frames });
        }
        match self.freqs {
            None => self.freqs = Some(freqs),
            Some(anterior) if anterior != freqs => {
                return Err(StitchingError::InconsistentWidth {
                    anterior,
                    actual: freqs,
                })
            }
            Some(_) => {}
        }

        // Se descartan los frames del borde, igual que hace el upstream
        // con `output[:, n_olap:-n_olap, :]`.
        //
        // El trozo se guarda sin la dimension de batch (`(142, freqs)`),
        // para que la concatenacion sea sobre el eje del tiempo: asi
        // `(n_ventanas, 142, freqs)` se convierte en
        // `(n_ventanas * 142, freqs)`, que es lo que hace el
        // `reshape` del upstream.
        let trozo = ventana
            .narrow(1, self.n_olap, self.frames_por_ventana)?
            .squeeze(0)?;
        self.tramos.push(trozo);
        Ok(())
    }

    /// Cierra la reconstruccion y devuelve la matriz de frames.
    pub fn finish(self) -> Result<Tensor, StitchingError> {
        if self.tramos.is_empty() {
            return Err(StitchingError::NoWindows);
        }
        let refs: Vec<&Tensor> = self.tramos.iter().collect();
        let matriz = Tensor::cat(&refs, 0)?;

        // Truncado final: `unwrapped_output[: int(n_expected_windows * 142), :]`.
        // Si sobran frames (porque el audio tiene mas ventanas de las que
        // aporta el calculo), se descartan.
        Ok(matriz.narrow(0, 0, self.n_frames_finales.min(matriz.dims2()?.0))?)
    }
}

/// Las tres salidas del modelo, ya reconstruidas.
///
/// Cada una es una matriz `(n_frames, n_freqs)`:
///
/// | Salida | Forma |
/// |---|---|
/// | `note` | `(n_frames, 88)` |
/// | `onset` | `(n_frames, 88)` |
/// | `contour` | `(n_frames, 264)` |
#[derive(Debug, Clone)]
pub struct ReconstructedOutput {
    /// Activacion de notas.
    pub note: Tensor,
    /// Activacion de onsets.
    pub onset: Tensor,
    /// Contornos de frecuencia.
    pub contour: Tensor,
}

/// Reconstruye las tres salidas del modelo, ventana a ventana.
///
/// Es la version incremental de `unwrap_output` para las tres salidas a la
/// vez: la prediccion de cada ventana se suelta en cuanto se ha
/// incorporado, en vez de acumularlas todas.
///
/// `n_frames_finales` es el numero de frames que sobreviven, que se calcula
/// con `expected_frames` antes de inferir.
pub struct ModelUnwrapper {
    note: Unwrapper,
    onset: Unwrapper,
    contour: Unwrapper,
}

impl ModelUnwrapper {
    /// Crea un reconstructor para las tres salidas.
    pub fn new(n_frames_finales: usize, n_overlapping_frames: usize) -> Self {
        Self {
            note: Unwrapper::new(n_frames_finales, n_overlapping_frames),
            onset: Unwrapper::new(n_frames_finales, n_overlapping_frames),
            contour: Unwrapper::new(n_frames_finales, n_overlapping_frames),
        }
    }

    /// Incorpora la prediccion de una ventana.
    pub fn push(&mut self, salida: &ModelOutput) -> Result<(), StitchingError> {
        // Las salidas del modelo ya son (1, 172, n_freqs).
        self.note.push(&salida.note)?;
        self.onset.push(&salida.onset)?;
        self.contour.push(&salida.contour)?;
        Ok(())
    }

    /// Cierra la reconstruccion.
    pub fn finish(self) -> Result<ReconstructedOutput, StitchingError> {
        Ok(ReconstructedOutput {
            note: self.note.finish()?,
            onset: self.onset.finish()?,
            contour: self.contour.finish()?,
        })
    }
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

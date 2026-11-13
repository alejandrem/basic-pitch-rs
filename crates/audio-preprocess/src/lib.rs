//! Preprocesamiento de audio para `basic-pitch-rs`.
//!
//! # Orden de las operaciones
//!
//! La referencia (`librosa.load(path, sr=22050, mono=True)`) aplica las
//! transformaciones en este orden exacto:
//!
//! 1. Decodificacion del archivo.
//! 2. Conversion a mono por **promedio** de los canales.
//! 3. Remuestreo a 22.050 Hz con `soxr` en modo `HQ`.
//!
//! Invertir el orden de los pasos 2 y 3 produce una senal distinta, y por
//! tanto un CQT distinto. Ver `docs/practica/fase-07-procesamiento-audio.md`.
//!
//! # Constantes del contrato
//!
//! Los valores de este modulo provienen del modelo ICASSP 2022 y estan
//! verificados contra `basic-pitch/inference.py` y `basic-pitch/constants.py`.

/// Frecuencia de muestreo del modelo.
pub const AUDIO_SAMPLE_RATE: u32 = 22_050;

/// Numero de canales que consume el modelo.
pub const AUDIO_CHANNELS: u16 = 1;

/// Duracion nominal de una ventana, en segundos.
pub const AUDIO_WINDOW_LENGTH: u64 = 2;

/// Desplazamiento entre frames de la anotacion, en muestras.
pub const FFT_HOP: usize = 256;

/// Frames de anotacion por segundo.
pub const ANNOTATIONS_FPS: usize = AUDIO_SAMPLE_RATE as usize / FFT_HOP; // 86

/// Frames que el modelo produce por ventana.
pub const ANNOT_N_FRAMES: usize = ANNOTATIONS_FPS * AUDIO_WINDOW_LENGTH as usize; // 172

/// Numero de muestras que el modelo consume por ventana.
///
/// ```text
/// 22050 * 2 - 256 = 43844
/// ```
pub const AUDIO_N_SAMPLES: usize =
    AUDIO_SAMPLE_RATE as usize * AUDIO_WINDOW_LENGTH as usize - FFT_HOP; // 43844

/// Frames superpuestos entre ventanas consecutivas.
pub const N_OVERLAPPING_FRAMES: usize = 30;

/// Longitud del solapamiento, en muestras.
pub const OVERLAP_LEN: usize = N_OVERLAPPING_FRAMES * FFT_HOP; // 7680

/// Desplazamiento entre ventanas consecutivas, en muestras.
pub const HOP_SIZE: usize = AUDIO_N_SAMPLES - OVERLAP_LEN; // 36164

/// Zeros de relleno que se anteponen al audio antes de ventanear.
pub const INITIAL_PADDING: usize = OVERLAP_LEN / 2; // 3840

/// Frames conservados por ventana tras descartar los solapados.
pub const FRAMES_KEPT_PER_WINDOW: usize = ANNOT_N_FRAMES - N_OVERLAPPING_FRAMES; // 142

pub mod mono;
pub mod resample;
pub mod windowing;

pub use mono::to_mono;
pub use resample::{resample, ResampleError};
pub use windowing::{AudioWindows, Window};

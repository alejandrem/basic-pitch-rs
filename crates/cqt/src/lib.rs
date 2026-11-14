//! Frontend acustico CQT de `basic-pitch-rs`.
//!
//! # Advertencia critica: 309 bins, no 264
//!
//! La CQT produce **309** bins de frecuencia. La reduccion a 264 ocurre
//! despues, dentro de Harmonic Stacking, que trunca el eje de frecuencias.
//!
//! ```text
//! CQT              -> (batch, 172, 309)
//! NormalizedLog    -> (batch, 172, 309)
//! BatchNorm inicial-> (batch, 172, 309, 1)
//! HarmonicStacking -> (batch, 172, 264, 8)
//! ```
//!
//! Implementar la CQT con 264 bins produce un modelo numéricamente
//! incorrecto, con error silencioso hasta las notas detectadas.
//!
//! # Kernels precalculados
//!
//! Los kernels de la CQT son parametros deterministas, no pesos entrenados.
//! Se extraen del ONNX oficial y se distribuyen como asset
//! (`cqt_kernels.safetensors`), de modo que no hace falta reimplementar
//! `scipy.signal.firwin2` ni el diseño de ventanas en Rust.
//!
//! Ver `docs/practica/fase-08-frontend-cqt.md`.

use serde::{Deserialize, Serialize};

/// Frecuencia de muestreo del modelo.
pub const SAMPLE_RATE: u32 = 22_050;

/// Frecuencia del bin mas bajo.
pub const FMIN: f64 = 27.5;

/// Bins por semitono.
pub const BINS_PER_SEMITONE: usize = 3;

/// Bins por octava.
pub const BINS_PER_OCTAVE: usize = 12 * BINS_PER_SEMITONE; // 36

/// Semitonos cubiertos por la CQT.
///
/// Se calcula como `min(ceil(12*log2(n_harmonics)) + 88, MAX_N_SEMITONES)`
/// con `n_harmonics = 8` y `MAX_N_SEMITONES = 103`.
///
/// No confundir con las 88 teclas del modelo (`ANNOTATIONS_N_SEMITONES`): la
/// CQT cubre mas recorrido porque tiene que llegar a los armonicos altos.
pub const CQT_N_SEMITONES: usize = 103;

/// Bins de frecuencia que produce la CQT.
pub const N_BINS: usize = CQT_N_SEMITONES * BINS_PER_SEMITONE; // 309

/// Bins por semitono del modelo.
pub const CONTOURS_BINS_PER_SEMITONE: usize = BINS_PER_SEMITONE;

/// Semitonos musicales del modelo (88 teclas de piano).
pub const ANNOTATIONS_N_SEMITONES: usize = 88;

/// Bins de la representacion de contornos, ya truncada.
pub const N_FREQ_BINS_CONTOURS: usize = ANNOTATIONS_N_SEMITONES * CONTOURS_BINS_PER_SEMITONE; // 264

/// Factor Q de la transformada.
pub fn q_factor() -> f64 {
    1.0 / (2f64.powf(1.0 / BINS_PER_OCTAVE as f64) - 1.0)
}

/// Numero de octavas que cubre la CQT.
pub fn n_octaves() -> usize {
    N_BINS.div_ceil(BINS_PER_OCTAVE)
}

/// Numero de filtros del kernel de la octava superior.
pub const N_FILTERS: usize = BINS_PER_OCTAVE; // 36

/// Longitud del kernel en el dominio del tiempo.
pub const N_FFT: usize = 256;

/// Desplazamiento entre frames de la CQT, en muestras.
pub const HOP_LENGTH: usize = 256;

/// Configuracion completa de la CQT.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CqtConfig {
    /// Frecuencia de muestreo.
    pub sample_rate: u32,
    /// Frecuencia del bin mas bajo, en Hz.
    pub fmin: f64,
    /// Numero de bins de frecuencia.
    pub n_bins: usize,
    /// Bins por octava.
    pub bins_per_octave: usize,
    /// Desplazamiento entre frames, en muestras.
    pub hop_length: usize,
    /// Longitud del kernel.
    pub n_fft: usize,
    /// Factor Q.
    pub q_factor: f64,
    /// Numero de octavas.
    pub n_octaves: usize,
    /// Numero de filtros del kernel de la octava superior.
    pub n_filters: usize,
    /// Si se aplica submuestreo temprano.
    pub early_downsample: bool,
    /// Factor de submuestreo temprano.
    pub downsample_factor: usize,
    /// Modo de relleno: `reflect`.
    pub pad_mode: String,
    /// Normalizacion de la base: 1 (L1), como en librosa.
    pub basis_norm: u8,
}

impl Default for CqtConfig {
    fn default() -> Self {
        Self {
            sample_rate: SAMPLE_RATE,
            fmin: FMIN,
            n_bins: N_BINS,
            bins_per_octave: BINS_PER_OCTAVE,
            hop_length: HOP_LENGTH,
            n_fft: N_FFT,
            q_factor: q_factor(),
            n_octaves: n_octaves(),
            n_filters: N_FILTERS,
            early_downsample: false,
            downsample_factor: 1,
            pad_mode: "reflect".to_string(),
            basis_norm: 1,
        }
    }
}

pub mod batch_norm;
pub mod kernels;
pub mod normalization;
pub mod transform;

pub use batch_norm::{BatchNormError, BatchNormParams};
pub use kernels::{CqtKernels, KernelsError};
pub use normalization::{normalized_log, NormalizedLogError};
pub use transform::{transform, TransformError};

/// Error del frontend CQT.
#[derive(Debug, thiserror::Error)]
pub enum CqtError {
    /// Los kernels no se pudieron cargar.
    #[error("error al cargar los kernels de la CQT: {0}")]
    Kernels(#[from] KernelsError),

    /// Error en la normalizacion logaritmica.
    #[error("error en NormalizedLog: {0}")]
    Normalization(#[from] NormalizedLogError),

    /// Error en el calculo de la CQT.
    #[error("error en el calculo de la CQT: {0}")]
    Transform(#[from] TransformError),

    /// Error en la BatchNorm inicial.
    #[error("error en la BatchNorm inicial: {0}")]
    BatchNorm(#[from] BatchNormError),
}

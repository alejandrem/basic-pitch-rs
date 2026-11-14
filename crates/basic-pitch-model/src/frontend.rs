//! El frontend acustico: de audio a representacion armonica.
//!
//! Es la cadena completa que la red neuronal espera recibir:
//!
//! ```text
//! audio (1, 43844)
//!   -> CQT              (1, 172, 309)      [crate basic-pitch-cqt]
//!   -> NormalizedLog    (1, 172, 309)      [crate basic-pitch-cqt]
//!   -> BatchNorm inicial(1, 172, 309)      [crate basic-pitch-cqt]
//!   -> HarmonicStacking (1, 8, 172, 264)   NCHW, el layout de Candle
//! ```
//!
//! La ultima forma es la que consume `BasicPitchModel::forward`.
//!
//! # Orden de las operaciones
//!
//! El orden importa y no es intercambiable:
//!
//! - El **relleno reflectante** de la CQT usa el modo `REFLECT` de
//!   TensorFlow, no ceros.
//! - La normalizacion por `sqrt(lengths)` se aplica a la parte real y a la
//!   imaginaria **antes** de tomar la magnitud.
//! - La BatchNorm se aplica **despues** de `NormalizedLog`, sobre el tensor
//!   con el eje de canales anadido, y con estadisticas **escalares** (un solo
//!   canal).
//! - El truncado a 264 bins ocurre **despues** de concatenar los 8 canales
//!   armonicos, no antes de desplazar.
//!
//! Ver `docs/practica/fase-08-frontend-cqt.md` y
//! `docs/practica/fase-09-harmonic-stacking.md`.

use std::path::Path;

use basic_pitch_cqt::{kernels::CqtKernels, normalized_log, transform, BatchNormParams, CqtConfig};
use candle_core::{Device, Tensor};

use crate::config::ModelConfig;
use crate::harmonic_stacking::harmonic_stacking;
use crate::weights::Weights;

/// Error del frontend acustico.
#[derive(Debug, thiserror::Error)]
pub enum FrontendError {
    /// No se pudieron cargar los kernels de la CQT.
    #[error("error al cargar los kernels de la CQT: {0}")]
    Kernels(#[from] basic_pitch_cqt::KernelsError),

    /// Error numerico de Candle.
    ///
    /// Va encuadrada (`Box`) a proposito: `candle_core::Error` es grande y,
    /// sin el `Box`, inflaria este enum y los que lo contienen.
    #[error("error de Candle en el frontend acustico: {0}")]
    Candle(#[source] Box<candle_core::Error>),

    /// Error en el calculo de la CQT.
    #[error("error en el calculo de la CQT: {0}")]
    Transform(#[from] basic_pitch_cqt::TransformError),

    /// Error en la normalizacion logaritmica.
    #[error("error en NormalizedLog: {0}")]
    Normalization(#[from] basic_pitch_cqt::NormalizedLogError),

    /// Error en la BatchNorm inicial.
    #[error("error en la BatchNorm inicial: {0}")]
    BatchNorm(#[from] basic_pitch_cqt::BatchNormError),

    /// Error en Harmonic Stacking.
    #[error("error en Harmonic Stacking: {0}")]
    HarmonicStacking(#[from] crate::harmonic_stacking::HarmonicStackingError),

    /// Error al cargar los pesos del modelo.
    #[error("error al cargar los pesos: {0}")]
    Weights(#[from] crate::weights::WeightsError),

    /// La configuracion del modelo no es compatible con el frontend.
    #[error("configuracion invalida: {0}")]
    Config(String),
}

/// Las salidas intermedias del frontend acustico.
///
/// Se exponen para poder comparar etapa a etapa contra la referencia
/// (`reference/reference_outputs/`), que es la unica forma de saber en que
/// punto se ha acumulado el error.
#[derive(Debug, Clone)]
pub struct FrontendStages {
    /// CQT sin normalizar: `(batch, 172, 309)`.
    pub cqt: Tensor,
    /// CQT tras `NormalizedLog`: `(batch, 172, 309)`.
    pub normalized_log: Tensor,
    /// CQT tras la BatchNorm inicial: `(batch, 172, 309)`.
    pub batch_norm: Tensor,
    /// Representacion armonica: `(batch, 8, 172, 264)` en NCHW.
    pub harmonic_stacking: Tensor,
}

/// El frontend acustico de Basic Pitch.
#[derive(Debug, Clone)]
pub struct AcousticFrontend {
    /// Kernels deterministas de la CQT.
    kernels: CqtKernels,
    /// Parametros de la BatchNorm inicial.
    batch_norm: BatchNormParams,
    /// Configuracion de la transformada.
    cqt_config: CqtConfig,
    /// Armonicos del Harmonic Stacking.
    harmonics: Vec<f64>,
    /// Bins por semitono.
    bins_per_semitone: usize,
    /// Bins de frecuencia de la salida.
    n_output_freqs: usize,
    /// Dispositivo de ejecucion.
    device: Device,
}

impl AcousticFrontend {
    /// Carga el frontend desde el directorio del modelo.
    ///
    /// Espera `model_config.json`, `model.safetensors` y
    /// `cqt_kernels.safetensors` en ese directorio.
    pub fn load(
        dir: &Path,
        config: &ModelConfig,
        weights: &Weights,
    ) -> Result<Self, FrontendError> {
        let kernels = CqtKernels::load(&dir.join("cqt_kernels.safetensors"))?;
        Self::new(kernels, config, weights)
    }

    /// Construye el frontend con los kernels ya cargados.
    pub fn new(
        kernels: CqtKernels,
        config: &ModelConfig,
        weights: &Weights,
    ) -> Result<Self, FrontendError> {
        // La BatchNorm inicial tiene un unico canal: sus cuatro parametros
        // son escalares.
        let gamma = weights.expect("batch_norm.input.gamma", &[1])?.clone();
        let beta = weights.expect("batch_norm.input.beta", &[1])?.clone();
        let moving_mean = weights
            .expect("batch_norm.input.moving_mean", &[1])?
            .clone();
        let moving_variance = weights
            .expect("batch_norm.input.moving_variance", &[1])?
            .clone();

        let batch_norm = BatchNormParams {
            gamma,
            beta,
            moving_mean,
            moving_variance,
            epsilon: config.batch_norm.epsilon,
        };

        let cqt_config = cqt_config_from_model(config)?;

        Ok(Self {
            kernels,
            batch_norm,
            cqt_config,
            harmonics: config.harmonic_stacking.harmonics.clone(),
            bins_per_semitone: config.harmonic_stacking.bins_per_semitone,
            n_output_freqs: config.harmonic_stacking.n_output_freqs,
            device: Device::Cpu,
        })
    }

    /// Ejecuta el frontend y devuelve **todas** las etapas intermedias.
    pub fn stages(&self, samples: &[f32]) -> Result<FrontendStages, FrontendError> {
        let cqt = transform(samples, &self.kernels, &self.cqt_config)?;
        let normalized = normalized_log(&cqt)?;
        let batch_norm = self.batch_norm.normalize(&normalized)?;

        // (batch, tiempo, frecuencia) -> (batch, 1, tiempo, frecuencia).
        let (batch, time, freq) = batch_norm.dims3()?;
        let nchw = batch_norm.reshape((batch, 1, time, freq))?;

        let harmonic = harmonic_stacking(
            &nchw,
            &self.harmonics,
            self.bins_per_semitone,
            self.n_output_freqs,
        )?;

        Ok(FrontendStages {
            cqt,
            normalized_log: normalized,
            batch_norm,
            harmonic_stacking: harmonic,
        })
    }

    /// Ejecuta el frontend y devuelve solo la representacion armonica.
    ///
    /// La forma es `(batch, 8, 172, 264)` en NCHW, que es lo que espera
    /// `BasicPitchModel::forward`.
    pub fn forward(&self, samples: &[f32]) -> Result<Tensor, FrontendError> {
        Ok(self.stages(samples)?.harmonic_stacking)
    }

    /// Dispositivo de ejecucion.
    pub fn device(&self) -> &Device {
        &self.device
    }
}

impl From<candle_core::Error> for FrontendError {
    fn from(error: candle_core::Error) -> Self {
        FrontendError::Candle(Box::new(error))
    }
}

/// Traduce la configuracion del modelo a la de la transformada.
///
/// Comprueba, ademas, que la configuracion sea compatible con la
/// implementacion: el modelo no usa submuestreo temprano y su relleno es
/// reflectante.
fn cqt_config_from_model(config: &ModelConfig) -> Result<CqtConfig, FrontendError> {
    let cqt = &config.cqt;

    if cqt.early_downsample {
        return Err(FrontendError::Config(
            "el modelo no usa submuestreo temprano en la CQT".to_string(),
        ));
    }
    if cqt.downsample_factor != 1 {
        return Err(FrontendError::Config(format!(
            "el factor de submuestreo debe ser 1, es {}",
            cqt.downsample_factor
        )));
    }
    if cqt.pad_mode != "reflect" {
        return Err(FrontendError::Config(format!(
            "el modo de relleno debe ser 'reflect', es '{}'",
            cqt.pad_mode
        )));
    }
    if cqt.basis_norm != 1 {
        return Err(FrontendError::Config(format!(
            "la normalizacion de la base debe ser 1 (L1), es {}",
            cqt.basis_norm
        )));
    }

    Ok(CqtConfig {
        sample_rate: cqt.sr,
        fmin: cqt.fmin,
        n_bins: cqt.n_bins,
        bins_per_octave: cqt.bins_per_octave,
        hop_length: cqt.hop_length,
        n_fft: cqt.n_fft,
        q_factor: cqt.q_factor,
        n_octaves: cqt.n_octaves,
        n_filters: cqt.n_filters,
        early_downsample: cqt.early_downsample,
        downsample_factor: cqt.downsample_factor,
        pad_mode: cqt.pad_mode.clone(),
        basis_norm: cqt.basis_norm,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn la_configuracion_de_la_cqt_se_traduce() {
        let config = ModelConfig::example();
        let cqt = cqt_config_from_model(&config).unwrap();
        assert_eq!(cqt.sample_rate, 22_050);
        assert_eq!(cqt.n_bins, 309);
        assert_eq!(cqt.hop_length, 256);
        assert_eq!(cqt.n_fft, 256);
        assert_eq!(cqt.n_octaves, 9);
        assert_eq!(cqt.n_filters, 36);
        assert!(!cqt.early_downsample);
    }

    #[test]
    fn una_configuracion_con_submuestreo_temprano_produce_error() {
        let mut config = ModelConfig::example();
        config.cqt.early_downsample = true;
        assert!(matches!(
            cqt_config_from_model(&config),
            Err(FrontendError::Config(_))
        ));
    }

    #[test]
    fn una_configuracion_con_padding_distinto_de_reflect_produce_error() {
        let mut config = ModelConfig::example();
        config.cqt.pad_mode = "constant".to_string();
        assert!(matches!(
            cqt_config_from_model(&config),
            Err(FrontendError::Config(_))
        ));
    }
}

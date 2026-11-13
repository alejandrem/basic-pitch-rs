//! Arquitectura neuronal de Basic Pitch, portada a Candle.
//!
//! # Grafo activo (verificado contra el SavedModel oficial)
//!
//! ```text
//! entrada (batch, 43844, 1)
//!   -> FlattenAudioCh            -> (batch, 43844)
//!   -> CQT2010v2                 -> (batch, 172, 309)
//!   -> NormalizedLog             -> (batch, 172, 309)
//!   -> expand_dims               -> (batch, 172, 309, 1)
//!   -> BatchNorm inicial (1 ch)  -> (batch, 172, 309, 1)
//!   -> HarmonicStacking          -> (batch, 172, 264, 8)
//!        |
//!        +--> conv contour (8, 3x39) -> BatchNorm(8) -> ReLU
//!        |        -> conv contours-reduced (1, 5x5) + sigmoid
//!        |        -> FlattenFreqCh  -> (batch, 172, 264)      == contour
//!        |        -> expand_dims    -> (batch, 172, 264, 1)
//!        |        -> conv notes1 (32, 7x7, stride 1x3) -> ReLU
//!        |        -> conv notes2 (1, 7x3) + sigmoid        -> (batch, 172, 88, 1)
//!        |             |
//!        |             +--> x_notes_pre
//!        |
//!        +--> conv onset1 (32, 5x5, stride 1x3) -> BatchNorm(32) -> ReLU
//!                 -> Concat(axis=3)([x_notes_pre, x_onset]) -> (batch, 172, 88, 33)
//!                 -> conv onset2 (1, 3x3) + sigmoid          -> (batch, 172, 88)
//! ```
//!
//! # Dos advertencias
//!
//! 1. La concatenacion de onsets produce **33 canales** (1 de `x_notes_pre` +
//!    32 de `x_onset`), en ese orden. Invertirlo produce un modelo que
//!    difiere del original sin error visible.
//! 2. La CQT produce **309** bins, no 264. El truncado a 264 ocurre en
//!    Harmonic Stacking.
//!
//! # Padding y stride
//!
//! `candle_nn::Conv2dConfig` solo admite un unico `padding` simetrico y un
//! unico `stride`, mientras que el modelo usa `padding="same"` y strides por
//! eje `(1, 3)`. La estrategia es **padding manual + conv stride-1 +
//! submuestreo**, que es algebraicamente exacta. Ver `layers.rs`.

use std::path::Path;

pub mod config;
pub mod contour;
pub mod harmonic_stacking;
pub mod layers;
pub mod model;
pub mod notes;
pub mod onsets;
pub mod weights;

pub use config::ModelConfig;
pub use model::{BasicPitchModel, ModelOutput};
pub use weights::{Weights, WeightsError};

/// Error del modelo.
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    /// Error al cargar los pesos.
    #[error("error al cargar los pesos: {0}")]
    Weights(#[from] WeightsError),

    /// Error de Candle durante la inferencia.
    #[error("error de Candle en el modelo: {0}")]
    Candle(#[from] candle_core::Error),

    /// Error de la configuracion del modelo.
    #[error("configuracion invalida: {0}")]
    Config(String),
}

/// Carga un modelo completo desde su directorio.
pub fn load_model(dir: &Path) -> Result<BasicPitchModel, ModelError> {
    let config = ModelConfig::load(&dir.join("model_config.json")).map_err(ModelError::Config)?;
    let weights = Weights::load(&dir.join("model.safetensors"))?;
    BasicPitchModel::new(config, weights)
}

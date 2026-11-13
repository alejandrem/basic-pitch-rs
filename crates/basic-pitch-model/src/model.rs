//! El modelo completo y su ejecucion.
//!
//! # Separacion entre frontend acustico y red neuronal
//!
//! La red neuronal **no** recibe audio. Recibe la representacion armonica que
//! produce el frontend acustico. El pipeline completo es:
//!
//! ```text
//! audio (batch, 43844, 1)
//!   -> CQT             (batch, 172, 309)      [crate basic-pitch-cqt]
//!   -> NormalizedLog   (batch, 172, 309)
//!   -> BatchNorm       (batch, 172, 309, 1)
//!   -> HarmonicStacking(batch, 8, 172, 264)  [NCHW en Candle]
//!   -> BasicPitchModel::forward(...)
//!        -> note, onset, contour
//! ```
//!
//! La implementacion del frontend acustico llega en la Fase 4 (P2). Este
//! modulo solo cubre la parte neuronal.

use candle_core::{Device, Tensor};

use crate::config::ModelConfig;
use crate::contour::ContourBranch;
use crate::notes::NotesBranch;
use crate::onsets::OnsetBranch;
use crate::weights::Weights;
use crate::ModelError;

/// Las tres salidas del modelo.
///
/// # Dimensiones
///
/// | Salida | Forma NHWC (TensorFlow) | Forma NCHW (Candle) |
/// |---|---|---|
/// | `note` | `(batch, 172, 88)` | `(batch, 88, 172)` |
/// | `onset` | `(batch, 172, 88)` | `(batch, 88, 172)` |
/// | `contour` | `(batch, 172, 264)` | `(batch, 264, 172)` |
///
/// Las tres son **salidas finales**, con el eje de canales ya eliminado. Las
/// formas intermedias de cada rama, que si conservan el eje de canales, estan
/// documentadas en los modulos `contour`, `notes` y `onsets`.
#[derive(Debug, Clone)]
pub struct ModelOutput {
    /// Activacion de notas: `(batch, 172, 88)`.
    pub note: candle_core::Tensor,
    /// Activacion de onsets: `(batch, 172, 88)`.
    pub onset: candle_core::Tensor,
    /// Contornos de frecuencia: `(batch, 172, 264)`.
    pub contour: candle_core::Tensor,
}

/// El modelo ICASSP 2022 de Basic Pitch.
#[derive(Debug, Clone)]
pub struct BasicPitchModel {
    /// Configuracion tecnica del modelo.
    pub config: ModelConfig,
    /// Rama de contornos.
    pub contour: ContourBranch,
    /// Rama de notas.
    pub notes: NotesBranch,
    /// Rama de onsets.
    pub onsets: OnsetBranch,
    /// Dispositivo de ejecucion.
    pub device: Device,
}

impl BasicPitchModel {
    /// Construye el modelo a partir de la configuracion y los pesos.
    pub fn new(config: ModelConfig, weights: Weights) -> Result<Self, ModelError> {
        config.validate().map_err(ModelError::Config)?;
        Ok(Self {
            contour: ContourBranch::from_weights(&weights)?,
            notes: NotesBranch::from_weights(&weights)?,
            onsets: OnsetBranch::from_weights(&weights)?,
            config,
            device: Device::Cpu,
        })
    }

    /// Ejecuta la **red neuronal** sobre la representacion armonica.
    ///
    /// # Entrada
    ///
    /// `harmonic_stacked` es la salida de Harmonic Stacking, en NCHW:
    /// `(batch, 8, 172, 264)`, es decir:
    ///
    /// - `batch`: numero de ventanas,
    /// - `8`: canales armónicos (`[0.5, 1, 2, 3, 4, 5, 6, 7]`),
    /// - `172`: frames temporales,
    /// - `264`: bins de frecuencia (ya truncados desde 309).
    ///
    /// **No** se acepta audio aqui: el frontend acustico (CQT,
    /// NormalizedLog, BatchNorm inicial y Harmonic Stacking) es
    /// responsabilidad del crate `basic-pitch-cqt` y se aplica antes.
    ///
    /// # Salida
    ///
    /// Las tres salidas del modelo: `note`, `onset` y `contour`.
    pub fn forward(&self, harmonic_stacked: &Tensor) -> Result<ModelOutput, ModelError> {
        let dims = harmonic_stacked.dims4()?;
        let expected = (
            self.config.harmonic_stacking.n_output_freqs,
            self.config.audio.annot_n_frames,
        );
        if dims.1 != self.config.harmonic_stacking.harmonics.len()
            || dims.2 != expected.1
            || dims.3 != expected.0
        {
            return Err(ModelError::Config(format!(
                "la representacion armonica debe tener forma (batch, {}, {}, {}), tiene {:?}",
                self.config.harmonic_stacking.harmonics.len(),
                expected.1,
                expected.0,
                dims
            )));
        }

        let harmonic_stacked = harmonic_stacked.to_device(&self.device)?;

        // --- Rama de contornos ---
        // (batch, 8, 172, 264) -> contour (batch, 172, 264)
        //                       -> x_contours_reduced (batch, 1, 172, 264)
        let (contour, x_contours_reduced) = self.contour.forward(&harmonic_stacked)?;

        // --- Rama de notas ---
        // (batch, 1, 172, 264) -> note (batch, 172, 88)
        //                       -> x_notes_pre (batch, 1, 172, 88)
        let (note, x_notes_pre) = self.notes.forward(&x_contours_reduced)?;

        // --- Rama de onsets ---
        // (batch, 8, 172, 264) + (batch, 1, 172, 88)
        //   -> concat en canales -> (batch, 33, 172, 88)
        //   -> onset (batch, 172, 88)
        let onset = self.onsets.forward(&harmonic_stacked, &x_notes_pre)?;

        Ok(ModelOutput {
            note,
            onset,
            contour,
        })
    }
}

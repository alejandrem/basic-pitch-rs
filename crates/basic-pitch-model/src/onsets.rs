//! Rama de onsets.
//!
//! ```text
//! HarmonicStacking (batch, 8, 172, 264)
//!   -> conv onset1 (32 filtros, kernel 5x5, stride (1,3), pad SAME (2,2)/(1,1))
//!   -> BatchNorm (32 canales, epsilon 0.001)
//!   -> ReLU
//!   -> Concat(axis=canales)([x_notes_pre, x_onset])  -> 33 canales
//!   -> conv onset2 (1 filtro, kernel 3x3, pad SAME (1,1)/(1,1))
//!   -> sigmoid
//!   -> FlattenFreqCh -> (batch, 172, 88)   == onset
//! ```
//!
//! # La concatenacion: 33 canales, en un orden fijo
//!
//! ```python
//! x_onset = tfkl.Concatenate(axis=3, name="concat")([x_notes_pre, x_onset])
//! ```
//!
//! - `x_notes_pre` es la salida **sigmoid de la conv de notas**, con forma
//!   `(batch, 1, 172, 88)`. **No** es la salida final `note`.
//! - `x_onset` es la salida de esta rama, `(batch, 32, 172, 88)`.
//! - El concatenado va sobre el eje de canales: `(batch, 33, 172, 88)`.
//!
//! El **orden es obligatorio**: primero `x_notes_pre` (canal 0), despues
//! `x_onset` (canales 1..32). Invertirlo produce un modelo que difiere del
//! original sin error visible.
//!
//! Confirmacion independiente en el ONNX oficial: el kernel de esta
//! convoluion final es `const_fold_opt__680` con dimensiones `[1, 33, 3, 3]`.

use candle_core::Tensor;
use candle_nn::ops::sigmoid;

use crate::contour::batch_norm;
use crate::layers::{conv2d_same, same_padding2d};
use crate::weights::Weights;

/// Nombres canonicos de los pesos de la rama.
pub mod weight_names {
    /// Kernel de la primera convolucion.
    pub const CONV1_WEIGHT: &str = "onset.conv1.weight";
    /// Bias de la primera convolucion.
    pub const CONV1_BIAS: &str = "onset.conv1.bias";
    /// Escala de la BatchNorm.
    pub const BN_GAMMA: &str = "onset.batch_norm.gamma";
    /// Desplazamiento de la BatchNorm.
    pub const BN_BETA: &str = "onset.batch_norm.beta";
    /// Media de la BatchNorm.
    pub const BN_MEAN: &str = "onset.batch_norm.moving_mean";
    /// Varianza de la BatchNorm.
    pub const BN_VAR: &str = "onset.batch_norm.moving_variance";
    /// Kernel de la convolucion de salida.
    pub const CONV2_WEIGHT: &str = "onset.conv2.weight";
    /// Bias de la convolucion de salida.
    pub const CONV2_BIAS: &str = "onset.conv2.bias";
}

/// Numero de filtros de la primera convolucion.
pub const CONV1_FILTERS: usize = 32;
/// Kernel de la primera convolucion, como (tiempo, frecuencia).
pub const CONV1_KERNEL: (usize, usize) = (5, 5);
/// Stride de la primera convolucion, como (tiempo, frecuencia).
pub const CONV1_STRIDE: (usize, usize) = (1, 3);
/// Numero de filtros de la convolucion de salida.
pub const CONV2_FILTERS: usize = 1;
/// Kernel de la convolucion de salida, como (tiempo, frecuencia).
pub const CONV2_KERNEL: (usize, usize) = (3, 3);
/// Canales de la rama de onsets antes de concatenar.
pub const ONSET_CHANNELS: usize = 32;
/// Canales de `x_notes_pre`.
pub const NOTES_PRE_CHANNELS: usize = 1;
/// Canales resultantes de la concatenacion.
pub const CONCAT_CHANNELS: usize = ONSET_CHANNELS + NOTES_PRE_CHANNELS; // 33
/// Epsilon de Keras para BatchNorm.
pub const BATCH_NORM_EPSILON: f64 = 0.001;

/// La rama de onsets.
#[derive(Debug, Clone)]
pub struct OnsetBranch {
    /// Kernel de la primera convolucion: `(32, 8, 5, 5)`.
    pub conv1_weight: Tensor,
    /// Bias de la primera convolucion: `(32)`.
    pub conv1_bias: Tensor,
    /// Escala de la BatchNorm: `(32)`.
    pub bn_gamma: Tensor,
    /// Desplazamiento de la BatchNorm: `(32)`.
    pub bn_beta: Tensor,
    /// Media de la BatchNorm: `(32)`.
    pub bn_mean: Tensor,
    /// Varianza de la BatchNorm: `(32)`.
    pub bn_var: Tensor,
    /// Kernel de la convolucion de salida: `(1, 33, 3, 3)`.
    pub conv2_weight: Tensor,
    /// Bias de la convolucion de salida: `(1)`.
    pub conv2_bias: Tensor,
}

impl OnsetBranch {
    /// Construye la rama a partir de los pesos cargados.
    pub fn from_weights(weights: &Weights) -> Result<Self, crate::ModelError> {
        Ok(Self {
            conv1_weight: weights
                .expect(weight_names::CONV1_WEIGHT, &[32, 8, 5, 5])?
                .clone(),
            conv1_bias: weights.expect(weight_names::CONV1_BIAS, &[32])?.clone(),
            bn_gamma: weights.expect(weight_names::BN_GAMMA, &[32])?.clone(),
            bn_beta: weights.expect(weight_names::BN_BETA, &[32])?.clone(),
            bn_mean: weights.expect(weight_names::BN_MEAN, &[32])?.clone(),
            bn_var: weights.expect(weight_names::BN_VAR, &[32])?.clone(),
            conv2_weight: weights
                .expect(weight_names::CONV2_WEIGHT, &[1, CONCAT_CHANNELS, 3, 3])?
                .clone(),
            conv2_bias: weights.expect(weight_names::CONV2_BIAS, &[1])?.clone(),
        })
    }

    /// Ejecuta la rama.
    ///
    /// # Formas (NCHW, el layout de Candle)
    ///
    /// | Tensor | Forma | Equivalente NHWC |
    /// |---|---|---|
    /// | `x` (entrada) | `(batch, 8, 172, 264)` | `(batch, 172, 264, 8)` |
    /// | `onset_features` | `(batch, 32, 172, 88)` | `(batch, 172, 88, 32)` |
    /// | `x_notes_pre` | `(batch, 1, 172, 88)` | `(batch, 172, 88, 1)` |
    /// | concatenado | `(batch, 33, 172, 88)` | `(batch, 172, 88, 33)` |
    /// | `onset` (salida) | `(batch, 172, 88)` | `(batch, 172, 88)` |
    ///
    /// El concatenado se hace sobre el **eje de canales**, con `x_notes_pre`
    /// primero (canal 0) y `onset_features` despues (canales 1..32).
    pub fn forward(&self, x: &Tensor, x_notes_pre: &Tensor) -> Result<Tensor, candle_core::Error> {
        // --- conv onset1 (32, 5x5, stride 1x3) ---
        let pad = same_padding2d((172, 264), CONV1_KERNEL, CONV1_STRIDE);
        let conv1 = conv2d_same(x, &self.conv1_weight, pad, CONV1_STRIDE)?;
        let conv1 = conv1.broadcast_add(&self.conv1_bias.reshape((1, 32, 1, 1))?)?;


        // --- BatchNorm (32 canales) + ReLU ---
        let bn = batch_norm(
            &conv1,
            &self.bn_gamma,
            &self.bn_beta,
            &self.bn_mean,
            &self.bn_var,
            BATCH_NORM_EPSILON,
        )?;
        let bn = bn.relu()?;

        // --- Concat(axis=canales)([x_notes_pre, x_onset]) -> 33 canales ---
        // El orden es obligatorio: primero `x_notes_pre`.
        let concatenated = Tensor::cat(&[x_notes_pre, &bn], 1)?;

        // --- conv onset2 (1, 3x3) + sigmoid ---
        let pad = same_padding2d((172, 88), CONV2_KERNEL, (1, 1));
        let conv2 = conv2d_same(&concatenated, &self.conv2_weight, pad, (1, 1))?;
        let conv2 = conv2.broadcast_add(&self.conv2_bias.reshape((1, 1, 1, 1))?)?;
        let conv2 = sigmoid(&conv2)?;

        // `onset` es el aplanado: (batch, 1, 172, 88) -> (batch, 172, 88).
        //
        // OJO: aqui NO hay que transponer. `FlattenFreqCh` de TensorFlow
        // aplana `(batch, 172, 88, 1)` a `(batch, 172, 88)`: elimina el eje
        // de canales y conserva el orden (tiempo, frecuencia). El
        // equivalente en NCHW es `squeeze(1)` y nada mas; un
        // `transpose(1, 2)` de mas devolveria (batch, 88, 172).
        let onset = conv2.squeeze(1)?.contiguous()?;


        Ok(onset)
    }
}

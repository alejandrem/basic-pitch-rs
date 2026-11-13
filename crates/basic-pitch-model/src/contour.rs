//! Rama de contornos.
//!
//! ```text
//! HarmonicStacking (batch, 172, 264, 8)
//!   -> conv contour   (8 filtros, kernel 3x39, pad SAME (1,1)/(19,19))
//!   -> BatchNorm (8 canales, epsilon 0.001)
//!   -> ReLU
//!   -> conv contours-reduced (1 filtro, kernel 5x5, pad SAME (2,2)/(2,2))
//!   -> sigmoid
//!   -> FlattenFreqCh -> (batch, 172, 264)   == contour
//! ```
//!
//! # Capa (5, 5): dos, y solo una esta activa
//!
//! | Constante | Kernel | Filtros | Estado |
//! |---|---|---:|---|
//! | `CONTOUR_KERNEL_SIZE_1` | (5, 5) | 32 | **COMENTADA - se omite** |
//! | `CONTOUR_KERNEL_SIZE_3` | (5, 5) | 1 | **ACTIVA - es la de arriba** |
//!
//! La capa comentada existe en el codigo original porque fue omitida
//! accidentalmente durante el entrenamiento del modelo publicado.

use candle_core::{DType, Tensor};
use candle_nn::ops::sigmoid;

use crate::layers::{conv2d_same, same_padding2d};
use crate::weights::Weights;

/// Nombres canonicos de los pesos de la rama.
pub mod weight_names {
    /// Kernel de la primera convolucion.
    pub const CONV1_WEIGHT: &str = "contour.conv1.weight";
    /// Bias de la primera convolucion.
    pub const CONV1_BIAS: &str = "contour.conv1.bias";
    /// Escala de la BatchNorm.
    pub const BN_GAMMA: &str = "contour.batch_norm.gamma";
    /// Desplazamiento de la BatchNorm.
    pub const BN_BETA: &str = "contour.batch_norm.beta";
    /// Media de la BatchNorm.
    pub const BN_MEAN: &str = "contour.batch_norm.moving_mean";
    /// Varianza de la BatchNorm.
    pub const BN_VAR: &str = "contour.batch_norm.moving_variance";
    /// Kernel de la convolucion de reduccion.
    pub const REDUCED_WEIGHT: &str = "contour.reduced.weight";
    /// Bias de la convolucion de reduccion.
    pub const REDUCED_BIAS: &str = "contour.reduced.bias";
}

/// Numero de filtros de la primera convolucion.
pub const CONV1_FILTERS: usize = 8;
/// Canales de entrada de la primera convolucion (los 8 armonicos).
pub const CONV1_IN_CHANNELS: usize = 8;
/// Kernel de la primera convolucion, como (tiempo, frecuencia).
pub const CONV1_KERNEL: (usize, usize) = (3, 39);
/// Numero de filtros de la convolucion de reduccion.
pub const REDUCED_FILTERS: usize = 1;
/// Kernel de la convolucion de reduccion, como (tiempo, frecuencia).
pub const REDUCED_KERNEL: (usize, usize) = (5, 5);
/// Epsilon de Keras para BatchNorm.
pub const BATCH_NORM_EPSILON: f64 = 0.001;

/// La rama de contornos.
#[derive(Debug, Clone)]
pub struct ContourBranch {
    /// Kernel de la primera convolucion: `(8, 8, 3, 39)`.
    pub conv1_weight: Tensor,
    /// Bias de la primera convolucion: `(8)`.
    pub conv1_bias: Tensor,
    /// Escala de la BatchNorm: `(8)`.
    pub bn_gamma: Tensor,
    /// Desplazamiento de la BatchNorm: `(8)`.
    pub bn_beta: Tensor,
    /// Media de la BatchNorm: `(8)`.
    pub bn_mean: Tensor,
    /// Varianza de la BatchNorm: `(8)`.
    pub bn_var: Tensor,
    /// Kernel de la convolucion de reduccion: `(1, 8, 5, 5)`.
    pub reduced_weight: Tensor,
    /// Bias de la convolucion de reduccion: `(1)`.
    pub reduced_bias: Tensor,
}

impl ContourBranch {
    /// Construye la rama a partir de los pesos cargados.
    pub fn from_weights(weights: &Weights) -> Result<Self, crate::ModelError> {
        Ok(Self {
            conv1_weight: weights
                .expect(weight_names::CONV1_WEIGHT, &[8, 8, 3, 39])?
                .clone(),
            conv1_bias: weights.expect(weight_names::CONV1_BIAS, &[8])?.clone(),
            bn_gamma: weights.expect(weight_names::BN_GAMMA, &[8])?.clone(),
            bn_beta: weights.expect(weight_names::BN_BETA, &[8])?.clone(),
            bn_mean: weights.expect(weight_names::BN_MEAN, &[8])?.clone(),
            bn_var: weights.expect(weight_names::BN_VAR, &[8])?.clone(),
            reduced_weight: weights
                .expect(weight_names::REDUCED_WEIGHT, &[1, 8, 5, 5])?
                .clone(),
            reduced_bias: weights.expect(weight_names::REDUCED_BIAS, &[1])?.clone(),
        })
    }

    /// Ejecuta la rama.
    ///
    /// # Formas (NCHW, el layout de Candle)
    ///
    /// | Tensor | Forma | Equivalente NHWC |
    /// |---|---|---|
    /// | `x` (entrada) | `(batch, 8, 172, 264)` | `(batch, 172, 264, 8)` |
    /// | `contour_conv` | `(batch, 8, 172, 264)` | `(batch, 172, 264, 8)` |
    /// | `contour_reduced` | `(batch, 1, 172, 264)` | `(batch, 172, 264, 1)` |
    /// | `contour` (salida) | `(batch, 172, 264)` | `(batch, 172, 264)` |
    ///
    /// El eje de canales se elimina en la salida final mediante
    /// `FlattenFreqCh`, que en TensorFlow aplana `(batch, 172, 264, 1)` a
    /// `(batch, 172, 264)` conservando el orden frecuencia-principal.
    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor), candle_core::Error> {
        // --- conv contour (8, 3x39) ---
        let pad = same_padding2d((172, 264), CONV1_KERNEL, (1, 1));
        let conv1 = conv2d_same(x, &self.conv1_weight, pad, (1, 1))?;
        let conv1 = conv1.broadcast_add(&self.conv1_bias.reshape((1, 8, 1, 1))?)?;

        // --- BatchNorm (8 canales) en modo inferencia ---
        let bn = batch_norm(
            &conv1,
            &self.bn_gamma,
            &self.bn_beta,
            &self.bn_mean,
            &self.bn_var,
            BATCH_NORM_EPSILON,
        )?;

        // --- ReLU ---
        let bn = bn.relu()?;

        // --- conv contours-reduced (1, 5x5) + sigmoid ---
        let pad = same_padding2d((172, 264), REDUCED_KERNEL, (1, 1));
        let reduced = conv2d_same(&bn, &self.reduced_weight, pad, (1, 1))?;
        let reduced = reduced.broadcast_add(&self.reduced_bias.reshape((1, 1, 1, 1))?)?;
        let reduced = sigmoid(&reduced)?;

        // La salida `contour` es el aplanado de frecuencia y canales:
        // (batch, 1, 172, 264) -> (batch, 172, 264).
        let contour = reduced.squeeze(1)?.transpose(1, 2)?.contiguous()?;

        Ok((contour, reduced))
    }
}

/// Aplica Batch Normalization en modo inferencia.
///
/// `x` es NCHW y los parametros tienen longitud `C` (los canales).
pub fn batch_norm(
    x: &Tensor,
    gamma: &Tensor,
    beta: &Tensor,
    mean: &Tensor,
    var: &Tensor,
    epsilon: f64,
) -> Result<Tensor, candle_core::Error> {
    let width = mean.dims1()?;
    let scale = var.affine(1.0, epsilon)?.sqrt()?;
    let normalized = x
        .broadcast_sub(&mean.reshape((1, width, 1, 1))?)?
        .broadcast_div(&scale.reshape((1, width, 1, 1))?)?;
    normalized
        .broadcast_mul(&gamma.reshape((1, width, 1, 1))?)?
        .broadcast_add(&beta.reshape((1, width, 1, 1))?)
}

/// Devuelve el dtype con el que se calculan las activaciones.
pub const ACTIVATION_DTYPE: DType = DType::F32;

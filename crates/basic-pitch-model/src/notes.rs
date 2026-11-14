//! Rama de notas.
//!
//! ```text
//! x_contours_reduced (batch, 1, 172, 264)
//!   -> conv notes1 (32 filtros, kernel 7x7, stride (1,3), pad SAME (3,3)/(2,2))
//!   -> ReLU                       (sin BatchNorm)
//!   -> conv notes2 (1 filtro, kernel 7x3, pad SAME (3,3)/(1,1))
//!   -> sigmoid
//!   -> FlattenFreqCh -> (batch, 172, 88)   == note
//! ```
//!
//! La salida intermedia `x_notes_pre` (antes de aplanar) tiene forma
//! `(batch, 1, 172, 88)` y se concatena con la rama de onsets para formar los
//! **33** canales de entrada de la convolucion final de onsets.

use candle_core::Tensor;
use candle_nn::ops::sigmoid;

use crate::layers::{conv2d_same, same_padding2d};
use crate::trace::{a_nhwc, ModelTrace};
use crate::weights::Weights;

/// Nombres canonicos de los pesos de la rama.
pub mod weight_names {
    /// Kernel de la primera convolucion.
    pub const CONV1_WEIGHT: &str = "notes.conv1.weight";
    /// Bias de la primera convolucion.
    pub const CONV1_BIAS: &str = "notes.conv1.bias";
    /// Kernel de la convolucion de salida.
    pub const CONV2_WEIGHT: &str = "notes.conv2.weight";
    /// Bias de la convolucion de salida.
    pub const CONV2_BIAS: &str = "notes.conv2.bias";
}

/// Numero de filtros de la primera convolucion.
pub const CONV1_FILTERS: usize = 32;
/// Kernel de la primera convolucion, como (tiempo, frecuencia).
pub const CONV1_KERNEL: (usize, usize) = (7, 7);
/// Stride de la primera convolucion, como (tiempo, frecuencia).
pub const CONV1_STRIDE: (usize, usize) = (1, 3);
/// Numero de filtros de la convolucion de salida.
pub const CONV2_FILTERS: usize = 1;
/// Kernel de la convolucion de salida, como (tiempo, frecuencia).
pub const CONV2_KERNEL: (usize, usize) = (7, 3);
/// Bins de la salida: 88 notas.
pub const N_NOTES: usize = 88;

/// La rama de notas.
#[derive(Debug, Clone)]
pub struct NotesBranch {
    /// Kernel de la primera convolucion: `(32, 1, 7, 7)`.
    pub conv1_weight: Tensor,
    /// Bias de la primera convolucion: `(32)`.
    pub conv1_bias: Tensor,
    /// Kernel de la convolucion de salida: `(1, 32, 7, 3)`.
    pub conv2_weight: Tensor,
    /// Bias de la convolucion de salida: `(1)`.
    pub conv2_bias: Tensor,
}

impl NotesBranch {
    /// Construye la rama a partir de los pesos cargados.
    pub fn from_weights(weights: &Weights) -> Result<Self, crate::ModelError> {
        Ok(Self {
            conv1_weight: weights
                .expect(
                    weight_names::CONV1_WEIGHT,
                    &[CONV1_FILTERS, 1, CONV1_KERNEL.0, CONV1_KERNEL.1],
                )?
                .clone(),
            conv1_bias: weights
                .expect(weight_names::CONV1_BIAS, &[CONV1_FILTERS])?
                .clone(),
            conv2_weight: weights
                .expect(
                    weight_names::CONV2_WEIGHT,
                    &[CONV2_FILTERS, CONV1_FILTERS, CONV2_KERNEL.0, CONV2_KERNEL.1],
                )?
                .clone(),
            conv2_bias: weights
                .expect(weight_names::CONV2_BIAS, &[CONV2_FILTERS])?
                .clone(),
        })
    }

    /// Ejecuta la rama.
    ///
    /// # Formas (NCHW, el layout de Candle)
    ///
    /// | Tensor | Forma | Equivalente NHWC |
    /// |---|---|---|
    /// | `x` (entrada) | `(batch, 1, 172, 264)` | `(batch, 172, 264, 1)` |
    /// | `notes_features` | `(batch, 32, 172, 88)` | `(batch, 172, 88, 32)` |
    /// | `x_notes_pre` | `(batch, 1, 172, 88)` | `(batch, 172, 88, 1)` |
    /// | `note` (salida) | `(batch, 172, 88)` | `(batch, 172, 88)` |
    ///
    /// `x_notes_pre` es la salida **sigmoid** de la conv de notas, y es la
    /// que se concatena con la rama de onsets (33 canales en total).
    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor), candle_core::Error> {
        self.forward_with_trace(x, &mut None)
    }

    /// Ejecuta la rama capturando los puntos de inspeccion de la Fase 7.
    ///
    /// Es **la misma implementacion** que `forward`: la traza solo recoge
    /// tensores que ya se han calculado, no cambia el camino de ejecucion ni
    /// un bit del resultado. No existe una segunda copia de la red que se
    /// pueda validar por error. Ver `crate::trace`.
    ///
    /// `trace` es `None` cuando no se quiere traza, y entonces no se hace
    /// **ni una copia extra**.
    pub fn forward_with_trace(
        &self,
        x: &Tensor,
        trace: &mut Option<ModelTrace>,
    ) -> Result<(Tensor, Tensor), candle_core::Error> {
        // --- conv notes1 (32, 7x7, stride 1x3) ---
        let pad = same_padding2d((172, 264), CONV1_KERNEL, CONV1_STRIDE);
        let conv1 = conv2d_same(x, &self.conv1_weight, pad, CONV1_STRIDE)?;
        let conv1 = conv1.broadcast_add(&self.conv1_bias.reshape((1, 32, 1, 1))?)?;

        // Punto de inspeccion `notes_features`: la salida CRUDA de la
        // conv, **antes** del ReLU. En Keras la conv y el ReLU son capas
        // distintas, y el punto de referencia es el de la conv.
        if let Some(t) = trace.as_mut() {
            t.notes_features = Some(a_nhwc(&conv1)?);
        }

        // --- ReLU (esta rama NO tiene BatchNorm) ---
        let conv1 = conv1.relu()?;

        // --- conv notes2 (1, 7x3) + sigmoid ---
        let pad = same_padding2d((172, N_NOTES), CONV2_KERNEL, (1, 1));
        let conv2 = conv2d_same(&conv1, &self.conv2_weight, pad, (1, 1))?;
        let conv2 = conv2.broadcast_add(&self.conv2_bias.reshape((1, 1, 1, 1))?)?;
        let conv2 = sigmoid(&conv2)?;

        // `note` es el aplanado: (batch, 1, 172, 88) -> (batch, 172, 88).
        //
        // OJO: aqui NO hay que transponer. `FlattenFreqCh` de TensorFlow
        // aplana `(batch, 172, 88, 1)` a `(batch, 172, 88)`: elimina el eje
        // de canales y conserva el orden (tiempo, frecuencia). El
        // equivalente en NCHW es `squeeze(1)` y nada mas; un
        // `transpose(1, 2)` de mas devolveria (batch, 88, 172).
        let note = conv2.squeeze(1)?.contiguous()?;

        // Punto de inspeccion `note`.
        if let Some(t) = trace.as_mut() {
            t.note = Some(note.clone());
        }

        Ok((note, conv2))
    }
}

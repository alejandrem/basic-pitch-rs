//! Traza de los puntos de inspeccion de la red.
//!
//! # Que es
//!
//! La Fase 7 (validacion de equivalencia) exige comparar la red de Rust con
//! TensorFlow **etapa a etapa**, no solo en la salida final:
//!
//! ```text
//! entrada -> CQT -> NormalizedLog -> BatchNorm -> HarmonicStacking
//!         -> contour_conv -> contour_reduced -> contour
//!         -> notes_features -> note
//!         -> onset_features -> onset
//! ```
//!
//! Para eso hace falta poder mirar **dentro** de la red. La `ModelTrace`
//! recoge exactamente los siete puntos que define la especificacion de la
//! Fase 7, y nada mas: no se capturan tensores indiscriminadamente.
//!
//! # Por que una traza y no una copia de la red
//!
//! La alternativa seria reimplementar la red en un modulo de validacion.
//! Eso tiene tres problemas:
//!
//! 1. **Duplicacion**: dos implementaciones que pueden divergir.
//! 2. **Falsa confianza**: se podria validar la copia y que la red de
//!    produccion tuviera un error.
//! 3. **Mantenimiento**: cada cambio futuro habria que aplicarlo en los dos
//!    sitios.
//!
//! Con la traza, lo que se valida es **la red que usa produccion**.
//!
//! # Como se usa
//!
//! ```ignore
//! let mut trace = ModelTrace::default();
//! let salida = model.forward_with_trace(&harmonic, Some(&mut trace))?;
//!
//! // `trace.contour_conv` es la salida de la conv, antes del BatchNorm
//! // y del ReLU.
//! assert_eq!(trace.contour_conv.as_ref().unwrap().dims4().unwrap(),
//!            (1, 172, 264, 8));
//! ```
//!
//! Sin traza, `forward()` hace exactamente lo mismo que antes y no
//! cambia ni un bit del resultado.
//!
//! # Las ramas
//!
//! Las tres ramas (`ContourBranch`, `NotesBranch`, `OnsetBranch`) tienen
//! dos metodos:
//!
//! ```ignore
//! pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
//!     self.forward_with_trace(input, &mut None)
//! }
//!
//! pub fn forward_with_trace(
//!     &self,
//!     input: &Tensor,
//!     trace: &mut Option<ModelTrace>,
//! ) -> Result<Tensor>
//! ```
//!
//! El `&mut Option<ModelTrace>` de las ramas (y no `Option<&mut ...>`) es
//! lo que permite pasar la misma traza a las tres ramas sin moverla: cada
//! una la toma prestada. El modelo recibe `Option<&mut ModelTrace>` y
//! vuelca al final lo que las ramas recogieron.
//!
//! # El layout
//!
//! Los tensores se guardan en el layout de los puntos de referencia
//! (**NHWC**: tiempo, frecuencia, canales), que es el que usan los `.npy`
//! de `reference/reference_outputs/`. Asi la comparacion con la referencia
//! es una comparacion elemento a elemento, sin ninguna conversion de por
//! medio que pueda equivocarse.
//!
//! Internamente la red trabaja en NCHW, de modo que la traza hace la
//! permutacion **solo cuando se pide**; sin traza, no hay ni una copia
//! extra.

use candle_core::Tensor;

/// Los siete puntos de inspeccion de la Fase 7.
///
/// Todos los campos son `Option`: `None` significa "este punto no se ha
/// capturado", y todos valen `None` si la red se ejecuta sin traza.
#[derive(Debug, Clone, Default)]
pub struct ModelTrace {
    /// Salida de la conv de contornos, **antes** del BatchNorm y del ReLU.
    ///
    /// Forma: `(batch, 172, 264, 8)`.
    pub contour_conv: Option<Tensor>,

    /// Salida de la conv que reduce los contornos, con su sigmoid.
    ///
    /// Forma: `(batch, 172, 264, 1)`.
    pub contour_reduced: Option<Tensor>,

    /// Contorno final, ya aplanado.
    ///
    /// Forma: `(batch, 172, 264)`.
    pub contour: Option<Tensor>,

    /// Salida de la conv de notas, **antes** del ReLU.
    ///
    /// Forma: `(batch, 172, 88, 32)`.
    pub notes_features: Option<Tensor>,

    /// Prediccion de notas.
    ///
    /// Forma: `(batch, 172, 88)`.
    pub note: Option<Tensor>,

    /// Salida de la conv de onsets, **antes** del BatchNorm y del ReLU.
    ///
    /// Forma: `(batch, 172, 88, 32)`.
    pub onset_features: Option<Tensor>,

    /// Prediccion de onsets.
    ///
    /// Forma: `(batch, 172, 88)`.
    pub onset: Option<Tensor>,
}

impl ModelTrace {
    /// Indica si se ha capturado algun punto.
    pub fn is_empty(&self) -> bool {
        self.contour_conv.is_none()
            && self.contour_reduced.is_none()
            && self.contour.is_none()
            && self.notes_features.is_none()
            && self.note.is_none()
            && self.onset_features.is_none()
            && self.onset.is_none()
    }

    /// Numero de puntos capturados.
    pub fn count(&self) -> usize {
        [
            self.contour_conv.is_some(),
            self.contour_reduced.is_some(),
            self.contour.is_some(),
            self.notes_features.is_some(),
            self.note.is_some(),
            self.onset_features.is_some(),
            self.onset.is_some(),
        ]
        .iter()
        .filter(|capturado| **capturado)
        .count()
    }
}

/// Convierte un tensor NCHW al layout de los puntos de referencia (NHWC).
///
/// `permute` cambia la disposicion de los ejes; `contiguous` compacta la
/// memoria para que el tensor se pueda comparar sin saltos.
pub(crate) fn a_nhwc(tensor: &Tensor) -> Result<Tensor, candle_core::Error> {
    tensor.permute((0, 2, 3, 1))?.contiguous()
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn una_traza_vacia_no_tiene_nada() {
        let trace = ModelTrace::default();
        assert!(trace.is_empty());
        assert_eq!(trace.count(), 0);
    }

    #[test]
    fn la_conversion_a_nhwc_intercambia_los_ejes() {
        // (batch, canales, tiempo, frecuencia) -> (batch, tiempo, frecuencia, canales)
        let tensor = Tensor::from_slice(
            &(0..(2 * 3 * 4 * 5)).map(|v| v as f32).collect::<Vec<_>>(),
            (2, 3, 4, 5),
            &Device::Cpu,
        )
        .unwrap();
        let nhwc = a_nhwc(&tensor).unwrap();
        assert_eq!(nhwc.dims4().unwrap(), (2, 4, 5, 3));

        // El valor [b, c, t, f] se mueve a [b, t, f, c].
        let datos = nhwc.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let original = tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for b in 0..2 {
            for c in 0..3 {
                for t in 0..4 {
                    for f in 0..5 {
                        let antes = original[((b * 3 + c) * 4 + t) * 5 + f];
                        let despues = datos[((b * 4 + t) * 5 + f) * 3 + c];
                        assert_eq!(antes, despues, "b={b} c={c} t={t} f={f}");
                    }
                }
            }
        }
    }
}

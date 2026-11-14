//! Batch Normalization en modo de inferencia.
//!
//! Reproduce el comportamiento de `tf.keras.layers.BatchNormalization` en
//! modo inferencia (que es el unico que usa el modelo preentrenado):
//!
//! ```text
//! scale = gamma / sqrt(moving_variance + epsilon)
//! out   = (x - moving_mean) * scale + beta
//! ```
//!
//! # La BatchNorm inicial
//!
//! Es la que se aplica justo despues de `NormalizedLog`, sobre un tensor de
//! forma `(batch, 172, 309, 1)`. **Un solo canal**, de modo que `gamma`,
//! `beta`, `moving_mean` y `moving_variance` tienen longitud **1**: son
//! escalares, no un vector por bin de frecuencia.
//!
//! La normalizacion se calcula sobre las dimensiones `(batch, tiempo,
//! frecuencia)`: una unica estadistica global por muestra.
//!
//! El `epsilon` es el de Keras, **0.001**. El atributo `epsilon=1e-5` del
//! nodo `FusedBatchNormV3` del ONNX oficial **no** es el valor de este
//! modelo, porque el ONNX fusiona la normalizacion dentro de la
//! convolucion.
//!
//! Ver `docs/practica/fase-08-frontend-cqt.md`.

use candle_core::Tensor;

/// Error en la normalizacion por lotes.
#[derive(Debug, thiserror::Error)]
pub enum BatchNormError {
    /// Error numerico de Candle.
    #[error("error de Candle en BatchNorm: {0}")]
    Candle(#[from] candle_core::Error),

    /// Un parametro tiene una forma inesperada.
    #[error("el parametro '{name}' tiene forma {actual:?}, se esperaba {expected:?}")]
    UnexpectedShape {
        /// Nombre del parametro.
        name: &'static str,
        /// Forma encontrada.
        actual: Vec<usize>,
        /// Forma esperada.
        expected: Vec<usize>,
    },
}

/// Parametros de una BatchNorm en modo de inferencia.
#[derive(Debug, Clone)]
pub struct BatchNormParams {
    /// Escala aprendida (`gamma`).
    pub gamma: Tensor,
    /// Desplazamiento aprendido (`beta`).
    pub beta: Tensor,
    /// Media en movimiento (`moving_mean`).
    pub moving_mean: Tensor,
    /// Varianza en movimiento (`moving_variance`).
    pub moving_variance: Tensor,
    /// `epsilon` de Keras.
    pub epsilon: f64,
}

impl BatchNormParams {
    /// Normaliza un tensor `(batch, tiempo, frecuencia)`.
    ///
    /// Los cuatro parametros son escalares (longitud 1), de modo que el
    /// resultado tiene la misma forma que la entrada.
    pub fn normalize(&self, input: &Tensor) -> Result<Tensor, BatchNormError> {
        for (name, tensor) in [
            ("gamma", &self.gamma),
            ("beta", &self.beta),
            ("moving_mean", &self.moving_mean),
            ("moving_variance", &self.moving_variance),
        ] {
            let dims = tensor.dims().to_vec();
            if dims != [1] {
                return Err(BatchNormError::UnexpectedShape {
                    name,
                    actual: dims,
                    expected: vec![1],
                });
            }
        }

        // scale = gamma / sqrt(moving_variance + epsilon)
        let variance = self.moving_variance.affine(1.0, self.epsilon)?;
        let scale = self.gamma.div(&variance.sqrt()?)?;

        // out = (x - moving_mean) * scale + beta
        let centered = input.broadcast_sub(&self.moving_mean)?;
        let scaled = centered.broadcast_mul(&scale)?;
        Ok(scaled.broadcast_add(&self.beta)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn params(gamma: f32, beta: f32, mean: f32, var: f32) -> BatchNormParams {
        BatchNormParams {
            gamma: Tensor::from_slice(&[gamma], (1,), &Device::Cpu).unwrap(),
            beta: Tensor::from_slice(&[beta], (1,), &Device::Cpu).unwrap(),
            moving_mean: Tensor::from_slice(&[mean], (1,), &Device::Cpu).unwrap(),
            moving_variance: Tensor::from_slice(&[var], (1,), &Device::Cpu).unwrap(),
            epsilon: 0.001,
        }
    }

    #[test]
    fn normaliza_con_la_formula_de_keras() {
        let input = Tensor::from_slice(&[1.0f32, 2.0, 3.0], (1, 1, 3), &Device::Cpu).unwrap();
        let p = params(2.0, 1.0, 1.0, 4.0);
        let out = p.normalize(&input).unwrap();
        let data = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let scale = 2.0f32 / (4.0f32 + 0.001).sqrt();
        for (i, value) in data.iter().enumerate() {
            let x = i as f32 + 1.0;
            let expected = (x - 1.0) * scale + 1.0;
            assert!((value - expected).abs() < 1e-6, "{value} != {expected}");
        }
    }

    #[test]
    fn la_forma_se_conserva() {
        let input = Tensor::zeros((1, 172, 309), candle_core::DType::F32, &Device::Cpu).unwrap();
        let out = params(1.0, 0.0, 0.0, 1.0).normalize(&input).unwrap();
        assert_eq!(out.dims3().unwrap(), (1, 172, 309));
    }

    #[test]
    fn un_parametro_con_forma_incorrecta_produce_error() {
        let input = Tensor::zeros((1, 2, 2), candle_core::DType::F32, &Device::Cpu).unwrap();
        let mut p = params(1.0, 0.0, 0.0, 1.0);
        p.gamma = Tensor::from_slice(&[1.0f32, 2.0], (2,), &Device::Cpu).unwrap();
        assert!(matches!(
            p.normalize(&input),
            Err(BatchNormError::UnexpectedShape { name: "gamma", .. })
        ));
    }

    #[test]
    fn el_epsilon_de_keras_es_el_que_se_usa() {
        // Con varianza 0, el epsilon evita la division por cero y el
        // resultado es finito.
        let input = Tensor::from_slice(&[1.0f32], (1, 1, 1), &Device::Cpu).unwrap();
        let out = params(1.0, 0.0, 0.0, 0.0).normalize(&input).unwrap();
        let data = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(data[0].is_finite());
        assert!((data[0] - 1.0 / 0.001f32.sqrt()).abs() < 1e-3);
    }
}

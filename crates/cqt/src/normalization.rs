//! NormalizedLog: la transformacion logaritmica normalizada del frontend.
//!
//! Reproduce `basic_pitch/layers/signal.py`, clase `NormalizedLog`:
//!
//! ```text
//! 1. potencia        = magnitud^2
//! 2. estabilizador   = potencia + 1e-10
//! 3. logaritmo       = log10(estabilizador)
//! 4. escala          = 10 * logaritmo
//! 5. minimo          = min(escala) por muestra
//! 6. desplazamiento  = escala - minimo
//! 7. maximo          = max(desplazamiento) por muestra
//! 8. normalizado     = desplazamiento / maximo   (div_no_nan)
//! ```
//!
//! Los pasos 5 y 7 se reducen sobre los ejes de tiempo y frecuencia, de modo
//! que la normalizacion es **global por muestra**, no por bin de frecuencia.
//!
//! No debe sustituirse por una normalizacion estandar de imagen ni por una
//! normalizacion de amplitud global.

use candle_core::Tensor;

/// Error en la normalizacion logaritmica.
#[derive(Debug, thiserror::Error)]
pub enum NormalizedLogError {
    /// Las dimensiones del tensor de entrada no son las esperadas.
    #[error("dimensiones inesperadas: {0:?}, se esperaba (batch, time, freq)")]
    UnexpectedShape(Vec<usize>),

    /// Error numerico de Candle.
    #[error("error de Candle en NormalizedLog: {0}")]
    Candle(#[from] candle_core::Error),
}

/// Valor de estabilizacion anadido antes del logaritmo.
pub const EPSILON: f64 = 1e-10;

/// Aplica `NormalizedLog` a un tensor `(batch, time, freq)`.
///
/// Devuelve un tensor con la misma forma, con valores en el intervalo `[0, 1]`
/// (cero cuando la muestra es constante).
pub fn normalized_log(input: &Tensor) -> Result<Tensor, NormalizedLogError> {
    let (batch, time, freq) = input.dims3()?;
    if freq == 0 {
        return Err(NormalizedLogError::UnexpectedShape(vec![batch, time, freq]));
    }

    // 1. Potencia: magnitud al cuadrado.
    let power = input.sqr()?;

    // 2. Estabilizador numerico.
    let stabilized = power.affine(1.0, EPSILON)?;

    // 3 y 4. Logaritmo en base 10, multiplicado por 10.
    let log_power = stabilized
        .log()?
        .affine(10.0 / std::f64::consts::LN_10, 0.0)?;

    // 5. Minimo por muestra, sobre los ejes de tiempo y frecuencia.
    //    Se aplana primero porque `min` reduce sobre un solo eje.
    let flat = log_power.reshape((batch, time * freq))?;
    let min_per_sample = flat.min(1)?.unsqueeze(1)?.unsqueeze(1)?;

    // 6. Desplazamiento.
    let offset = log_power.broadcast_sub(&min_per_sample)?;

    // 7. Maximo por muestra.
    let offset_flat = offset.reshape((batch, time * freq))?;
    let max_per_sample = offset_flat.max(1)?.unsqueeze(1)?.unsqueeze(1)?;

    // 8. Division segura (equivale a `tf.math.divide_no_nan`).
    //    Cuando el maximo es 0, el numerador tambien lo es: se sustituye el
    //    denominador por 1 para evitar la division por cero, y el resultado
    //    queda en 0, que es el valor que produce `divide_no_nan`.
    let shape = max_per_sample.shape().dims();
    let dtype = max_per_sample.dtype();
    let device = max_per_sample.device();

    let zeros = Tensor::zeros(shape, dtype, device)?;
    let is_zero = max_per_sample.eq(&zeros)?;
    let is_zero_f = is_zero.to_dtype(dtype)?;
    let one = Tensor::ones(shape, dtype, device)?;
    let not_zero = one.broadcast_sub(&is_zero_f)?;

    // safe_denominator = denom * not_zero + is_zero
    let safe_denominator = max_per_sample
        .broadcast_mul(&not_zero)?
        .broadcast_add(&is_zero_f)?;

    Ok(offset.broadcast_div(&safe_denominator)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn la_salida_queda_normalizada_entre_cero_y_uno() {
        let input = Tensor::from_slice(
            &[0.1f32, 0.2, 0.5, 0.9, 1.0, 0.3, 0.7, 0.05],
            (1, 4, 2),
            &Device::Cpu,
        )
        .unwrap();
        let out = normalized_log(&input).unwrap();
        let data = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(data.iter().all(|v| *v >= 0.0 && *v <= 1.0));
    }

    #[test]
    fn el_minimo_de_cada_muestra_es_cero() {
        let input = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (1, 3, 2), &Device::Cpu)
            .unwrap();
        let out = normalized_log(&input).unwrap();
        let data = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let min = data.iter().copied().fold(f32::INFINITY, f32::min);
        assert!(min.abs() < 1e-6, "minimo = {min}");
    }

    #[test]
    fn una_muestra_constante_no_produce_nan() {
        // El denominador es 0: `divide_no_nan` debe devolver 0, no NaN.
        let input = Tensor::from_slice(&[0.5f32; 4], (1, 2, 2), &Device::Cpu).unwrap();
        let out = normalized_log(&input).unwrap();
        let data = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(data.iter().all(|v| v.is_finite()), "valores: {data:?}");
        assert!(data.iter().all(|v| v.abs() < 1e-6), "valores: {data:?}");
    }
}

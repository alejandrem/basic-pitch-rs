//! Convoluciones 2D con la semantica de TensorFlow.
//!
//! # Por que no se usa `candle_nn::Conv2d` directamente
//!
//! `candle_nn::Conv2dConfig` solo admite:
//!
//! ```text
//! pub struct Conv2dConfig {
//!     pub padding: usize,   // un unico valor, simetrico, en AMBOS ejes
//!     pub stride: usize,    // un unico valor, en AMBOS ejes
//!     ...
//! }
//! ```
//!
//! El modelo original usa `padding="same"` (asimétrico cuando el stride > 1)
//! y strides por eje `(1, 3)`. Ninguna de las dos cosas es representable.
//!
//! # La estrategia: padding manual + conv stride-1 + submuestreo
//!
//! Con el mismo padding manual aplicado antes de la conv:
//!
//! ```text
//! y_salida[o] = sum_j w[j] * x_pad[o*s + j]
//! y_conv1[m]  = sum_j w[j] * x_pad[m + j]
//! => y_salida[o] = y_conv1[o*s]
//! ```
//!
//! Es decir, la convolución con stride `s` es **exactamente** la convolución
//! con stride 1 tomando cada s-ésimo resultado. La equivalencia es
//! algebraica, no aproximada.
//!
//! Lo que **no** se promete es igualdad bit a bit entre backends: TensorFlow,
//! Candle y el motor de convolución de Rust acumulan en órdenes distintos, de
//! modo que diferencias del orden de `1e-6` en `float32` son esperables. La
//! equivalencia debe verificarse numericamente en la Fase 7.
//!
//! # Orientacion
//!
//! Las funciones de este modulo trabajan en **NCHW** (el layout de Candle):
//! `(batch, canales, tiempo, frecuencia)`. El layout NHWC de TensorFlow es
//! `(batch, tiempo, frecuencia, canales)`.

use candle_core::Tensor;

/// Padding SAME de TensorFlow para un eje.
///
/// ```text
/// out      = ceil(in / stride)
/// pad_tot  = max((out - 1) * stride + kernel - in, 0)
/// pad_ant  = pad_tot / 2      (division entera)
/// pad_post = pad_tot - pad_ant
/// ```
///
/// Devuelve `(pad_antes, pad_despues)`.
pub fn same_padding(in_size: usize, kernel: usize, stride: usize) -> (usize, usize) {
    assert!(stride > 0, "el stride debe ser mayor que cero");
    let out = in_size.div_ceil(stride);
    let pad_total = ((out - 1) * stride + kernel).saturating_sub(in_size);
    let pad_before = pad_total / 2;
    (pad_before, pad_total - pad_before)
}

/// Padding SAME de TensorFlow para los dos ejes de una conv2d.
///
/// Devuelve `[(tiempo_antes, tiempo_despues), (frec_antes, frec_despues)]`.
pub fn same_padding2d(
    in_size: (usize, usize),
    kernel: (usize, usize),
    stride: (usize, usize),
) -> [(usize, usize); 2] {
    [
        same_padding(in_size.0, kernel.0, stride.0),
        same_padding(in_size.1, kernel.1, stride.1),
    ]
}

/// Aplica un relleno de ceros a un tensor NCHW.
///
/// `pad` es `[(tiempo_antes, tiempo_despues), (frecuencia_antes, frecuencia_despues)]`.
pub fn pad2d(x: &Tensor, pad: [(usize, usize); 2]) -> Result<Tensor, candle_core::Error> {
    let (b, c, t, _f) = x.dims4()?;
    let (pt, pb) = pad[0];
    let (pf, pfb) = pad[1];

    if pt == 0 && pb == 0 && pf == 0 && pfb == 0 {
        return Ok(x.clone());
    }

    let mut result = x.clone();

    // Eje de frecuencia (ultima dimension).
    if pf > 0 {
        let zeros = Tensor::zeros((b, c, t, pf), x.dtype(), x.device())?;
        result = Tensor::cat(&[&zeros, &result], 3)?;
    }
    if pfb > 0 {
        let (b2, c2, t2, _) = result.dims4()?;
        let zeros = Tensor::zeros((b2, c2, t2, pfb), x.dtype(), x.device())?;
        result = Tensor::cat(&[&result, &zeros], 3)?;
    }

    // Eje de tiempo.
    if pt > 0 {
        let (b2, c2, _, f2) = result.dims4()?;
        let zeros = Tensor::zeros((b2, c2, pt, f2), x.dtype(), x.device())?;
        result = Tensor::cat(&[&zeros, &result], 2)?;
    }
    if pb > 0 {
        let (b2, c2, _, f2) = result.dims4()?;
        let zeros = Tensor::zeros((b2, c2, pb, f2), x.dtype(), x.device())?;
        result = Tensor::cat(&[&result, &zeros], 2)?;
    }

    Ok(result)
}

/// Submuestreo en el eje de frecuencia de un tensor NCHW.
///
/// Equivale a `x[:, :, :, 0::stride]`.
pub fn subsample_freq(x: &Tensor, stride: usize) -> Result<Tensor, candle_core::Error> {
    if stride == 1 {
        return Ok(x.clone());
    }
    let (_, _, _, f) = x.dims4()?;
    let mut indices: Vec<u32> = Vec::new();
    let mut i = 0usize;
    while i < f {
        indices.push(i as u32);
        i += stride;
    }
    let index = Tensor::from_slice(&indices, (indices.len(),), x.device())?;
    // `index_select` sobre el eje 3 (frecuencia) devuelve un tensor con la
    // misma forma salvo en ese eje.
    x.index_select(&index, 3)
}

/// Convolucion 2D con la semantica de TensorFlow.
///
/// `x` es NCHW `(batch, in_channels, tiempo, frecuencia)`.
/// `kernel` es `(out_channels, in_channels, kernel_tiempo, kernel_frecuencia)`.
///
/// `stride` es `(stride_tiempo, stride_frecuencia)`. **Solo se admite
/// `stride_tiempo == 1`**: el submuestreo se aplica unicamente en el eje de
/// frecuencia, que es lo que necesita este modelo. La asercion evita que
/// alguien use accidentalmente un stride temporal que la funcion no
/// implementa y obtenga un resultado silenciosamente incorrecto.
///
/// `pad` es el padding SAME por eje: `[(tiempo_antes, tiempo_despues), (frec_antes, frec_despues)]`.
pub fn conv2d_same(
    x: &Tensor,
    kernel: &Tensor,
    pad: [(usize, usize); 2],
    stride: (usize, usize),
) -> Result<Tensor, candle_core::Error> {
    assert_eq!(
        stride.0, 1,
        "conv2d_same solo implementa stride temporal 1; se ha pedido {stride:?}"
    );

    // 1. Relleno manual.
    let padded = pad2d(x, pad)?;

    // 2. Convolucion con stride 1 y sin relleno adicional.
    let convolved = padded.conv2d(kernel, 0, 1, 1, 1)?;

    // 3. Submuestreo en el eje de frecuencia.
    if stride.1 == 1 {
        return Ok(convolved);
    }
    subsample_freq(&convolved, stride.1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn el_padding_same_coincide_con_tensorflow() {
        // notes conv1: kernel 7, stride 3, entrada 264 -> salida 88.
        assert_eq!(same_padding(264, 7, 3), (2, 2));
        // onset conv1: kernel 5, stride 3, entrada 264 -> salida 88.
        assert_eq!(same_padding(264, 5, 3), (1, 1));
        // contour conv: kernel 39, stride 1, entrada 264.
        assert_eq!(same_padding(264, 39, 1), (19, 19));
        // Eje temporal: kernel 7, stride 1, entrada 172.
        assert_eq!(same_padding(172, 7, 1), (3, 3));
        // notes conv2: kernel 3, stride 1, entrada 88.
        assert_eq!(same_padding(88, 3, 1), (1, 1));
    }

    #[test]
    fn el_relleno_agrega_los_valores_esperados() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (1, 1, 2, 2), &Device::Cpu).unwrap();
        let padded = pad2d(&x, [(1, 1), (1, 1)]).unwrap();
        assert_eq!(padded.dims4().unwrap(), (1, 1, 4, 4));
        let data = padded.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // El tensor original es (1, 2, 2) = [1, 2, 3, 4]; con relleno de un
        // elemento en cada borde queda en el centro.
        assert_eq!(data[0], 0.0);
        assert_eq!(data[5], 1.0);
        assert_eq!(data[6], 2.0);
        assert_eq!(data[9], 3.0);
        assert_eq!(data[10], 4.0);
        assert_eq!(data[15], 0.0);
    }

    #[test]
    fn el_submuestreo_toma_cada_tercero() {
        let x = Tensor::from_slice(
            &(0..24).map(|v| v as f32).collect::<Vec<_>>(),
            (1, 1, 2, 12),
            &Device::Cpu,
        )
        .unwrap();
        let out = subsample_freq(&x, 3).unwrap();
        assert_eq!(out.dims4().unwrap(), (1, 1, 2, 4));
        let data = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // Primer plano (tiempo 0): 0, 3, 6, 9
        assert_eq!(&data[0..4], &[0.0, 3.0, 6.0, 9.0]);
        // Segundo plano (tiempo 1): 12, 15, 18, 21
        assert_eq!(&data[4..8], &[12.0, 15.0, 18.0, 21.0]);
    }

    #[test]
    fn la_convolucion_con_stride_3_conserva_88_frecuencias() {
        // Entrada (1, 1, 172, 264), kernel (1, 1, 7, 7).
        let x = Tensor::zeros((1, 1, 172, 264), DType::F32, &Device::Cpu).unwrap();
        let kernel = Tensor::zeros((1, 1, 7, 7), DType::F32, &Device::Cpu).unwrap();
        let out = conv2d_same(&x, &kernel, [(3, 3), (2, 2)], (1, 3)).unwrap();
        let dims = out.dims4().unwrap();
        assert_eq!(dims.2, 172, "el tiempo no debe cambiar");
        assert_eq!(dims.3, 88, "la frecuencia debe quedar en 88");
    }
}

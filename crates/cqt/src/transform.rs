//! La transformada CQT multirresolucion.
//!
//! Reproduce `basic_pitch/layers/nnaudio.py`, clase `CQT2010v2`, con los
//! parametros del modelo ICASSP 2022 (`sr=22050`, `hop_length=256`,
//! `fmin=27.5`, `n_bins=309`, `bins_per_octave=36`).
//!
//! # El algoritmo
//!
//! 1. Se calcula la CQT de la **octava superior** con el kernel de 36
//!    filtros y el hop completo (256 muestras).
//! 2. Para bajar una octava, el audio se **submuestrea por 2** con el filtro
//!    `lowpass` y se vuelve a aplicar el mismo kernel, con la mitad de hop.
//! 3. Se repite hasta cubrir las 9 octavas (`n_octaves = ceil(309/36)`).
//! 4. Las octavas se concatenan de la mas grave a la mas aguda y se conservan
//!    los **309** bins de mayor frecuencia.
//! 5. Se normaliza por `sqrt(lengths)` y se toma la magnitud.
//!
//! # Detalles que importan
//!
//! - El relleno de la senal antes de cada conv es **reflectante** (modo
//!   `REFLECT` de TensorFlow), no de ceros: `[a b c] -> [b a b c b]`.
//! - El submuestreo usa el filtro `lowpass` con **padding de ceros manual**
//!   de 127 muestras a cada lado (`(256-1)//2`), no `SAME`.
//! - La parte imaginaria se calcula con el kernel imaginario y se **cambia de
//!   signo**: `CQT_imag = -conv(imag_kernel)`.
//! - La normalizacion por `sqrt(lengths)` se aplica a la parte real y a la
//!   imaginaria **antes** de tomar la magnitud.
//! - El numero de frames es 172 en todas las octavas: el relleno reflectante
//!   y el submuestreo estan disenados para que eso ocurra.
//!
//! # Nota sobre el batch
//!
//! En la referencia, la normalizacion por `sqrt(lengths)` se aplica con la
//! forma `(309, 1, 1)` sobre un tensor `(batch, 309, frames)`, lo que solo
//! funciona con `batch == 1` (TensorFlow alinearia los ejes por la derecha y
//! exigiria `batch == 309` en caso contrario). Aqui la operacion se hace en
//! NCHW, con la forma `(1, 309, 1)`, que funciona con cualquier batch.
//!
//! Ver `docs/practica/fase-08-frontend-cqt.md`.

use candle_core::{Device, Tensor};

use crate::kernels::CqtKernels;
use crate::CqtConfig;

/// Error del calculo de la CQT.
#[derive(Debug, thiserror::Error)]
pub enum TransformError {
    /// Error numerico de Candle.
    #[error("error de Candle en la CQT: {0}")]
    Candle(#[from] candle_core::Error),

    /// La senal de entrada no tiene muestras.
    #[error("la senal de entrada esta vacia")]
    EmptyInput,

    /// El relleno reflectante no cabe en la senal.
    ///
    /// TensorFlow exige que el relleno sea **menor** que la dimension de la
    /// senal: con un relleno igual o mayor, el modo `REFLECT` no esta
    /// definido.
    #[error("el relleno reflectante de {pad} muestras no cabe en una senal de {len} muestras")]
    ReflectPadTooLarge {
        /// Relleno solicitado.
        pad: usize,
        /// Longitud de la senal.
        len: usize,
    },

    /// Las octavas generadas no alcanzan para los bins pedidos.
    #[error("las octavas generadas dan {total_bins} bins y se piden {n_bins}")]
    NotEnoughBins {
        /// Bins generados.
        total_bins: usize,
        /// Bins pedidos.
        n_bins: usize,
    },

    /// El factor de normalizacion no corresponde a los bins pedidos.
    #[error("el factor de normalizacion tiene {actual} entradas y se esperan {expected}")]
    NormalizationMismatch {
        /// Entradas del factor de normalizacion.
        actual: usize,
        /// Entradas esperadas.
        expected: usize,
    },

    /// El numero de filtros del kernel no coincide con la configuracion.
    #[error("el kernel tiene {actual} filtros y la configuracion pide {expected}")]
    FilterMismatch {
        /// Filtros del kernel.
        actual: usize,
        /// Filtros pedidos.
        expected: usize,
    },
}

/// Calcula la CQT de una senal mono.
///
/// # Argumentos
///
/// - `samples`: la senal a 22.050 Hz.
/// - `kernels`: los kernels deterministas (`cqt_kernels.safetensors`).
/// - `config`: la configuracion de la transformada.
///
/// # Salida
///
/// `(batch, frames, n_bins)`, es decir `(1, 172, 309)` para una ventana del
/// modelo. El eje de frecuencia va **en orden ascendente**: el bin 0 es la
/// frecuencia mas grave (`fmin = 27.5 Hz`).
pub fn transform(
    samples: &[f32],
    kernels: &CqtKernels,
    config: &CqtConfig,
) -> Result<Tensor, TransformError> {
    if samples.is_empty() {
        return Err(TransformError::EmptyInput);
    }

    let n_filters = kernels.real.dims4().map(|d| d.0).unwrap_or(0);
    if n_filters != config.n_filters {
        return Err(TransformError::FilterMismatch {
            actual: n_filters,
            expected: config.n_filters,
        });
    }
    let n_norm = kernels.lengths_sqrt.dims3().map(|d| d.0).unwrap_or(0);
    if n_norm != config.n_bins {
        return Err(TransformError::NormalizationMismatch {
            actual: n_norm,
            expected: config.n_bins,
        });
    }

    let device = Device::Cpu;

    // Pesos de las convoluciones, en el layout de Candle: (out, in, kernel).
    // Los kernels vienen en `[36, 1, 1, 256]`, que es el mismo layout con un
    // eje de tamano 1 de mas.
    let real_w = kernels.real.reshape((config.n_filters, 1, config.n_fft))?;
    let imag_w = kernels.imag.reshape((config.n_filters, 1, config.n_fft))?;
    let lowpass_w = kernels.lowpass_filter.reshape((1, 1, config.n_fft))?;

    // La senal, en el layout (batch, canales, muestras).
    let x = Tensor::from_slice(samples, (1, 1, samples.len()), &device)?;

    let pad = config.n_fft / 2;
    let mut hop = config.hop_length;

    // Una octava: parte real y parte imaginaria, ambas (1, 36, frames).
    let mut octaves: Vec<(Tensor, Tensor)> = Vec::with_capacity(config.n_octaves);
    octaves.push(cqt_octave(&x, &real_w, &imag_w, hop, pad)?);

    // Bajar octavas: submuestreo por 2 y hop a la mitad.
    let mut x_down = x;
    for _ in 1..config.n_octaves {
        x_down = downsample_by_two(&x_down, &lowpass_w)?;
        hop /= 2;
        octaves.push(cqt_octave(&x_down, &real_w, &imag_w, hop, pad)?);
    }

    // Ensamblado: de la octava mas grave a la mas aguda.
    let mut reals: Vec<&Tensor> = Vec::with_capacity(octaves.len());
    let mut imags: Vec<&Tensor> = Vec::with_capacity(octaves.len());
    for (real, imag) in octaves.iter().rev() {
        reals.push(real);
        imags.push(imag);
    }
    let real = Tensor::cat(&reals, 1)?;
    let imag = Tensor::cat(&imags, 1)?;

    // Se conservan los ultimos `n_bins`: los de mayor frecuencia. La
    // referencia hace exactamente lo mismo (`CQT[:, -n_bins:, :]`).
    let total_bins = real.dims3()?.1;
    if total_bins < config.n_bins {
        return Err(TransformError::NotEnoughBins {
            total_bins,
            n_bins: config.n_bins,
        });
    }
    let start = total_bins - config.n_bins;
    let real = real.narrow(1, start, config.n_bins)?;
    let imag = imag.narrow(1, start, config.n_bins)?;

    // Normalizacion por `sqrt(lengths)`, **antes** de la magnitud.
    let scale = kernels.lengths_sqrt.reshape((1, config.n_bins, 1))?;
    let real = real.broadcast_mul(&scale)?;
    let imag = imag.broadcast_mul(&scale)?;

    // Magnitud: sqrt(real^2 + imag^2).
    let magnitude = real.sqr()?.add(&imag.sqr()?)?.sqrt()?;

    // De (batch, frecuencia, frames) a (batch, frames, frecuencia).
    Ok(magnitude.permute((0, 2, 1))?)
}

/// CQT de una octava: parte real y parte imaginaria.
///
/// El relleno es reflectante y la conv es `VALID`, con el hop de la octava.
fn cqt_octave(
    x: &Tensor,
    real_w: &Tensor,
    imag_w: &Tensor,
    hop: usize,
    pad: usize,
) -> Result<(Tensor, Tensor), candle_core::Error> {
    let padded = reflect_pad_1d(x, pad)?;
    let real = padded.conv1d(real_w, 0, hop, 1, 1)?;
    // `CQT_imag = -conv1d(x, kernel_imag)`: el signo se invierte.
    let imag = padded.conv1d(imag_w, 0, hop, 1, 1)?.neg()?;
    Ok((real, imag))
}

/// Submuestreo por 2 con el filtro `lowpass`.
///
/// Reproduce `downsampling_by_n(x, filter, 2, match_torch_exactly=True)`:
///
/// 1. Relleno de **ceros** de `(kernel_len - 1) // 2` muestras a cada lado.
/// 2. Convolucion `VALID` con salto 2.
///
/// El relleno de ceros es manual a proposito: la referencia lo hace asi para
/// coincidir con Torch, y no con el `SAME` de TensorFlow.
fn downsample_by_two(x: &Tensor, lowpass: &Tensor) -> Result<Tensor, candle_core::Error> {
    let (batch, channels, _) = x.dims3()?;
    let kernel_len = lowpass.dims3()?.2;
    let pad = (kernel_len - 1) / 2;

    let zeros = Tensor::zeros((batch, channels, pad), x.dtype(), x.device())?;
    let padded = Tensor::cat(&[&zeros, x, &zeros], 2)?;
    padded.conv1d(lowpass, 0, 2, 1, 1)
}

/// Relleno reflectante de una senal (modo `REFLECT` de TensorFlow).
///
/// El reflejo no repite el valor del borde: `[a b c]` con un elemento de relleno
/// queda `[b a b c b]`. Equivale al modo `reflect` de NumPy.
///
/// `x` es `(batch, canales, muestras)` y el relleno se aplica al ultimo eje.
fn reflect_pad_1d(x: &Tensor, pad: usize) -> Result<Tensor, candle_core::Error> {
    if pad == 0 {
        return Ok(x.clone());
    }
    let (_, _, len) = x.dims3()?;
    if pad >= len {
        return Err(candle_core::Error::Msg(format!(
            "el relleno reflectante de {pad} muestras no cabe en una senal de {len} muestras"
        )));
    }

    let total = len + 2 * pad;
    let mut indices = Vec::with_capacity(total);
    for j in 0..total {
        let k = j as i64 - pad as i64;
        indices.push(reflect_index(k, len) as u32);
    }

    let index = Tensor::from_slice(&indices, (total,), x.device())?;
    x.index_select(&index, 2)
}

/// Indice de una senal reflectada.
///
/// Refleja sin repetir el borde: los indices negativos se invierten y los que
/// se pasan por el final se calculan como `2 * (n - 1) - k`.
fn reflect_index(k: i64, len: usize) -> usize {
    let n = len as i64;
    let mut k = k;
    if k < 0 {
        k = -k;
    }
    if k >= n {
        k = 2 * (n - 1) - k;
    }
    k as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    /// Configuracion de la CQT del modelo.
    fn config() -> CqtConfig {
        CqtConfig::default()
    }

    #[test]
    fn el_relleno_reflectante_no_repite_el_borde() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0], (1, 1, 3), &Device::Cpu).unwrap();
        let padded = reflect_pad_1d(&x, 1).unwrap();
        let data = padded.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // [1, 2, 3] -> [2, 1, 2, 3, 2]
        assert_eq!(data, vec![2.0, 1.0, 2.0, 3.0, 2.0]);
    }

    #[test]
    fn el_relleno_reflectante_con_dos_elementos() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0], (1, 1, 3), &Device::Cpu).unwrap();
        let padded = reflect_pad_1d(&x, 2).unwrap();
        let data = padded.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // [1, 2, 3] -> [3, 2, 1, 2, 3, 2, 1]
        assert_eq!(data, vec![3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn el_indice_reflectado_es_correcto() {
        // n = 3: los indices validos son 0, 1, 2.
        // Con relleno de 2, la senal queda [2, 1, 0, 1, 2, 1, 0].
        assert_eq!(reflect_index(-1, 3), 1);
        assert_eq!(reflect_index(-2, 3), 2);
        // Mas alla del primer reflejo: se vuelve a reflejar.
        assert_eq!(reflect_index(-3, 3), 1);
        assert_eq!(reflect_index(3, 3), 1);
        assert_eq!(reflect_index(4, 3), 0);
    }

    #[test]
    fn un_relleno_mayor_que_la_senal_produce_error() {
        let x = Tensor::from_slice(&[1.0f32, 2.0], (1, 1, 2), &Device::Cpu).unwrap();
        assert!(reflect_pad_1d(&x, 2).is_err());
    }

    #[test]
    fn el_submuestreo_reduce_a_la_mitad() {
        // Filtro identidad de 4 taps: solo importa el factor de escala.
        let x = Tensor::from_slice(
            &(0..100).map(|v| v as f32).collect::<Vec<_>>(),
            (1, 1, 100),
            &Device::Cpu,
        )
        .unwrap();
        let kernel = Tensor::from_slice(&[1.0f32, 0.0, 0.0, 0.0], (1, 1, 4), &Device::Cpu).unwrap();
        let out = downsample_by_two(&x, &kernel).unwrap();
        // Con relleno de 1 a cada lado: (100 + 2 - 4) / 2 + 1 = 50.
        assert_eq!(out.dims3().unwrap().2, 50);
    }

    #[test]
    fn la_senal_vacia_produce_error() {
        let kernels = CqtKernels {
            real: Tensor::zeros((36, 1, 1, 256), DType::F32, &Device::Cpu).unwrap(),
            imag: Tensor::zeros((36, 1, 1, 256), DType::F32, &Device::Cpu).unwrap(),
            lowpass_filter: Tensor::zeros((1, 1, 1, 256), DType::F32, &Device::Cpu).unwrap(),
            lengths_sqrt: Tensor::ones((309, 1, 1), DType::F32, &Device::Cpu).unwrap(),
            conv_bias: Tensor::zeros(36, DType::F32, &Device::Cpu).unwrap(),
        };
        assert!(matches!(
            transform(&[], &kernels, &config()),
            Err(TransformError::EmptyInput)
        ));
    }
}

//! Carga de los kernels deterministas de la CQT.
//!
//! Los kernels se extraen del ONNX oficial y se distribuyen como asset
//! (`cqt_kernels.safetensors`). Son parametros deterministas de la
//! transformada, **no** pesos entrenados.

use std::path::Path;

use candle_core::{DType, Device, Tensor};
use safetensors::{Dtype, SafeTensorError, SafeTensors};

/// Error al cargar los kernels de la CQT.
#[derive(Debug, thiserror::Error)]
pub enum KernelsError {
    /// No se pudo leer el archivo Safetensors.
    #[error("no se pudo leer el archivo de kernels: {0}")]
    Io(#[from] std::io::Error),

    /// El archivo no es un Safetensors valido.
    #[error("archivo de kernels invalido: {0}")]
    Safetensors(#[from] SafeTensorError),

    /// Falta un tensor esperado.
    #[error("falta el tensor '{0}' en el archivo de kernels")]
    Missing(&'static str),

    /// Un tensor tiene una forma inesperada.
    #[error("el tensor '{name}' tiene forma {actual:?}, se esperaba {expected:?}")]
    UnexpectedShape {
        /// Nombre del tensor.
        name: &'static str,
        /// Forma encontrada.
        actual: Vec<usize>,
        /// Forma esperada.
        expected: Vec<usize>,
    },

    /// Un tensor no es `float32`.
    #[error("el tensor '{0}' no es float32")]
    WrongDtype(&'static str),

    /// Error de Candle al construir el tensor.
    #[error("error de Candle al construir el tensor '{name}': {source}")]
    Candle {
        /// Nombre del tensor.
        name: &'static str,
        /// Error original.
        source: candle_core::Error,
    },
}

/// Los kernels deterministas de la CQT.
#[derive(Debug, Clone)]
pub struct CqtKernels {
    /// Kernel de la octava superior, parte real. Forma `[36, 1, 1, 256]`.
    pub real: Tensor,
    /// Kernel de la octava superior, parte imaginaria. Forma `[36, 1, 1, 256]`.
    pub imag: Tensor,
    /// Filtro lowpass `firwin2` para el submuestreo. Forma `[1, 1, 1, 256]`.
    pub lowpass_filter: Tensor,
    /// Factor de normalizacion `sqrt(lengths)`. Forma `[309, 1, 1]`.
    pub lengths_sqrt: Tensor,
    /// Bias de las convoluciones de la CQT (ceros). Forma `[36]`.
    pub conv_bias: Tensor,
}

impl CqtKernels {
    /// Carga los kernels desde un archivo Safetensors.
    pub fn load(path: &Path) -> Result<Self, KernelsError> {
        let data = std::fs::read(path)?;
        let tensors = SafeTensors::deserialize(&data)?;

        let real = tensor(&tensors, "cqt.kernels.real", &[36, 1, 1, 256])?;
        let imag = tensor(&tensors, "cqt.kernels.imag", &[36, 1, 1, 256])?;
        let lowpass_filter = tensor(&tensors, "cqt.lowpass_filter", &[1, 1, 1, 256])?;
        let lengths_sqrt = tensor(&tensors, "cqt.lengths_sqrt", &[309, 1, 1])?;
        let conv_bias = tensor(&tensors, "cqt.conv_bias", &[36])?;

        Ok(Self {
            real,
            imag,
            lowpass_filter,
            lengths_sqrt,
            conv_bias,
        })
    }

    /// Numero de filtros del kernel.
    pub fn n_filters(&self) -> usize {
        self.real.dims1().unwrap_or(0)
    }

    /// Longitud del kernel.
    pub fn kernel_len(&self) -> usize {
        self.real.dims4().map(|d| d.3).unwrap_or(0)
    }
}

/// Extrae un tensor `float32` de un archivo Safetensors y lo comprueba.
fn tensor(
    tensors: &SafeTensors<'_>,
    name: &'static str,
    expected: &[usize],
) -> Result<Tensor, KernelsError> {
    let view = tensors
        .tensor(name)
        .map_err(|_| KernelsError::Missing(name))?;

    let shape = view.shape().to_vec();
    if shape != expected {
        return Err(KernelsError::UnexpectedShape {
            name,
            actual: shape,
            expected: expected.to_vec(),
        });
    }
    if view.dtype() != Dtype::F32 {
        return Err(KernelsError::WrongDtype(name));
    }

    let values = f32_values(view.data());
    Tensor::from_slice(&values, shape, &Device::Cpu)
        .map_err(|source| KernelsError::Candle { name, source })
}

/// Convierte bytes little-endian en valores `f32`.
fn f32_values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

/// El dtype con el que se calculan las activaciones.
pub const ACTIVATION_DTYPE: DType = DType::F32;

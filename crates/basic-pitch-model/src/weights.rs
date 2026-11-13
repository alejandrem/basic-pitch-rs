//! Carga y validacion de los pesos del modelo desde Safetensors.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{Device, Tensor};
use safetensors::{Dtype, SafeTensorError, SafeTensors};

/// Error al cargar los pesos.
#[derive(Debug, thiserror::Error)]
pub enum WeightsError {
    /// No se pudo leer el archivo.
    #[error("no se pudo leer el archivo de pesos: {0}")]
    Io(#[from] std::io::Error),

    /// El archivo no es un Safetensors valido.
    #[error("archivo de pesos invalido: {0}")]
    Safetensors(#[from] SafeTensorError),

    /// Falta un tensor esperado.
    #[error("falta el tensor '{name}' en el archivo de pesos")]
    Missing {
        /// Nombre del tensor ausente.
        name: String,
    },

    /// Un tensor tiene una forma inesperada.
    #[error("el tensor '{name}' tiene forma {actual:?}, se esperaba {expected:?}")]
    UnexpectedShape {
        /// Nombre del tensor.
        name: String,
        /// Forma encontrada.
        actual: Vec<usize>,
        /// Forma esperada.
        expected: Vec<usize>,
    },

    /// Un tensor no es `float32`.
    #[error("el tensor '{name}' no es float32")]
    WrongDtype {
        /// Nombre del tensor.
        name: String,
    },

    /// Error de Candle al construir el tensor.
    #[error("error de Candle al construir el tensor '{name}': {source}")]
    Candle {
        /// Nombre del tensor.
        name: String,
        /// Error original.
        source: candle_core::Error,
    },
}

/// Los pesos del modelo, indexados por su nombre canonico en Rust.
#[derive(Debug, Clone)]
pub struct Weights {
    tensors: HashMap<String, Tensor>,
}

impl Weights {
    /// Carga los pesos desde un archivo Safetensors.
    pub fn load(path: &Path) -> Result<Self, WeightsError> {
        let data = std::fs::read(path)?;
        let safetensors = SafeTensors::deserialize(&data)?;

        let mut tensors = HashMap::new();
        for (name, view) in safetensors.tensors() {
            if view.dtype() != Dtype::F32 {
                return Err(WeightsError::WrongDtype {
                    name: name.to_string(),
                });
            }
            let bytes = view.data();
            let mut values = Vec::with_capacity(bytes.len() / 4);
            for chunk in bytes.chunks_exact(4) {
                values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            let shape = view.shape().to_vec();
            let tensor = Tensor::from_slice(&values, shape, &Device::Cpu).map_err(|source| {
                WeightsError::Candle {
                    name: name.to_string(),
                    source,
                }
            })?;
            tensors.insert(name.to_string(), tensor);
        }

        Ok(Self { tensors })
    }

    /// Devuelve un tensor por su nombre.
    pub fn get(&self, name: &str) -> Option<&Tensor> {
        self.tensors.get(name)
    }

    /// Devuelve un tensor, o error si falta o tiene una forma inesperada.
    pub fn expect(&self, name: &str, expected: &[usize]) -> Result<&Tensor, WeightsError> {
        let tensor = self
            .tensors
            .get(name)
            .ok_or_else(|| WeightsError::Missing {
                name: name.to_string(),
            })?;
        let actual = tensor.dims().to_vec();
        if actual != expected {
            return Err(WeightsError::UnexpectedShape {
                name: name.to_string(),
                actual,
                expected: expected.to_vec(),
            });
        }
        Ok(tensor)
    }

    /// Nombres de los tensores cargados.
    pub fn names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.tensors.keys().map(|s| s.as_str()).collect();
        names.sort_unstable();
        names
    }

    /// Numero de tensores cargados.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Indica si no se cargo ningun tensor.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}

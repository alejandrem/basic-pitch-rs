//! Lector minimalista de archivos `.npy` de NumPy.
//!
//! Sirve para leer los puntos de comparacion de
//! `reference/reference_outputs/`, que son el **oraculo** de la validacion
//! numerica: los ficheros `.npy` que TensorFlow guardo en la Fase 0.
//!
//! Solo soporta lo que la referencia produce: vectores de `float32` en
//! orden C. No es una libreria general de E/S numerica.
//!
//! # Formato soportado
//!
//! ```text
//! offset 0  : \x93NUMPY          (6 bytes de firma)
//! offset 6  : version mayor, version menor
//! offset 8  : longitud de la cabecera (u16 en v1, u32 en v2/v3)
//!             cabecera ASCII: {'descr': '<f4', 'fortran_order': False,
//!                              'shape': (1, 172, 309), }
//!             datos: float32 little-endian, en orden C
//! ```
//!
//! Es un crate **de desarrollo**: no forma parte del runtime del motor y no
//! se publica.

use std::path::Path;

/// Error al leer un archivo `.npy`.
#[derive(Debug, thiserror::Error)]
pub enum NpyError {
    /// No se pudo leer el archivo.
    #[error("no se pudo leer el archivo .npy: {0}")]
    Io(#[from] std::io::Error),

    /// La firma no es la de NumPy.
    #[error("el archivo no es un .npy valido (firma incorrecta)")]
    InvalidMagic,

    /// La version del formato no esta soportada.
    #[error("version de .npy no soportada: {0}")]
    UnsupportedVersion(u8),

    /// La cabecera no es ASCII valido.
    #[error("la cabecera del .npy no es texto valido: {0}")]
    InvalidHeader(String),

    /// Falta un campo obligatorio de la cabecera.
    #[error("falta el campo '{0}' en la cabecera del .npy")]
    MissingField(&'static str),

    /// El tipo de dato no es `float32` little-endian.
    #[error("el .npy tiene el tipo '{0}', solo se soporta '<f4'")]
    UnsupportedDtype(String),

    /// Los datos estan en orden Fortran.
    #[error("el .npy esta en orden Fortran, que no se soporta")]
    FortranOrder,

    /// El numero de bytes no corresponde a la forma declarada.
    #[error("el .npy declara {expected} bytes y tiene {actual}")]
    SizeMismatch {
        /// Bytes esperados.
        expected: usize,
        /// Bytes encontrados.
        actual: usize,
    },
}

/// Un array de `float32` leido de un `.npy`.
#[derive(Debug, Clone, PartialEq)]
pub struct NpyArray {
    /// Forma del array.
    pub shape: Vec<usize>,
    /// Datos en orden C (row-major).
    pub data: Vec<f32>,
}

impl NpyArray {
    /// Numero total de elementos.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Indica si el array esta vacio.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// Lee un array de `float32` de un archivo `.npy`.
pub fn read_f32(path: &Path) -> Result<NpyArray, NpyError> {
    let bytes = std::fs::read(path)?;

    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err(NpyError::InvalidMagic);
    }

    let major = bytes[6];
    let (header_len, header_start) = match major {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10usize),
        2 | 3 => (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12usize,
        ),
        other => return Err(NpyError::UnsupportedVersion(other)),
    };

    let header_end = header_start + header_len;
    if header_end > bytes.len() {
        return Err(NpyError::InvalidHeader(
            "la cabecera excede el archivo".into(),
        ));
    }
    let header = std::str::from_utf8(&bytes[header_start..header_end])
        .map_err(|e| NpyError::InvalidHeader(e.to_string()))?;

    let descr = field(header, "'descr'").ok_or(NpyError::MissingField("descr"))?;
    if descr != "<f4" && descr != "=f4" && descr != "f4" {
        return Err(NpyError::UnsupportedDtype(descr.to_string()));
    }

    let fortran =
        field(header, "'fortran_order'").ok_or(NpyError::MissingField("fortran_order"))?;
    if fortran.trim() != "False" {
        return Err(NpyError::FortranOrder);
    }

    let shape_text = shape_field(header).ok_or(NpyError::MissingField("shape"))?;
    let shape = parse_shape(shape_text)?;

    let expected: usize = shape.iter().product();
    let data_bytes = &bytes[header_end..];
    if data_bytes.len() < expected * 4 {
        return Err(NpyError::SizeMismatch {
            expected: expected * 4,
            actual: data_bytes.len(),
        });
    }

    let data = data_bytes
        .chunks_exact(4)
        .take(expected)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    Ok(NpyArray { shape, data })
}

/// Extrae el valor de un campo de la cabecera.
///
/// La cabecera es un diccionario de Python; los valores van entre comillas
/// simples o son literales.
fn field<'a>(header: &'a str, key: &str) -> Option<&'a str> {
    let start = header.find(key)? + key.len();
    let rest = header[start..].trim_start();
    let rest = rest.strip_prefix(':')?.trim_start();

    if let Some(stripped) = rest.strip_prefix('\'') {
        let end = stripped.find('\'')?;
        Some(&stripped[..end])
    } else {
        // Literal sin comillas: `False`, `(1, 172, 309)`, ...
        let end = rest.find([',', '}']).unwrap_or(rest.len());
        Some(&rest[..end])
    }
}

/// Extrae la tupla de la forma: `(1, 172, 309)`.
///
/// No se puede usar `field` aqui: el valor contiene comas, y `field` corta
/// en la primera.
fn shape_field(header: &str) -> Option<&str> {
    let start = header.find("'shape'")? + "'shape'".len();
    let rest = header[start..].trim_start();
    let rest = rest.strip_prefix(':')?.trim_start();
    let rest = rest.strip_prefix('(')?;

    let mut depth = 1usize;
    for (i, c) in rest.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&rest[..i]);
                }
            }
            _ => {}
        }
    }
    None
}

/// Interpreta la tupla de la forma: `(1, 172, 309)`.
fn parse_shape(text: &str) -> Result<Vec<usize>, NpyError> {
    // Se tolera que la tupla venga con o sin parentesis.
    let inner = text
        .trim()
        .trim_start_matches('(')
        .trim_end_matches(')')
        .trim();

    if inner.is_empty() {
        return Ok(Vec::new());
    }

    inner
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.parse::<usize>()
                .map_err(|e| NpyError::InvalidHeader(format!("forma invalida '{text}': {e}")))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Escribe un `.npy` minimo con los datos dados.
    fn write_npy(path: &Path, shape: &[usize], data: &[f32]) {
        let shape_text = shape
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let mut header =
            format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({shape_text}), }}");
        // La cabecera se alinea a 64 bytes (como hace NumPy).
        while (10 + header.len() + 1) % 64 != 0 {
            header.push(' ');
        }
        header.push('\n');

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"\x93NUMPY");
        bytes.push(1);
        bytes.push(0);
        bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        for value in data {
            bytes.extend_from_slice(&value.to_le_bytes());
        }

        let mut file = std::fs::File::create(path).unwrap();
        file.write_all(&bytes).unwrap();
    }

    #[test]
    fn lee_un_vector_de_float32() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v.npy");
        write_npy(&path, &[4], &[1.0, 2.0, 3.0, 4.0]);

        let array = read_f32(&path).unwrap();
        assert_eq!(array.shape, vec![4]);
        assert_eq!(array.data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn lee_un_array_tridimensional() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.npy");
        let data: Vec<f32> = (0..6).map(|v| v as f32).collect();
        write_npy(&path, &[1, 2, 3], &data);

        let array = read_f32(&path).unwrap();
        assert_eq!(array.shape, vec![1, 2, 3]);
        assert_eq!(array.data, data);
    }

    #[test]
    fn un_archivo_que_no_es_npy_produce_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("x.bin");
        std::fs::write(&path, b"esto no es un npy").unwrap();
        assert!(matches!(read_f32(&path), Err(NpyError::InvalidMagic)));
    }

    #[test]
    fn un_archivo_inexistente_produce_error_de_es() {
        let dir = tempfile::tempdir().unwrap();
        assert!(matches!(
            read_f32(&dir.path().join("no-existe.npy")),
            Err(NpyError::Io(_))
        ));
    }

    #[test]
    fn la_forma_se_interpreta_bien() {
        assert_eq!(parse_shape("(1, 172, 309)").unwrap(), vec![1, 172, 309]);
        assert_eq!(parse_shape("(43844,)").unwrap(), vec![43844]);
        assert_eq!(parse_shape("()").unwrap(), Vec::<usize>::new());
    }
}

//! Metricas de comparacion entre dos tensores.
//!
//! La Fase 7 (validacion de equivalencia) pide seis metricas:
//!
//! | Metrica | Para que sirve |
//! |---|---|
//! | MAE | el error tipico |
//! | error absoluto maximo | el peor caso |
//! | MSE | el error cuadratico medio |
//! | RMSE | la raiz del MSE, en las mismas unidades que la senal |
//! | similitud coseno | si la **direccion** coincide, aunque las magnitudes no |
//! | diferencia relativa maxima | el error respecto a la escala de la referencia |
//!
//! # La diferencia relativa
//!
//! Se define como `max_abs_diff / max_abs_referencia`: el error
//! respecto a la **escala** de la referencia, no respecto a cada valor.
//! Es la que permite comparar etapas con escalas muy distintas (una
//! activacion entre 0 y 1, un contour entre 0 y 1) sin que el numero
//! absoluto diga nada.
//!
//! # La similitud coseno
//!
//! `cos(a, b) = (a . b) / (|a| |b|)`. Vale 1 cuando los dos tensores son
//! paralelos, es decir, cuando tienen la misma forma aunque difieran en
//! la escala. Es la metrica que el cross-check de la Fase 0 uso entre
//! TensorFlow y ONNX, donde dio 1.0000000000.

/// Error al comparar dos tensores.
#[derive(Debug, thiserror::Error)]
pub enum MetricsError {
    /// Los tensores tienen distinta longitud.
    #[error("los tensores tienen longitudes distintas: {a} y {b}")]
    LengthMismatch {
        /// Longitud del primero.
        a: usize,
        /// Longitud del segundo.
        b: usize,
    },
}

/// Las seis metricas de la comparacion.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Metrics {
    /// Error absoluto medio.
    pub mae: f64,
    /// Error cuadratico medio.
    pub mse: f64,
    /// Raiz del error cuadratico medio.
    pub rmse: f64,
    /// Error absoluto maximo.
    pub max_abs_diff: f64,
    /// Diferencia relativa maxima (respecto a la escala de la referencia).
    pub max_rel_diff: f64,
    /// Similitud coseno.
    pub cosine_similarity: f64,
}

impl Metrics {
    /// Formatea las metricas en una linea, para las tablas de los
    /// validadores.
    pub fn linea(&self, nombre: &str) -> String {
        format!(
            "{nombre:<22} MAE {:.3e}  MSE {:.3e}  RMSE {:.3e}  max {:.3e}  rel {:.3e}  cos {:.10}",
            self.mae,
            self.mse,
            self.rmse,
            self.max_abs_diff,
            self.max_rel_diff,
            self.cosine_similarity
        )
    }
}

/// Compara dos tensores planos.
pub fn compare(a: &[f32], b: &[f32]) -> Result<Metrics, MetricsError> {
    if a.len() != b.len() {
        return Err(MetricsError::LengthMismatch {
            a: a.len(),
            b: b.len(),
        });
    }
    if a.is_empty() {
        return Ok(Metrics {
            mae: 0.0,
            mse: 0.0,
            rmse: 0.0,
            max_abs_diff: 0.0,
            max_rel_diff: 0.0,
            cosine_similarity: 1.0,
        });
    }

    let mut suma_abs = 0.0f64;
    let mut suma_cuadrada = 0.0f64;
    let mut max_abs = 0.0f64;
    let mut max_ref = 0.0f64;
    let mut producto = 0.0f64;
    let mut norma_a = 0.0f64;
    let mut norma_b = 0.0f64;

    for (x, y) in a.iter().zip(b.iter()) {
        let x = *x as f64;
        let y = *y as f64;
        let d = (x - y).abs();
        suma_abs += d;
        suma_cuadrada += d * d;
        if d > max_abs {
            max_abs = d;
        }
        if y.abs() > max_ref {
            max_ref = y.abs();
        }
        producto += x * y;
        norma_a += x * x;
        norma_b += y * y;
    }

    let n = a.len() as f64;
    let mse = suma_cuadrada / n;
    let cosine_similarity = if norma_a > 0.0 && norma_b > 0.0 {
        producto / (norma_a.sqrt() * norma_b.sqrt())
    } else {
        // Si alguno de los dos tensores es identicamente cero, la
        // similitud no esta definida. Se devuelve 1.0 solo si los dos
        // son cero (son el mismo tensor).
        if norma_a == 0.0 && norma_b == 0.0 {
            1.0
        } else {
            0.0
        }
    };

    Ok(Metrics {
        mae: suma_abs / n,
        mse,
        rmse: mse.sqrt(),
        max_abs_diff: max_abs,
        max_rel_diff: if max_ref > 0.0 {
            max_abs / max_ref
        } else {
            0.0
        },
        cosine_similarity,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dos_tensores_identicos_dan_error_cero_y_coseno_uno() {
        let x = vec![0.1f32, 0.5, -0.3, 0.9];
        let m = compare(&x, &x).unwrap();
        assert_eq!(m.mae, 0.0);
        assert_eq!(m.mse, 0.0);
        assert_eq!(m.rmse, 0.0);
        assert_eq!(m.max_abs_diff, 0.0);
        assert_eq!(m.max_rel_diff, 0.0);
        assert!((m.cosine_similarity - 1.0).abs() < 1e-12);
    }

    #[test]
    fn el_error_medio_es_el_promedio_de_las_diferencias() {
        let a = vec![0.0f32, 1.0];
        let b = vec![0.5f32, 1.5];
        let m = compare(&a, &b).unwrap();
        assert!((m.mae - 0.5).abs() < 1e-12);
        assert!((m.max_abs_diff - 0.5).abs() < 1e-12);
        assert!((m.mse - 0.25).abs() < 1e-12);
        assert!((m.rmse - 0.5).abs() < 1e-12);
    }

    #[test]
    fn la_diferencia_relativa_es_respecto_a_la_escala_de_la_referencia() {
        let a = vec![0.0f32, 0.1];
        let b = vec![0.0f32, 1.0];
        let m = compare(&a, &b).unwrap();
        // max_abs = 0.9, escala de la referencia = 1.0.
        // La tolerancia es de f32: 0.1 no es representable exacto.
        assert!((m.max_rel_diff - 0.9).abs() < 1e-6, "{}", m.max_rel_diff);
    }

    #[test]
    fn la_similitud_coseno_ignora_la_escala() {
        // b es el doble que a: misma direccion, distinta magnitud.
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![2.0f32, 4.0, 6.0];
        let m = compare(&a, &b).unwrap();
        assert!((m.cosine_similarity - 1.0).abs() < 1e-12);
        assert!(m.mae > 1.0, "pero el error absoluto si es grande");
    }

    #[test]
    fn longitudes_distintas_producen_error() {
        assert!(matches!(
            compare(&[1.0, 2.0], &[1.0]),
            Err(MetricsError::LengthMismatch { a: 2, b: 1 })
        ));
    }

    #[test]
    fn dos_vectores_nulos_dan_coseno_uno() {
        let m = compare(&[], &[]).unwrap();
        assert_eq!(m.cosine_similarity, 1.0);
    }
}

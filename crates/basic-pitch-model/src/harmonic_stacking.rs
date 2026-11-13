//! Harmonic Stacking: la expansion armonica del frontend.
//!
//! # Entrada y salida
//!
//! ```text
//! entrada  : (batch, 1, 172, 309)   NCHW, los 309 bins de la CQT
//! salida   : (batch, 8, 172, 264)   NCHW, truncado a 264 bins, 8 canales
//! ```
//!
//! En NHWC (TensorFlow) la salida equivalente es `(batch, 172, 264, 8)`.
//!
//! El truncado a 264 ocurre **despues** de concatenar los 8 canales, no antes
//! de desplazar.
//!
//! # El relleno con ceros es asimetrico
//!
//! Tomando como referencia `basic_pitch/nn.py`, clase `HarmonicStacking`:
//!
//! | Caso | Operacion | Efecto |
//! |---|---|---|
//! | `shift == 0` | `padded = x` | canal identico a la entrada |
//! | `shift > 0` | `pad(x[:, :, shift:, :], [[0,0],[0,0],[0,shift],[0,0]])` | descarta los `shift` bins **bajos**, rellena con ceros los **altos** |
//! | `shift < 0` | `pad(x[:, :, :shift, :], [[0,0],[0,0],[-shift,0],[0,0]])` | rellena con ceros los bins **bajos**, conserva el resto |
//!
//! En ambos casos, el desplazamiento equivale a `out[i] = x[i + shift]`,
//! utilizando ceros cuando el indice queda fuera de rango. La diferencia entre
//! ambos casos esta en **donde** se colocan esos ceros.
//!
//! # Desplazamientos
//!
//! ```text
//! shift = round(12 × bins_per_semitone × log2(harmonic))
//! ```
//!
//! | Armonico | 0.5 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
//! |---|---|---|---|---|---|---|---|---|
//! | `shift` | -36 | 0 | 36 | 57 | 72 | 84 | 93 | 101 |

use candle_core::Tensor;

/// Calcula los desplazamientos de una lista de armonicos.
///
/// Reproduce `int(tf.math.round(12 * bins_per_semitone * log2(h)))`.
///
/// `tf.math.round` redondea al par en los empates, igual que
/// `f64::round_ties_even` de Rust.
pub fn harmonic_shifts(harmonics: &[f64], bins_per_semitone: usize) -> Vec<i64> {
    harmonics
        .iter()
        .map(|h| (12.0 * bins_per_semitone as f64 * h.log2()).round_ties_even() as i64)
        .collect()
}

/// Aplica el desplazamiento de un canal a una matriz de bins.
///
/// `bins` es el eje de frecuencias de una muestra: un vector plano.
/// Devuelve el canal desplazado, con ceros en las posiciones fuera de rango.
pub fn shift_channel(bins: &[f32], shift: i64) -> Vec<f32> {
    let n = bins.len();
    let mut out = vec![0.0f32; n];
    for (i, slot) in out.iter_mut().enumerate() {
        let source = i as i64 + shift;
        if source >= 0 && (source as usize) < n {
            *slot = bins[source as usize];
        }
    }
    out
}

/// Error en la operacion de Harmonic Stacking.
#[derive(Debug, thiserror::Error)]
pub enum HarmonicStackingError {
    /// La forma de la entrada no es la esperada.
    #[error("se esperaba (batch, 1, time, {expected_bins}) y se recibio {actual:?}")]
    UnexpectedShape {
        /// Forma recibida.
        actual: Vec<usize>,
        /// Numero de bins esperado.
        expected_bins: usize,
    },

    /// Un desplazamiento esta fuera de rango.
    #[error("el desplazamiento {shift} del armonico {harmonic} no cabe en {n_bins} bins")]
    ShiftOutOfRange {
        /// Armonico correspondiente.
        harmonic: f64,
        /// Desplazamiento.
        shift: i64,
        /// Numero de bins de la entrada.
        n_bins: usize,
    },

    /// Error de Candle.
    #[error("error de Candle en Harmonic Stacking: {0}")]
    Candle(#[from] candle_core::Error),
}

/// Aplica Harmonic Stacking a la representacion CQT.
///
/// # Argumentos
///
/// - `cqt`: la CQT en NCHW, `(batch, 1, time, n_bins)` con `n_bins = 309`.
/// - `harmonics`: los armónicos a usar, por ejemplo `[0.5, 1.0, ..., 7.0]`.
/// - `bins_per_semitone`: 3.
/// - `n_output_freqs`: bins de la salida, 264.
///
/// # Salida
///
/// `(batch, harmonics.len(), time, n_output_freqs)` en NCHW.
///
/// # Errores
///
/// Falla si la forma de la entrada no es la esperada, o si algum
/// desplazamiento no cabe en la entrada.
pub fn harmonic_stacking(
    cqt: &Tensor,
    harmonics: &[f64],
    bins_per_semitone: usize,
    n_output_freqs: usize,
) -> Result<Tensor, HarmonicStackingError> {
    let (batch, channels, time, n_bins) = cqt.dims4()?;
    if channels != 1 {
        return Err(HarmonicStackingError::UnexpectedShape {
            actual: vec![batch, channels, time, n_bins],
            expected_bins: n_bins,
        });
    }
    if n_output_freqs > n_bins {
        return Err(HarmonicStackingError::UnexpectedShape {
            actual: vec![batch, channels, time, n_bins],
            expected_bins: n_bins,
        });
    }

    let shifts = harmonic_shifts(harmonics, bins_per_semitone);
    let device = cqt.device();

    let mut channels_out: Vec<Tensor> = Vec::with_capacity(shifts.len());

    for (harmonic, shift) in harmonics.iter().zip(shifts.iter()) {
        // --- shift == 0: el canal es identico a la entrada ---
        let shifted = if *shift == 0 {
            cqt.clone()
        } else if *shift > 0 {
            // Descarta los `shift` bins bajos y rellena con ceros los altos.
            let shift_usize = *shift as usize;
            if shift_usize >= n_bins {
                return Err(HarmonicStackingError::ShiftOutOfRange {
                    harmonic: *harmonic,
                    shift: *shift,
                    n_bins,
                });
            }
            let kept = cqt.narrow(3, shift_usize, n_bins - shift_usize)?;
            let zeros = Tensor::zeros((batch, 1, time, shift_usize), cqt.dtype(), device)?;
            Tensor::cat(&[&kept, &zeros], 3)?
        } else {
            // Rellena con ceros los bins bajos y conserva el resto.
            let drop = (-*shift) as usize;
            if drop >= n_bins {
                return Err(HarmonicStackingError::ShiftOutOfRange {
                    harmonic: *harmonic,
                    shift: *shift,
                    n_bins,
                });
            }
            let kept = cqt.narrow(3, 0, n_bins - drop)?;
            let zeros = Tensor::zeros((batch, 1, time, drop), cqt.dtype(), device)?;
            Tensor::cat(&[&zeros, &kept], 3)?
        };

        channels_out.push(shifted);
    }

    // Concatenacion en el eje de canales: (batch, 8, time, n_bins).
    let refs: Vec<&Tensor> = channels_out.iter().collect();
    let stacked = Tensor::cat(&refs, 1)?;

    // Truncado a los primeros `n_output_freqs` bins de frecuencia.
    // Ocurre DESPUES de concatenar, igual que en la referencia.
    Ok(stacked.narrow(3, 0, n_output_freqs)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn cqt_de_prueba(n_bins: usize) -> Tensor {
        // Valores identificables por posicion: bins[i] = i.
        let mut data = Vec::with_capacity(n_bins);
        for i in 0..n_bins {
            data.push(i as f32);
        }
        Tensor::from_slice(&data, (1, 1, 1, n_bins), &Device::Cpu).unwrap()
    }

    #[test]
    fn los_desplazamientos_coinciden_con_el_referencia() {
        let harmonics = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        assert_eq!(
            harmonic_shifts(&harmonics, 3),
            vec![-36, 0, 36, 57, 72, 84, 93, 101]
        );
    }

    #[test]
    fn un_desplazamiento_positivo_desplaza_hacia_binarios_bajos() {
        let bins = [10.0, 20.0, 30.0, 40.0, 50.0];
        let out = shift_channel(&bins, 2);
        assert_eq!(out, vec![30.0, 40.0, 50.0, 0.0, 0.0]);
    }

    #[test]
    fn un_desplazamiento_negativo_desplaza_hacia_binarios_altos() {
        let bins = [10.0, 20.0, 30.0, 40.0, 50.0];
        let out = shift_channel(&bins, -2);
        assert_eq!(out, vec![0.0, 0.0, 10.0, 20.0, 30.0]);
    }

    #[test]
    fn un_desplazamiento_cero_devuelve_el_canal_intacto() {
        let bins = [1.0, 2.0, 3.0];
        assert_eq!(shift_channel(&bins, 0), bins.to_vec());
    }

    #[test]
    fn un_desplazamiento_mayor_que_la_longitud_deja_todo_a_cero() {
        let bins = [1.0, 2.0, 3.0];
        assert_eq!(shift_channel(&bins, 10), vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn la_salida_tiene_la_forma_esperada() {
        let cqt = cqt_de_prueba(309);
        let harmonics = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let out = harmonic_stacking(&cqt, &harmonics, 3, 264).unwrap();
        assert_eq!(out.dims4().unwrap(), (1, 8, 1, 264));
    }

    #[test]
    fn el_canal_del_armonico_1_es_la_entrada_truncada() {
        let cqt = cqt_de_prueba(309);
        let out = harmonic_stacking(&cqt, &[1.0], 3, 264).unwrap();
        let data = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // El armonico 1 tiene shift 0: el canal es la entrada sin truncar.
        for (i, value) in data.iter().enumerate() {
            assert_eq!(*value, i as f32, "bin {i}");
        }
    }

    #[test]
    fn el_desplazamiento_positivo_se_implementa_en_tensores() {
        // El armonico 2 tiene shift +36: out[i] = x[i + 36].
        let cqt = cqt_de_prueba(309);
        let out = harmonic_stacking(&cqt, &[2.0], 3, 264).unwrap();
        let data = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (i, value) in data.iter().enumerate() {
            let expected = if i + 36 < 309 { (i + 36) as f32 } else { 0.0 };
            assert_eq!(*value, expected, "bin {i}");
        }
    }

    #[test]
    fn el_desplazamiento_negativo_se_implementa_en_tensores() {
        // El armonico 0.5 tiene shift -36: out[i] = x[i - 36], ceros al inicio.
        let cqt = cqt_de_prueba(309);
        let out = harmonic_stacking(&cqt, &[0.5], 3, 264).unwrap();
        let data = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (i, value) in data.iter().enumerate() {
            let expected = if i >= 36 { (i - 36) as f32 } else { 0.0 };
            assert_eq!(*value, expected, "bin {i}");
        }
    }

    #[test]
    fn la_version_con_tensores_coincide_con_la_de_vectores() {
        // La implementacion con tensores debe ser equivalente a
        // `shift_channel` aplicada canal a canal.
        let n_bins = 309usize;
        let cqt = cqt_de_prueba(n_bins);
        let harmonics = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let shifts = harmonic_shifts(&harmonics, 3);

        let out = harmonic_stacking(&cqt, &harmonics, 3, 264).unwrap();
        let data = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let reference: Vec<f32> = (0..n_bins).map(|i| i as f32).collect();
        for (channel, shift) in shifts.iter().enumerate() {
            let expected = shift_channel(&reference, *shift);
            for i in 0..264 {
                assert_eq!(
                    data[channel * 264 + i],
                    expected[i],
                    "canal {channel} bin {i}"
                );
            }
        }
    }

    #[test]
    fn una_entrada_con_varios_canales_produce_error() {
        let cqt = Tensor::zeros((1, 2, 172, 309), candle_core::DType::F32, &Device::Cpu).unwrap();
        assert!(matches!(
            harmonic_stacking(&cqt, &[1.0], 3, 264),
            Err(HarmonicStackingError::UnexpectedShape { .. })
        ));
    }
}

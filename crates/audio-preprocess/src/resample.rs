//! Remuestreo de audio con `soxr` en modo de alta calidad.
//!
//! La referencia usa `librosa.load(..., sr=22050)`, que en librosa 0.10
//! remuestrea con `res_type="soxr_hq"`. Se usa el mismo motor para que la
//! senal sea comparable muestra a muestra.
//!
//! `soxr_hq` corresponde a `SOXR_HQ`, que es la receta de 20 bits
//! (`QualityRecipe::high()`).
//!
//! **Aviso:** usar `soxr` garantiza el mismo algoritmo, pero no igualdad
//! exacta: el decodificador de audio puede diferir del de la referencia
//! (librosa usa `soundfile`/`libsndfile`). La diferencia se mide en la Fase 7
//! y se documenta como tolerancia asumida.

use std::sync::Mutex;

use soxr::format::Mono;
use soxr::params::{QualityRecipe, QualitySpec, RuntimeSpec};
use soxr::Soxr;

/// Cerrojo global que serializa el uso de `libsoxr`.
///
/// # Por que hace falta
///
/// `libsoxr` mantiene un **descriptor de FFT compartido** entre todas sus
/// instancias (`soxr_fft`). Dos instancias que se usan a la vez desde hilos
/// distintos corruptos ese estado compartido: la salida resultante tiene la
/// longitud correcta pero la **amplitud y la fase equivocadas**.
///
/// Es un problema de la biblioteca, no de este codigo, y es observable: una
/// senal de 100 Hz remuestreada a 22.050 Hz devuelve una amplitud de ~0.07 en
/// lugar de 1.0 cuando hay remuestreos concurrentes.
///
/// Serializar el acceso lo elimina por completo y, de paso, garantiza que el
/// resultado sea **reproducible**: el motor procesa las ventanas de forma
/// secuencial, asi que el coste es nulo en la practica.
static SOXR_LOCK: Mutex<()> = Mutex::new(());

/// Error durante el remuestreo.
#[derive(Debug, thiserror::Error)]
pub enum ResampleError {
    /// Las frecuencias de muestreo no son validas.
    #[error("frecuencia de muestreo invalida: origen={from}, destino={to}")]
    InvalidRate {
        /// Frecuencia de origen.
        from: u32,
        /// Frecuencia de destino.
        to: u32,
    },

    /// `soxr` no pudo crear el remuestreador.
    #[error("soxr no pudo crear el remuestreador: {0}")]
    Setup(String),

    /// `soxr` fallo al procesar los datos.
    #[error("fallo de soxr durante el remuestreo: {0}")]
    Process(String),
}

/// Remuestrea una senal mono de `from_rate` a `to_rate`.
///
/// La longitud de salida sigue la regla de `librosa.resample` con
/// `fix=True`: exactamente `ceil(len * to_rate / from_rate)` muestras.
///
/// Si las frecuencias coinciden, la senal se devuelve sin cambios.
pub fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>, ResampleError> {
    if from_rate == 0 || to_rate == 0 {
        return Err(ResampleError::InvalidRate {
            from: from_rate,
            to: to_rate,
        });
    }
    if from_rate == to_rate {
        return Ok(samples.to_vec());
    }

    // Se serializa el acceso a libsoxr (ver `SOXR_LOCK`).
    let _guard = SOXR_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let expected_len = (samples.len() as u64 * to_rate as u64).div_ceil(from_rate as u64) as usize;

    // Un solo hilo: el modo multihilo de libsoxr reparte el filtro entre
    // hilos, y el punto de corte depende del planificacion, de modo que el
    // resultado no es reproducible. Para un proyecto cuyo objetivo es la
    // fidelidad numerica frente a la referencia, el determinismo es
    // obligatorio.
    let quality = QualitySpec::new(QualityRecipe::high());
    let runtime = RuntimeSpec::new(1);
    let mut resampler =
        Soxr::<Mono<f32>>::new_with_params(from_rate as f64, to_rate as f64, quality, runtime)
            .map_err(|e| ResampleError::Setup(format!("{e:?}")))?;

    let mut out: Vec<f32> = Vec::with_capacity(expected_len + 1024);

    // `soxr_process` consume solo la parte de la entrada que cabe en el
    // buffer de salida, asi que se itera hasta consumirla entera.
    let mut chunk = vec![0.0f32; (expected_len.max(4096) + 4096).next_power_of_two()];
    let mut input_pos = 0usize;

    while input_pos < samples.len() {
        let processed = resampler
            .process(&samples[input_pos..], &mut chunk)
            .map_err(|e| ResampleError::Process(format!("{e:?}")))?;

        if processed.output_frames > 0 {
            out.extend_from_slice(&chunk[..processed.output_frames]);
        }

        if processed.input_frames == 0 {
            // El buffer de salida era insuficiente: se duplica y se reintenta.
            if chunk.len() > samples.len() * 4 {
                return Err(ResampleError::Process(
                    "soxr no consume la entrada con el buffer disponible".into(),
                ));
            }
            chunk.resize(chunk.len() * 2, 0.0);
            continue;
        }

        input_pos += processed.input_frames;
    }

    // Se drenan las muestras retenidas por el filtro.
    loop {
        chunk.fill(0.0);
        let drained = resampler
            .drain(&mut chunk)
            .map_err(|e| ResampleError::Process(format!("{e:?}")))?;
        if drained == 0 {
            break;
        }
        out.extend_from_slice(&chunk[..drained]);
    }

    // `librosa.resample` recorta a la longitud exacta esperada.
    out.resize(expected_len, 0.0);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn la_frecuencia_igual_no_altera_la_senal() {
        let samples = vec![0.1f32, -0.2, 0.3];
        let out = resample(&samples, 22_050, 22_050).unwrap();
        assert_eq!(out, samples);
    }

    #[test]
    fn la_longitud_es_la_esperada_por_la_regla_de_librosa() {
        // 44100 -> 22050 reduce a la mitad.
        let samples = vec![0.0f32; 1000];
        let out = resample(&samples, 44_100, 22_050).unwrap();
        assert_eq!(out.len(), 500);

        // 22050 -> 44100 duplica.
        let out = resample(&samples, 22_050, 44_100).unwrap();
        assert_eq!(out.len(), 2000);
    }

    #[test]
    fn la_longitud_se_redondea_hacia_arriba() {
        // ceil(1001 * 22050 / 44100) = 501
        let samples = vec![0.0f32; 1001];
        let out = resample(&samples, 44_100, 22_050).unwrap();
        assert_eq!(out.len(), 501);
    }

    #[test]
    fn el_remuestreo_concurrente_es_correcto() {
        // Regresion: `libsoxr` comparte el descriptor de FFT entre
        // instancias, asi que dos remuestreos simultaneos desde hilos
        // distintos pueden corromper la salida (amplitud y fase erroneas,
        // con la longitud correcta). El cerrojo global de este modulo lo
        // evita; este test lo verifica.
        let n = 44_100usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / 44_100.0;
                (2.0 * std::f32::consts::PI * 100.0 * t).sin()
            })
            .collect();

        let handles: Vec<_> = (0..8)
            .map(|_| {
                let samples = samples.clone();
                std::thread::spawn(move || {
                    let out = resample(&samples, 44_100, 22_050).unwrap();
                    assert_eq!(out.len(), 22_050);
                    let peak = out.iter().fold(f32::NEG_INFINITY, |a, b| a.max(*b));
                    // Si la salida esta corrupta, la amplitud se desploma.
                    assert!(peak > 0.99, "amplitud corrupta: {peak}");
                    let mut crossings = 0usize;
                    for i in 1..out.len() {
                        if out[i - 1] <= 0.0 && out[i] > 0.0 {
                            crossings += 1;
                        }
                    }
                    assert!(
                        (crossings as i32 - 99).abs() <= 2,
                        "se esperaban ~99 cruces, se obtuvieron {crossings}"
                    );
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn una_frecuencia_invalida_produce_error() {
        assert!(matches!(
            resample(&[0.0], 0, 22_050),
            Err(ResampleError::InvalidRate { .. })
        ));
    }

    #[test]
    fn una_sine_survive_al_remuestreo() {
        // 100 Hz a 44100 Hz, un segundo.
        let n = 44_100;
        let samples: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / 44_100.0;
                (2.0 * std::f32::consts::PI * 100.0 * t).sin()
            })
            .collect();

        let out = resample(&samples, 44_100, 22_050).unwrap();
        assert_eq!(out.len(), 22_050);

        // La frecuencia fundamental debe seguir siendo ~100 Hz: se cuentan
        // los cruces por cero en un segundo de la salida.
        let mut crossings = 0usize;
        for i in 1..out.len() {
            if out[i - 1] <= 0.0 && out[i] > 0.0 {
                crossings += 1;
            }
        }
        assert!(
            (crossings as i32 - 100).abs() <= 2,
            "se esperaban ~100 cruces, se obtuvieron {crossings}"
        );
    }
}

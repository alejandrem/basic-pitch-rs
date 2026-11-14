//! La cadena completa, de un archivo de audio real a la entrada de la red,
//! comparada con TensorFlow.
//!
//! Este es el test definitivo de las Fases 3, 4 y 5: no valida una etapa
//! suelta, sino **todas juntas**, empezando por el archivo de audio.
//!
//! ```text
//! archivo .wav
//!   -> decodificacion      (Symphonia)      } la referencia usa
//!   -> mono por promedio   }                } soundfile + soxr_hq
//!   -> remuestreo a 22050  (soxr HQ)         }
//!   -> ventana de 43844    (relleno de 3840 ceros)
//!   -> CQT                }
//!   -> NormalizedLog      }  contra el      }
//!   -> BatchNorm inicial  }  oraculo de     }
//!   -> Harmonic Stacking  }  TensorFlow      }
//! ```
//!
//! El oraculo son los `.npy` de `reference/audio_reference/`, generados por
//! `tools/weight-export/reference_audio_frontend.py` con el SavedModel
//! oficial y el ventaneo literal de `basic_pitch/inference.py`.
//!
//! Se comparan las ventanas 0 y 1 de dos archivos reales: la 0 incluye el
//! relleno inicial (y empieza antes del audio) y la 1 ya esta dentro del
//! audio.

use std::path::{Path, PathBuf};

use basic_pitch_audio_io::decode_file;
use basic_pitch_audio_preprocess::{resample, to_mono, AudioWindows, AUDIO_SAMPLE_RATE};
use basic_pitch_model::{config::ModelConfig, weights::Weights, AcousticFrontend};
use basic_pitch_reference_npy::read_f32;
use candle_core::Tensor;

/// Raiz del repositorio.
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Directorio de los recursos de audio del upstream.
fn resources_dir() -> PathBuf {
    repo_root().join("basic-pitch/tests/resources")
}

/// Directorio del oraculo de audio.
fn oracle_dir() -> PathBuf {
    repo_root().join("reference/audio_reference")
}

/// Directorio del modelo.
fn model_dir() -> PathBuf {
    repo_root().join("models/basic-pitch/icassp-2022")
}

/// Si falta el audio del upstream o el oraculo, no hay nada que comparar.
fn skip_si_falta_algo() -> bool {
    if !resources_dir().exists() || !oracle_dir().exists() {
        eprintln!(
            "OMITIDO: falta {} o {}",
            resources_dir().display(),
            oracle_dir().display()
        );
        return true;
    }
    false
}

/// Compara dos tensores contra el oraculo y devuelve (mae, max_abs).
fn comparar(rust: &Tensor, referencia: &[f32]) -> (f64, f64) {
    let rust = rust
        .flatten_all()
        .expect("aplanar")
        .to_vec1::<f32>()
        .expect("float32");
    assert_eq!(
        rust.len(),
        referencia.len(),
        "la etapa tiene {} valores y la referencia {}",
        rust.len(),
        referencia.len()
    );
    let mut suma = 0.0f64;
    let mut maximo = 0.0f64;
    for (a, b) in rust.iter().zip(referencia.iter()) {
        let d = (*a as f64 - *b as f64).abs();
        suma += d;
        if d > maximo {
            maximo = d;
        }
    }
    (suma / rust.len().max(1) as f64, maximo)
}

/// Compara el pipeline completo de Rust con el oraculo de una ventana.
fn comprobar_ventana(nombre: &str, indice_ventana: usize) {
    // --- El frontend de Rust ---
    let config = ModelConfig::load(&model_dir().join("model_config.json")).expect("configuracion");
    let weights = Weights::load(&model_dir().join("model.safetensors")).expect("pesos");
    let frontend = AcousticFrontend::load(&model_dir(), &config, &weights).expect("frontend");

    // --- El audio, como lo procesa Rust ---
    let decoded = decode_file(&resources_dir().join(format!("{nombre}.wav"))).expect("decodificar");
    let mono = to_mono(&decoded);
    let remuestreado =
        resample(&mono.samples, mono.sample_rate, AUDIO_SAMPLE_RATE).expect("remuestrear");
    let windows = AudioWindows::new(remuestreado, AUDIO_SAMPLE_RATE);
    let ventana = windows.iter().nth(indice_ventana).expect("ventana");

    let stages = frontend.stages(&ventana.samples).expect("frontend");

    // --- El oraculo ---
    let prefijo = format!("{nombre}_v{indice_ventana}_");
    let cqt_ref = read_f32(&oracle_dir().join(format!("{prefijo}cqt.npy"))).expect("cqt");
    let log_ref =
        read_f32(&oracle_dir().join(format!("{prefijo}normalized_log.npy"))).expect("log");
    let bn_ref = read_f32(&oracle_dir().join(format!("{prefijo}batch_norm.npy"))).expect("bn");
    let harm_ref =
        read_f32(&oracle_dir().join(format!("{prefijo}harmonic_stacking.npy"))).expect("harmonic");

    // --- Comparacion ---
    let (cqt_mae, cqt_max) = comparar(&stages.cqt, &cqt_ref.data);
    let (log_mae, log_max) = comparar(&stages.normalized_log, &log_ref.data);
    let (bn_mae, bn_max) = comparar(&stages.batch_norm, &bn_ref.data);
    let harm_nhwc = stages
        .harmonic_stacking
        .permute((0, 2, 3, 1))
        .expect("permute");
    let (harm_mae, harm_max) = comparar(&harm_nhwc, &harm_ref.data);

    println!(
        "{prefijo}cqt  mae={cqt_mae:.3e} max={cqt_max:.3e}\n\
         {prefijo}log mae={log_mae:.3e} max={log_max:.3e}\n\
         {prefijo}bn   mae={bn_mae:.3e} max={bn_max:.3e}\n\
         {prefijo}harm mae={harm_mae:.3e} max={harm_max:.3e}",
    );

    // Tolerancias: las mismas que en el test del oraculo sintetico, que
    // estan justificadas en docs/practica/fase-08-frontend-cqt.md.
    //
    // La CQT incluye ahora la decodificacion y el remuestreo, asi que el
    // error no es solo de la conv: se acumula el de Symphonia frente a
    // soundfile (2.7e-8 de media) y el de la cadena de remuestreo.
    assert!(cqt_max < 2e-5, "la CQT difiere en {cqt_max:.3e}");
    assert!(cqt_mae < 1e-6, "el error medio de la CQT es {cqt_mae:.3e}");
    assert!(log_max < 5e-3, "el NormalizedLog difiere en {log_max:.3e}");
    assert!(
        log_mae < 1e-4,
        "el error medio del NormalizedLog es {log_mae:.3e}"
    );
    assert!(bn_max < 1e-2, "la BatchNorm difiere en {bn_max:.3e}");
    assert!(
        harm_max < 1e-2,
        "el Harmonic Stacking difiere en {harm_max:.3e}"
    );
}

#[test]
fn la_cadena_completa_coincide_con_tensorflow_ventana_0() {
    if skip_si_falta_algo() {
        return;
    }
    // La ventana 0 incluye el relleno inicial de 3840 ceros.
    comprobar_ventana("vocadito_10", 0);
}

#[test]
fn la_cadena_completa_coincide_con_tensorflow_ventana_1() {
    if skip_si_falta_algo() {
        return;
    }
    // La ventana 1 ya esta dentro del audio.
    comprobar_ventana("vocadito_10", 1);
}

#[test]
fn el_segundo_audio_tambien_coincide() {
    if skip_si_falta_algo() {
        return;
    }
    comprobar_ventana("vocadito_14", 0);
}

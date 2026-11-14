//! Comparacion del frontend acustico de Rust contra la referencia de
//! TensorFlow, etapa a etapa.
//!
//! Los puntos de comparacion son los `.npy` que la Fase 0 guardo en
//! `reference/reference_outputs/`. Son el **oraculo**: lo que produjo
//! TensorFlow con el modelo oficial.
//!
//! # Tolerancias
//!
//! Las tolerancias no son arbitrarias: reproducen el comportamiento medido
//! del frontend y estan justificadas en `docs/practica/fase-08-frontend-cqt.md`.
//!
//! - **CQT**: el error es de `float32` puro. La convolucion acumula 256
//!   sumandos, y TensorFlow y Candle los acumulan en orden distinto: la
//!   diferencia medida es de 5.7e-6 **absolutos** sobre valores que llegan
//!   a 18, es decir, 3.2e-7 relativos: el epsilon de `float32`.
//! - **NormalizedLog, BatchNorm y Harmonic Stacking**: el error es mayor
//!   porque el logaritmo **amplifica** el ruido de la CQT en los bins donde
//!   la senal es casi silenciosa. Con un valor de CQT de 5e-6, la
//!   sensibilidad `d(salida)/d(CQT)` vale ~2800, de modo que un error de
//!   5e-7 en la CQT se convierte en 1.4e-3 en la salida normalizada. Es una
//!   sensibilidad de la propia referencia, no un defecto del port.
//!
//! Si el directorio de referencia no existe (por ejemplo, en una copia
//! publicada del crate), los tests se omiten con un aviso.

use std::path::{Path, PathBuf};

use basic_pitch_model::{config::ModelConfig, weights::Weights, AcousticFrontend};
use basic_pitch_reference_npy::read_f32;
use candle_core::Tensor;

/// Raiz del repositorio.
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Directorio del modelo desplegable.
fn model_dir() -> PathBuf {
    repo_root().join("models/basic-pitch/icassp-2022")
}

/// Directorio de los puntos de comparacion.
fn reference_dir() -> PathBuf {
    repo_root().join("reference/reference_outputs")
}

/// Si la referencia no esta disponible, no hay nada que comparar.
fn skip_si_no_hay_referencia() -> bool {
    if reference_dir().exists() {
        return false;
    }
    eprintln!(
        "OMITIDO: no se encuentra {} (no hay referencia que comparar)",
        reference_dir().display()
    );
    true
}

/// Carga el frontend y el audio de la referencia.
fn cargar() -> Option<(AcousticFrontend, Vec<f32>)> {
    let config = ModelConfig::load(&model_dir().join("model_config.json")).expect("configuracion");
    let weights = Weights::load(&model_dir().join("model.safetensors")).expect("pesos");
    let frontend = AcousticFrontend::load(&model_dir(), &config, &weights).expect("frontend");
    let audio = read_f32(&reference_dir().join("input_audio.npy")).expect("audio de referencia");
    Some((frontend, audio.data))
}

/// Compara dos tensores contra un `.npy` de referencia.
///
/// Devuelve la diferencia absoluta maxima.
fn max_diff(rust: &Tensor, reference: &[f32]) -> f64 {
    let rust = rust
        .flatten_all()
        .expect("aplanar")
        .to_vec1::<f32>()
        .expect("float32");
    assert_eq!(
        rust.len(),
        reference.len(),
        "la etapa tiene {} valores y la referencia {}",
        rust.len(),
        reference.len()
    );
    rust.iter()
        .zip(reference.iter())
        .map(|(a, b)| (*a as f64 - *b as f64).abs())
        .fold(0.0f64, f64::max)
}

/// Error absoluto medio.
fn mae(rust: &Tensor, reference: &[f32]) -> f64 {
    let rust = rust
        .flatten_all()
        .expect("aplanar")
        .to_vec1::<f32>()
        .expect("float32");
    let sum: f64 = rust
        .iter()
        .zip(reference.iter())
        .map(|(a, b)| (*a as f64 - *b as f64).abs())
        .sum();
    sum / rust.len().max(1) as f64
}

#[test]
fn la_cqt_coincide_con_la_referencia() {
    if skip_si_no_hay_referencia() {
        return;
    }
    let (frontend, audio) = cargar().unwrap();
    let stages = frontend.stages(&audio).unwrap();
    let reference = read_f32(&reference_dir().join("cqt.npy")).unwrap();

    assert_eq!(stages.cqt.dims3().unwrap(), (1, 172, 309));
    assert_eq!(reference.shape, vec![1, 172, 309]);

    // 5.7e-6 medidos: el ruido de float32 de la conv de 256 sumandos.
    // El margen x4 cubre la variabilidad entre backends.
    let diff = max_diff(&stages.cqt, &reference.data);
    assert!(
        diff < 2e-5,
        "la CQT difiere de la referencia en {diff:.3e}, se esperaba menos de 2e-5"
    );
}

#[test]
fn el_normalized_log_coincide_con_la_referencia() {
    if skip_si_no_hay_referencia() {
        return;
    }
    let (frontend, audio) = cargar().unwrap();
    let stages = frontend.stages(&audio).unwrap();
    let reference = read_f32(&reference_dir().join("normalized_log.npy")).unwrap();

    // 1.4e-3 medidos: el logaritmo amplifica el ruido de la CQT en los bins
    // silenciosos. La mediana del error es de 2.7e-6.
    let diff = max_diff(&stages.normalized_log, &reference.data);
    assert!(
        diff < 5e-3,
        "el NormalizedLog difiere en {diff:.3e}, se esperaba menos de 5e-3"
    );
    let mean = mae(&stages.normalized_log, &reference.data);
    assert!(
        mean < 1e-4,
        "el error medio del NormalizedLog es {mean:.3e}, se esperaba menos de 1e-4"
    );
}

#[test]
fn la_batch_norm_inicial_coincide_con_la_referencia() {
    if skip_si_no_hay_referencia() {
        return;
    }
    let (frontend, audio) = cargar().unwrap();
    let stages = frontend.stages(&audio).unwrap();
    let reference = read_f32(&reference_dir().join("batch_norm_input.npy")).unwrap();

    // 3.4e-3 medidos: la BatchNorm escala la entrada normalizada, asi que
    // hereda su error ampliado.
    let diff = max_diff(&stages.batch_norm, &reference.data);
    assert!(
        diff < 1e-2,
        "la BatchNorm difiere en {diff:.3e}, se esperaba menos de 1e-2"
    );
}

#[test]
fn el_harmonic_stacking_coincide_con_la_referencia() {
    if skip_si_no_hay_referencia() {
        return;
    }
    let (frontend, audio) = cargar().unwrap();
    let stages = frontend.stages(&audio).unwrap();
    let reference = read_f32(&reference_dir().join("harmonic_stacking.npy")).unwrap();

    // La referencia lo guarda en NHWC (1, 172, 264, 8); Rust en NCHW.
    let nhwc = stages.harmonic_stacking.permute((0, 2, 3, 1)).unwrap();
    assert_eq!(nhwc.dims4().unwrap(), (1, 172, 264, 8));
    assert_eq!(reference.shape, vec![1, 172, 264, 8]);

    let diff = max_diff(&nhwc, &reference.data);
    assert!(
        diff < 1e-2,
        "el Harmonic Stacking difiere en {diff:.3e}, se esperaba menos de 1e-2"
    );
}

#[test]
fn la_representacion_armonica_tiene_la_forma_que_exige_el_modelo() {
    if skip_si_no_hay_referencia() {
        return;
    }
    let (frontend, audio) = cargar().unwrap();
    let stages = frontend.stages(&audio).unwrap();

    // (batch, 8 canales, 172 frames, 264 bins) en NCHW.
    assert_eq!(stages.harmonic_stacking.dims4().unwrap(), (1, 8, 172, 264));
}

#[test]
fn un_tono_puro_pega_en_el_bin_correcto() {
    if skip_si_no_hay_referencia() {
        return;
    }
    let (frontend, _) = cargar().unwrap();

    // 440 Hz (La4) es exactamente el bin 144: la frecuencia del bin i es
    // fmin * 2^(i/36) = 27.5 * 2^(i/36), y 440/27.5 = 2^4.
    let n = 43_844;
    let samples: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f64 / 22_050.0;
            (2.0 * std::f64::consts::PI * 440.0 * t).sin() as f32
        })
        .collect();

    let stages = frontend.stages(&samples).unwrap();
    let cqt = stages.cqt.flatten_all().unwrap().to_vec1::<f32>().unwrap();

    // El maximo de toda la CQT debe estar en el bin 144.
    let mut best_bin = 0usize;
    let mut best_value = f32::NEG_INFINITY;
    // Los 309 bins del primer frame.
    for (bin, value) in cqt.iter().take(309).enumerate() {
        if *value > best_value {
            best_value = *value;
            best_bin = bin;
        }
    }
    assert_eq!(
        best_bin, 144,
        "el tono de 440 Hz deberia pegar en el bin 144, pero pego en el {best_bin}"
    );
}

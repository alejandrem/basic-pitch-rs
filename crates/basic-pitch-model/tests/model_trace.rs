//! La traza de los puntos de inspeccion de la red.
//!
//! Tres cosas se comprueban:
//!
//! 1. **Que captura lo que debe**: exactamente los siete puntos que define
//!    la Fase 7, con las formas correctas, y nada mas.
//! 2. **Que no perturba la red**: con traza y sin traza, las salidas son
//!    **identicas bit a bit**. Si no lo fueran, la traza estaria cambiando
//!    la computacion y la validacion no serviria de nada.
//! 3. **Que captura los tensores correctos**: cada punto se compara con su
//!    `.npy` de referencia.
//!
//! El punto 3 es el que da sentido a todo: sin el, la traza seria un
//! hypothesis sobre lo que la red calcula. Con el, esta demostrado que
//! `contour_conv` es la conv **antes** del BatchNorm y del ReLU (y no
//! despues), que el layout es NHWC, y que el orden de los canales de la
//! concatenacion es el correcto.

use std::path::{Path, PathBuf};

use basic_pitch_model::ModelTrace;
use basic_pitch_model::{config::ModelConfig, weights::Weights, AcousticFrontend, BasicPitchModel};
use basic_pitch_reference_npy::read_f32;

/// Raiz del repositorio.
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Directorio del modelo desplegable.
fn model_dir() -> PathBuf {
    repo_root().join("models/basic-pitch/icassp-2022")
}

/// Directorio de los puntos de referencia.
fn reference_dir() -> PathBuf {
    repo_root().join("reference/reference_outputs")
}

/// Si faltan los assets, no hay nada que probar.
fn skip_si_faltan() -> bool {
    if !model_dir().join("model.safetensors").exists() || !reference_dir().exists() {
        eprintln!("OMITIDO: faltan el modelo o la referencia");
        return true;
    }
    false
}

/// Carga el modelo, el frontend y el audio de la referencia.
fn cargar() -> (BasicPitchModel, AcousticFrontend, Vec<f32>) {
    let dir = model_dir();
    let config = ModelConfig::load(&dir.join("model_config.json")).expect("configuracion");
    let weights = Weights::load(&dir.join("model.safetensors")).expect("pesos");
    let frontend = AcousticFrontend::load(&dir, &config, &weights).expect("frontend");
    let model = BasicPitchModel::new(
        config,
        weights,
        basic_pitch_cqt::CqtKernels::load(&dir.join("cqt_kernels.safetensors")).expect("kernels"),
    )
    .expect("modelo");
    let audio = read_f32(&reference_dir().join("input_audio.npy")).expect("audio");
    (model, frontend, audio.data)
}

/// Diferencia maxima absoluta entre un tensor y una referencia plana.
fn max_abs_diff(rust: &candle_core::Tensor, reference: &[f32]) -> f64 {
    let rust = rust
        .flatten_all()
        .expect("aplanar")
        .to_vec1::<f32>()
        .expect("float32");
    assert_eq!(
        rust.len(),
        reference.len(),
        "la forma no coincide: {} valores contra {}",
        rust.len(),
        reference.len()
    );
    rust.iter()
        .zip(reference.iter())
        .map(|(a, b)| (*a as f64 - *b as f64).abs())
        .fold(0.0f64, f64::max)
}

/// Error absoluto medio entre un tensor y una referencia plana.
fn mae(rust: &candle_core::Tensor, reference: &[f32]) -> f64 {
    let rust = rust
        .flatten_all()
        .expect("aplanar")
        .to_vec1::<f32>()
        .expect("float32");
    let suma: f64 = rust
        .iter()
        .zip(reference.iter())
        .map(|(a, b)| (*a as f64 - *b as f64).abs())
        .sum();
    suma / rust.len().max(1) as f64
}

#[test]
fn la_traeza_captura_exactamente_los_siete_puntos() {
    if skip_si_faltan() {
        return;
    }
    let (model, frontend, audio) = cargar();
    let stages = frontend.stages(&audio).expect("frontend");

    let mut trace = ModelTrace::default();
    assert!(trace.is_empty());

    model
        .forward_with_trace(&stages.harmonic_stacking, Some(&mut trace))
        .expect("forward");

    assert_eq!(trace.count(), 7, "los siete puntos de la Fase 7");
    assert!(!trace.is_empty());

    // Las formas son las de los puntos de referencia (NHWC).
    assert_eq!(
        trace.contour_conv.as_ref().unwrap().dims4().unwrap(),
        (1, 172, 264, 8)
    );
    assert_eq!(
        trace.contour_reduced.as_ref().unwrap().dims4().unwrap(),
        (1, 172, 264, 1)
    );
    assert_eq!(
        trace.contour.as_ref().unwrap().dims3().unwrap(),
        (1, 172, 264)
    );
    assert_eq!(
        trace.notes_features.as_ref().unwrap().dims4().unwrap(),
        (1, 172, 88, 32)
    );
    assert_eq!(trace.note.as_ref().unwrap().dims3().unwrap(), (1, 172, 88));
    assert_eq!(
        trace.onset_features.as_ref().unwrap().dims4().unwrap(),
        (1, 172, 88, 32)
    );
    assert_eq!(trace.onset.as_ref().unwrap().dims3().unwrap(), (1, 172, 88));
}

#[test]
fn la_traeza_no_cambia_el_resultado_ni_un_bit() {
    if skip_si_faltan() {
        return;
    }
    let (model, frontend, audio) = cargar();
    let stages = frontend.stages(&audio).expect("frontend");

    let sin_traza = model.forward(&stages.harmonic_stacking).expect("forward");

    let mut trace = ModelTrace::default();
    let con_traza = model
        .forward_with_trace(&stages.harmonic_stacking, Some(&mut trace))
        .expect("forward");

    for (nombre, a, b) in [
        ("note", &sin_traza.note, &con_traza.note),
        ("onset", &sin_traza.onset, &con_traza.onset),
        ("contour", &sin_traza.contour, &con_traza.contour),
    ] {
        let a = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(a, b, "{nombre}: la traza ha cambiado el resultado");
    }
}

#[test]
fn cada_punto_de_la_traeza_coincide_con_su_referencia() {
    if skip_si_faltan() {
        return;
    }
    let (model, frontend, audio) = cargar();
    let stages = frontend.stages(&audio).expect("frontend");

    let mut trace = ModelTrace::default();
    model
        .forward_with_trace(&stages.harmonic_stacking, Some(&mut trace))
        .expect("forward");

    // Tolerancias: cuatro veces el error medido.
    //
    // Los puntos pre-activacion (`contour_conv`, `onset_features`) tienen un
    // error mayor que los post-activacion, y no es casualidad: su rango
    // dinamico es mucho mas amplio (el `contour_conv` va de -2.7 a 4.6, y
    // un ReLU no puede dar negativo), asi que el mismo error absoluto de
    // la convolution se ve mas grande. El BatchNorm de la capa siguiente
    // lo comprime.
    let casos: [(&str, &candle_core::Tensor, &str, f64, f64); 7] = [
        (
            "contour_conv",
            trace.contour_conv.as_ref().unwrap(),
            "contour_conv.npy",
            8e-3,
            2e-4,
        ),
        (
            "contour_reduced",
            trace.contour_reduced.as_ref().unwrap(),
            "contour_reduced.npy",
            2e-4,
            1e-5,
        ),
        (
            "contour",
            trace.contour.as_ref().unwrap(),
            "contour.npy",
            2e-4,
            1e-5,
        ),
        (
            "notes_features",
            trace.notes_features.as_ref().unwrap(),
            "notes_features.npy",
            3e-4,
            1e-5,
        ),
        ("note", trace.note.as_ref().unwrap(), "note.npy", 7e-5, 1e-6),
        (
            "onset_features",
            trace.onset_features.as_ref().unwrap(),
            "onset_features.npy",
            8e-3,
            2e-4,
        ),
        (
            "onset",
            trace.onset.as_ref().unwrap(),
            "onset.npy",
            2e-4,
            1e-5,
        ),
    ];

    for (nombre, tensor, fichero, tol_max, tol_mae) in casos {
        let referencia = read_f32(&reference_dir().join(fichero)).expect(fichero);
        let maximo = max_abs_diff(tensor, &referencia.data);
        let medio = mae(tensor, &referencia.data);
        assert!(
            maximo < tol_max,
            "{nombre}: la diferencia maxima es {maximo:.3e}, se esperaba menos de {tol_max:.0e}"
        );
        assert!(
            medio < tol_mae,
            "{nombre}: el error medio es {medio:.3e}, se esperaba menos de {tol_mae:.0e}"
        );
    }
}

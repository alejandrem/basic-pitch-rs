//! La reconstruccion temporal de Rust contra la del upstream.
//!
//! Este es el test de **armonia** entre los tres bloques:
//!
//! ```text
//! P1  audio   -> decodificacion -> mono -> remuestreo -> ventanas
//! P2  ventanas -> CQT -> NormalizedLog -> BatchNorm -> HarmonicStacking -> red
//! P4  ventanas -> predicciones -> reconstruccion temporal
//! ```
//!
//! La referencia no es una reimplementacion: es la funcion
//! `basic_pitch.inference.unwrap_output` **del upstream**, ejecutada sobre
//! audio real por `tools/weight-export/reference_reconstruction.py`.
//!
//! Si las tres piezas no encajasen, el numero de frames o los valores
//! saldrian distintos, y este test lo diria.

use std::path::{Path, PathBuf};

use basic_pitch_audio_io::decode_file;
use basic_pitch_audio_preprocess::{resample, to_mono, AudioWindows, AUDIO_SAMPLE_RATE, HOP_SIZE};
use basic_pitch_model::{config::ModelConfig, weights::Weights, AcousticFrontend, BasicPitchModel};
use basic_pitch_reference_npy::read_f32;
use basic_pitch_transcription::stitching::{expected_frames, ModelUnwrapper, N_OVERLAPPING_FRAMES};
use candle_core::Tensor;

/// Raiz del repositorio.
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Directorio del modelo desplegable.
fn model_dir() -> PathBuf {
    repo_root().join("models/basic-pitch/icassp-2022")
}

/// Directorio del oraculo de reconstruccion.
fn oracle_dir() -> PathBuf {
    repo_root().join("reference/reconstruction")
}

/// Directorio de los recursos de audio del upstream.
fn resources_dir() -> PathBuf {
    repo_root().join("basic-pitch/tests/resources")
}

/// Si faltan los assets, no hay nada que comparar.
fn skip_si_faltan() -> bool {
    if !model_dir().join("model.safetensors").exists()
        || !oracle_dir().exists()
        || !resources_dir().exists()
    {
        eprintln!("OMITIDO: faltan el modelo, el oraculo o los recursos de audio");
        return true;
    }
    false
}

/// Carga el modelo y el frontend.
fn cargar() -> (BasicPitchModel, AcousticFrontend) {
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
    (model, frontend)
}

/// Compara dos tensores y devuelve (max_abs, mae).
fn comparar(rust: &Tensor, referencia: &[f32]) -> (f64, f64) {
    let r = rust
        .flatten_all()
        .expect("aplanar")
        .to_vec1::<f32>()
        .expect("float32");
    assert_eq!(
        r.len(),
        referencia.len(),
        "la forma no coincide: {} contra {}",
        r.len(),
        referencia.len()
    );
    let mut maximo = 0.0f64;
    let mut suma = 0.0f64;
    for (a, b) in r.iter().zip(referencia.iter()) {
        let d = (*a as f64 - *b as f64).abs();
        if d > maximo {
            maximo = d;
        }
        suma += d;
    }
    (maximo, suma / r.len().max(1) as f64)
}

/// Recorre la cadena completa y compara con el oraculo del upstream.
fn comprobar(nombre: &str) {
    let (model, frontend) = cargar();

    // --- P1: audio -> ventanas ---
    let decoded = decode_file(&resources_dir().join(format!("{nombre}.wav"))).expect("decodificar");
    let mono = to_mono(&decoded);
    let remuestreado =
        resample(&mono.samples, mono.sample_rate, AUDIO_SAMPLE_RATE).expect("remuestrear");
    let windows = AudioWindows::new(remuestreado, AUDIO_SAMPLE_RATE);
    let original_length = windows.original_length();

    // --- P2: ventanas -> predicciones, y P4: reconstruccion ---
    let n_frames = expected_frames(original_length, HOP_SIZE);
    let mut reconstruccion = ModelUnwrapper::new(n_frames, N_OVERLAPPING_FRAMES);

    for ventana in windows.iter() {
        let armonica = frontend.forward(&ventana.samples).expect("frontend");
        let salida = model.forward(&armonica).expect("forward");
        reconstruccion.push(&salida).expect("reconstruir");
    }
    let salida = reconstruccion.finish().expect("cerrar");

    // --- La forma tiene que coincidir con la del upstream ---
    let referencia_note = read_f32(&oracle_dir().join(format!("{nombre}_note.npy"))).expect("note");
    assert_eq!(
        salida.note.dims2().unwrap(),
        (referencia_note.shape[0], referencia_note.shape[1]),
        "{nombre}: la forma de la reconstruccion no coincide con la del upstream"
    );
    assert_eq!(
        salida.note.dims2().unwrap().0,
        n_frames,
        "{nombre}: los frames reconstruidos no son los esperados"
    );

    // --- Y los valores ---
    let casos = [
        ("note", &salida.note, format!("{nombre}_note.npy")),
        ("onset", &salida.onset, format!("{nombre}_onset.npy")),
        ("contour", &salida.contour, format!("{nombre}_contour.npy")),
    ];
    for (etapa, tensor, fichero) in casos {
        let referencia = read_f32(&oracle_dir().join(&fichero)).expect(&fichero);
        let (maximo, medio) = comparar(tensor, &referencia.data);
        println!("  {nombre:12} {etapa:8} max={maximo:.3e} mae={medio:.3e}");
        assert!(
            maximo < 1e-2,
            "{nombre}/{etapa}: la diferencia maxima es {maximo:.3e}"
        );
        assert!(
            medio < 1e-4,
            "{nombre}/{etapa}: el error medio es {medio:.3e}"
        );
    }
}

#[test]
fn la_reconstruccion_del_primer_audio_coincide_con_el_upstream() {
    if skip_si_faltan() {
        return;
    }
    comprobar("vocadito_10");
}

#[test]
fn la_reconstruccion_del_segundo_audio_coincide_con_el_upstream() {
    if skip_si_faltan() {
        return;
    }
    comprobar("vocadito_14");
}

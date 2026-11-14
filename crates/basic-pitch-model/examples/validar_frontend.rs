//! Comparacion etapa a etapa del frontend acustico de Rust contra la
//! referencia de TensorFlow.
//!
//! Es la herramienta de validacion de las Fases 3, 4 y 5 (P2). Lee los
//! puntos de comparacion que la Fase 0 guardo en
//! `reference/reference_outputs/` y mide cuanto se parece cada etapa del
//! frontend de Rust.
//!
//! ```bash
//! cargo run --example validar_frontend -p basic-pitch-model
//! ```
//!
//! # Por que etapa a etapa
//!
//! Comparar solo el MIDI final oculta en que punto se ha acumulado el
//! error: un CQT ligeramente desviado y una BatchNorm mal implementada
//! producen una transcripcion parecida pero fausseada. Comparando cada
//! etapa se ve **donde** aparece la discrepancia.
//!
//! # La entrada es la misma
//!
//! Se usa el audio de la referencia (`input_audio.npy`), que son las
//! 43.844 muestras que la referencia alimento al modelo. Asi el error que
//! se mide es **solo** del frontend, sin contaminar con las diferencias
//! de decodificacion y remuestreo entre Symphonia y soundfile.

use std::path::{Path, PathBuf};

use basic_pitch_model::{config::ModelConfig, weights::Weights, AcousticFrontend};
use basic_pitch_reference_npy::{read_f32, write_f32};
use candle_core::Tensor;

/// Informe de la validacion, guardado en `reference/validation_results.json`.
///
/// La fase 4 pide que "cada resultado se guarde con sus dimensiones, tipo
/// de datos y metricas de error". Imprimir por pantalla no es guardar.
fn guardar_informe(
    ruta: &Path,
    etapas: &[(&str, &[usize], &[usize], Metrics)],
) -> std::io::Result<()> {
    let mut json = String::new();
    json.push_str("{\n");
    json.push_str("  \"descripcion\": \"Validacion etapa a etapa del frontend acustico de Rust contra TensorFlow.\",\n");
    json.push_str(
        "  \"generado_por\": \"cargo run --example validar_frontend -p basic-pitch-model\",\n",
    );
    json.push_str("  \"origen_referencia\": \"reference/reference_outputs/ (SavedModel oficial, commit fa5997af)\",\n");
    json.push_str("  \"etapas\": {\n");

    for (indice, (nombre, forma_rust, forma_ref, metricas)) in etapas.iter().enumerate() {
        let coma = if indice + 1 == etapas.len() { "" } else { "," };
        json.push_str(&format!(
            "    \"{nombre}\": {{\n      \"shape_rust\": {forma_rust:?},\n      \"shape_referencia\": {forma_ref:?},\n      \"dtype\": \"float32\",\n      \"mae\": {:.6e},\n      \"rmse\": {:.6e},\n      \"max_abs_diff\": {:.6e},\n      \"max_rel_diff\": {:.6e},\n      \"tolerancia\": \"{tolerancia}\"\n    }}{coma}\n",
            metricas.mae,
            metricas.rmse,
            metricas.max_abs,
            metricas.max_rel(),
            tolerancia = tolerancia_de(nombre),
        ));
    }

    json.push_str("  }\n}\n");
    std::fs::write(ruta, json)
}

/// La tolerancia con la que se valida cada etapa.
fn tolerancia_de(nombre: &str) -> &'static str {
    match nombre {
        "CQT" => "2e-5",
        "NormalizedLog" => "5e-3",
        _ => "1e-2",
    }
}

/// Raiz del repositorio.
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Directorio con el modelo desplegable.
fn model_dir() -> PathBuf {
    repo_root().join("models/basic-pitch/icassp-2022")
}

/// Directorio con los puntos de comparacion de la referencia.
fn reference_dir() -> PathBuf {
    repo_root().join("reference/reference_outputs")
}

/// Metricas de la comparacion entre dos vectores.
#[derive(Debug, Clone, Copy)]
struct Metrics {
    /// Error absoluto medio.
    mae: f64,
    /// Raiz del error cuadratico medio.
    rmse: f64,
    /// Diferencia absoluta maxima.
    max_abs: f64,
    /// Valor absoluto maximo de la referencia (la escala de los datos).
    ref_max_abs: f64,
}

impl Metrics {
    /// Diferencia relativa maxima, respecto a la escala de la referencia.
    fn max_rel(&self) -> f64 {
        if self.ref_max_abs == 0.0 {
            0.0
        } else {
            self.max_abs / self.ref_max_abs
        }
    }
}

/// Compara dos vectores de la misma longitud.
fn compare(rust: &[f32], reference: &[f32]) -> Metrics {
    assert_eq!(
        rust.len(),
        reference.len(),
        "las etapas no tienen el mismo numero de valores"
    );
    let mut sum_abs = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut max_abs = 0.0f64;
    let mut ref_max_abs = 0.0f64;

    for (a, b) in rust.iter().zip(reference.iter()) {
        let diff = (*a as f64 - *b as f64).abs();
        sum_abs += diff;
        sum_sq += diff * diff;
        if diff > max_abs {
            max_abs = diff;
        }
        let abs = b.abs() as f64;
        if abs > ref_max_abs {
            ref_max_abs = abs;
        }
    }

    let n = rust.len().max(1) as f64;
    Metrics {
        mae: sum_abs / n,
        rmse: (sum_sq / n).sqrt(),
        max_abs,
        ref_max_abs,
    }
}

/// Aplana un tensor a un vector, en orden C.
fn flat(tensor: &Tensor) -> Vec<f32> {
    tensor
        .flatten_all()
        .expect("no se pudo aplanar el tensor")
        .to_vec1::<f32>()
        .expect("el tensor no es float32")
}

/// Imprime la linea de una etapa.
fn print_stage(name: &str, rust_shape: &[usize], ref_shape: &[usize], metrics: Metrics) {
    println!(
        "{name:<20} rust {rust_shape:?}  ref {ref_shape:?}  \
         MAE {mae:.3e}  RMSE {rmse:.3e}  max {max_abs:.3e}  rel {rel:.3e}",
        mae = metrics.mae,
        rmse = metrics.rmse,
        max_abs = metrics.max_abs,
        rel = metrics.max_rel(),
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // `--dump <directorio>` escribe las etapas de Rust en `.npy`, para
    // analizarlas con Python (sensibilidad, amplificacion del error, ...).
    let args: Vec<String> = std::env::args().collect();
    let dump_dir = args
        .windows(2)
        .find(|w| w[0] == "--dump")
        .map(|w| PathBuf::from(&w[1]));

    let model_dir = model_dir();
    let reference_dir = reference_dir();

    println!("modelo    : {}", model_dir.display());
    println!("referencia: {}", reference_dir.display());
    println!();

    // --- Carga del modelo ---
    let config = ModelConfig::load(&model_dir.join("model_config.json"))?;
    let weights = Weights::load(&model_dir.join("model.safetensors"))?;
    let frontend = AcousticFrontend::load(&model_dir, &config, &weights)?;

    // --- Audio de la referencia (las mismas muestras quevio TensorFlow) ---
    let audio = read_f32(&reference_dir.join("input_audio.npy"))?;
    let audio_min = audio
        .data
        .iter()
        .map(|v| *v as f64)
        .fold(f64::INFINITY, f64::min);
    let audio_max = audio
        .data
        .iter()
        .map(|v| *v as f64)
        .fold(f64::NEG_INFINITY, f64::max);
    println!(
        "audio de referencia: {:?} ({} muestras, min {audio_min:.6}, max {audio_max:.6})",
        audio.shape,
        audio.data.len(),
    );
    println!();

    // --- Ejecucion del frontend ---
    let stages = frontend.stages(&audio.data)?;

    // --- Comparacion etapa a etapa ---
    println!("=== Comparacion con la referencia de TensorFlow ===");
    println!();

    let cqt_ref = read_f32(&reference_dir.join("cqt.npy"))?;
    let cqt_metrics = compare(&flat(&stages.cqt), &cqt_ref.data);
    print_stage("CQT", stages.cqt.dims(), &cqt_ref.shape, cqt_metrics);

    let log_ref = read_f32(&reference_dir.join("normalized_log.npy"))?;
    let log_metrics = compare(&flat(&stages.normalized_log), &log_ref.data);
    print_stage(
        "NormalizedLog",
        stages.normalized_log.dims(),
        &log_ref.shape,
        log_metrics,
    );

    // Diagnostico: NormalizedLog aplicada a la CQT **de la referencia**.
    // Si la diferencia cae, el error viene de la CQT amplificado por el
    // logaritmo, y no de la implementacion de NormalizedLog.
    let cqt_ref_tensor =
        Tensor::from_slice(&cqt_ref.data, (1, 172, 309), &candle_core::Device::Cpu)?;
    let log_from_ref = basic_pitch_cqt::normalized_log(&cqt_ref_tensor)?;
    print_stage(
        "  (diagnostico)",
        log_from_ref.dims(),
        &log_ref.shape,
        compare(&flat(&log_from_ref), &log_ref.data),
    );

    let bn_ref = read_f32(&reference_dir.join("batch_norm_input.npy"))?;
    let bn_metrics = compare(&flat(&stages.batch_norm), &bn_ref.data);
    print_stage(
        "BatchNorm inicial",
        stages.batch_norm.dims(),
        &bn_ref.shape,
        bn_metrics,
    );

    // La referencia guarda el Harmonic Stacking en NHWC (batch, 172, 264, 8);
    // Rust lo mantiene en NCHW (batch, 8, 172, 264).
    let harmonic_nhwc = stages.harmonic_stacking.permute((0, 2, 3, 1))?;
    let harmonic_ref = read_f32(&reference_dir.join("harmonic_stacking.npy"))?;
    let harmonic_metrics = compare(&flat(&harmonic_nhwc), &harmonic_ref.data);
    print_stage(
        "HarmonicStacking",
        stages.harmonic_stacking.dims(),
        &harmonic_ref.shape,
        harmonic_metrics,
    );

    println!();

    // --- Informe persistido ---
    //
    // La fase 4 exige que cada resultado se guarde con sus dimensiones,
    // su tipo de datos y sus metricas de error.
    let informe = reference_dir
        .parent()
        .unwrap()
        .join("validation_results.json");
    guardar_informe(
        &informe,
        &[
            ("CQT", stages.cqt.dims(), &cqt_ref.shape, cqt_metrics),
            (
                "NormalizedLog",
                stages.normalized_log.dims(),
                &log_ref.shape,
                log_metrics,
            ),
            (
                "BatchNorm inicial",
                stages.batch_norm.dims(),
                &bn_ref.shape,
                bn_metrics,
            ),
            (
                "HarmonicStacking",
                stages.harmonic_stacking.dims(),
                &harmonic_ref.shape,
                harmonic_metrics,
            ),
        ],
    )
    .ok();
    println!("Informe guardado en {}", informe.display());
    println!();

    // --- Volcado opcional ---
    if let Some(dir) = dump_dir {
        std::fs::create_dir_all(&dir)?;
        write_f32(
            &dir.join("cqt_rust.npy"),
            stages.cqt.dims(),
            &flat(&stages.cqt),
        )?;
        write_f32(
            &dir.join("normalized_log_rust.npy"),
            stages.normalized_log.dims(),
            &flat(&stages.normalized_log),
        )?;
        write_f32(
            &dir.join("batch_norm_rust.npy"),
            stages.batch_norm.dims(),
            &flat(&stages.batch_norm),
        )?;
        write_f32(
            &dir.join("harmonic_stacking_rust.npy"),
            stages.harmonic_stacking.dims(),
            &flat(&stages.harmonic_stacking),
        )?;
        println!("Etapas de Rust volcadas en {}", dir.display());
        println!();
    }

    Ok(())
}

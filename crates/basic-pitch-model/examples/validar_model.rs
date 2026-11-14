//! El validador: compara **todo** el pipeline con TensorFlow, etapa a etapa.
//!
//! ```bash
//! cargo run --release --example validar_model -p basic-pitch-model
//! ```
//!
//! # Que mide
//!
//! Las once etapas que define la Fase 7, con las seis metricas que tambien
//! pide (MAE, MSE, RMSE, maximo absoluto, diferencia relativa y similitud
//! coseno):
//!
//! | # | Etapa | De donde sale |
//! |---|---|---|
//! | 1 | `CQT` | el frontend acustico |
//! | 2 | `NormalizedLog` | el frontend acustico |
//! | 3 | `BatchNorm inicial` | el frontend acustico |
//! | 4 | `HarmonicStacking` | el frontend acustico |
//! | 5 | `contour_conv` | la traza de la rama de contornos (pre-activacion) |
//! | 6 | `contour_reduced` | la traza de la rama de contornos |
//! | 7 | `contour` | la traza de la rama de contornos |
//! | 8 | `notes_features` | la traza de la rama de notas (pre-activacion) |
//! | 9 | `note` | la traza de la rama de notas |
//! | 10 | `onset_features` | la traza de la rama de onsets (pre-activacion) |
//! | 11 | `onset` | la traza de la rama de onsets |
//!
//! # Que escribe
//!
//! `reference/validation_results.json`, que es el **reporte de diferencias**
//! que pide el entregable de la Fase 7: con la forma de Rust y de la
//! referencia, el tipo de datos, las seis metricas y la tolerancia con la
//! que se valida cada etapa.

use std::path::{Path, PathBuf};

use basic_pitch_model::config::ModelConfig;
use basic_pitch_model::weights::Weights;
use basic_pitch_model::{AcousticFrontend, BasicPitchModel, ModelTrace};
use basic_pitch_reference_npy::read_f32;
use basic_pitch_validation::{compare, Metrics};
use candle_core::Tensor;

/// Raiz del repositorio.
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Directorio del modelo.
fn model_dir() -> PathBuf {
    repo_root().join("models/basic-pitch/icassp-2022")
}

/// Directorio de los puntos de referencia.
fn reference_dir() -> PathBuf {
    repo_root().join("reference/reference_outputs")
}

/// Una etapa validada.
struct Etapa {
    nombre: &'static str,
    forma_rust: Vec<usize>,
    forma_ref: Vec<usize>,
    metricas: Metrics,
    tolerancia_max: f64,
}

/// Aplana un tensor a un vector, en orden C.
fn flat(tensor: &Tensor) -> Result<Vec<f32>, candle_core::Error> {
    tensor.flatten_all()?.to_vec1::<f32>()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = model_dir();
    let ref_dir = reference_dir();

    let config = ModelConfig::load(&dir.join("model_config.json"))?;
    let weights = Weights::load(&dir.join("model.safetensors"))?;
    let frontend = AcousticFrontend::load(&dir, &config, &weights)?;
    let model = BasicPitchModel::new(
        config,
        weights,
        basic_pitch_cqt::CqtKernels::load(&dir.join("cqt_kernels.safetensors"))?,
    )?;

    // El audio de la referencia: las mismas 43844 muestras quevio
    // TensorFlow.
    let audio = read_f32(&ref_dir.join("input_audio.npy"))?;

    // El frontend acustico.
    let stages = frontend.stages(&audio.data)?;

    // La red, con la traza de sus etapas internas.
    let mut trace = ModelTrace::default();
    model.forward_with_trace(&stages.harmonic_stacking, Some(&mut trace))?;

    // Cada etapa, con su punto de referencia y su tolerancia.
    //
    // Las tolerancias estan justificadas en
    // docs/practica/fase-08-frontend-cqt.md (apartado 4.7) y en
    // docs/practica/fase-13-posprocesamiento-musical.md.
    let casos: Vec<(&str, Tensor, &str, f64)> = vec![
        ("CQT", stages.cqt.clone(), "cqt.npy", 2e-5),
        (
            "NormalizedLog",
            stages.normalized_log.clone(),
            "normalized_log.npy",
            5e-3,
        ),
        (
            "BatchNorm inicial",
            stages.batch_norm.clone(),
            "batch_norm_input.npy",
            1e-2,
        ),
        (
            "HarmonicStacking",
            // El frontend trabaja en NCHW y la referencia guarda NHWC:
            // hay que permutar para comparar elemento a elemento.
            stages
                .harmonic_stacking
                .permute((0, 2, 3, 1))?
                .contiguous()?,
            "harmonic_stacking.npy",
            1e-2,
        ),
        (
            "contour_conv",
            trace
                .contour_conv
                .clone()
                .expect("la traza deberia tener contour_conv"),
            "contour_conv.npy",
            8e-3,
        ),
        (
            "contour_reduced",
            trace
                .contour_reduced
                .clone()
                .expect("la traza deberia tener contour_reduced"),
            "contour_reduced.npy",
            2e-4,
        ),
        (
            "contour",
            trace
                .contour
                .clone()
                .expect("la traza deberia tener contour"),
            "contour.npy",
            2e-4,
        ),
        (
            "notes_features",
            trace
                .notes_features
                .clone()
                .expect("la traza deberia tener notes_features"),
            "notes_features.npy",
            3e-4,
        ),
        (
            "note",
            trace.note.clone().expect("la traza deberia tener note"),
            "note.npy",
            7e-5,
        ),
        (
            "onset_features",
            trace
                .onset_features
                .clone()
                .expect("la traza deberia tener onset_features"),
            "onset_features.npy",
            8e-3,
        ),
        (
            "onset",
            trace.onset.clone().expect("la traza deberia tener onset"),
            "onset.npy",
            2e-4,
        ),
    ];

    println!("Validacion del pipeline completo contra TensorFlow");
    println!("================================================");
    println!();

    let mut etapas: Vec<Etapa> = Vec::new();
    let mut fallos = 0usize;

    for (nombre, tensor, fichero, tolerancia) in casos {
        let referencia = read_f32(&ref_dir.join(fichero))?;
        let rust = flat(&tensor)?;
        let metricas = compare(&rust, &referencia.data)?;

        let ok = metricas.max_abs_diff <= tolerancia;
        if !ok {
            fallos += 1;
        }
        println!(
            "{} {:<20} {}",
            if ok { "OK  " } else { "FALLO" },
            nombre,
            metricas.linea("")
        );

        etapas.push(Etapa {
            nombre,
            forma_rust: tensor.dims().to_vec(),
            forma_ref: referencia.shape.clone(),
            metricas,
            tolerancia_max: tolerancia,
        });
    }

    // El informe.
    let destino = repo_root().join("reference/validation_results.json");
    let mut mapa = serde_json::Map::new();
    for etapa in &etapas {
        mapa.insert(
            etapa.nombre.to_string(),
            serde_json::json!({
                "shape_rust": etapa.forma_rust,
                "shape_referencia": etapa.forma_ref,
                "dtype": "float32",
                "mae": etapa.metricas.mae,
                "mse": etapa.metricas.mse,
                "rmse": etapa.metricas.rmse,
                "max_abs_diff": etapa.metricas.max_abs_diff,
                "max_rel_diff": etapa.metricas.max_rel_diff,
                "cosine_similarity": etapa.metricas.cosine_similarity,
                "tolerancia": format!("{:.0e}", etapa.tolerancia_max),
            }),
        );
    }

    let informe = serde_json::json!({
        "descripcion": "Validacion etapa a etapa de Rust contra TensorFlow.",
        "generado_por": "cargo run --release --example validar_model -p basic-pitch-model",
        "origen_referencia": "reference/reference_outputs/ (SavedModel oficial, commit fa5997af)",
        "nota": "Las etapas del frontend las produce el crate cqt; las de la red, la traza de las ramas.",
        "etapas": mapa,
    });
    std::fs::write(&destino, serde_json::to_string_pretty(&informe)? + "\n")?;
    println!();
    println!("Informe escrito en {}", destino.display());

    if fallos > 0 {
        eprintln!("\n{fallos} etapas fuera de tolerancia.");
        std::process::exit(1);
    }
    println!("Las {} etapas coinciden con TensorFlow.", etapas.len());
    Ok(())
}

//! CLI de `basic-pitch-rs`.
//!
//! Estado actual del proyecto (Fase 2): el workspace compila, los pesos estan
//! exportados y validados, y el motor esta en construccion. Este comando
//! informa del estado y valida los artefactos del modelo.

use std::path::PathBuf;
use std::process::ExitCode;

use basic_pitch_model::ModelConfig;

const DEFAULT_MODEL_DIR: &str = "models/basic-pitch/icassp-2022";

fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let mut args = std::env::args().skip(1);
    let command = args.next().unwrap_or_else(|| "info".to_string());

    match command.as_str() {
        "info" => {
            let model_dir = args
                .next()
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_DIR));

            if let Err(error) = print_info(&model_dir) {
                eprintln!("error: {error}");
                return ExitCode::FAILURE;
            }
            ExitCode::SUCCESS
        }
        "--help" | "-h" | "help" => {
            print_help();
            ExitCode::SUCCESS
        }
        other => {
            eprintln!("comando desconocido: {other}\n");
            print_help();
            ExitCode::FAILURE
        }
    }
}

fn print_help() {
    println!("basic-pitch-rs - transcripcion musical en Rust");
    println!();
    println!("USO:");
    println!("    basic-pitch <comando>");
    println!();
    println!("COMANDOS:");
    println!("    info [dir]    Muestra la configuracion del modelo y valida sus invariantes");
    println!("    help          Muestra esta ayuda");
    println!();
    println!("El directorio del modelo por defecto es:");
    println!("    {DEFAULT_MODEL_DIR}");
}

fn print_info(model_dir: &std::path::Path) -> Result<(), String> {
    println!("basic-pitch-rs");
    println!("===============");
    println!();

    let config_path = model_dir.join("model_config.json");
    let weights_path = model_dir.join("model.safetensors");
    let cqt_path = model_dir.join("cqt_kernels.safetensors");

    if !config_path.exists() {
        return Err(format!(
            "no se encuentra {}; ejecuta primero tools/weight-export/export_weights.py",
            config_path.display()
        ));
    }

    let config = ModelConfig::load(&config_path)?;
    println!(
        "Modelo:      {} / {}",
        config.model.name, config.model.variant
    );
    println!("Referencia:  commit {}", config.model.reference_commit);
    println!("Directorio:  {}", model_dir.display());
    println!();

    println!("Contrato tecnico:");
    println!(
        "  audio            : {} muestras @ {} Hz, {} canal(es)",
        config.audio.audio_n_samples, config.audio.sample_rate, config.audio.channels
    );
    println!(
        "  CQT              : {} bins ({} semitonos) <- la salida se trunca a {} en Harmonic Stacking",
        config.frequency.cqt_n_bins,
        config.frequency.cqt_n_semitones,
        config.frequency.n_freq_bins_contours
    );
    println!(
        "  armónicos        : {:?} -> shifts {:?}",
        config.harmonic_stacking.harmonics, config.harmonic_stacking.shifts
    );
    println!(
        "  concat de onsets : {:?} -> {} canales",
        config.concat.order, config.concat.channels
    );
    println!(
        "  BatchNorm        : epsilon {} ({}), {} capas",
        config.batch_norm.epsilon,
        config.batch_norm.epsilon_source,
        config.batch_norm.layers.len()
    );
    println!(
        "  ventanas         : overlap {} muestras, hop {} muestras, {} -> {} frames",
        config.windowing.overlap_len,
        config.windowing.hop_size,
        config.windowing.frames_per_window,
        config.windowing.frames_kept_per_window
    );
    println!();

    println!("Capas convolucionales activas:");
    for layer in &config.layers {
        println!(
            "  {:20} kernel {:2?} filtros {:3} entrada {:3} stride {:?} pad {:?} act {:8} bn {}",
            layer.name,
            layer.kernel_tf,
            layer.filters,
            layer.in_channels,
            layer.stride_tf,
            layer.pad_same_tf,
            layer.activation,
            if layer.batch_norm { "si" } else { "no" }
        );
    }
    println!();

    for (label, path) in [("pesos    ", &weights_path), ("kernels  ", &cqt_path)] {
        if path.exists() {
            println!(
                "{label}: {} ({} bytes)",
                path.display(),
                std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
            );
        } else {
            println!("{label}: {} (AUSENTE)", path.display());
        }
    }

    println!();
    println!("Invariantes del contrato: OK (validados al cargar la configuracion)");
    Ok(())
}

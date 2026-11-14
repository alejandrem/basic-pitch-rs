//! CLI de `basic-pitch-rs`.
//!
//! Estado actual del proyecto: el frontend acustico (Fases 3, 4 y 5), la red
//! neuronal, la reconstruccion temporal (Fase 8) y la deteccion de notas
//! (Fase 9) estan implementados y validados contra la referencia de
//! TensorFlow.
//!
//! ```text
//! basic-pitch info [dir]                Muestra el contrato del modelo
//! basic-pitch transcribe <audio> [-o salida.mid]
//!                                        Transcribe un archivo a MIDI
//! ```

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use basic_pitch_midi_io::MidiWriter;
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
        "transcribe" => {
            let mut audio: Option<PathBuf> = None;
            let mut output: Option<PathBuf> = None;
            let mut model_dir = PathBuf::from(DEFAULT_MODEL_DIR);

            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "-o" | "--output" => match args.next() {
                        Some(ruta) => output = Some(PathBuf::from(ruta)),
                        None => {
                            eprintln!("error: -o necesita una ruta");
                            return ExitCode::FAILURE;
                        }
                    },
                    "-m" | "--model" => match args.next() {
                        Some(ruta) => model_dir = PathBuf::from(ruta),
                        None => {
                            eprintln!("error: -m necesita una ruta");
                            return ExitCode::FAILURE;
                        }
                    },
                    "-h" | "--help" => {
                        print_help();
                        return ExitCode::SUCCESS;
                    }
                    otra => {
                        if audio.is_none() {
                            audio = Some(PathBuf::from(otra));
                        } else {
                            eprintln!("error: sobran argumentos: {otra}");
                            return ExitCode::FAILURE;
                        }
                    }
                }
            }

            let Some(audio) = audio else {
                eprintln!("error: falta el archivo de audio\n");
                print_help();
                return ExitCode::FAILURE;
            };

            match transcribe(&audio, output.as_deref(), &model_dir) {
                Ok(mensaje) => {
                    println!("{mensaje}");
                    ExitCode::SUCCESS
                }
                Err(error) => {
                    eprintln!("error: {error}");
                    ExitCode::FAILURE
                }
            }
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

/// Transcribe un archivo de audio y escribe el MIDI.
fn transcribe(audio: &Path, output: Option<&Path>, model_dir: &Path) -> Result<String, String> {
    let transcriptor = basic_pitch_transcription::Transcriber::load(model_dir)
        .map_err(|e| format!("no se pudo cargar el modelo: {e}"))?;

    let resultado = transcriptor
        .transcribe(audio)
        .map_err(|e| format!("no se pudo transcribir {audio:?}: {e}"))?;

    let destino = output
        .map(PathBuf::from)
        .unwrap_or_else(|| audio.with_extension("mid"));

    // El evento musical es UNO solo (el canonico de midi-io), con la
    // velocidad ya calculada a partir de la amplitud del modelo.
    MidiWriter::new()
        .write(&resultado.notes, &destino)
        .map_err(|e| format!("no se pudo escribir {:?}: {e}", destino))?;

    Ok(format!(
        "{} -> {}  ({} notas, {:.2} s de audio)",
        audio.display(),
        destino.display(),
        resultado.notes.len(),
        resultado.duration_seconds
    ))
}

fn print_help() {
    println!("basic-pitch-rs - transcripcion musical en Rust");
    println!();
    println!("USO:");
    println!("    basic-pitch <comando>");
    println!();
    println!("COMANDOS:");
    println!(
        "    info [dir]              Muestra la configuracion del modelo y valida sus invariantes"
    );
    println!("    transcribe <audio>      Transcribe un archivo de audio a MIDI");
    println!("        -o <salida.mid>    Ruta del MIDI de salida (por defecto, junto al audio)");
    println!("        -m <dir_modelo>    Directorio del modelo");
    println!("    help                    Muestra esta ayuda");
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

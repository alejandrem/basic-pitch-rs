//! CLI de `basic-pitch-rs`.
//!
//! ```text
//! basic-pitch info [dir]                Muestra el contrato del modelo
//! basic-pitch transcribe <audio> [opciones]
//!                                        Transcribe un archivo a MIDI
//! ```
//!
//! Los umbrales de la deteccion de notas se exponen como opciones; la lista
//! completa, con sus valores por defecto, esta en `basic-pitch transcribe --help`.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use basic_pitch_model::ModelConfig;
use basic_pitch_transcription::note_detection::{
    DEFAULT_ENERGY_TOL, DEFAULT_FRAME_THRESHOLD, DEFAULT_MINIMUM_NOTE_LENGTH_MS,
    DEFAULT_ONSET_THRESHOLD,
};
use basic_pitch_transcription::NoteDetectionConfig;
use clap::{Args, Parser, Subcommand};

const DEFAULT_MODEL_DIR: &str = "models/basic-pitch/icassp-2022";

/// Transcripcion musical (audio a MIDI) en Rust, port de Spotify Basic Pitch.
#[derive(Parser)]
#[command(name = "basic-pitch", version)]
struct Cli {
    /// Sin subcomando se ejecuta `info`.
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Muestra la configuracion del modelo y valida sus invariantes
    Info {
        /// Directorio del modelo
        #[arg(value_name = "DIR")]
        dir: Option<PathBuf>,
    },
    /// Transcribe un archivo de audio a MIDI
    Transcribe(TranscribeArgs),
}

#[derive(Args)]
struct TranscribeArgs {
    /// Archivo de audio (WAV, FLAC, OGG, MP3 o AAC)
    audio: PathBuf,

    /// Ruta del MIDI de salida (por defecto, junto al audio)
    #[arg(short, long, value_name = "SALIDA.mid")]
    output: Option<PathBuf>,

    /// Directorio del modelo
    #[arg(short, long, value_name = "DIR", default_value = DEFAULT_MODEL_DIR)]
    model: PathBuf,

    /// Umbral de activacion de onset (0 a 1)
    ///
    /// Subirlo da menos notas espurias, pero pierde notas reales con ataque
    /// suave.
    #[arg(long, value_name = "N", value_parser = fraccion, default_value_t = DEFAULT_ONSET_THRESHOLD)]
    onset_threshold: f64,

    /// Umbral de activacion de nota (0 a 1)
    ///
    /// Subirlo gana precision y pierde recall. Con una pieza de prueba
    /// sintetica, de 0.3 a 0.5 la precision pasa de 0.81 a 0.93 y el recall de
    /// 0.87 a 0.73.
    #[arg(long, value_name = "N", value_parser = fraccion, default_value_t = DEFAULT_FRAME_THRESHOLD)]
    frame_threshold: f64,

    /// Duracion minima de nota, en milisegundos
    #[arg(long, value_name = "MS", value_parser = no_negativo, default_value_t = DEFAULT_MINIMUM_NOTE_LENGTH_MS)]
    min_note_length_ms: f64,

    /// Tolerancia de energia de Melodia, en frames
    #[arg(long, value_name = "FRAMES", default_value_t = DEFAULT_ENERGY_TOL)]
    energy_tol: usize,

    /// Frecuencia minima permitida, en Hz
    #[arg(long, value_name = "HZ", value_parser = positivo)]
    min_freq: Option<f64>,

    /// Frecuencia maxima permitida, en Hz
    #[arg(long, value_name = "HZ", value_parser = positivo)]
    max_freq: Option<f64>,

    /// Desactiva el truco de Melodia: solo notas con onset claro
    #[arg(long)]
    no_melodia: bool,

    /// No infiere onsets adicionales a partir de los cambios de nota
    #[arg(long)]
    no_infer_onsets: bool,
}

impl TranscribeArgs {
    /// Configuracion de la deteccion de notas que describen las opciones.
    fn note_detection(&self) -> NoteDetectionConfig {
        NoteDetectionConfig {
            onset_threshold: self.onset_threshold,
            frame_threshold: self.frame_threshold,
            minimum_note_length_ms: self.min_note_length_ms,
            energy_tol: self.energy_tol,
            infer_onsets: !self.no_infer_onsets,
            melodia_trick: !self.no_melodia,
            min_freq: self.min_freq,
            max_freq: self.max_freq,
        }
    }
}

fn numero(texto: &str) -> Result<f64, String> {
    texto
        .parse::<f64>()
        .map_err(|_| format!("`{texto}` no es un numero"))
}

/// Un numero entre 0 y 1.
fn fraccion(texto: &str) -> Result<f64, String> {
    let valor = numero(texto)?;
    if (0.0..=1.0).contains(&valor) {
        Ok(valor)
    } else {
        Err(format!("{valor} debe estar entre 0 y 1"))
    }
}

/// Un numero finito mayor o igual que cero.
fn no_negativo(texto: &str) -> Result<f64, String> {
    let valor = numero(texto)?;
    if valor.is_finite() && valor >= 0.0 {
        Ok(valor)
    } else {
        Err(format!("{valor} debe ser un numero no negativo"))
    }
}

/// Un numero finito mayor que cero.
fn positivo(texto: &str) -> Result<f64, String> {
    let valor = numero(texto)?;
    if valor.is_finite() && valor > 0.0 {
        Ok(valor)
    } else {
        Err(format!("{valor} debe ser un numero positivo"))
    }
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    match cli.command.unwrap_or(Command::Info { dir: None }) {
        Command::Info { dir } => {
            let model_dir = dir.unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_DIR));

            if let Err(error) = print_info(&model_dir) {
                eprintln!("error: {error}");
                return ExitCode::FAILURE;
            }
            ExitCode::SUCCESS
        }
        Command::Transcribe(args) => match transcribe(&args) {
            Ok(mensaje) => {
                println!("{mensaje}");
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("error: {error}");
                ExitCode::FAILURE
            }
        },
    }
}

/// Transcribe un archivo de audio y escribe el MIDI.
fn transcribe(args: &TranscribeArgs) -> Result<String, String> {
    let mut transcriptor = basic_pitch_transcription::Transcriber::load(&args.model)
        .map_err(|e| format!("no se pudo cargar el modelo: {e}"))?;
    transcriptor.config.note_detection = args.note_detection();

    let resultado = transcriptor
        .transcribe(&args.audio)
        .map_err(|e| format!("no se pudo transcribir {:?}: {e}", args.audio))?;

    let destino = args
        .output
        .clone()
        .unwrap_or_else(|| args.audio.with_extension("mid"));

    // El MIDI ya viene generado en memoria dentro del resultado (Fase 12):
    // solo hay que volcarlo al disco.
    std::fs::write(&destino, &resultado.midi_data)
        .map_err(|e| format!("no se pudo escribir {:?}: {e}", destino))?;

    Ok(format!(
        "{} -> {}  ({} notas, {:.2} s de audio)",
        args.audio.display(),
        destino.display(),
        resultado.notes.len(),
        resultado.duration_seconds
    ))
}

fn print_info(model_dir: &Path) -> Result<(), String> {
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

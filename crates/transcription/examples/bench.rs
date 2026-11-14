//! Cronometro del pipeline completo sobre un archivo real.
//!
//! Es una "pasada en seco": procesa el archivo entero (decodificacion,
//! ventanas, red y reconstruccion) y dice cuanto tarda y que ha detectado
//! el modelo. La deteccion de notas todavia no esta implementada (Fase 9),
//! asi que no produce MIDI: eso es lo que queda.
//!
//! ```bash
//! cargo run --release --example bench -p basic-pitch-transcription -- cancion.mp3
//! ```

use std::path::Path;
use std::time::Instant;

use basic_pitch_audio_io::decode_file;
use basic_pitch_audio_preprocess::{resample, to_mono, AudioWindows, AUDIO_SAMPLE_RATE, HOP_SIZE};
use basic_pitch_model::{config::ModelConfig, weights::Weights, AcousticFrontend, BasicPitchModel};
use basic_pitch_transcription::stitching::{expected_frames, ModelUnwrapper, N_OVERLAPPING_FRAMES};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let audio = Path::new(
        args.get(1)
            .map(|s| s.as_str())
            .unwrap_or("basic-pitch/tests/resources/vocadito_10.wav"),
    );
    let raiz = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let dir = raiz.join("models/basic-pitch/icassp-2022");

    let t0 = Instant::now();
    let config = ModelConfig::load(&dir.join("model_config.json"))?;
    let weights = Weights::load(&dir.join("model.safetensors"))?;
    let frontend = AcousticFrontend::load(&dir, &config, &weights)?;
    let model = BasicPitchModel::new(
        config,
        weights,
        basic_pitch_cqt::CqtKernels::load(&dir.join("cqt_kernels.safetensors"))?,
    )?;
    let carga = t0.elapsed();

    // --- P1: audio -> ventanas ---
    let t1 = Instant::now();
    let decoded = decode_file(audio)?;
    let mono = to_mono(&decoded);
    let remuestreado = resample(&mono.samples, mono.sample_rate, AUDIO_SAMPLE_RATE)?;
    let windows = AudioWindows::new(remuestreado, AUDIO_SAMPLE_RATE);
    let original_length = windows.original_length();

    // --- P2 + P4: ventanas -> predicciones -> reconstruccion ---
    let n_frames = expected_frames(original_length, HOP_SIZE);
    let mut reconstruccion = ModelUnwrapper::new(n_frames, N_OVERLAPPING_FRAMES);
    for ventana in windows.iter() {
        let armonica = frontend.forward(&ventana.samples)?;
        let salida = model.forward(&armonica)?;
        reconstruccion.push(&salida)?;
    }
    let salida = reconstruccion.finish()?;
    let inferencia = t1.elapsed();

    let duracion = original_length as f64 / AUDIO_SAMPLE_RATE as f64;
    println!("archivo       : {}", audio.display());
    println!(
        "decodificado  : {} canales, {} Hz, {:.2} s",
        decoded.channels, decoded.sample_rate, duracion
    );
    println!(
        "ventanas      : {} de {} muestras",
        windows.len(),
        AUDIO_SAMPLE_RATE
    );
    println!("frames        : {}", n_frames);
    println!("carga modelo  : {:?}", carga);
    println!("inferencia    : {:?}", inferencia);
    println!(
        "tiempo real   : {:.2} x el tiempo del audio",
        inferencia.as_secs_f64() / duracion
    );

    // --- Lo que el modelo ha detectado (la deteccion de notas es de la
    //     Fase 9; aqui solo se informa) ---
    let notas = &salida.note;
    let (frames, bins) = (notas.dims2()?.0, notas.dims2()?.1);
    let datos = notas.flatten_all()?.to_vec1::<f32>()?;

    let mut max_por_nota = vec![0.0f32; bins];
    for f in 0..frames {
        for b in 0..bins {
            let v = datos[f * bins + b];
            if v > max_por_nota[b] {
                max_por_nota[b] = v;
            }
        }
    }
    let activas = max_por_nota.iter().filter(|v| **v > 0.5).count();
    let maxima = max_por_nota.iter().cloned().fold(0.0f32, f32::max);
    println!();
    println!("--- Lo que ve el modelo (la deteccion de notas es de la Fase 9) ---");
    println!("notas con activacion > 0.5 : {activas} de {bins}");
    println!("activacion maxima          : {maxima:.4}");

    println!();
    println!("Para tener el MIDI faltan tres piezas:");
    println!("  1. Fase 9: deteccion polifonica de notas (to_notes esta vacio)");
    println!("  2. Fase 11: conectar el escritor MIDI (ya existe y funciona)");
    println!("  3. Fase 12: el comando `basic-pitch transcribe` en el CLI");
    Ok(())
}

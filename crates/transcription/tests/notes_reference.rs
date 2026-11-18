//! La deteccion de notas de Rust contra la del upstream.
//!
//! El oraculo lo genera `tools/weight-export/reference_notes.py` con
//! `basic_pitch.inference.predict` **del upstream, sin modificar**. No es
//! una reimplementacion: es el juez.
//!
//! Se comparan las notas una a una: mismo numero, mismos pitches, mismos
//! tiempos de inicio y fin, y misma amplitud.

use std::path::{Path, PathBuf};

use basic_pitch_audio_preprocess::{AUDIO_N_SAMPLES, AUDIO_SAMPLE_RATE, FFT_HOP};
use basic_pitch_reference_npy::read_f32;
use basic_pitch_transcription::note_detection::{
    output_to_notes_polyphonic, NoteDetectionConfig, N_NOTES,
};
use basic_pitch_transcription::stitching::frame_to_time;
use candle_core::Tensor;

mod common;
use common::{omitir, vocadito_path};

/// Raiz del repositorio.
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Directorio del oraculo de notas.
fn oracle_dir() -> PathBuf {
    repo_root().join("reference/notes")
}

/// Si falta el oraculo, no hay nada que comparar.
fn skip_si_falta() -> bool {
    if !oracle_dir().exists() {
        omitir("no existe el oraculo de notas");
        return true;
    }
    false
}

/// Compara las notas detectadas por Rust con las del upstream.
fn comprobar(nombre: &str) {
    let referencia = read_f32(&oracle_dir().join(format!("{nombre}_notes.npy"))).expect("oraculo");
    let n_notas = referencia.shape[0];
    assert_eq!(
        referencia.shape[1], 4,
        "columnas: inicio, fin, pitch, amplitud"
    );

    // --- Las activaciones del modelo, las mismas que usa el upstream ---
    //
    // Se leen de la reconstruccion de referencia, que el upstream genero con
    // su propio `unwrap_output`.
    let raiz = repo_root();
    let dir = raiz.join("models/basic-pitch/icassp-2022");
    let config_model = basic_pitch_model::config::ModelConfig::load(&dir.join("model_config.json"))
        .expect("configuracion");
    let weights =
        basic_pitch_model::weights::Weights::load(&dir.join("model.safetensors")).expect("pesos");
    let frontend =
        basic_pitch_model::AcousticFrontend::load(&dir, &config_model, &weights).expect("frontend");
    let model = basic_pitch_model::BasicPitchModel::new(
        config_model,
        weights,
        basic_pitch_cqt::CqtKernels::load(&dir.join("cqt_kernels.safetensors")).expect("kernels"),
    )
    .expect("modelo");

    // El audio del oraculo.
    let audio = match nombre {
        "vocadito_10" => vocadito_path(),
        "peso_del_silencio" => raiz.join("reference/audio_reference/peso_del_silencio.wav"),
        otro => panic!("audio desconocido: {otro}"),
    };
    let decoded = basic_pitch_audio_io::decode_file(&audio).expect("decodificar");
    let mono = basic_pitch_audio_preprocess::to_mono(&decoded);
    let remuestreado = basic_pitch_audio_preprocess::resample(
        &mono.samples,
        mono.sample_rate,
        basic_pitch_audio_preprocess::AUDIO_SAMPLE_RATE,
    )
    .expect("remuestrear");
    let windows = basic_pitch_audio_preprocess::AudioWindows::new(
        remuestreado,
        basic_pitch_audio_preprocess::AUDIO_SAMPLE_RATE,
    );

    let mut reconstruccion = basic_pitch_transcription::stitching::ModelUnwrapper::new(
        basic_pitch_transcription::stitching::expected_frames(
            windows.original_length(),
            basic_pitch_audio_preprocess::HOP_SIZE,
        ),
        basic_pitch_transcription::stitching::N_OVERLAPPING_FRAMES,
    );
    for ventana in windows.iter() {
        let armonica = frontend.forward(&ventana.samples).expect("frontend");
        let salida = model.forward(&armonica).expect("forward");
        reconstruccion.push(&salida).expect("reconstruir");
    }
    let salida = reconstruccion.finish().expect("cerrar");

    // --- Las activaciones que alimentan la deteccion ---
    let frames = a_filas(&salida.note);
    let onsets = a_filas(&salida.onset);
    let n_frames = frames.len() / N_NOTES;

    let notas = output_to_notes_polyphonic(
        &frames,
        &onsets,
        n_frames,
        N_NOTES,
        &NoteDetectionConfig::default(),
    );

    // --- Comparacion ---
    println!(
        "  {nombre:20} Rust {} notas   upstream {} notas",
        notas.len(),
        n_notas
    );
    assert_eq!(notas.len(), n_notas, "{nombre}: numero de notas distinto");

    let mut max_pitch = 0usize;
    let mut max_inicio = 0.0f64;
    let mut max_fin = 0.0f64;
    let mut max_amp = 0.0f64;

    for (i, nota) in notas.iter().enumerate() {
        let (inicio, fin, pitch, amplitud) = (
            referencia.data[i * 4],
            referencia.data[i * 4 + 1],
            referencia.data[i * 4 + 2],
            referencia.data[i * 4 + 3],
        );

        // El upstream devuelve los tiempos en segundos; Rust, en frames.
        // Se comparan los **tiempos**, con la misma conversion que usa el
        // motor (`frame_to_time`), que es la del upstream.
        //
        // Convertir un tiempo del upstream a frames no serviria: la
        // numeracion de ventanas divide entre 172 y solo sobreviven 142, y
        // esa inconsistencia hace la conversion circular.
        let tiempo_inicio = frame_to_time(
            nota.start_frame,
            AUDIO_SAMPLE_RATE,
            FFT_HOP,
            AUDIO_N_SAMPLES,
        );
        let tiempo_fin = frame_to_time(nota.end_frame, AUDIO_SAMPLE_RATE, FFT_HOP, AUDIO_N_SAMPLES);

        assert_eq!(
            nota.pitch as i64, pitch as i64,
            "{nombre}: nota {i}: pitch distinto"
        );
        // Tolerancias justificadas:
        //
        // - **Tiempos**: el oraculo se guarda en float32, y un tiempo de
        //   ~177 s en float32 tiene ~1.5e-5 s de error. 1e-4 s es mil
        //   veces mas fino que un frame (11.6 ms), asi que no puede
        //   esconder una nota distinta.
        // - **Amplitud**: la amplitud es la media de las activaciones de
        //   la red, y la red difiere de TensorFlow en ~1e-5 (medido: el
        //   error de la CQT es de 5.7e-6 y se propaga).
        assert!(
            (tiempo_inicio - inicio as f64).abs() < 1e-4,
            "{nombre}: nota {i}: tiempo inicial distinto (Rust {tiempo_inicio}, upstream {inicio})"
        );
        assert!(
            (tiempo_fin - fin as f64).abs() < 1e-4,
            "{nombre}: nota {i}: tiempo final distinto (Rust {tiempo_fin}, upstream {fin})"
        );
        max_pitch = max_pitch.max((nota.pitch as i64 - pitch as i64).unsigned_abs() as usize);
        max_inicio = max_inicio.max((tiempo_inicio - inicio as f64).abs());
        max_fin = max_fin.max((tiempo_fin - fin as f64).abs());
        max_amp = max_amp.max((nota.amplitude as f64 - amplitud as f64).abs());
        assert!(
            (nota.amplitude as f64 - amplitud as f64).abs() < 1e-4,
            "{nombre}: nota {i}: amplitud distinta (Rust {}, upstream {amplitud})",
            nota.amplitude
        );
    }

    println!(
        "    desajustes: pitch {max_pitch}, inicio {max_inicio:.0} frames, fin {max_fin:.0} frames, amplitud {max_amp:.2e}"
    );
}

/// Convierte un tensor (n_frames, n_freqs) en un vector row-major.
fn a_filas(tensor: &Tensor) -> Vec<f32> {
    tensor
        .flatten_all()
        .expect("aplanar")
        .to_vec1::<f32>()
        .expect("float32")
}

#[test]
fn las_notas_del_vocadito_coinciden_con_el_upstream() {
    if skip_si_falta() {
        return;
    }
    if !vocadito_path().exists() {
        omitir("falta el audio vocadito_10.wav");
        return;
    }
    comprobar("vocadito_10");
}

/// La cancion completa del usuario (177 s, 110 ventanas).
///
/// Va marcada como `#[ignore]` porque en modo debug tarda varios minutos,
/// y porque el audio renderizado **no** se versiona (es material personal
/// del autor). Para ejecutarla:
///
/// ```bash
/// .venv/bin/python tools/weight-export/render_midi.py \
///     "/home/ale/OneDrive/Music/peso del silencio/peso del silencio.mid" \
///     reference/audio_reference/peso_del_silencio.wav
/// .venv/bin/python tools/weight-export/reference_notes.py \
///     reference/audio_reference/peso_del_silencio.wav
/// cargo test --release -p basic-pitch-transcription --test notes_reference -- --ignored
/// ```
#[test]
#[ignore = "comprobacion local: el audio renderizado no se versiona"]
fn las_notas_de_la_cancion_coinciden_con_el_upstream() {
    if skip_si_falta() {
        return;
    }
    comprobar("peso_del_silencio");
}

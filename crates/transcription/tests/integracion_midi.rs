//! Pruebas de integracion del motor completo (Fase 13).
//!
//! Comprueban que el flujo
//!
//! ```text
//! audio -> Transcriber -> TranscriptionResult -> MIDI en memoria
//! ```
//!
//! produce un archivo MIDI que un lector **independiente** (`midly`) puede
//! abrir y que contiene las notas esperadas.

use std::path::{Path, PathBuf};

use basic_pitch_transcription::Transcriber;
use midly::{MidiMessage, Smf, TrackEventKind};

mod common;
use common::{omitir, vocadito_path};

/// Raiz del repositorio.
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Si faltan los assets, no hay nada que probar.
fn skip_si_faltan() -> bool {
    if !repo_root()
        .join("models/basic-pitch/icassp-2022/model.safetensors")
        .exists()
        || !vocadito_path().exists()
    {
        omitir("faltan el modelo o el audio de referencia");
        return true;
    }
    false
}

#[test]
fn la_transcripcion_genera_un_midi_legible_por_un_lector_independiente() {
    if skip_si_faltan() {
        return;
    }
    let raiz = repo_root();
    let transcriptor =
        Transcriber::load(&raiz.join("models/basic-pitch/icassp-2022")).expect("modelo");

    let audio = vocadito_path();
    let resultado = transcriptor.transcribe(&audio).expect("transcripcion");

    // El upstream encuentra 28 notas en este audio.
    assert_eq!(resultado.notes.len(), 28, "numero de notas distinto");
    assert!(
        !resultado.midi_data.is_empty(),
        "el MIDI no deberia estar vacio"
    );

    // Lector independiente: midly, que no es el escritor del proyecto.
    let smf = Smf::parse(&resultado.midi_data).expect("MIDI invalido");
    assert_eq!(smf.tracks.len(), 1, "se esperaba un unico track");

    let note_ons = smf.tracks[0]
        .iter()
        .filter(|e| {
            matches!(
                &e.kind,
                TrackEventKind::Midi {
                    message: MidiMessage::NoteOn { vel, .. },
                    ..
                } if vel.as_int() > 0
            )
        })
        .count();
    assert_eq!(note_ons, 28, "el MIDI no contiene las 28 notas");

    // Tempo y duracion coherentes.
    assert!(resultado.duration_seconds > 9.0 && resultado.duration_seconds < 9.2);
}

#[test]
fn load_with_device_funciona_y_device_es_cpu() {
    if skip_si_faltan() {
        return;
    }
    let raiz = repo_root();
    let transcriptor = Transcriber::load_with_device(
        &raiz.join("models/basic-pitch/icassp-2022"),
        candle_core::Device::Cpu,
    )
    .expect("modelo");
    assert!(transcriptor.device().is_cpu());
}

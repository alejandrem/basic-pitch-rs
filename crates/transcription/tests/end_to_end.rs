//! La cadena completa, de un archivo de audio a las predicciones.
//!
//! Es el test de **armonia** de extremo a extremo: si alguno de los tres
//! bloques (P1 audio, P2 frontend + red, P4 reconstruccion) no encaja con
//! los otros, este test falla.
//!
//! ```text
//! archivo .wav
//!   -> decodificacion -> mono -> remuestreo -> ventanas      [P1]
//!   -> frontend acustico + red neuronal                     [P2]
//!   -> reconstruccion temporal                              [P4]
//!   -> (deteccion de notas)                                 [Fase 9]
//! ```
//!
//! La deteccion de notas todavia no existe, asi que el resultado no tiene
//! notas: lo que se comprueba es que la cadena **se ejecuta entera** y que
//! las dimensiones de la reconstruccion son las esperadas.

use std::path::{Path, PathBuf};

use basic_pitch_transcription::Transcriber;

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
fn la_cadena_completa_se_ejecuta_sobre_un_archivo_real() {
    if skip_si_faltan() {
        return;
    }
    let raiz = repo_root();
    let transcriptor =
        Transcriber::load(&raiz.join("models/basic-pitch/icassp-2022")).expect("modelo");

    let audio = vocadito_path();
    let resultado = transcriptor.transcribe(&audio).expect("transcripcion");

    // 9.1 segundos de audio a 22050 Hz.
    let duracion = resultado.duration_seconds;
    assert!(
        (duracion - 9.1).abs() < 0.05,
        "duracion inesperada: {duracion}"
    );

    // La deteccion de notas (Fase 9) ya funciona: el upstream encuentra 28
    // notas en este audio.
    assert!(
        !resultado.notes.is_empty(),
        "la deteccion de notas no deberia devolver una lista vacia"
    );
    for nota in &resultado.notes {
        assert!(
            (21..=108).contains(&nota.pitch),
            "pitch fuera de rango: {}",
            nota.pitch
        );
        assert!(
            nota.start_time <= nota.end_time,
            "la nota empieza despues de acabar"
        );
        assert!(
            (0.0..=1.0).contains(&nota.amplitude),
            "amplitud fuera de rango: {}",
            nota.amplitude
        );
    }
}

#[test]
fn un_archivo_inexistente_produce_error() {
    let transcriptor =
        Transcriber::load(&repo_root().join("models/basic-pitch/icassp-2022")).expect("modelo");
    let error = transcriptor
        .transcribe(repo_root().join("no-existe.wav"))
        .expect_err("deberia fallar");
    assert!(error.to_string().contains("no se encontro"));
}

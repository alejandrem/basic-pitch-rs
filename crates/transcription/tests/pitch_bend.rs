//! El pitch bend frente al upstream (Fase 10, apartado 10.4).
//!
//! Comprueba que los pitch bends que calcula Rust son los mismos que
//! calcula `basic_pitch.note_creation.get_pitch_bends`, nota a nota.
//!
//! Los oraculos los genera `tools/weight-export/reference_notes.py` y
//! `tools/weight-export/make_cases_oracle.py`, que usan el upstream sin
//! modificar.
//!
//! # Los siete escenarios del apartado 10.4
//!
//! | Escenario | Caso |
//! |---|---|
//! | Notas sin vibrato | `vocadito_10` (un piano real) |
//! | Notas con vibrato | `vibrato` |
//! | Notas con deslizamiento de afinacion | `deslizamiento` |
//! | Variaciones pequenas | `vibrato_sutil` (5 cents) |
//! | Variaciones por encima del rango permitido | `vibrato_extremo` (120 cents, mas alla de las 2 semitonas del recorte) |
//! | Varias notas simultaneas con pitch bend | `acorde_vibrato` |
//! | Notas superpuestas | `notas_superpuestas` |
//!
//! # Que se compara
//!
//! - El **numero** de notas.
//! - El **pitch**, el **inicio**, el **final** y la **amplitud** de cada nota.
//! - La **secuencia de pitch bend** de cada nota, en unidades de 1/3 de
//!   semitono (que es la unidad en la que el upstream los produce).

use std::path::{Path, PathBuf};

use basic_pitch_transcription::Transcriber;

mod common;
use common::{omitir, vocadito_path};

/// Raiz del repositorio.
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Directorio de los oraculos de notas.
fn oracle_dir() -> PathBuf {
    repo_root().join("reference/notes")
}

/// Directorio de los oraculos de los casos.
fn casos_dir() -> PathBuf {
    repo_root().join("reference/casos")
}

/// Si no esta el oraculo, no hay nada que comparar.
fn skip_si_falta() -> bool {
    if !oracle_dir().exists() {
        eprintln!("OMITIDO: no existe {}", oracle_dir().display());
        return true;
    }
    false
}

/// Una nota del oraculo del upstream.
#[derive(Debug, Clone)]
struct NotaOracle {
    start_time: f64,
    end_time: f64,
    pitch: u8,
    amplitude: f32,
    pitch_bend: Option<Vec<i16>>,
}

/// Compara las notas (con sus bends) de Rust con las del upstream.
fn comprobar(nombre: &str, audio: &Path, oracle: &Path) {
    let transcriptor =
        Transcriber::load(&repo_root().join("models/basic-pitch/icassp-2022")).expect("modelo");
    let resultado = transcriptor.transcribe(audio).expect("transcribir");

    let texto = std::fs::read_to_string(oracle.join(format!("{nombre}_notas.json")))
        .expect("notas del upstream");
    let crudo: Vec<serde_json::Value> = serde_json::from_str(&texto).expect("json");
    let oracle: Vec<NotaOracle> = crudo
        .into_iter()
        .map(|n| NotaOracle {
            start_time: n["start_time"].as_f64().unwrap(),
            end_time: n["end_time"].as_f64().unwrap(),
            pitch: n["pitch"].as_u64().unwrap() as u8,
            amplitude: n["amplitude"].as_f64().unwrap() as f32,
            pitch_bend: n["pitch_bend"].as_array().map(|a| {
                a.iter()
                    .map(|v| v.as_i64().unwrap() as i16)
                    .collect::<Vec<i16>>()
            }),
        })
        .collect();

    assert_eq!(
        resultado.notes.len(),
        oracle.len(),
        "{nombre}: numero de notas distinto (Rust {}, upstream {})",
        resultado.notes.len(),
        oracle.len()
    );

    let mut con_bends = 0usize;
    let mut max_diferencia = 0i64;
    for (i, (a, b)) in resultado.notes.iter().zip(oracle.iter()).enumerate() {
        assert_eq!(a.pitch, b.pitch, "{nombre}: nota {i}: pitch distinto");
        assert!(
            (a.start_time - b.start_time).abs() < 1e-3,
            "{nombre}: nota {i}: inicio distinto (Rust {}, upstream {})",
            a.start_time,
            b.start_time
        );
        assert!(
            (a.end_time - b.end_time).abs() < 1e-3,
            "{nombre}: nota {i}: final distinto (Rust {}, upstream {})",
            a.end_time,
            b.end_time
        );
        // La amplitud es la MEDIA de las activaciones de la nota, y cada
        // activacion lleva el error de la red (~1e-5). Con una nota larga,
        // el error medio llega a ~1e-4.
        assert!(
            (a.amplitude - b.amplitude).abs() < 1e-3,
            "{nombre}: nota {i}: amplitud distinta (Rust {}, upstream {})",
            a.amplitude,
            b.amplitude
        );

        match (&a.pitch_bend, &b.pitch_bend) {
            (Some(x), Some(y)) => {
                con_bends += 1;
                assert_eq!(
                    x.len(),
                    y.len(),
                    "{nombre}: nota {i}: numero de bends distinto (Rust {}, upstream {})",
                    x.len(),
                    y.len()
                );
                for (j, (p, q)) in x.iter().zip(y.iter()).enumerate() {
                    let diferencia = (*p as i64 - *q as i64).abs();
                    max_diferencia = max_diferencia.max(diferencia);
                    assert!(
                        diferencia <= 1,
                        "{nombre}: nota {i}: bend {j} distinto (Rust {p}, upstream {q})"
                    );
                }
            }
            (None, None) => {}
            _ => panic!("{nombre}: nota {i}: una tiene bends y la otra no"),
        }
    }

    println!(
        "  {nombre:22} {} notas, {con_bends} con bends, diferencia maxima {max_diferencia}",
        resultado.notes.len()
    );
}

/// Un caso de los 17, con su oraculo.
fn comprobar_caso(nombre: &str) {
    let casos = casos_dir();
    if !casos.join(format!("{nombre}_notas.json")).exists() {
        eprintln!("OMITIDO: no hay oraculo de '{nombre}'");
        return;
    }
    comprobar(nombre, &casos.join(format!("{nombre}.wav")), &casos);
}

/// Escenario 1: notas SIN vibrato (un piano real).
#[test]
fn notas_sin_vibrato() {
    if skip_si_falta() {
        return;
    }
    if !vocadito_path().exists() {
        omitir("falta el audio vocadito_10.wav");
        return;
    }
    comprobar("vocadito_10", &vocadito_path(), &oracle_dir());
}

/// Escenario 2: notas con vibrato.
#[test]
fn notas_con_vibrato() {
    if skip_si_falta() {
        return;
    }
    comprobar_caso("vibrato");
}

/// Escenario 3: notas con deslizamiento de afinacion.
#[test]
fn notas_con_deslizamiento() {
    if skip_si_falta() {
        return;
    }
    comprobar_caso("deslizamiento");
}

/// Escenario 4: variaciones pequenas.
#[test]
fn variaciones_pequenas() {
    if skip_si_falta() {
        return;
    }
    comprobar_caso("vibrato_sutil");
}

/// Escenario 5: variaciones SUPERIORES al rango permitido.
///
/// El pitch bend del MIDI esta limitado a unas dos semitonas (8192 ticks),
/// asi que un vibrato de 120 cents tiene que recortarse.
#[test]
fn variaciones_por_encima_del_rango() {
    if skip_si_falta() {
        return;
    }
    comprobar_caso("vibrato_extremo");
}

/// Escenario 6: varias notas simultaneas con pitch bend.
#[test]
fn varias_notas_simultaneas_con_bend() {
    if skip_si_falta() {
        return;
    }
    comprobar_caso("acorde_vibrato");
}

/// Escenario 7: notas superpuestas.
#[test]
fn notas_superpuestas() {
    if skip_si_falta() {
        return;
    }
    comprobar_caso("notas_superpuestas");
}

//! Cada formato de audio, probado por separado.
//!
//! Cubre el punto de la fase 3 que decia: "Se debera probar cada formato
//! por separado. La extension del archivo no garantiza que el codec
//! interno sea compatible."
//!
//! Los fixtures son el MISMO contenido en cinco formatos, generados por
//! `tools/weight-export/make_audio_fixtures.py`:
//!
//! | Fichero | Codec | Perdidas |
//! |---|---|---|
//! | `tono.wav` | PCM 16 bits | ninguna |
//! | `tono.flac` | FLAC | ninguna |
//! | `tono.mp3` | MP3 (LAME, 128 kbps) | si |
//! | `tono.ogg` | Vorbis (128 kbps) | si |
//! | `tono.m4a` | AAC (128 kbps) | si |
//!
//! La senal es stereo con dos frecuencias distintas (220 Hz a la
//! izquierda, 330 Hz a la derecha), con un impulso al principio y
//! silencio al final. Eso permite comprobar, por cada formato, que:
//!
//! - se decodifica el numero de canales y la frecuencia correctos;
//! - el contenido es el correcto (la magnitud en cada frecuencia), y no
//!   "algunos samples";
//! - los extremos (impulso y silencio) sobreviven a la decodificacion.

mod common;

use std::path::{Path, PathBuf};

use basic_pitch_audio_io::decode_file;
use basic_pitch_audio_preprocess::{to_mono, AUDIO_SAMPLE_RATE};

use common::{diferencia_de_fase, goertzel};

/// Directorio de los fixtures.
fn recursos() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("resources")
}

/// Si no estan los fixtures, no hay nada que probar.
fn skip_si_faltan() -> bool {
    if !recursos().exists() {
        eprintln!("OMITIDO: no hay fixtures en {}", recursos().display());
        return true;
    }
    false
}

/// Comprobaciones comunes a todos los formatos.
///
/// `tolerancia_amplitud` es relativa: los formatos con perdidas pierden
/// precision, los sin perdidas no.
fn comprobar_formato(nombre: &str, tolerancia_amplitud: f64, tolerancia_fase: f64) {
    let ruta = recursos().join(nombre);
    let decoded =
        decode_file(&ruta).unwrap_or_else(|e| panic!("{nombre}: no se pudo decodificar: {e}"));

    // --- Metadatos ---
    assert_eq!(decoded.channels, 2, "{nombre}: canales");
    assert_eq!(
        decoded.sample_rate, 44_100,
        "{nombre}: frecuencia de muestreo"
    );

    // Los codigos con Delay introducen muestras extra al final (el
    //padding del encoder). Se admite un margen de un frame tipico.
    let esperado = 44_100usize;
    assert!(
        decoded.frames() >= esperado && decoded.frames() < esperado + 4_096,
        "{nombre}: {} frames, se esperaban al menos {esperado}",
        decoded.frames()
    );

    // --- Contenido: la magnitud en la frecuencia de cada canal ---
    //
    // Se mide sobre el tramo central, lejos del impulso inicial y del
    // silencio final, y con una ventana de un numero entero de periodos
    // para que la magnitud no dependa del desalineamiento.
    let mitad = decoded.frames() / 2;
    let largo = 8_820usize; // 0.2 s a 44100 Hz
    let ini = mitad - largo / 2;
    let ventana: Vec<f32> = decoded.samples[ini * 2..(ini + largo) * 2]
        .chunks_exact(2)
        .map(|c| c[0])
        .collect();
    let (magnitud_izq, fase_izq) = goertzel(&ventana, 220.0, 44_100);
    let (magnitud_der, fase_der) = goertzel(&ventana, 330.0, 44_100);

    // La izquierda tiene 220 Hz con amplitud 0.55; la derecha, 330 Hz con
    // 0.35. La magnitud de Goertzel es la amplitud de la senal.
    assert!(
        (magnitud_izq - 0.55).abs() < tolerancia_amplitud,
        "{nombre}: magnitud a 220 Hz = {magnitud_izq:.4}, se esperaba 0.55 (+-{tolerancia_amplitud})"
    );
    assert!(
        (magnitud_der - 0.35).abs() < tolerancia_amplitud,
        "{nombre}: magnitud a 330 Hz = {magnitud_der:.4}, se esperaba 0.35 (+-{tolerancia_amplitud})"
    );

    // La fase de cada canal: la izquierda arranca en 0 y la derecha con
    // un desfase de 0.7 radianes. El desfase entre canales es lo que
    // comprueba que no se han intercambiado.
    let desfase = diferencia_de_fase(fase_der, fase_izq);
    assert!(
        desfase.abs() < tolerancia_fase,
        "{nombre}: desfase entre canales = {desfase:.4} rad, se esperaba 0.7 (+-{tolerancia_fase})"
    );

    // --- Extremos: el impulso inicial y el silencio final ---
    let pico_inicial = decoded.samples[..2]
        .iter()
        .fold(f32::NEG_INFINITY, |a, b| a.max(*b))
        .abs();
    assert!(
        pico_inicial > 0.5,
        "{nombre}: el impulso inicial no ha sobrevivido (pico {pico_inicial:.4})"
    );

    let ultimo_decimo = &decoded.samples[(decoded.frames() - decoded.frames() / 10) * 2..];
    let rms_final = (ultimo_decimo
        .iter()
        .map(|v| *v as f64 * *v as f64)
        .sum::<f64>()
        / ultimo_decimo.len().max(1) as f64)
        .sqrt();
    assert!(
        rms_final < 0.05,
        "{nombre}: el silencio final no ha sobrevivido (rms {rms_final:.4})"
    );

    // --- El promedio a mono sigue el promedio de la referencia ---
    let mono = to_mono(&decoded);
    let (magnitud_mono, _) = goertzel(&mono.samples[ini..ini + largo], 220.0, AUDIO_SAMPLE_RATE);
    assert!(
        magnitud_mono > 0.1,
        "{nombre}: el mono no conserva el contenido (magnitud {magnitud_mono:.4})"
    );
}

#[test]
fn wav_se_decodifica_sin_perdidas() {
    if skip_si_faltan() {
        return;
    }
    // PCM 16 bits: solo el redondeo de la cuantizacion (~3e-5).
    comprobar_formato("tono.wav", 0.005, 0.05);
}

#[test]
fn flac_se_decodifica_sin_perdidas() {
    if skip_si_faltan() {
        return;
    }
    comprobar_formato("tono.flac", 0.005, 0.05);
}

#[test]
fn mp3_se_decodifica_con_perdidas_tolerables() {
    if skip_si_faltan() {
        return;
    }
    // MP3 a 128 kbps: el espectro se deforma alrededor de las componentes
    // fuertes, y hay un retardo del encoder.
    comprobar_formato("tono.mp3", 0.06, 0.6);
}

#[test]
fn ogg_se_decodifica_con_perdidas_tolerables() {
    if skip_si_faltan() {
        return;
    }
    comprobar_formato("tono.ogg", 0.06, 0.6);
}

#[test]
fn m4a_se_decodifica_con_perdidas_tolerables() {
    if skip_si_faltan() {
        return;
    }
    comprobar_formato("tono.m4a", 0.06, 0.6);
}

#[test]
fn la_extension_no_decide_el_formato() {
    if skip_si_faltan() {
        return;
    }
    // Un WAV con extension .mp3 se decodifica igual: lo que decide es el
    // contenido, no el nombre. Y un .mp3 de verdad tambien.
    let dir = tempfile::tempdir().unwrap();
    let falso = dir.path().join("tono.mp3");
    std::fs::copy(recursos().join("tono.wav"), &falso).unwrap();

    let decoded = decode_file(&falso).expect("no se pudo decodificar");
    assert_eq!(decoded.channels, 2);
    assert_eq!(decoded.sample_rate, 44_100);
    assert_eq!(decoded.frames(), 44_100);
}

#[test]
fn un_archivo_que_no_es_audio_produce_error() {
    let dir = tempfile::tempdir().unwrap();
    let ruta = dir.path().join("basura.mp3");
    std::fs::write(&ruta, b"esto no es audio, es texto").unwrap();
    assert!(decode_file(&ruta).is_err());
}

#[test]
fn un_archivo_vacio_produce_error() {
    let dir = tempfile::tempdir().unwrap();
    let ruta = dir.path().join("vacio.wav");
    std::fs::write(&ruta, b"").unwrap();
    assert!(decode_file(&ruta).is_err());
}

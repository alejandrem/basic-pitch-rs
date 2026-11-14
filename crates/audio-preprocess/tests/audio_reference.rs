//! La decodificacion, la conversion a mono y el remuestreo, comparados con
//! la referencia.
//!
//! El oraculo son los `.npy` de `reference/audio_reference/`: la senal mono a
//! 22.050 Hz tal y como la produce la referencia (`librosa.load(sr=22050,
//! mono=True)`, es decir `soundfile` para decodificar y `soxr_hq` para
//! remuestrar).
//!
//! # Por que este test existe
//!
//! El audio de la referencia del frontend (`input_audio.npy`) ya viene
//! decodificado, en mono y remuestreado: no permite validar esa parte. Con
//! estos archivos de audio reales si.
//!
//! # Que mide, exactamente
//!
//! La diferencia entre **Symphonia** (el decodificador de Rust) y
//! **soundfile/libsndfile** (el de la referencia). El remuestreador es el
//! mismo (`soxr` en modo HQ), y el promedio de canales tambien, asi que lo
//! que queda es la diferencia de decodificacion y la del orden de las
//! operaciones internas de cada uno.
//!
//! Los tests se omiten con un aviso si no estan los recursos.

use std::path::{Path, PathBuf};

use basic_pitch_audio_io::decode_file;
use basic_pitch_audio_preprocess::{resample, to_mono, AUDIO_SAMPLE_RATE};
use basic_pitch_reference_npy::read_f32;

/// Raiz del repositorio.
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Directorio de los recursos de audio del upstream.
fn resources_dir() -> PathBuf {
    repo_root().join("basic-pitch/tests/resources")
}

/// Directorio del oraculo de audio.
fn oracle_dir() -> PathBuf {
    repo_root().join("reference/audio_reference")
}

/// Si falta el audio del upstream o el oraculo, no hay nada que comparar.
fn skip_si_falta_algo() -> bool {
    if !resources_dir().exists() || !oracle_dir().exists() {
        eprintln!(
            "OMITIDO: falta {} o {}",
            resources_dir().display(),
            oracle_dir().display()
        );
        return true;
    }
    false
}

/// Compara dos senales y devuelve (mae, max_abs, rms_de_la_referencia).
fn comparar(rust: &[f32], referencia: &[f32]) -> (f64, f64, f64) {
    assert_eq!(
        rust.len(),
        referencia.len(),
        "la senal de Rust tiene {} muestras y la referencia {}",
        rust.len(),
        referencia.len()
    );
    let mut suma = 0.0f64;
    let mut maximo = 0.0f64;
    let mut rms = 0.0f64;
    for (a, b) in rust.iter().zip(referencia.iter()) {
        let d = (*a as f64 - *b as f64).abs();
        suma += d;
        if d > maximo {
            maximo = d;
        }
        rms += (*b as f64) * (*b as f64);
    }
    (
        suma / rust.len().max(1) as f64,
        maximo,
        (rms / referencia.len().max(1) as f64).sqrt(),
    )
}

/// Compara el pipeline completo de Rust con el oraculo de la referencia.
fn comprobar(nombre: &str) {
    let audio = resources_dir().join(format!("{nombre}.wav"));
    let oracle = oracle_dir().join(format!("{nombre}_mono22050.npy"));

    // 1. Decodificacion.
    let decoded = decode_file(&audio).expect("no se pudo decodificar");

    // 2. Conversion a mono por promedio.
    let mono = to_mono(&decoded);

    // 3. Remuestreo a 22.050 Hz con soxr en modo HQ.
    let remuestreado = resample(&mono.samples, mono.sample_rate, AUDIO_SAMPLE_RATE)
        .expect("no se pudo remuestrear");

    // 4. Comparacion con la referencia.
    let referencia = read_f32(&oracle).expect("no se pudo leer el oraculo").data;
    let (mae, maximo, rms) = comparar(&remuestreado, &referencia);

    println!(
        "{nombre}: {} muestras, mae={mae:.3e}, max={maximo:.3e}, rms de la referencia={rms:.6e}",
        remuestreado.len()
    );

    // Tolerancias justificadas:
    //
    // - El MAE tiene que estar tres ordenes de magnitud por debajo del RMS
    //   de la senal: una diferencia de 1e-4 en una senal de 0.15 de RMS es
    //   de -70 dB, inaudible y por debajo del ruido de cuantizacion del
    //   propio WAV de 16 bits (que es de ~3e-5).
    // - El maximo absoluto tiene que ser pequeno en terminos absolutos: por
    //   debajo de 1e-3, que es el orden del error de cuantizacion del
    //   archivo original multiplicado por el factor de remuestreo.
    assert!(
        mae < rms * 1e-3,
        "{nombre}: el error medio {mae:.3e} es demasiado grande para un RMS de {rms:.3e}"
    );
    assert!(
        maximo < 1e-3,
        "{nombre}: la diferencia maxima {maximo:.3e} es demasiado grande"
    );
}

#[test]
fn el_audio_estereo_se_decodifica_y_remuestrea_como_la_referencia() {
    if skip_si_falta_algo() {
        return;
    }
    comprobar("vocadito_10");
}

#[test]
fn el_segundo_audio_tambien_coincide() {
    if skip_si_falta_algo() {
        return;
    }
    comprobar("vocadito_14");
}

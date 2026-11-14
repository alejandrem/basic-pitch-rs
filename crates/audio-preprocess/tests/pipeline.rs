//! La cadena de procesamiento de audio de la referencia, de punta a punta.
//!
//! Cubre el criterio de aceptacion de la Fase 3: las muestras, las
//! dimensiones, el relleno y las posiciones temporales deben coincidir con las
//! reglas de `basic_pitch/inference.py`.
//!
//! La cadena es:
//!
//! ```text
//! archivo -> decodificacion -> mono (PROMEDIO) -> remuestreo a 22050 -> ventanas
//! ```
//!
//! El orden de los dos ultimos pasos es obligatorio: mono **antes** de
//! remuestrar, como hace `librosa.load(path, sr=22050, mono=True)`.

use std::path::Path;

use basic_pitch_audio_io::decode_file;
use basic_pitch_audio_preprocess::{
    resample, to_mono, AudioWindows, AUDIO_N_SAMPLES, AUDIO_SAMPLE_RATE, HOP_SIZE, INITIAL_PADDING,
};

/// Escribe un WAV PCM de 16 bits.
///
/// Es un escritor minimo, solo para tests: el objetivo es tener un archivo
/// real que decodificar, no producir audio de calidad.
fn escribir_wav(path: &Path, muestras: &[f32], canales: u16, sr: u32) {
    let bytes = (muestras.len() * 2) as u32;
    let mut out = Vec::with_capacity(44 + bytes as usize);

    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + bytes).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM sin compresion
    out.extend_from_slice(&canales.to_le_bytes());
    out.extend_from_slice(&sr.to_le_bytes());
    out.extend_from_slice(&(sr * canales as u32 * 2).to_le_bytes()); // bytes por segundo
    out.extend_from_slice(&(canales * 2).to_le_bytes()); // alineacion de bloque
    out.extend_from_slice(&16u16.to_le_bytes()); // bits por muestra
    out.extend_from_slice(b"data");
    out.extend_from_slice(&bytes.to_le_bytes());
    for muestra in muestras {
        let valor = (muestra.clamp(-1.0, 1.0) * 32_767.0) as i16;
        out.extend_from_slice(&valor.to_le_bytes());
    }

    std::fs::write(path, out).expect("no se pudo escribir el WAV");
}

/// Senal estereo de dos canales: 100 Hz en el izquierdo, la mitad en el
/// derecho. El promedio de canales da 0.75 * seno.
fn stereo_100hz(n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n * 2);
    for i in 0..n {
        let t = i as f64 / 44_100.0;
        let seno = (2.0 * std::f64::consts::PI * 100.0 * t).sin() as f32;
        out.push(seno);
        out.push(seno * 0.5);
    }
    out
}

#[test]
fn la_cadena_completa_produce_las_ventanas_del_modelo() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tono.wav");

    let n = 44_100 * 3; // 3 segundos
    escribir_wav(&path, &stereo_100hz(n), 2, 44_100);

    // 1. Decodificacion.
    let decoded = decode_file(&path).expect("no se pudo decodificar");
    assert_eq!(decoded.channels, 2);
    assert_eq!(decoded.sample_rate, 44_100);
    assert_eq!(decoded.frames(), n);

    // 2. Conversion a mono por PROMEDIO (no suma, no un canal).
    let mono = to_mono(&decoded);
    assert_eq!(mono.channels, 1);
    assert_eq!(mono.frames(), n);
    let t = 0.0f64;
    let esperado = 0.75 * (2.0 * std::f64::consts::PI * 100.0 * t).sin() as f32;
    assert!(
        (mono.samples[0] - esperado).abs() < 1e-4,
        "el promedio de canales no es el esperado: {} != {esperado}",
        mono.samples[0]
    );

    // 3. Remuestreo a 22.050 Hz.
    let remuestreado = resample(&mono.samples, mono.sample_rate, AUDIO_SAMPLE_RATE)
        .expect("no se pudo remuestrear");
    assert_eq!(remuestreado.len(), n / 2);
    // La amplitud se conserva: el remuestreador no normaliza.
    let pico = remuestreado
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    assert!(pico > 0.7 && pico < 0.8, "pico inesperado: {pico}");

    // 4. Ventaneo.
    let windows = AudioWindows::new(remuestreado.clone(), AUDIO_SAMPLE_RATE);
    let esperado_ventanas = (remuestreado.len() + INITIAL_PADDING).div_ceil(HOP_SIZE);
    assert_eq!(windows.len(), esperado_ventanas);

    let todas: Vec<_> = windows.iter().collect();
    for ventana in &todas {
        assert_eq!(ventana.samples.len(), AUDIO_N_SAMPLES);
    }

    // La primera ventana empieza ANTES del audio original (relleno inicial).
    assert_eq!(todas[0].start_sample, -(INITIAL_PADDING as i64));
    assert!(todas[0].samples[..INITIAL_PADDING]
        .iter()
        .all(|v| v.abs() < f32::EPSILON));

    // Y su contenido, a partir del relleno, es el audio remuestreado.
    for (i, esperado) in remuestreado.iter().take(1_000).enumerate() {
        assert!(
            (todas[0].samples[INITIAL_PADDING + i] - esperado).abs() < 1e-6,
            "la ventana no contiene el audio remuestreado en la posicion {i}"
        );
    }

    // Las ventanas consecutivas se solapan: el desplazamiento es HOP_SIZE.
    assert_eq!(
        todas[1].start_sample - todas[0].start_sample,
        HOP_SIZE as i64
    );
    // Y el solapamiento en muestras es AUDIO_N_SAMPLES - HOP_SIZE.
    assert_eq!(AUDIO_N_SAMPLES - HOP_SIZE, 7_680);
}

#[test]
fn un_audio_mono_no_cambia_al_convertirlo() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mono.wav");

    let n = 22_050;
    let muestras: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f64 / 22_050.0;
            (2.0 * std::f64::consts::PI * 220.0 * t).sin() as f32
        })
        .collect();
    escribir_wav(&path, &muestras, 1, 22_050);

    let decoded = decode_file(&path).expect("no se pudo decodificar");
    assert_eq!(decoded.channels, 1);
    let mono = to_mono(&decoded);
    assert_eq!(mono.samples.len(), decoded.samples.len());

    // Sin remuestreo (ya esta a 22.050 Hz), la senal no se toca.
    let igual = resample(&mono.samples, mono.sample_rate, AUDIO_SAMPLE_RATE).unwrap();
    assert_eq!(igual.len(), n);
    assert_eq!(igual, mono.samples);
}

#[test]
fn un_archivo_que_no_existe_produce_error() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("no-existe.wav");
    assert!(decode_file(&path).is_err());
}

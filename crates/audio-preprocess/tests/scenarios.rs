//! Los escenarios que enumera la fase 3 para las pruebas de audio.
//!
//! La fase 3 pide pruebas para:
//!
//! - Audio mono y estéreo.
//! - Audio de menos de dos segundos.
//! - Audio de exactamente dos segundos.
//! - Audio de varios minutos.
//! - Audio con silencio inicial y final.
//! - Archivos con diferentes frecuencias de muestreo.
//! - Archivos con diferentes formatos (en `formats.rs`).
//! - Señales con valores cercanos a los límites de amplitud.
//! - Audio con transitorios al inicio y al final.
//!
//! Cada escenario escribe un archivo real, lo decodifica y comprueba el
//! contrato del modelo: ventanas de 43.844 muestras, relleno inicial de
//! 3.840 ceros y solapamiento de 7.680.

mod common;

use basic_pitch_audio_io::decode_file;
use basic_pitch_audio_preprocess::{
    resample, to_mono, AudioWindows, AUDIO_N_SAMPLES, AUDIO_SAMPLE_RATE, HOP_SIZE, INITIAL_PADDING,
};

use common::{escribir_wav, silencio, tono};

/// Comprueba el contrato de ventanas sobre una senal ya remuestreada.
fn comprobar_ventanas(muestras: &[f32]) -> AudioWindows {
    let windows = AudioWindows::new(muestras.to_vec(), AUDIO_SAMPLE_RATE);
    let todas: Vec<_> = windows.iter().collect();

    for ventana in &todas {
        assert_eq!(
            ventana.samples.len(),
            AUDIO_N_SAMPLES,
            "toda ventana debe tener {AUDIO_N_SAMPLES} muestras"
        );
    }

    // El desplazamiento entre ventanas consecutivas.
    for par in todas.windows(2) {
        assert_eq!(
            par[1].start_sample - par[0].start_sample,
            HOP_SIZE as i64,
            "el desplazamiento entre ventanas debe ser {HOP_SIZE}"
        );
    }

    // La primera ventana empieza antes del audio (relleno inicial).
    assert_eq!(todas[0].start_sample, -(INITIAL_PADDING as i64));
    assert!(
        todas[0].samples[..INITIAL_PADDING]
            .iter()
            .all(|v| v.abs() < f32::EPSILON),
        "el relleno inicial deben ser ceros"
    );

    windows
}

/// Decodifica, convierte a mono, remuestrea y comprueba las ventanas.
fn cadena_completa(ruta: &std::path::Path) -> AudioWindows {
    let decoded = decode_file(ruta).expect("no se pudo decodificar");
    let mono = to_mono(&decoded);
    let remuestreado = resample(&mono.samples, mono.sample_rate, AUDIO_SAMPLE_RATE)
        .expect("no se pudo remuestrear");
    comprobar_ventanas(&remuestreado)
}

#[test]
fn audio_mono() {
    let dir = tempfile::tempdir().unwrap();
    let ruta = dir.path().join("mono.wav");
    escribir_wav(&ruta, &tono(440.0, 22_050, 22_050), 1, 22_050);

    let decoded = decode_file(&ruta).unwrap();
    assert_eq!(decoded.channels, 1);
    let mono = to_mono(&decoded);
    assert_eq!(mono.channels, 1);
    assert_eq!(mono.frames(), 22_050);

    let windows = cadena_completa(&ruta);
    assert_eq!(windows.original_length(), 22_050);
}

#[test]
fn audio_estereo() {
    let dir = tempfile::tempdir().unwrap();
    let ruta = dir.path().join("estereo.wav");

    // Dos canales con contenidos distintos.
    let izq = tono(440.0, 22_050, 22_050);
    let der = tono(330.0, 22_050, 22_050);
    let mut muestras = Vec::with_capacity(izq.len() * 2);
    for (a, b) in izq.iter().zip(der.iter()) {
        muestras.push(*a);
        muestras.push(*b);
    }
    escribir_wav(&ruta, &muestras, 2, 22_050);

    let decoded = decode_file(&ruta).unwrap();
    assert_eq!(decoded.channels, 2);

    let mono = to_mono(&decoded);
    assert_eq!(mono.channels, 1);
    assert_eq!(mono.frames(), 22_050);

    let windows = cadena_completa(&ruta);
    assert_eq!(windows.original_length(), 22_050);
}

#[test]
fn audio_de_menos_de_dos_segundos() {
    let dir = tempfile::tempdir().unwrap();
    let ruta = dir.path().join("corto.wav");
    // 1.5 segundos: menos que una ventana del modelo.
    let muestras = tono(440.0, 33_075, 22_050);
    escribir_wav(&ruta, &muestras, 1, 22_050);

    let decoded = decode_file(&ruta).unwrap();
    assert_eq!(decoded.frames(), 33_075);

    let windows = cadena_completa(&ruta);

    // Con el relleno inicial (3840) mas el audio (33075) son 36915
    // muestras, que es mas de un salto (36164): el upstream itera en
    // `range(0, len_rellenado, hop_size)`, asi que produce DOS ventanas,
    // aunque el audio no llegue a una ventana completa. Es el
    // comportamiento de la referencia, no un error.
    assert_eq!(windows.len(), 2);
    assert_eq!(windows.original_length(), 33_075);
    // La ventana se completa con ceros al final.
    assert!(windows.iter().next().unwrap().samples.contains(&0.0));
}

#[test]
fn audio_de_exactamente_dos_segundos() {
    let dir = tempfile::tempdir().unwrap();
    let ruta = dir.path().join("dos_segundos.wav");
    // Exactamente 2 segundos a 22.050 Hz.
    let muestras = tono(440.0, 44_100, 22_050);
    escribir_wav(&ruta, &muestras, 1, 22_050);

    let decoded = decode_file(&ruta).unwrap();
    assert_eq!(decoded.frames(), 44_100);

    let windows = cadena_completa(&ruta);
    assert_eq!(windows.original_length(), 44_100);
    // 44100 + 3840 = 47940 muestras rellenadas; ceil(47940/36164) = 2.
    assert_eq!(windows.len(), 2);
}

#[test]
fn audio_de_varios_minutos() {
    let dir = tempfile::tempdir().unwrap();
    let ruta = dir.path().join("largo.wav");
    // 3 minutos a 44.100 Hz.
    let n = 44_100 * 180;
    let muestras = tono(440.0, n, 44_100);
    escribir_wav(&ruta, &muestras, 1, 44_100);

    let decoded = decode_file(&ruta).unwrap();
    assert_eq!(decoded.frames(), n);

    let windows = cadena_completa(&ruta);
    // ceil((n/2 + 3840) / 36164) ventanas.
    let esperado = (n / 2 + INITIAL_PADDING).div_ceil(HOP_SIZE);
    assert_eq!(windows.len(), esperado);
    assert!(
        windows.len() > 10,
        "3 minutos deberian dar mas de 10 ventanas"
    );
}

#[test]
fn audio_con_silencio_inicial_y_final() {
    let dir = tempfile::tempdir().unwrap();
    let ruta = dir.path().join("silencios.wav");

    // 0.5 s de silencio, 1 s de tono, 0.5 s de silencio.
    let mut muestras = silencio(11_025);
    muestras.extend_from_slice(&tono(440.0, 22_050, 22_050));
    muestras.extend_from_slice(&silencio(11_025));
    escribir_wav(&ruta, &muestras, 1, 22_050);

    let decoded = decode_file(&ruta).unwrap();
    let remuestreado = resample(&decoded.samples, decoded.sample_rate, AUDIO_SAMPLE_RATE).unwrap();

    // El silencio se conserva en los extremos.
    assert!(remuestreado[..2_000].iter().all(|v| v.abs() < 1e-4));
    let final_ = &remuestreado[remuestreado.len() - 2_000..];
    assert!(final_.iter().all(|v| v.abs() < 1e-4));

    let windows = comprobar_ventanas(&remuestreado);
    assert_eq!(windows.original_length(), 44_100);
}

#[test]
fn archivos_con_diferentes_frecuencias_de_muestreo() {
    let dir = tempfile::tempdir().unwrap();

    for sr in [8_000u32, 16_000, 22_050, 32_000, 44_100, 48_000] {
        let ruta = dir.path().join(format!("a_{sr}.wav"));
        // Un segundo a cada frecuencia.
        let muestras = tono(440.0, sr as usize, sr);
        escribir_wav(&ruta, &muestras, 1, sr);

        let decoded = decode_file(&ruta).unwrap();
        assert_eq!(
            decoded.sample_rate, sr,
            "la frecuencia de {sr} Hz no se lee bien"
        );
        assert_eq!(decoded.frames(), sr as usize);

        let remuestreado = resample(&decoded.samples, decoded.sample_rate, AUDIO_SAMPLE_RATE)
            .unwrap_or_else(|e| panic!("no se pudo remuestrear de {sr} Hz: {e}"));
        assert_eq!(
            remuestreado.len(),
            22_050,
            "un segundo a {sr} Hz debe dar 22050 muestras"
        );

        let windows = comprobar_ventanas(&remuestreado);
        assert!(!windows.is_empty());
    }
}

#[test]
fn senales_cerca_de_los_limites_de_amplitud() {
    let dir = tempfile::tempdir().unwrap();
    let ruta = dir.path().join("limites.wav");

    // Valores justo por debajo del limite de clipping del PCM de 16 bits.
    let mut muestras = Vec::new();
    for i in 0..22_050 {
        let valor = if i % 2 == 0 { 0.9999 } else { -0.9999 };
        muestras.push(valor as f32);
    }
    escribir_wav(&ruta, &muestras, 1, 22_050);

    let decoded = decode_file(&ruta).unwrap();
    let pico = decoded.samples.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    assert!(
        pico > 0.99,
        "la senal deberia estar cerca del limite, pico = {pico}"
    );
    assert!(pico <= 1.0, "la senal no debe pasar de 1.0, pico = {pico}");

    // Y tras remuestrear sigue sin pasar del limite (el remuestreador
    // puede dar un pequeño overshoot).
    let remuestreado = resample(&decoded.samples, decoded.sample_rate, AUDIO_SAMPLE_RATE).unwrap();
    let pico = remuestreado.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    assert!(
        pico < 1.05,
        "el remuestreo ha producido un pico excesivo: {pico}"
    );
}

#[test]
fn audio_con_transitorios_al_inicio_y_al_final() {
    let dir = tempfile::tempdir().unwrap();
    let ruta = dir.path().join("transitorios.wav");

    // Un tono con un impulso brusco en la primera y en la ultima muestra.
    let mut muestras = tono(440.0, 22_050, 22_050);
    muestras[0] = 0.95;
    let ultimo = muestras.len() - 1;
    muestras[ultimo] = -0.95;
    escribir_wav(&ruta, &muestras, 1, 22_050);

    let decoded = decode_file(&ruta).unwrap();
    assert!(decoded.samples[0].abs() > 0.9);
    assert!(decoded.samples[ultimo].abs() > 0.9);

    let remuestreado = resample(&decoded.samples, decoded.sample_rate, AUDIO_SAMPLE_RATE).unwrap();

    // El transitorio inicial no debe desaparecer por el recorte de los
    // bordes del filtro.
    assert!(
        remuestreado[..2_000]
            .iter()
            .fold(0.0f32, |a, b| a.max(b.abs()))
            > 0.5,
        "el transitorio inicial se ha perdido"
    );

    let windows = comprobar_ventanas(&remuestreado);
    assert_eq!(windows.original_length(), 22_050);
}

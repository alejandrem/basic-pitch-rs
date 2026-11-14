//! Utilidades compartidas por los tests de audio.
//!
//! Este modulo no es un test en si mismo: Cargo no compila los
//! subdirectorios de `tests/` como binarios de prueba.

#![allow(dead_code)]

use std::path::Path;

/// Escribe un WAV PCM de 16 bits.
///
/// Es un escritor minimo, solo para tests: el objetivo es tener archivos
/// reales que decodificar, no producir audio de calidad.
pub fn escribir_wav(path: &Path, muestras: &[f32], canales: u16, sr: u32) {
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

    // Los datos PCM se generan de una vez: escribir muestra a muestra
    // hacia el vector de salida es lentisimo en modo debug, y los tests
    // generan archivos de varios minutos.
    out.extend(
        muestras
            .iter()
            .flat_map(|muestra| {
                let valor = (muestra.clamp(-1.0, 1.0) * 32_767.0) as i16;
                valor.to_le_bytes()
            })
            .collect::<Vec<u8>>(),
    );

    std::fs::write(path, out).expect("no se pudo escribir el WAV");
}

/// Senal mono de un tono, con envolvente suave al principio.
pub fn tono(freq: f64, n: usize, sr: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let t = i as f64 / sr as f64;
            let envolvente = (t / 0.02).min(1.0);
            (2.0 * std::f64::consts::PI * freq * t).sin() as f32 * envolvente as f32
        })
        .collect()
}

/// Senal de un tono, **sin envolvente**.
///
/// Para medir fase y amplitud con precision hace falta una senal que sea
/// exactamente un seno en toda su duracion: una envolvente introduce un
/// transitorio que falsea la estimacion de la fase.
pub fn seno_puro(freq: f64, n: usize, sr: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let t = i as f64 / sr as f64;
            (2.0 * std::f64::consts::PI * freq * t).sin() as f32
        })
        .collect()
}

/// Senal de silencio.
pub fn silencio(n: usize) -> Vec<f32> {
    vec![0.0; n]
}

/// Magnitud y fase de una frecuencia concreta, con el algoritmo de Goertzel.
///
/// Se usa para comprobar el contenido espectral de una senal decodificada
/// sin depender de la alineacion temporal (los codigos con perdidas
/// introducen un retardo que desplaza la senal).
///
/// La medicion se hace con ventana de Hann y se normaliza por la ganancia
/// coherente de esa ventana. Sin ella, la magnitud solo seria exacta si la
/// ventana contuviese un numero entero de periodos de la frecuencia
/// medida, cosa que no ocurre en general: `22050 / 440 = 50.11` periodos
/// por ventana de 1000 muestras, y el "sangrado" espectral de la ventana
/// rectangular reduce la magnitud hasta un factor 2.
///
/// La ventana de Hann es simetrica, asi que **no desplaza la fase**: la
/// fase que se devuelve es la de la senal.
pub fn goertzel(x: &[f32], freq: f64, sr: u32) -> (f64, f64) {
    let n = x.len() as f64;
    if n == 0.0 {
        return (0.0, 0.0);
    }
    let k = 2.0 * std::f64::consts::PI * freq / sr as f64;
    let coefficiente = 2.0 * k.cos();
    let mut s1 = 0.0f64;
    let mut s2 = 0.0f64;
    let mut ganancia = 0.0f64;
    for (i, valor) in x.iter().enumerate() {
        // Ventana de Hann.
        let w = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (n - 1.0)).cos();
        ganancia += w;
        let s0 = *valor as f64 * w + coefficiente * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    let real = s1 - s2 * k.cos();
    let imaginaria = s2 * k.sin();
    // El factor 2 compensa que una senoide de amplitud A tiene una
    // transformada de magnitud N*A/2 en su bin.
    let magnitud = 2.0 * (real * real + imaginaria * imaginaria).sqrt() / ganancia;
    let fase = imaginaria.atan2(real);
    (magnitud, fase)
}

/// Posiciones de los cruces de cero ascendentes, con precision
/// sub-muestra.
///
/// Para medir el retardo de grupo de forma robusta: la fase estimada por
/// Goertzel tiene un sesgo que depende de la posicion de la ventana (si
/// la ventana no contiene un numero entero de periodos), y ese sesgo se
/// confunde con una deriva de fase. Los cruces de cero no tienen ese
/// problema: si el retardo es constante, el espaciado entre ellos es
/// exactamente el semiperiodo.
pub fn cruces_de_cero(x: &[f32]) -> Vec<f64> {
    let mut out = Vec::new();
    for i in 1..x.len() {
        if x[i - 1] <= 0.0 && x[i] > 0.0 {
            let a = x[i - 1] as f64;
            let b = x[i] as f64;
            out.push((i - 1) as f64 + (-a) / (b - a));
        }
    }
    out
}

/// Diferencia angular entre dos fases, en radianes, normalizada a
/// `(-pi, pi]`.
pub fn diferencia_de_fase(a: f64, b: f64) -> f64 {
    let mut d = a - b;
    while d > std::f64::consts::PI {
        d -= 2.0 * std::f64::consts::PI;
    }
    while d <= -std::f64::consts::PI {
        d += 2.0 * std::f64::consts::PI;
    }
    d
}

//! El remuestreador, en los puntos que la fase 3 senala como delicados.
//!
//! La fase 3 pide atencion explicita a:
//!
//! - **Calidad del filtro**: la salida debe ser fiel a la entrada, sin
//!   imagen especular ni ruido.
//! - **Retardo de grupo**: el retardo debe ser *constante*, no variable.
//!   Un retardo que cambia con el tiempo deforma la señal.
//! - **Tratamiento de los extremos**: los bordes no deben producir
//!   transitorios espurios.
//! - **Numero final de muestras**: la longitud sigue la regla de
//!   `librosa.resample` con `fix=True`.
//! - **Diferencias de amplitud**: el remuestreador no normaliza.
//! - **Frecuencias de muestreo poco habituales**: ratios que no son
//!   potencias de dos, y razones muy distintas.
//!
//! El motor es `soxr` en modo HQ, el mismo que el `soxr_hq` de librosa
//! 0.10.2, que es el de la referencia.

mod common;

use basic_pitch_audio_preprocess::resample;

use common::{cruces_de_cero, goertzel, seno_puro, tono};

/// Longitud esperada segun la regla de `librosa.resample` con `fix=True`.
fn longitud_esperada(entrada: usize, desde: u32, hasta: u32) -> usize {
    (entrada as u64 * hasta as u64).div_ceil(desde as u64) as usize
}

#[test]
fn la_calidad_del_filtro_conserva_la_frecuencia_sin_ruido() {
    // 440 Hz a 44100 Hz, remuestreada a 22050 Hz.
    let n = 44_100;
    let entrada = tono(440.0, n, 44_100);
    let salida = resample(&entrada, 44_100, 22_050).unwrap();
    assert_eq!(salida.len(), longitud_esperada(n, 44_100, 22_050));

    // La magnitud en 440 Hz debe ser la misma (el filtro no cambia la
    // amplitud de la frecuencia fundamental).
    let ventana = &salida[4_000..8_000];
    let (magnitud, _) = goertzel(ventana, 440.0, 22_050);
    assert!(
        (magnitud - 1.0).abs() < 0.01,
        "la amplitud fundamental cambio: {magnitud:.4}"
    );

    // Y no debe aparecer energia espuria en una frecuencia ausente. Se
    // mide a 1000 Hz, que no esta en la señal: el ruido es minusculo.
    let (ruido, _) = goertzel(ventana, 1_000.0, 22_050);
    assert!(
        ruido < 0.01,
        "hay ruido espurio a 1000 Hz: {ruido:.4} (la relacion senal-ruido debe ser alta)"
    );
}

#[test]
fn el_retardo_de_grupo_es_constante() {
    // El retardo de grupo del filtro tiene que ser el mismo al principio
    // y al final de la senal. Si variase, la fase rotaria a lo largo del
    // tiempo y la senal se deformaria.
    //
    // Se mide con los cruces de cero ascendentes, no con la fase de
    // Goertzel: la fase de Goertzel tiene un sesgo que depende de la
    // posicion de la ventana (cuando la ventana no contiene un numero
    // entero de periodos de la frecuencia medida), y ese sesgo es
    // indistinguible de una deriva de fase. Con los cruces de cero, si
    // el retardo es constante, el espaciado es exactamente el
    // semiperiodo.
    let n = 44_100;
    let entrada = seno_puro(440.0, n, 44_100);
    let salida = resample(&entrada, 44_100, 22_050).unwrap();

    let cruces = cruces_de_cero(&salida);
    assert!(
        cruces.len() > 50,
        "solo se han encontrado {} cruces de cero",
        cruces.len()
    );

    // Un seno cruza el cero en sentido ascendente UNA VEZ POR PERIODO.
    let periodo = 22_050.0 / 440.0;
    let mut minimo = f64::INFINITY;
    let mut maximo = f64::NEG_INFINITY;
    for par in cruces.windows(2) {
        let espaciado = par[1] - par[0];
        minimo = minimo.min(espaciado);
        maximo = maximo.max(espaciado);
    }

    // La variacion del espaciado es la medida del retardo variable. Con
    // un filtro de fase lineal, la unica fuente de variacion es el
    // redondeo a la muestra.
    let variacion = maximo - minimo;
    assert!(
        variacion < 0.05,
        "el espaciado entre cruces varia {variacion:.4} muestras ({minimo:.4}..{maximo:.4}): el retardo de grupo no es constante"
    );
    assert!(
        (maximo - periodo).abs() < 0.02,
        "el periodo medido {maximo:.4} no es el esperado {periodo:.4}"
    );
}

#[test]
fn los_extremos_no_producen_transitorios_espurios() {
    // Un impulso en el centro de la señal: la salida debe tener un pico
    // en el sitio correspondiente, y los bordes no deben Burst fuera de
    // rango.
    let n = 44_100;
    let mut entrada = vec![0.0f32; n];
    entrada[n / 2] = 1.0;
    let salida = resample(&entrada, 44_100, 22_050).unwrap();

    // Los primeros y ultimos 64 samples no deben superar la amplitud de
    // la entrada: un filtro que se comporta mal en los bordes produciria picos.
    let borde_inicial = salida[..64].iter().fold(0.0f32, |a, b| a.max(b.abs()));
    let borde_final = salida[salida.len() - 64..]
        .iter()
        .fold(0.0f32, |a, b| a.max(b.abs()));
    assert!(
        borde_inicial < 0.5,
        "transitorio espurio al principio: {borde_inicial:.4}"
    );
    assert!(
        borde_final < 0.5,
        "transitorio espurio al final: {borde_final:.4}"
    );

    // Y el pico debe estar donde corresponde: alrededor de la mitad.
    let pico = salida
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
        .unwrap();
    assert!(
        (pico.0 as i64 - n as i64 / 4).abs() < 2_000,
        "el impulso ha aparecido en la posicion {} en vez de en la mitad",
        pico.0
    );
}

#[test]
fn el_remuestreador_no_normaliza_la_amplitud() {
    // Una señal muy debil tiene que seguir siendo debil: no hay
    // normalización de pico ni compresión, porque la referencia no las
    // hace.
    let n = 44_100;
    for escala in [0.001f32, 0.01, 0.5] {
        let entrada: Vec<f32> = tono(440.0, n, 44_100).iter().map(|v| v * escala).collect();
        let salida = resample(&entrada, 44_100, 22_050).unwrap();
        let pico = salida.iter().fold(0.0f32, |a, b| a.max(b.abs()));
        assert!(
            (pico - escala).abs() < escala * 0.05,
            "la amplitud {escala} se ha convertido en {pico}: el remuestreador normaliza"
        );
    }
}

#[test]
fn la_longitud_sigue_la_regla_de_librosa() {
    // `librosa.resample` con `fix=True` devuelve exactamente
    // `ceil(len * to_rate / from_rate)` muestras.
    for (desde, hasta) in [
        (44_100u32, 22_050u32),
        (22_050, 44_100),
        (48_000, 22_050),
        (8_000, 22_050),
        (11_025, 22_050),
        (96_000, 22_050),
        (22_050, 8_000),
        (22_050, 16_000),
        (32_000, 44_100),
    ] {
        let n = 4_411;
        let entrada = tono(440.0, n, desde);
        let salida = resample(&entrada, desde, hasta).unwrap();
        assert_eq!(
            salida.len(),
            longitud_esperada(n, desde, hasta),
            "longitud incorrecta para {desde} -> {hasta}"
        );
    }
}

#[test]
fn las_frecuencias_poco_habituales_funcionan() {
    // Frecucias que no son potencias de dos, razones muy distintas y
    //audio con duraciones raras.
    let casos = [
        (8_000u32, 22_050u32, 1_000usize),
        (11_025, 22_050, 2_753),
        (48_000, 22_050, 4_097),
        (96_000, 22_050, 8_193),
        (22_050, 8_000, 5_001),
        (22_050, 44_100, 999),
        (17_000, 22_050, 3_333),
        (22_050, 11_025, 777),
    ];

    for (desde, hasta, n) in casos {
        let entrada = seno_puro(440.0, n, desde);
        let salida = resample(&entrada, desde, hasta).unwrap();

        assert_eq!(
            salida.len(),
            longitud_esperada(n, desde, hasta),
            "longitud incorrecta para {desde} -> {hasta}"
        );
        assert!(
            salida.iter().all(|v| v.is_finite()),
            "{desde} -> {hasta}: la salida tiene valores no finitos"
        );

        // La frecuencia fundamental se conserva. Se mide en el interior
        // de la salida: en los bordes el filtro no esta en regimen
        // permanente y la transitorio falsearia la medicion.
        let mitad = salida.len() / 2;
        let ventana = &salida[mitad.saturating_sub(2_000)..(mitad + 2_000).min(salida.len())];
        let (magnitud, _) = goertzel(ventana, 440.0, hasta);
        assert!(
            magnitud > 0.9,
            "{desde} -> {hasta}: la frecuencia no se conserva (magnitud {magnitud:.4})"
        );
    }
}

#[test]
fn una_senal_de_una_sola_muestra_no_produce_panic() {
    for (desde, hasta) in [(44_100u32, 22_050u32), (8_000, 22_050), (22_050, 8_000)] {
        let salida = resample(&[0.5], desde, hasta).unwrap();
        assert_eq!(salida.len(), longitud_esperada(1, desde, hasta));
    }
}

#[test]
fn una_senal_vacia_no_produce_panic() {
    let salida = resample(&[], 44_100, 22_050).unwrap();
    assert!(salida.is_empty());
}

//! La reconstruccion temporal, con los siete escenarios de la fase.
//!
//! El apartado 8.4 de `docs/practica/fase-12-inferencia-duracion-arbitraria.md`
//! pide pruebas con:
//!
//! 1. Una cancion que no llegue a completar una ventana.
//! 2. Una cancion con exactamente una ventana.
//! 3. Una cancion que genere dos ventanas.
//! 4. Una cancion con muchas ventanas.
//! 5. Una nota que comience antes de un limite.
//! 6. Una nota que termine despues de un limite.
//! 7. Una nota sostenida durante varias ventanas.
//!
//! # Como se prueban
//!
//! A nivel de **frames**, que es donde vive la reconstruccion. Cada frame
//! lleva codificado su indice global en el bin 0, y un "nota" ficticia
//! ocupa el bin 1 de un rango de frames. Asi:
//!
//! - el bin 0 demuestra que los frames salen **en orden, sin huecos y sin
//!   duplicados** (un recorte mal hecho se ve al instante);
//! - el bin 1 demuestra que la nota **cruza el limite** sin perder frames
//!   ni duplicarse.
//!
//! Los frames del borde de cada ventana (los 15 que se descartan) se
//! rellenan con un valor centinela: si la reconstruccion no los descarta,
//! el test falla.

use basic_pitch_transcription::stitching::{
    expected_frames, unwrap_output, ANNOT_N_FRAMES, FRAMES_KEPT_PER_WINDOW, N_OLAP,
    N_OVERLAPPING_FRAMES,
};
use candle_core::{DType, Device, Tensor};

/// Desplazamiento entre ventanas, en muestras.
const HOP_SIZE: usize = 36_164;

/// Longitud de una ventana, en muestras.
const VENTANA: usize = 43_844;

/// Relleno inicial, en muestras.
const RELLENO: usize = 3_840;

/// Numero de bins de frecuencia de la matriz de prueba.
const FREQS: usize = 8;

/// Valor centinela para los frames que deben descartarse.
const CENTINELA: f32 = -999_999.0;

/// Cuantas ventanas produce el ventaneo para un audio de `muestras`.
fn num_ventanas(muestras: usize) -> usize {
    (muestras + RELLENO).div_ceil(HOP_SIZE)
}

/// Construye la matriz global de frames y la reparte en ventanas.
///
/// La matriz global es `(n_frames_globales, FREQS)`:
///
/// - el bin 0 lleva el indice global del frame;
/// - el bin 1 lleva la activacion de la nota ficticia;
/// - el resto van a cero.
///
/// Al repartirla, los frames del borde de cada ventana (los `N_OLAP` de
/// cada lado) se rellenan con `CENTINELA`, porque la reconstruccion tiene
/// que descartarlos.
fn repartir(muestras: usize, nota: Option<(usize, usize)>) -> (Tensor, usize) {
    let n_ventanas = num_ventanas(muestras);
    let n_frames_globales = n_ventanas * FRAMES_KEPT_PER_WINDOW;

    // Matriz global.
    let mut global = vec![0.0f32; n_frames_globales * FREQS];
    for g in 0..n_frames_globales {
        global[g * FREQS] = g as f32;
        if let Some((inicio, fin)) = nota {
            if g >= inicio && g < fin {
                global[g * FREQS + 1] = 1.0;
            }
        }
    }

    // Reparto en ventanas: la ventana w aporta los frames globales
    // [w*142, w*142 + 142), y sus 15 frames de cada borde se descartan.
    let mut ventanas = vec![0.0f32; n_ventanas * ANNOT_N_FRAMES * FREQS];
    for w in 0..n_ventanas {
        for j in 0..ANNOT_N_FRAMES {
            let destino = (w * ANNOT_N_FRAMES + j) * FREQS;
            if (N_OLAP..N_OLAP + FRAMES_KEPT_PER_WINDOW).contains(&j) {
                let g = w * FRAMES_KEPT_PER_WINDOW + (j - N_OLAP);
                for b in 0..FREQS {
                    ventanas[destino + b] = global[g * FREQS + b];
                }
            } else {
                for b in 0..FREQS {
                    ventanas[destino + b] = CENTINELA;
                }
            }
        }
    }

    let tensor = Tensor::from_slice(&ventanas, (n_ventanas, ANNOT_N_FRAMES, FREQS), &Device::Cpu)
        .expect("tensor");
    (tensor, n_frames_globales)
}

/// Reconstruye y comprueba que los frames salen en orden y sin huecos.
fn comprobar_orden(muestras: usize, nombre: &str) -> Tensor {
    let (ventanas, _) = repartir(muestras, None);
    let esperada = expected_frames(muestras, HOP_SIZE);

    let salida = unwrap_output(&ventanas, muestras, N_OVERLAPPING_FRAMES, HOP_SIZE)
        .unwrap_or_else(|e| panic!("{nombre}: {e}"));

    assert_eq!(
        salida.dims2().unwrap(),
        (esperada, FREQS),
        "{nombre}: forma inesperada"
    );

    let datos = salida.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let n_frames = salida.dims2().unwrap().0;

    // Ningun frame del borde puede haber sobrevivido: los que se
    // descartaban estaban marcados con el centinela.
    for (i, valor) in datos.iter().enumerate() {
        assert_ne!(
            *valor, CENTINELA,
            "{nombre}: ha sobrevivido el valor {i} de un frame del borde, que deberia descartarse"
        );
    }

    // El bin 0 de cada frame lleva su indice global: si la
    // reconstruccion perdiera o duplicara frames, se veria aqui.
    for frame in 0..n_frames {
        let valor = datos[frame * FREQS];
        assert_eq!(
            valor, frame as f32,
            "{nombre}: el frame {frame} deberia llevar su indice global"
        );
    }
    salida
}

/// Comprueba que una nota ficticia cruza un limite de ventana sin perder
/// frames ni duplicarse.
fn comprobar_nota(muestras: usize, nota: (usize, usize), nombre: &str) {
    let (ventanas, _) = repartir(muestras, Some(nota));
    let salida = unwrap_output(&ventanas, muestras, N_OVERLAPPING_FRAMES, HOP_SIZE)
        .unwrap_or_else(|e| panic!("{nombre}: {e}"));

    let datos = salida.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let (inicio, fin) = nota;

    for frame in 0..salida.dims2().unwrap().0 {
        let activacion = datos[frame * FREQS + 1];
        let esperado = if frame >= inicio && frame < fin {
            1.0
        } else {
            0.0
        };
        assert_eq!(
            activacion, esperado,
            "{nombre}: el frame {frame} deberia tener activacion {esperado}, tiene {activacion}"
        );
    }
}

// ---------------------------------------------------------------------------
// Los siete escenarios del apartado 8.4
// ---------------------------------------------------------------------------

#[test]
fn escenario_1_una_cancion_que_no_llega_a_completar_una_ventana() {
    // 1 segundo: menos que una ventana del modelo.
    let muestras = 22_050;
    assert_eq!(num_ventanas(muestras), 1);

    let salida = comprobar_orden(muestras, "escenario 1");
    // int(22050 / 36164 * 142) = int(86.55) = 86 frames.
    assert_eq!(salida.dims2().unwrap().0, 86);
}

#[test]
fn escenario_2_una_cancion_con_exactamente_una_ventana() {
    // Exactamente 2 segundos: una ventana completa de audio.
    let muestras = VENTANA;
    assert_eq!(
        num_ventanas(muestras),
        2,
        "con el relleno inicial salen dos ventanas"
    );

    let salida = comprobar_orden(muestras, "escenario 2");
    // int(43844 / 36164 * 142) = int(172.17) = 172 frames.
    assert_eq!(salida.dims2().unwrap().0, 172);
}

#[test]
fn escenario_3_una_cancion_que_genere_dos_ventanas() {
    let muestras = 2 * HOP_SIZE;
    assert_eq!(
        num_ventanas(muestras),
        3,
        "el relleno inicial añade una ventana"
    );

    let salida = comprobar_orden(muestras, "escenario 3");
    // int(72328 / 36164 * 142) = 284 frames.
    assert_eq!(salida.dims2().unwrap().0, 284);
}

#[test]
fn escenario_4_una_cancion_con_muchas_ventanas() {
    // Un minuto de audio.
    let muestras = 60 * HOP_SIZE;
    assert!(num_ventanas(muestras) > 55);

    let salida = comprobar_orden(muestras, "escenario 4");
    // 60 ventanas * 142 = 8520 frames.
    assert_eq!(salida.dims2().unwrap().0, 8520);
}

#[test]
fn escenario_5_una_nota_que_comience_antes_de_un_limite() {
    // La nota empieza en el frame 130 y acaba en el 160: el limite entre
    // ventanas esta en el frame 142.
    let muestras = 2 * HOP_SIZE;
    comprobar_nota(muestras, (130, 160), "escenario 5");
}

#[test]
fn escenario_6_una_nota_que_termine_despues_de_un_limite() {
    // La nota empieza antes del limite y acaba despues.
    let muestras = 2 * HOP_SIZE;
    comprobar_nota(muestras, (120, 200), "escenario 6");
}

#[test]
fn escenario_7_una_nota_sostenida_durante_varias_ventanas() {
    // Una nota de 400 frames: atraviesa tres ventanas.
    let muestras = 4 * HOP_SIZE;
    comprobar_nota(muestras, (100, 500), "escenario 7");
}

#[test]
fn el_numero_de_frames_esperados_usa_la_longitud_original() {
    // El upstream recorta con la longitud ORIGINAL (antes del relleno
    // inicial de 3840), no con la longitud rellenada.
    let muestras = 100_000;
    let con_relleno = (muestras + RELLENO).div_ceil(HOP_SIZE);
    let esperado = (muestras as f64 / HOP_SIZE as f64 * 142.0) as usize;
    assert_eq!(expected_frames(muestras, HOP_SIZE), esperado);
    assert!(
        esperado < con_relleno * FRAMES_KEPT_PER_WINDOW,
        "el truncado tiene que recortar frames"
    );
}

#[test]
fn una_matriz_vacia_produce_error() {
    let vacia = Tensor::zeros((0, ANNOT_N_FRAMES, FREQS), DType::F32, &Device::Cpu).unwrap();
    assert!(unwrap_output(&vacia, 22_050, N_OVERLAPPING_FRAMES, HOP_SIZE).is_err());
}

#[test]
fn una_ventana_con_malos_frames_produce_error() {
    let mala = Tensor::zeros((1, 100, FREQS), DType::F32, &Device::Cpu).unwrap();
    assert!(unwrap_output(&mala, 22_050, N_OVERLAPPING_FRAMES, HOP_SIZE).is_err());
}

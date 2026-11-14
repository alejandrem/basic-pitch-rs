//! Cada desplazamiento armonico, probado de manera independiente.
//!
//! La fase 5 pide: "Se probara cada desplazamiento de manera independiente
//! con entradas artificiales. Por ejemplo, se utilizaran matrices con
//! valores identificables por posicion para comprobar que los canales se
//! desplazan y rellenan correctamente."
//!
//! El oraculo de cada test es la formula literal de `basic_pitch/nn.py`:
//!
//! ```text
//! shift == 0  ->  out = x
//! shift > 0   ->  out = pad(x[:, :, shift:, :], [[0,0],[0,0],[0,shift],[0,0]])
//! shift < 0   ->  out = pad(x[:, :, :shift, :], [[0,0],[0,0],[-shift,0],[0,0]])
//! ```
//!
//! Es decir, `out[i] = x[i + shift]`, con ceros donde el indice se sale
//! de rango, y la **asimetria** decide donde van esos ceros: al final
//! para los desplazamientos positivos y al principio para los negativos.
//!
//! La asimetria importa de verdad: despues de truncar a 264 bins, la
//! posicion de los ceros es lo que distingue un desplazamiento de otro.

use basic_pitch_model::harmonic_stacking::{harmonic_shifts, harmonic_stacking};
use candle_core::{Device, Tensor};

/// Los ocho armonicos del modelo y su desplazamiento en bins.
const ARMONICOS: [(f64, i64); 8] = [
    (0.5, -36),
    (1.0, 0),
    (2.0, 36),
    (3.0, 57),
    (4.0, 72),
    (5.0, 84),
    (6.0, 93),
    (7.0, 101),
];

/// Entrada artificial: el bin i tiene el valor `i + 1`.
///
/// Los valores empiezan en 1 y no en 0 a proposito: si la entrada
/// tuviera ceros, no se podria distinguir un cero del relleno de un cero
/// que viene de la entrada, y las comprobaciones de asimetria no
/// significarian nada.
fn entrada_de_prueba(n_bins: usize) -> Tensor {
    let datos: Vec<f32> = (0..n_bins).map(|i| (i + 1) as f32).collect();
    Tensor::from_slice(&datos, (1, 1, 1, n_bins), &Device::Cpu).unwrap()
}

/// El valor esperado del bin `i` del canal desplazado.
///
/// Reproduce la formula literal de la referencia.
fn esperado(bin: usize, shift: i64, n_bins: usize) -> f32 {
    let origen = bin as i64 + shift;
    if origen >= 0 && (origen as usize) < n_bins {
        (origen + 1) as f32
    } else {
        0.0
    }
}

/// Comprueba un desplazamiento sobre la entrada de 309 bins.
fn comprobar_desplazamiento(harmonic: f64, shift: i64) {
    let n_bins = 309usize;
    let salida = harmonic_stacking(&entrada_de_prueba(n_bins), &[harmonic], 3, 264).unwrap();
    assert_eq!(salida.dims4().unwrap(), (1, 1, 1, 264));
    let datos = salida.flatten_all().unwrap().to_vec1::<f32>().unwrap();

    for (bin, valor) in datos.iter().enumerate() {
        let esperado = esperado(bin, shift, n_bins);
        assert_eq!(
            *valor, esperado,
            "armonico {harmonic} (shift {shift}), bin {bin}: {valor} != {esperado}"
        );
    }

    // La asimetria: donde van los ceros.
    //
    // Con shift > 0, out[i] = x[i + shift] es cero para i >= n - shift: los
    // ceros van AL FINAL. Ese limite puede quedar fuera de la salida
    // truncada (si n - shift >= 264), en cuyo caso no hay ningun cero.
    //
    // Con shift < 0, out[i] = x[i + shift] es cero para i < -shift: los
    // ceros van AL PRINCIPIO.
    if shift > 0 {
        let limite = n_bins - shift as usize;
        let primer_cero = (0..264).find(|i| esperado(*i, shift, n_bins) == 0.0);
        assert_eq!(
            primer_cero,
            Some(limite).filter(|l| *l < 264),
            "armonico {harmonic}: los ceros deberian empezar en el bin {limite}"
        );
        // El ultimo bin con contenido.
        let ultimo = (0..264)
            .rev()
            .find(|i| esperado(*i, shift, n_bins) != 0.0)
            .unwrap();
        assert_eq!(
            ultimo,
            limite.min(264) - 1,
            "armonico {harmonic}: ultimo bin con contenido"
        );
    } else if shift < 0 {
        let primero_con_contenido = (0..264)
            .find(|i| esperado(*i, shift, n_bins) != 0.0)
            .unwrap();
        assert_eq!(
            primero_con_contenido,
            (-shift) as usize,
            "armonico {harmonic}: los ceros deberian ocupar los primeros {} bins",
            (-shift) as usize
        );
    }
}

#[test]
fn armonico_c_medio_desplaza_36_bins_hacia_arriba() {
    // 0.5 -> shift -36: out[i] = x[i - 36], con 36 ceros al principio.
    comprobar_desplazamiento(0.5, -36);
    let salida = harmonic_stacking(&entrada_de_prueba(309), &[0.5], 3, 264).unwrap();
    let datos = salida.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert!(
        datos[..36].iter().all(|v| *v == 0.0),
        "los 36 primeros deben ser ceros"
    );
    assert_eq!(datos[36], 1.0, "el bin 36 es el bin 0 de la entrada");
    assert_eq!(datos[36 + 10], 11.0);
    // Y al final no hay ceros: el desplazamiento negativo rellena al
    // principio, no al final. Es la asimetria.
    assert!(datos[263] != 0.0, "al final no deberia haber ceros");
}

#[test]
fn armonico_uno_no_desplaza_nada() {
    // 1 -> shift 0: el canal es la entrada, truncada.
    comprobar_desplazamiento(1.0, 0);
    let salida = harmonic_stacking(&entrada_de_prueba(309), &[1.0], 3, 264).unwrap();
    let datos = salida.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert!(datos.iter().all(|v| *v != 0.0), "no deberia haber ceros");
    for (i, valor) in datos.iter().enumerate() {
        assert_eq!(*valor, (i + 1) as f32, "bin {i}");
    }
}

#[test]
fn armonico_dos_desplaza_36_bins_hacia_abajo() {
    // 2 -> shift +36: out[i] = x[i + 36], con ceros al final.
    comprobar_desplazamiento(2.0, 36);
    let salida = harmonic_stacking(&entrada_de_prueba(309), &[2.0], 3, 264).unwrap();
    let datos = salida.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(datos[0], 37.0);
    // Con shift +36 sobre 309 bins, el primer cero estaria en el bin 273:
    // fuera de la salida truncada a 264. No hay ningun cero.
    assert!(
        datos.iter().all(|v| *v != 0.0),
        "no deberia haber ceros en la salida truncada"
    );
    assert_eq!(datos[263], 300.0, "el ultimo bin es el 299 de la entrada");
}

#[test]
fn armonico_tres_desplaza_57_bins() {
    comprobar_desplazamiento(3.0, 57);
}

#[test]
fn armonico_cuatro_desplaza_72_bins() {
    comprobar_desplazamiento(4.0, 72);
}

#[test]
fn armonico_cinco_desplaza_84_bins() {
    comprobar_desplazamiento(5.0, 84);
}

#[test]
fn armonico_seis_desplaza_93_bins() {
    comprobar_desplazamiento(6.0, 93);
}

#[test]
fn armonico_siete_desplaza_101_bins() {
    comprobar_desplazamiento(7.0, 101);
    let salida = harmonic_stacking(&entrada_de_prueba(309), &[7.0], 3, 264).unwrap();
    let datos = salida.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(datos[0], 102.0);
    // Los ceros empiezan en el bin 309 - 101 = 208.
    assert_eq!(
        datos[207], 309.0,
        "el ultimo bin con contenido es el 308 de la entrada"
    );
    assert!(
        datos[208..].iter().all(|v| *v == 0.0),
        "del bin 208 en adelante hay ceros"
    );
}

#[test]
fn los_desplazamientos_se_calculo_con_la_formula_de_la_referencia() {
    // shift = round(12 * bins_per_semitone * log2(harmonico))
    let armonicos: Vec<f64> = ARMONICOS.iter().map(|(h, _)| *h).collect();
    let esperados: Vec<i64> = ARMONICOS.iter().map(|(_, s)| *s).collect();
    assert_eq!(harmonic_shifts(&armonicos, 3), esperados);
}

#[test]
fn los_ocho_canales_estan_en_el_orden_de_los_armonicos() {
    // El orden de los canales es el orden de la lista de armonicos, y no
    // se puede reordenar sin romper el modelo.
    let salida = harmonic_stacking(&entrada_de_prueba(309), &[0.5, 1.0, 2.0], 3, 264).unwrap();
    assert_eq!(salida.dims4().unwrap(), (1, 3, 1, 264));

    let datos = salida.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    // Canal 0 (0.5): empieza con 36 ceros.
    assert!(datos[..36].iter().all(|v| *v == 0.0));
    // Canal 1 (1.0): la entrada intacta.
    assert_eq!(datos[264], 1.0);
    assert_eq!(datos[264 + 10], 11.0);
    // Canal 2 (2.0): empieza en 36.
    assert_eq!(datos[2 * 264], 37.0);
}

#[test]
fn el_truncado_ocurre_despues_de_concatenar() {
    // Si el truncado ocurriera antes de desplazar, el canal del
    // armonico 7 (shift +101) no tendria contenido: la entrada truncada
    // a 264 bins solo llega hasta el 263, y se necesitan hasta el 364.
    let salida = harmonic_stacking(&entrada_de_prueba(309), &[7.0], 3, 264).unwrap();
    let datos = salida.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(
        datos[0], 102.0,
        "el primer bin debe ser el 101 de la entrada"
    );
    assert_eq!(
        datos[162], 264.0,
        "el ultimo bin con contenido es el 263 de la entrada"
    );
}

#[test]
fn un_desplazamiento_que_no_cabe_produce_error() {
    // Con 309 bins, un desplazamiento de 309 o mas no cabe.
    let armonicos = [1_000.0f64];
    let salida = harmonic_stacking(&entrada_de_prueba(309), &armonicos, 3, 264);
    assert!(
        salida.is_err(),
        "un desplazamiento fuera de rango debe fallar"
    );
}

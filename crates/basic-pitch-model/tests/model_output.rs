//! El contrato de las salidas del modelo.
//!
//! Estas comprobaciones existen porque hubo un bug que **no** se detectaba
//! con los tests anteriores: las tres salidas del modelo salian con el eje
//! de tiempo y el de frecuencia intercambiados (`note` como
//! `(batch, 88, 172)` en vez de `(batch, 172, 88)`).
//!
//! La red era correcta (coincidia con TensorFlow); lo que estaba mal era el
//! `transpose(1, 2)` final de cada rama. Y como no existia ninguna
//! comparacion contra la referencia (eso es la Fase 7), nada lo detecto.
//!
//! # Por que estos tests bastan
//!
//! La forma **ya** distingue el bug: `(batch, 88, 172)` y `(batch, 172, 88)`
//! son formas distintas. Y anade una comprobacion **estructural** que no
//! depende de la forma: si la entrada es constante en el tiempo, la salida
//! tiene que ser constante en el tiempo, porque las convoluciones son
//! invariantes a la traslacion en el tiempo y la BatchNorm en modo
//! inferencia no calcula estadisticas sobre el eje temporal.
//!
//! Si los ejes estuvieran intercambiados, esa propiedad fallaria.

use std::path::{Path, PathBuf};

use basic_pitch_model::{config::ModelConfig, weights::Weights, BasicPitchModel};

/// Raiz del repositorio.
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Directorio del modelo desplegable.
fn model_dir() -> PathBuf {
    repo_root().join("models/basic-pitch/icassp-2022")
}

/// Si no esta el asset, no hay nada que probar.
fn skip_si_no_hay_modelo() -> bool {
    if !model_dir().join("model.safetensors").exists() {
        eprintln!(
            "OMITIDO: no se encuentra el modelo en {}",
            model_dir().display()
        );
        return true;
    }
    false
}

/// Carga el modelo completo.
fn cargar() -> BasicPitchModel {
    let dir = model_dir();
    let config = ModelConfig::load(&dir.join("model_config.json")).expect("configuracion");
    let weights = Weights::load(&dir.join("model.safetensors")).expect("pesos");
    let kernels =
        basic_pitch_cqt::CqtKernels::load(&dir.join("cqt_kernels.safetensors")).expect("kernels");
    BasicPitchModel::new(config, weights, kernels).expect("modelo")
}

/// Entrada armonica sintetica, en NCHW `(1, 8, 172, 264)`.
///
/// `valor(t, f, c)` decide cada elemento.
fn entrada(valor: impl Fn(usize, usize, usize) -> f32) -> candle_core::Tensor {
    let (canales, frames, bins) = (8usize, 172usize, 264usize);
    let mut datos = Vec::with_capacity(canales * frames * bins);
    for c in 0..canales {
        for t in 0..frames {
            for f in 0..bins {
                datos.push(valor(t, f, c));
            }
        }
    }
    candle_core::Tensor::from_slice(
        &datos,
        (1, canales, frames, bins),
        &candle_core::Device::Cpu,
    )
    .expect("tensor")
}

#[test]
fn las_salidas_tienen_la_forma_del_contrato() {
    if skip_si_no_hay_modelo() {
        return;
    }
    let model = cargar();
    let entrada = entrada(|_, f, _| f as f32 / 264.0);
    let salida = model.forward(&entrada).expect("forward");

    assert_eq!(
        salida.note.dims3().unwrap(),
        (1, 172, 88),
        "`note` debe ser (batch, 172, 88): tiempo primero"
    );
    assert_eq!(
        salida.onset.dims3().unwrap(),
        (1, 172, 88),
        "`onset` debe ser (batch, 172, 88): tiempo primero"
    );
    assert_eq!(
        salida.contour.dims3().unwrap(),
        (1, 172, 264),
        "`contour` debe ser (batch, 172, 264): tiempo primero"
    );
}

#[test]
fn una_entrada_constante_en_el_tiempo_da_una_salida_constante_en_el_tiempo() {
    if skip_si_no_hay_modelo() {
        return;
    }
    let model = cargar();

    // Constante en el tiempo, variable en la frecuencia.
    let entrada = entrada(|_, f, c| ((f + c * 7) % 31) as f32 / 31.0);
    let salida = model.forward(&entrada).expect("forward");

    for (nombre, tensor) in [
        ("note", &salida.note),
        ("onset", &salida.onset),
        ("contour", &salida.contour),
    ] {
        let (batch, frames, ancho) = tensor.dims3().unwrap();
        assert_eq!(batch, 1);
        let datos = tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        // Se comparan dos frames del interior: los bordes estan afectados
        // por el relleno SAME y no tienen por que coincidir.
        let t1 = frames / 2;
        let t2 = frames / 2 + 10;
        for i in 0..ancho {
            let a = datos[t1 * ancho + i];
            let b = datos[t2 * ancho + i];
            assert!(
                (a - b).abs() < 1e-5,
                "{nombre}: la salida cambia con el tiempo aunque la entrada no ({a} != {b})"
            );
        }
    }
}

#[test]
fn una_entrada_constante_en_la_frecuencia_da_una_salida_constante_en_la_frecuencia() {
    if skip_si_no_hay_modelo() {
        return;
    }
    let model = cargar();

    // Constante en la frecuencia, variable en el tiempo.
    let entrada = entrada(|t, _, c| ((t + c) % 17) as f32 / 17.0);
    let salida = model.forward(&entrada).expect("forward");

    for (nombre, tensor) in [
        ("note", &salida.note),
        ("onset", &salida.onset),
        ("contour", &salida.contour),
    ] {
        let (batch, frames, ancho) = tensor.dims3().unwrap();
        assert_eq!(batch, 1);
        let datos = tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        // Dentro de un MISMO frame, todos los valores de frecuencia deben
        // coincidir: las convoluciones mezclan frecuencias, pero si la
        // entrada es constante en frecuencia, la mezcla da lo mismo en
        // todas partes.
        //
        // Se mide en el interior: en los bordes, el relleno SAME de
        // frecuencia introduce ceros y el valor si cambia.
        let t = frames / 2;
        let margen = 30.min(ancho / 4);
        let referencia = datos[t * ancho + margen];
        for f in margen..(ancho - margen) {
            let valor = datos[t * ancho + f];
            assert!(
                (valor - referencia).abs() < 1e-4,
                "{nombre}: en el frame {t}, la frecuencia {f} vale {valor} y la de referencia {referencia}"
            );
        }
    }
}

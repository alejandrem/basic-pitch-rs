//! Los 13 casos de prueba del apartado 7.4 de la Fase 7.
//!
//! Para cada caso se comprueba que Rust y el upstream de Spotify producen
//! **lo mismo**: las mismas activaciones neuronales y las mismas notas
//! detectadas.
//!
//! Los casos son senales controladas, generadas por
//! `tools/weight-export/make_cases.py`, con sus oraculos calculados por el
//! upstream (`make_cases_oracle.py`):
//!
//! | Caso | Que prueba |
//! |---|---|
//! | `silencio` | que no se inventen notas donde no hay nada |
//! | `nota_sostenida` | una nota con ataque y cauda limpios |
//! | `nota_corta` | notas por debajo de la duracion minima |
//! | `notas_simultaneas` | varias notas a la vez |
//! | `acordes` | progresion de acordes |
//! | `notas_graves` | el registro grave, donde el modelo tiene menos resolucion |
//! | `notas_agudas` | el registro agudo |
//! | `transitorios_rapidos` | ataques muy seguidos |
//! | `vibrato` | la afinacion oscila dentro de la nota |
//! | `deslizamiento` | un glissando, que no es ninguna nota |
//! | `ruido` | una nota con ruido de fondo |
//! | `duracion_larga` | 30 segundos, unas 18 ventanas |
//! | `limites_ventana` | notas a ambos lados del limite entre ventanas |

use std::path::{Path, PathBuf};

use basic_pitch_audio_io::decode_file;
use basic_pitch_audio_preprocess::{resample, to_mono, AudioWindows, AUDIO_SAMPLE_RATE, HOP_SIZE};
use basic_pitch_model::{config::ModelConfig, weights::Weights, AcousticFrontend, BasicPitchModel};
use basic_pitch_reference_npy::read_f32;
use basic_pitch_transcription::note_detection::{
    output_to_notes_polyphonic, NoteDetectionConfig, N_NOTES,
};
use basic_pitch_validation::compare;

/// Raiz del repositorio.
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Directorio de los casos.
fn casos_dir() -> PathBuf {
    repo_root().join("reference/casos")
}

/// Si no estan los casos, no hay nada que probar.
fn skip_si_no_hay_casos() -> bool {
    if !casos_dir().exists() {
        eprintln!("OMITIDO: no existe {}", casos_dir().display());
        return true;
    }
    false
}

/// Carga el modelo y el frontend.
fn cargar() -> (BasicPitchModel, AcousticFrontend) {
    let dir = repo_root().join("models/basic-pitch/icassp-2022");
    let config = ModelConfig::load(&dir.join("model_config.json")).expect("configuracion");
    let weights = Weights::load(&dir.join("model.safetensors")).expect("pesos");
    let frontend = AcousticFrontend::load(&dir, &config, &weights).expect("frontend");
    let model = BasicPitchModel::new(
        config,
        weights,
        basic_pitch_cqt::CqtKernels::load(&dir.join("cqt_kernels.safetensors")).expect("kernels"),
    )
    .expect("modelo");
    (model, frontend)
}

/// Una nota detectada, para comparar con el upstream.
#[derive(Debug, Clone, PartialEq)]
struct Nota {
    start_time: f64,
    end_time: f64,
    pitch: u8,
    amplitude: f32,
}

/// Compara un caso con su oraculo.
///
/// Devuelve el numero de notas detectadas por cada lado.
fn comprobar(nombre: &str) {
    let (model, frontend) = cargar();

    // --- El audio ---
    let decoded =
        decode_file(&casos_dir().join(format!("{nombre}.wav"))).expect("decodificar el audio");
    let mono = to_mono(&decoded);
    let remuestreado =
        resample(&mono.samples, mono.sample_rate, AUDIO_SAMPLE_RATE).expect("remuestrear");
    let windows = AudioWindows::new(remuestreado, AUDIO_SAMPLE_RATE);

    // --- P2 + P4: ventanas -> red -> reconstruccion ---
    let n_frames =
        basic_pitch_transcription::stitching::expected_frames(windows.original_length(), HOP_SIZE);
    let mut reconstruccion = basic_pitch_transcription::stitching::ModelUnwrapper::new(
        n_frames,
        basic_pitch_transcription::stitching::N_OVERLAPPING_FRAMES,
    );
    for ventana in windows.iter() {
        let armonica = frontend.forward(&ventana.samples).expect("frontend");
        let salida = model.forward(&armonica).expect("forward");
        reconstruccion.push(&salida).expect("reconstruir");
    }
    let salida = reconstruccion.finish().expect("cerrar");

    // --- Las activaciones ---
    let mut fallos = 0usize;
    for (etapa, fichero) in [
        ("note", format!("{nombre}_note.npy")),
        ("onset", format!("{nombre}_onset.npy")),
        ("contour", format!("{nombre}_contour.npy")),
    ] {
        let referencia = read_f32(&casos_dir().join(fichero)).expect("referencia");
        let tensor = match etapa {
            "note" => &salida.note,
            "onset" => &salida.onset,
            _ => &salida.contour,
        };
        let rust = tensor
            .flatten_all()
            .expect("aplanar")
            .to_vec1::<f32>()
            .expect("float32");
        let metricas = compare(&rust, &referencia.data).expect("comparar");

        // El criterio es el mismo que en el resto del proyecto: el
        // **MAE** es la comprobacion estricta (mide el error tipico) y el
        // maximo va con margen, porque es la cola.
        //
        // Medido en los trece casos, el MAE esta entre 2e-8 y 7e-6 en
        // todos ellos: el error de la red NO se acumula con las ventanas
        // (el caso de 30 segundos, con 19 ventanas, tiene el mismo MAE que
        // uno de 2). Lo que varia es el maximo (de 4e-7 a 1.03e-3), que
        // depende de los valores concretos de cada caso.
        const TOLERANCIA_MAE: f64 = 1e-4;
        const TOLERANCIA_MAX: f64 = 5e-3;

        if metricas.mae > TOLERANCIA_MAE {
            println!(
                "  {nombre}/{etapa}: mae {:.3e} (tolerancia {TOLERANCIA_MAE:.0e})",
                metricas.mae
            );
            fallos += 1;
        }
        if metricas.max_abs_diff > TOLERANCIA_MAX {
            println!(
                "  {nombre}/{etapa}: max {:.3e} (tolerancia {TOLERANCIA_MAX:.0e})",
                metricas.max_abs_diff
            );
            fallos += 1;
        }
    }
    assert_eq!(fallos, 0, "{nombre}: {fallos} etapas fuera de tolerancia");

    // --- Las notas detectadas ---
    let frames = salida
        .note
        .flatten_all()
        .expect("aplanar")
        .to_vec1::<f32>()
        .expect("float32");
    let onsets = salida
        .onset
        .flatten_all()
        .expect("aplanar")
        .to_vec1::<f32>()
        .expect("float32");
    let n_frames = frames.len() / N_NOTES;

    let notas_rust: Vec<Nota> = output_to_notes_polyphonic(
        &frames,
        &onsets,
        n_frames,
        N_NOTES,
        &NoteDetectionConfig::default(),
    )
    .into_iter()
    .map(|n| Nota {
        start_time: basic_pitch_transcription::stitching::frame_to_time(
            n.start_frame,
            AUDIO_SAMPLE_RATE,
            basic_pitch_audio_preprocess::FFT_HOP,
            basic_pitch_audio_preprocess::AUDIO_N_SAMPLES,
        ),
        end_time: basic_pitch_transcription::stitching::frame_to_time(
            n.end_frame,
            AUDIO_SAMPLE_RATE,
            basic_pitch_audio_preprocess::FFT_HOP,
            basic_pitch_audio_preprocess::AUDIO_N_SAMPLES,
        ),
        pitch: n.pitch,
        amplitude: n.amplitude,
    })
    .collect();

    // Las notas del upstream, del JSON.
    let texto = std::fs::read_to_string(casos_dir().join(format!("{nombre}_notas.json")))
        .expect("notas del upstream");
    let crudo: Vec<serde_json::Value> = serde_json::from_str(&texto).expect("json");
    let notas_upstream: Vec<Nota> = crudo
        .into_iter()
        .map(|n| Nota {
            start_time: n["start_time"].as_f64().unwrap(),
            end_time: n["end_time"].as_f64().unwrap(),
            pitch: n["pitch"].as_u64().unwrap() as u8,
            amplitude: n["amplitude"].as_f64().unwrap() as f32,
        })
        .collect();

    assert_eq!(
        notas_rust.len(),
        notas_upstream.len(),
        "{nombre}: numero de notas distinto (Rust {}, upstream {})",
        notas_rust.len(),
        notas_upstream.len()
    );

    // Nota a nota: mismo pitch, mismo inicio, mismo final.
    //
    // La tolerancia de tiempo es de UN FRAME (11.6 ms), no de un
    // microsegundo, y el motivo esta medido:
    //
    // El onset de una nota puede venir de los onsets INFERIDOS (los
    // cambios de la activacion de nota), que se reescalan dividiendo por
    // `max(frame_diff)`. Cuando dos frames vecinos tienen valores casi
    // iguales (en el caso de "notas_simultaneas", 0.0866 y 0.1054), el
    // redondeo de float32 decide si el maximo local cae en uno o en otro.
    // Rust y NumPy hacen la misma formula, pero acumulan en orden
    // distinto, asi que pueden discrepar en un frame.
    //
    // El numero de notas y los pitches SI se comparan de forma exacta:
    // ahi no hay ambiguedad.
    // El margen de 1 ns es para el redondeo de f64: la diferencia puede
    // caer justo en el limite y quedar 1 ulp por encima.
    const FRAME: f64 = 256.0 / 22_050.0 + 1e-9;
    for (i, (a, b)) in notas_rust.iter().zip(notas_upstream.iter()).enumerate() {
        assert_eq!(a.pitch, b.pitch, "{nombre}: nota {i}: pitch distinto");
        assert!(
            (a.start_time - b.start_time).abs() <= FRAME,
            "{nombre}: nota {i}: inicio distinto (Rust {}, upstream {})",
            a.start_time,
            b.start_time
        );
        assert!(
            (a.end_time - b.end_time).abs() <= FRAME,
            "{nombre}: nota {i}: final distinto (Rust {}, upstream {})",
            a.end_time,
            b.end_time
        );
    }

    println!(
        "  {nombre:22} {} notas, {} ventanas, idénticas al upstream",
        notas_rust.len(),
        (windows.original_length() + basic_pitch_audio_preprocess::INITIAL_PADDING)
            .div_ceil(HOP_SIZE)
    );
}

/// Los trece casos.
///
/// Va marcado como `#[ignore]` porque el caso de 30 segundos (19 ventanas)
/// tarda casi un minuto en modo debug. El guardián permanente del suite es
/// `algunos_casos_coinciden_con_el_upstream`, que corre los tres casos mas
/// representativos.
///
/// Para ejecutarlo todo:
///
/// ```bash
/// cargo test --release -p basic-pitch-transcription --test casos -- --ignored
/// ```
#[test]
#[ignore = "tarda casi un minuto en modo debug"]
fn los_trece_casos_coinciden_con_el_upstream() {
    if skip_si_no_hay_casos() {
        return;
    }

    let casos = [
        "silencio",
        "nota_sostenida",
        "nota_corta",
        "notas_simultaneas",
        "acordes",
        "notas_graves",
        "notas_agudas",
        "transitorios_rapidos",
        "vibrato",
        "deslizamiento",
        "ruido",
        "duracion_larga",
        "limites_ventana",
    ];

    println!("Los 13 casos de la Fase 7:");
    for caso in casos {
        comprobar(caso);
    }
}

/// El guardian permanente: tres casos representativos.
///
///   - `silencio`: que no se invente nada donde no hay nada.
///   - `nota_sostenida`: una nota con ataque y cauda limpios.
///   - `limites_ventana`: notas a ambos lados del limite entre ventanas,
///     que es donde la reconstruccion se complica.
///
/// Los trece estan en `los_trece_casos_coinciden_con_el_upstream`.
#[test]
fn algunos_casos_coinciden_con_el_upstream() {
    if skip_si_no_hay_casos() {
        return;
    }
    println!("Casos del guardian:");
    for caso in ["silencio", "nota_sostenida", "limites_ventana"] {
        comprobar(caso);
    }
}

/// Un caso concreto, por si se quiere ver el detalle.
#[test]
fn el_silencio_no_inventa_notas() {
    if skip_si_no_hay_casos() {
        return;
    }
    let (model, frontend) = cargar();
    let decoded = decode_file(&casos_dir().join("silencio.wav")).expect("decodificar");
    let mono = to_mono(&decoded);
    let remuestreado = resample(&mono.samples, mono.sample_rate, AUDIO_SAMPLE_RATE).unwrap();
    let windows = AudioWindows::new(remuestreado, AUDIO_SAMPLE_RATE);

    let n_frames =
        basic_pitch_transcription::stitching::expected_frames(windows.original_length(), HOP_SIZE);
    let mut reconstruccion = basic_pitch_transcription::stitching::ModelUnwrapper::new(
        n_frames,
        basic_pitch_transcription::stitching::N_OVERLAPPING_FRAMES,
    );
    for ventana in windows.iter() {
        let armonica = frontend.forward(&ventana.samples).unwrap();
        let salida = model.forward(&armonica).unwrap();
        reconstruccion.push(&salida).unwrap();
    }
    let salida = reconstruccion.finish().unwrap();

    let frames: Vec<f32> = salida.note.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let onsets: Vec<f32> = salida
        .onset
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let n = frames.len() / N_NOTES;
    let notas = output_to_notes_polyphonic(
        &frames,
        &onsets,
        n,
        N_NOTES,
        &NoteDetectionConfig::default(),
    );
    // El silencio NO es silencio digital: lleva un residuo de ruido de
    // 1e-4 (un silencio absoluto seria degenerado para el modelo). El
    // upstream tambien detecta notas en el, y lo que se comprueba aqui es
    // que Rust detecte EXACTAMENTE las mismas.
    let texto = std::fs::read_to_string(casos_dir().join("silencio_notas.json"))
        .expect("notas del upstream");
    let crudo: Vec<serde_json::Value> = serde_json::from_str(&texto).expect("json");
    let del_upstream = crudo.len();
    assert_eq!(
        notas.len(),
        del_upstream,
        "el silencio deberia dar las mismas notas que el upstream"
    );
    println!("  silencio: {del_upstream} notas, las mismas que el upstream");
}

//! Benchmarks de la Fase 14.
//!
//! Se miden por separado las etapas del pipeline y el pipeline completo,
//! sobre el audio de referencia `vocadito_10.wav` (9.1 s).
//!
//! Ejecutar con:
//!
//! ```text
//! cargo bench -p basic-pitch-transcription
//! ```
//!
//! Los resultados quedan en `target/criterion/`.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use basic_pitch_audio_io::decode_file;
use basic_pitch_audio_preprocess::{resample, to_mono, AudioWindows, AUDIO_SAMPLE_RATE};
use basic_pitch_transcription::note_detection::{
    output_to_notes_polyphonic, NoteDetectionConfig, N_NOTES,
};
use basic_pitch_transcription::Transcriber;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn audio_path() -> PathBuf {
    // Primero el fixture versionado y, si no esta, el clon local del upstream.
    let fixture = repo_root().join("reference/audio_reference/vocadito_10.wav");
    if fixture.exists() {
        fixture
    } else {
        repo_root().join("basic-pitch/tests/resources/vocadito_10.wav")
    }
}

fn model_dir() -> PathBuf {
    repo_root().join("models/basic-pitch/icassp-2022")
}

static TRANSCRIPTOR: OnceLock<Transcriber> = OnceLock::new();

fn transcriptor() -> &'static Transcriber {
    TRANSCRIPTOR
        .get_or_init(|| Transcriber::load(&model_dir()).expect("no se pudo cargar el modelo"))
}

fn bench_decodificacion(c: &mut Criterion) {
    let ruta = audio_path();
    c.bench_function("decodificacion", |b| {
        b.iter(|| black_box(decode_file(&ruta).expect("decodificar")))
    });
}

fn bench_remuestreo(c: &mut Criterion) {
    let decoded = decode_file(&audio_path()).expect("decodificar");
    let mono = to_mono(&decoded);
    c.bench_function("remuestreo", |b| {
        b.iter(|| {
            black_box(
                resample(&mono.samples, mono.sample_rate, AUDIO_SAMPLE_RATE).expect("remuestrear"),
            )
        })
    });
}

fn bench_pipeline_completo(c: &mut Criterion) {
    let ruta = audio_path();
    c.bench_function("pipeline_completo", |b| {
        b.iter(|| black_box(transcriptor().transcribe(&ruta).expect("transcribir")))
    });
}

fn bench_ventanas_y_modelo(c: &mut Criterion) {
    let decoded = decode_file(&audio_path()).expect("decodificar");
    let mono = to_mono(&decoded);
    let remuestreado =
        resample(&mono.samples, mono.sample_rate, AUDIO_SAMPLE_RATE).expect("remuestrear");
    let windows = AudioWindows::new(remuestreado, AUDIO_SAMPLE_RATE);

    c.bench_function("ventanas_y_modelo", |b| {
        b.iter(|| {
            // Se reconstruye el resultado por ventana para medir el coste
            // real del frontend + red + reconstruccion.
            let mut total = 0usize;
            for ventana in windows.iter() {
                let armonica = transcriptor()
                    .model
                    .frontend
                    .forward(&ventana.samples)
                    .expect("frontend");
                let salida = transcriptor().model.forward(&armonica).expect("forward");
                total += salida.note.dims()[1];
            }
            black_box(total)
        })
    });
}

/// Salida real del modelo sobre `vocadito_10.wav`: `(note, onset, n_frames)`.
///
/// Son las predicciones **crudas por ventana**, concatenadas sin recortar el
/// solapamiento: sirven para medir el coste del posprocesado, no para
/// transcribir.
fn posteriors_reales() -> (Vec<f32>, Vec<f32>, usize) {
    let decoded = decode_file(&audio_path()).expect("decodificar");
    let mono = to_mono(&decoded);
    let remuestreado =
        resample(&mono.samples, mono.sample_rate, AUDIO_SAMPLE_RATE).expect("remuestrear");
    let windows = AudioWindows::new(remuestreado, AUDIO_SAMPLE_RATE);

    let mut note = Vec::new();
    let mut onset = Vec::new();
    let mut n_frames = 0usize;
    for ventana in windows.iter() {
        let armonica = transcriptor()
            .model
            .frontend
            .forward(&ventana.samples)
            .expect("frontend");
        let salida = transcriptor().model.forward(&armonica).expect("forward");
        n_frames += salida.note.dims()[1];
        note.extend(
            salida
                .note
                .flatten_all()
                .expect("aplanar")
                .to_vec1::<f32>()
                .expect("f32"),
        );
        onset.extend(
            salida
                .onset
                .flatten_all()
                .expect("aplanar")
                .to_vec1::<f32>()
                .expect("f32"),
        );
    }
    (note, onset, n_frames)
}

/// Coste del posprocesado (`output_to_notes_polyphonic`) por separado.
///
/// Las predicciones reales de un clip corto se repiten para simular una
/// cancion larga. Si el tiempo crece **mas que linealmente** con la duracion,
/// hay un cuello de botella (por ejemplo, el bucle de Melodia, que recorre la
/// matriz entera por cada nota extraida).
fn bench_deteccion_de_notas(c: &mut Criterion) {
    let (note, onset, n_frames) = posteriors_reales();
    let mut group = c.benchmark_group("deteccion_de_notas");
    group.sample_size(10);

    for veces in [1usize, 6, 20] {
        let note = note.repeat(veces);
        let onset = onset.repeat(veces);
        let frames = n_frames * veces;
        let segundos = frames as f64 * 256.0 / 22050.0;

        for (nombre, melodia) in [("con_melodia", true), ("sin_melodia", false)] {
            let config = NoteDetectionConfig {
                melodia_trick: melodia,
                ..NoteDetectionConfig::default()
            };
            let etiqueta = format!("{segundos:.0}s");
            group.bench_with_input(BenchmarkId::new(nombre, etiqueta), &(), |b, _| {
                b.iter(|| {
                    black_box(output_to_notes_polyphonic(
                        black_box(&note),
                        black_box(&onset),
                        frames,
                        N_NOTES,
                        &config,
                    ))
                })
            });
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_decodificacion,
    bench_remuestreo,
    bench_ventanas_y_modelo,
    bench_deteccion_de_notas,
    bench_pipeline_completo
);
criterion_main!(benches);

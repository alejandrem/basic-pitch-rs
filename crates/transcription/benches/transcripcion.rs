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
use basic_pitch_transcription::Transcriber;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn audio_path() -> PathBuf {
    repo_root().join("basic-pitch/tests/resources/vocadito_10.wav")
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

criterion_group!(
    benches,
    bench_decodificacion,
    bench_remuestreo,
    bench_ventanas_y_modelo,
    bench_pipeline_completo
);
criterion_main!(benches);

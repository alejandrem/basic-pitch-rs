//! Orquestacion de la inferencia sobre archivos de audio.

use std::path::Path;

use basic_pitch_audio_io::decode_file;
use basic_pitch_audio_preprocess::{
    resample, to_mono, AudioWindows, AUDIO_N_SAMPLES, AUDIO_SAMPLE_RATE, FFT_HOP, HOP_SIZE,
    INITIAL_PADDING,
};
use basic_pitch_model::BasicPitchModel;

use crate::note_detection::{output_to_notes_polyphonic, NoteEvent, N_NOTES};
use crate::stitching::{
    expected_frames, frame_to_time, ModelUnwrapper, ReconstructedOutput, N_OVERLAPPING_FRAMES,
};
use crate::InferenceConfig;

/// Error durante la transcripcion.
#[derive(Debug, thiserror::Error)]
pub enum TranscriptionError {
    /// Error al decodificar el audio.
    #[error("error de audio: {0}")]
    Audio(#[from] basic_pitch_audio_io::DecodeError),

    /// Error al remuestrear.
    #[error("error de remuestreo: {0}")]
    Resample(#[from] basic_pitch_audio_preprocess::ResampleError),

    /// Error del modelo.
    ///
    /// Las variantes grandes van encuadradas (`Box`) para que el enum no
    /// ocupe mas de lo necesario en cada `Result` que lo devuelve.
    #[error("error del modelo: {0}")]
    Model(#[source] Box<basic_pitch_model::ModelError>),

    /// Error del frontend acustico (CQT, NormalizedLog, BatchNorm o
    /// Harmonic Stacking).
    #[error("error del frontend acustico: {0}")]
    Frontend(#[source] Box<basic_pitch_model::FrontendError>),

    /// Error de Candle.
    #[error("error de Candle: {0}")]
    Candle(#[from] candle_core::Error),

    /// Error en la reconstruccion temporal.
    #[error("error en la reconstruccion temporal: {0}")]
    Stitching(#[from] crate::stitching::StitchingError),

    /// El archivo no existe.
    #[error("no se encontro el archivo: {0}")]
    NotFound(String),

    /// Error al escribir la salida.
    #[error("error de salida: {0}")]
    Io(#[from] std::io::Error),
}

impl From<basic_pitch_model::ModelError> for TranscriptionError {
    fn from(error: basic_pitch_model::ModelError) -> Self {
        TranscriptionError::Model(Box::new(error))
    }
}

impl From<basic_pitch_model::FrontendError> for TranscriptionError {
    fn from(error: basic_pitch_model::FrontendError) -> Self {
        TranscriptionError::Frontend(Box::new(error))
    }
}

/// El resultado de una transcripcion.
#[derive(Debug, Clone)]
pub struct TranscriptionResult {
    /// Eventos de nota detectados, ordenados por tiempo.
    pub notes: Vec<NoteEvent>,
    /// Duracion del audio, en segundos.
    pub duration_seconds: f64,
}

/// El motor de transcripcion.
#[derive(Debug)]
pub struct Transcriber {
    /// Modelo neuronal.
    pub model: BasicPitchModel,
    /// Configuracion de la inferencia.
    pub config: InferenceConfig,
}

impl Transcriber {
    /// Crea un transcriptor a partir de un modelo ya cargado.
    pub fn new(model: BasicPitchModel, config: InferenceConfig) -> Self {
        Self { model, config }
    }

    /// Carga el modelo desde un directorio y crea el transcriptor.
    pub fn load(model_dir: &Path) -> Result<Self, TranscriptionError> {
        let model = basic_pitch_model::load_model(model_dir)?;
        Ok(Self::new(model, InferenceConfig::default()))
    }

    /// Transcribe un archivo de audio.
    ///
    /// La cadena de operaciones es la de la referencia:
    /// decodificar -> mono (promedio) -> remuestrar -> ventanas.
    pub fn transcribe(&self, audio_path: &Path) -> Result<TranscriptionResult, TranscriptionError> {
        if !audio_path.exists() {
            return Err(TranscriptionError::NotFound(
                audio_path.display().to_string(),
            ));
        }

        // 1. Decodificacion.
        let decoded = decode_file(audio_path)?;

        // 2. Conversion a mono por promedio, ANTES del remuestreo.
        let mono = to_mono(&decoded);

        // 3. Remuestreo a 22.050 Hz con soxr.
        let resampled = resample(&mono.samples, mono.sample_rate, AUDIO_SAMPLE_RATE)?;

        let duration_seconds = resampled.len() as f64 / AUDIO_SAMPLE_RATE as f64;

        // 4. Ventaneo.
        let windows = AudioWindows::new(resampled, AUDIO_SAMPLE_RATE);

        // 5. Inferencia y reconstruccion.
        let predictions = self.run_model(&windows)?;
        let notes = self.to_notes(&predictions, windows.original_length());

        Ok(TranscriptionResult {
            notes,
            duration_seconds,
        })
    }

    /// Ejecuta el modelo sobre cada ventana y reconstruye las predicciones.
    ///
    /// Cada ventana pasa por el frontend acustico completo:
    ///
    /// ```text
    /// audio (43844)
    ///   -> CQT              (1, 172, 309)
    ///   -> NormalizedLog    (1, 172, 309)
    ///   -> BatchNorm inicial(1, 172, 309)
    ///   -> HarmonicStacking (1, 8, 172, 264)   NCHW
    ///   -> BasicPitchModel::forward(...)        -> note, onset, contour
    /// ```
    ///
    /// # Por que no se acumulan las predicciones
    ///
    /// La prediccion de cada ventana se incorpora **inmediatamente** a la
    /// reconstruccion temporal, y asi se puede soltar. Acumularlas todas
    /// costaria 296 KiB por ventana, es decir, unos 634 MiB por hora de
    /// audio.
    ///
    /// Es el requisito del apartado 8.1 de la fase de inferencia sobre
    /// audios de duracion arbitraria.
    fn run_model(&self, windows: &AudioWindows) -> Result<ReconstructedOutput, TranscriptionError> {
        // El numero de frames que sobreviven se conoce **antes** de inferir:
        // depende solo de la longitud del audio.
        let n_frames = expected_frames(windows.original_length(), HOP_SIZE);
        let mut reconstruccion = ModelUnwrapper::new(n_frames, N_OVERLAPPING_FRAMES);

        for ventana in windows.iter() {
            // El frontend acustico produce la representacion armonica en
            // NCHW: `(1, 8, 172, 264)`.
            let armonica = self.model.frontend.forward(&ventana.samples)?;
            let salida = self.model.forward(&armonica)?;
            reconstruccion.push(&salida)?;
        }

        Ok(reconstruccion.finish()?)
    }

    /// Convierte las predicciones en eventos de nota.
    ///
    /// La deteccion polifonica es la de `output_to_notes_polyphonic` del
    /// upstream (Fase 9).
    ///
    /// Las notas se detectan sobre la salida **note** y los onsets sobre la
    /// salida **onset**: las dos tienen 88 valores por frame. El contour
    /// (264 valores) no interviene aqui: solo se usa para el pitch bend
    /// (Fase 10) y para saber cuantos frames hay.
    fn to_notes(
        &self,
        predictions: &ReconstructedOutput,
        _original_length: usize,
    ) -> Vec<NoteEvent> {
        let n_freqs = N_NOTES;
        let frames = predictions.note.flatten_all().expect("aplanar");
        let onsets = predictions.onset.flatten_all().expect("aplanar");
        let frames = frames.to_vec1::<f32>().expect("float32");
        let onsets = onsets.to_vec1::<f32>().expect("float32");
        let n_frames = frames.len() / n_freqs;

        let notas = output_to_notes_polyphonic(
            &frames,
            &onsets,
            n_frames,
            n_freqs,
            &self.config.note_detection,
        );

        // Los frames se convierten a tiempo con la conversion del upstream,
        // que incluye el desplazamiento por ventana (y su inconsistencia
        // 172/142, que hay que conservar).
        notas
            .into_iter()
            .map(|nota| {
                NoteEvent::from_amplitude(
                    frame_to_time(
                        nota.start_frame,
                        AUDIO_SAMPLE_RATE,
                        FFT_HOP,
                        AUDIO_N_SAMPLES,
                    ),
                    frame_to_time(nota.end_frame, AUDIO_SAMPLE_RATE, FFT_HOP, AUDIO_N_SAMPLES),
                    nota.pitch,
                    nota.amplitude,
                )
            })
            .collect()
    }
}

/// Devuelve la longitud del audio original antes del relleno inicial.
pub fn original_length(resampled_len: usize) -> usize {
    resampled_len.saturating_sub(INITIAL_PADDING)
}

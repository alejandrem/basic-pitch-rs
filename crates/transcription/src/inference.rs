//! Orquestacion de la inferencia sobre archivos de audio.

use std::path::Path;

use basic_pitch_audio_io::decode_file;
use basic_pitch_audio_preprocess::{
    resample, to_mono, AudioWindows, AUDIO_SAMPLE_RATE, HOP_SIZE, INITIAL_PADDING,
};
use basic_pitch_model::{BasicPitchModel, ModelOutput};

use crate::note_detection::NoteEvent;
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
    #[error("error del modelo: {0}")]
    Model(#[from] basic_pitch_model::ModelError),

    /// Error de Candle.
    #[error("error de Candle: {0}")]
    Candle(#[from] candle_core::Error),

    /// El archivo no existe.
    #[error("no se encontro el archivo: {0}")]
    NotFound(String),

    /// Error al escribir la salida.
    #[error("error de salida: {0}")]
    Io(#[from] std::io::Error),
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

    /// Ejecuta el modelo sobre cada ventana y concatena las predicciones.
    ///
    /// # Pendiente (Fase 4 / P2)
    ///
    /// La red neuronal `BasicPitchModel::forward` recibe la **representacion
    /// armonica**, no el audio: el frontend acustico (CQT -> NormalizedLog ->
    /// BatchNorm inicial -> Harmonic Stacking) debe ejecutarse antes. Ese
    /// frontend todavia no esta implementado, asi que aqui solo se deja
    /// constancia del contrato.
    fn run_model(&self, _windows: &AudioWindows) -> Result<Vec<ModelOutput>, TranscriptionError> {
        // Cuando exista el frontend acustico, este es el sitio:
        //
        //   let cqt          = cqt::transform(&window.samples, &kernels)?;
        //   let normalized   = cqt::normalized_log(&cqt)?;
        //   let batch_normed = batch_norm(&normalized, &input_bn_params)?;
        //   let harmonic     = harmonic_stacking(&batch_normed, &harmonics, 3, 264)?;
        //   let output       = self.model.forward(&harmonic)?;
        //
        // Todo en NCHW: `harmonic` es `(1, 8, 172, 264)`.
        Ok(Vec::new())
    }

    /// Convierte las predicciones en eventos de nota.
    ///
    /// La deteccion polifonica completa se implementa en la Fase 9; aqui se
    /// dejan los parametros y la estructura prepared.
    fn to_notes(&self, _predictions: &[ModelOutput], _original_length: usize) -> Vec<NoteEvent> {
        // La implementacion completa llega en la Fase 9.
        Vec::new()
    }
}

/// Devuelve la longitud del audio original antes del relleno inicial.
pub fn original_length(resampled_len: usize) -> usize {
    resampled_len.saturating_sub(INITIAL_PADDING)
}

/// Desplazamiento entre ventanas consecutivas, en muestras.
pub fn window_hop() -> usize {
    HOP_SIZE
}

//! Orquestacion de la inferencia sobre archivos de audio.

use std::path::Path;

use basic_pitch_audio_io::decode_file;
use basic_pitch_audio_preprocess::{
    resample, to_mono, AudioWindows, AUDIO_N_SAMPLES, AUDIO_SAMPLE_RATE, FFT_HOP, HOP_SIZE,
    INITIAL_PADDING,
};
use basic_pitch_model::BasicPitchModel;
use candle_core::{Device, Tensor};

use crate::note_detection::{output_to_notes_polyphonic, NoteEvent, N_NOTES};
use crate::pitch_bend::PitchBendEstimator;
use crate::stitching::{
    expected_frames, frame_to_time, ModelUnwrapper, ReconstructedOutput, N_OVERLAPPING_FRAMES,
};
use crate::InferenceConfig;
use basic_pitch_midi_io::MidiWriteError;

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

    /// Error al generar el MIDI.
    #[error("error al generar el MIDI: {0}")]
    Midi(#[from] MidiWriteError),
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
    /// Archivo MIDI generado en memoria, listo para guardar o enviar.
    pub midi_data: Vec<u8>,
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

    /// Dispositivo de ejecucion del modelo.
    ///
    /// La implementacion actual es CPU-only.
    pub fn device(&self) -> &Device {
        &self.model.device
    }

    /// Carga el modelo desde un directorio y crea el transcriptor.
    pub fn load(model_dir: &Path) -> Result<Self, TranscriptionError> {
        let model = basic_pitch_model::load_model(model_dir)?;
        Ok(Self::new(model, InferenceConfig::default()))
    }

    /// Carga el modelo desde un directorio, indicando el dispositivo de
    /// ejecucion.
    ///
    /// La implementacion actual es CPU-only: el parametro se acepta para
    /// mantener la firma de la API estable (Fase 12) y se reserva para
    /// futuras versiones con backend acelerado de Candle. Si se pasa un
    /// dispositivo distinto de la CPU, se usa la CPU.
    pub fn load_with_device(model_dir: &Path, _device: Device) -> Result<Self, TranscriptionError> {
        Self::load(model_dir)
    }

    /// Transcribe un archivo de audio.
    ///
    /// La cadena de operaciones es la de la referencia:
    /// decodificar -> mono (promedio) -> remuestrar -> ventanas.
    pub fn transcribe<P: AsRef<Path>>(
        &self,
        audio_path: P,
    ) -> Result<TranscriptionResult, TranscriptionError> {
        let audio_path = audio_path.as_ref();
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

        // Generacion del MIDI en memoria (Fase 12): el resultado lleva los
        // bytes listos para guardarlos o enviarlos sin tocar el sistema de
        // archivos. El escritor se configura con el tempo y la politica de
        // pitch bends de la configuracion.
        let mut writer = basic_pitch_midi_io::MidiWriter::new();
        writer.tempo = self.config.midi_tempo;
        writer.multiple_pitch_bends = self.config.multiple_pitch_bends;
        let midi_data = writer.to_bytes(&notes)?;

        Ok(TranscriptionResult {
            notes,
            midi_data,
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
        let frames = predictions
            .note
            .flatten_all()
            .expect("aplanar")
            .to_vec1::<f32>()
            .expect("float32");
        let onsets = predictions
            .onset
            .flatten_all()
            .expect("aplanar")
            .to_vec1::<f32>()
            .expect("float32");
        let n_frames = frames.len() / n_freqs;

        let notas = output_to_notes_polyphonic(
            &frames,
            &onsets,
            n_frames,
            n_freqs,
            &self.config.note_detection,
        );

        // --- El pitch bend (Fase 10) ---
        //
        // La salida de contornos tiene tres bins por semitono, con lo que
        // se puede estimar como se mueve la afinacion DENTRO de una nota.
        //
        // El estimador trabaja con la matriz de contornos por FILAS (una
        // por frame), asi que se construye desde la salida reconstruida.
        let contornos = a_filas(&predictions.contour, basic_pitch_cqt::N_FREQ_BINS_CONTOURS);
        let estimador = PitchBendEstimator::default();

        // Las notas con sus bends, en el formato que espera el upstream:
        // (inicio, final, pitch, amplitud, bends).
        let con_bends: Vec<NotaConBend> = notas
            .into_iter()
            .map(|nota| {
                let inicio = frame_to_time(
                    nota.start_frame,
                    AUDIO_SAMPLE_RATE,
                    FFT_HOP,
                    AUDIO_N_SAMPLES,
                );
                let final_ =
                    frame_to_time(nota.end_frame, AUDIO_SAMPLE_RATE, FFT_HOP, AUDIO_N_SAMPLES);
                // Los bends se guardan en unidades de 1/3 de semitono,
                // que es lo que devuelve el upstream en
                // `get_pitch_bends`. La conversion a ticks de MIDI
                // ocurre al escribir el fichero, que es donde la hace el
                // upstream tambien (`note_events_to_midi`).
                let shifts =
                    estimador.estimate(&contornos, nota.start_frame, nota.end_frame, nota.pitch);
                let bends = if shifts.is_empty() {
                    None
                } else {
                    Some(shifts)
                };
                (inicio, final_, nota.pitch, nota.amplitude, bends)
            })
            .collect();

        con_bends
            .into_iter()
            .map(|(inicio, final_, pitch, amplitud, bends)| {
                NoteEvent::from_amplitude(inicio, final_, pitch, amplitud).with_pitch_bend(bends)
            })
            .collect()
    }
}

/// Una nota con su secuencia de pitch bend.
///
/// El tipo es el intermedio entre la deteccion (que produce notas en
/// frames) y el evento musical (que lleva tiempos y velocidad).
type NotaConBend = (f64, f64, u8, f32, Option<Vec<i16>>);

/// Convierte un tensor `(n_frames, n_freqs)` en filas.
fn a_filas(tensor: &Tensor, n_freqs: usize) -> Vec<Vec<f32>> {
    let datos = tensor
        .flatten_all()
        .expect("aplanar")
        .to_vec1::<f32>()
        .expect("float32");
    datos
        .chunks_exact(n_freqs)
        .map(|fila| fila.to_vec())
        .collect()
}

/// Devuelve la longitud del audio original antes del relleno inicial.
pub fn original_length(resampled_len: usize) -> usize {
    resampled_len.saturating_sub(INITIAL_PADDING)
}

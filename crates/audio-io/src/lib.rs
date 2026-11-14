//! Decodificacion de audio para `basic-pitch-rs`.
//!
//! Este crate convierte un archivo de audio en muestras `f32` en memoria. No
//! aplica ninguna transformacion: el remuestreo a 22.050 Hz y la conversion a
//! mono se realizan despues, en `basic-pitch-audio-preprocess`, **en ese
//! orden**, porque es el orden que usa la referencia.
//!
//! # Formatos
//!
//! Se soportan los formatos que decodifica `symphonia`: WAV, FLAC, OGG, MP3 y
//! AAC/M4A. La extension del archivo no garantiza nada: el contenido se
//! identifica por su firma.

use std::fs::File;
use std::path::Path;

use symphonia::core::audio::{SampleBuffer, SignalSpec};
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

/// Error al decodificar un archivo de audio.
#[derive(Debug, thiserror::Error)]
pub enum DecodeError {
    /// No se pudo abrir el archivo.
    #[error("no se pudo abrir el archivo de audio: {0}")]
    Io(#[from] std::io::Error),

    /// El formato no es compatible con el decodificador.
    #[error("formato de audio no soportado: {0}")]
    UnsupportedFormat(String),

    /// Error durante la decodificacion.
    #[error("error de decodificacion: {0}")]
    Decode(#[from] SymphoniaError),
}

/// Una senal de audio en memoria, todavia sin transformar.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioBuffer {
    /// Muestras intercaladas por canal, en `f32`.
    ///
    /// Si `channels == 1`, el vector es mono plano. Si hay varios canales, los
    /// valores de un mismo instante son contiguos (intercalado).
    pub samples: Vec<f32>,
    /// Frecuencia de muestreo original, en Hz.
    pub sample_rate: u32,
    /// Numero de canales.
    pub channels: u16,
}

impl AudioBuffer {
    /// Crea un buffer mono a partir de muestras planas.
    pub fn mono(samples: Vec<f32>, sample_rate: u32) -> Self {
        Self {
            samples,
            sample_rate,
            channels: 1,
        }
    }

    /// Numero de muestras por canal.
    pub fn frames(&self) -> usize {
        if self.channels == 0 {
            0
        } else {
            self.samples.len() / self.channels as usize
        }
    }

    /// Devuelve el canal `index` de una senal multicanal.
    ///
    /// Devuelve `None` si el canal no existe.
    pub fn channel(&self, index: u16) -> Option<Vec<f32>> {
        if index >= self.channels {
            return None;
        }
        let ch = index as usize;
        Some(
            self.samples
                .iter()
                .skip(ch)
                .step_by(self.channels as usize)
                .copied()
                .collect(),
        )
    }
}

/// Decodifica un archivo de audio a muestras `f32`.
///
/// La decodificacion conserva los canales originales y la frecuencia de
/// muestreo original: ni el remuestreo ni la mezcla a mono se aplican aqui.
pub fn decode_file(path: &Path) -> Result<AudioBuffer, DecodeError> {
    let file = File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let hint = Hint::new();
    let probed = symphonia::default::get_probe().format(
        &hint,
        mss,
        &FormatOptions::default(),
        &MetadataOptions::default(),
    )?;
    let mut format = probed.format;

    let track = format
        .default_track()
        .ok_or_else(|| DecodeError::UnsupportedFormat("el archivo no tiene pistas".into()))?;
    let track_id = track.id;
    let params = track.codec_params.clone();

    let sample_rate = params.sample_rate.unwrap_or(0);
    let channels = params.channels.map(|c| c.count() as u16).unwrap_or(1);

    let mut decoder = symphonia::default::get_codecs().make(&params, &DecoderOptions::default())?;

    let mut samples: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(SymphoniaError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                break;
            }
            Err(e) => return Err(DecodeError::Decode(e)),
        };

        if packet.track_id() != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(decoded) => {
                // Se copia cada paquete a un buffer con capacidad suficiente
                // y se anaden sus muestras intercaladas al resultado.
                let frames = decoded.frames();
                if frames == 0 {
                    continue;
                }
                let spec = SignalSpec::new(decoded.spec().rate, decoded.spec().channels);
                let mut buffer = SampleBuffer::<f32>::new(frames as u64, spec);
                buffer.copy_interleaved_ref(decoded);
                samples.extend_from_slice(buffer.samples());
            }
            Err(SymphoniaError::DecodeError(_)) => {
                // Un paquete corrupto se salta; no aborta la decodificacion.
                tracing::warn!("paquete de audio omitido por error de decodificacion");
            }
            Err(SymphoniaError::ResetRequired) => {
                decoder.reset();
            }
            Err(e) => return Err(DecodeError::Decode(e)),
        }
    }

    Ok(AudioBuffer {
        samples,
        sample_rate,
        channels,
    })
}

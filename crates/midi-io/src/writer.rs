//! Escritura de archivos MIDI.
//!
//! # Convenciones
//!
//! - Resolucion: 220 ticks por negra (la que usa `pretty_midi` por defecto).
//! - Tempo inicial: 120 BPM, igual que `DEFAULT_MINIMUM_MIDI_TEMPO`.
//! - Programa: `Electric Piano 1` (programa GM 4), como en la referencia.
//! - Pitch bend: valor de 14 bits con 8192 como centro; el rango util son
//!   +-2 semitonas.
//!
//! Cuando hay varias notas simultaneas con pitch bend, el formato MIDI
//! tradicional aplica el bend por canal, no por nota. La referencia lo
//! resuelve asignando un instrumento (y por tanto un canal) distinto a cada
//! nota MIDI distinta. Este comportamiento se reproduce en
//! [`MidiWriter::with_multiple_pitch_bends`].

use std::path::Path;

use midly::num::{u14, u15, u24, u28, u4, u7};
use midly::{
    Format, Header, MetaMessage, MidiMessage, PitchBend, Smf, Timing, Track, TrackEvent,
    TrackEventKind,
};

use crate::events::{EventError, NoteEvent};

/// Error al escribir un archivo MIDI.
#[derive(Debug, thiserror::Error)]
pub enum MidiWriteError {
    /// Error de E/S.
    #[error("error de E/S: {0}")]
    Io(#[from] std::io::Error),

    /// Error de `midly`.
    #[error("error de MIDI: {0}")]
    Midly(#[from] midly::Error),

    /// Un evento no es representable en MIDI.
    #[error("evento invalido: {0}")]
    Event(#[from] EventError),
}

/// Resolucion por defecto, en ticks por negra.
///
/// Es la que usa `pretty_midi`, con la que la referencia genera sus archivos.
pub const DEFAULT_RESOLUTION: u16 = 220;

/// Tempo inicial por defecto, en pulsos por minuto.
pub const DEFAULT_TEMPO: f64 = 120.0;

/// Programa GM de `Electric Piano 1`.
pub const ELECTRIC_PIANO_1: u8 = 4;

/// Valor de pitch bend que representa el centro.
pub const PITCH_BEND_CENTER: i16 = 8192;

/// Escribe eventos musicales en un archivo MIDI.
#[derive(Debug, Clone)]
pub struct MidiWriter {
    /// Resolucion, en ticks por negra.
    pub resolution: u16,
    /// Tempo inicial, en pulsos por minuto.
    pub tempo: f64,
    /// Si se asigna un canal distinto a cada nota MIDI.
    ///
    /// Es lo que hace la referencia cuando `multiple_pitch_bends` es verdadero.
    pub multiple_pitch_bends: bool,
}

impl Default for MidiWriter {
    fn default() -> Self {
        Self {
            resolution: DEFAULT_RESOLUTION,
            tempo: DEFAULT_TEMPO,
            multiple_pitch_bends: false,
        }
    }
}

impl MidiWriter {
    /// Crea un escritor con la configuracion por defecto.
    pub fn new() -> Self {
        Self::default()
    }

    /// Habilita los pitch bends simultaneos.
    ///
    /// Con esta opcion, cada nota MIDI distinta recibe su propio canal, de
    /// modo que dos notas superpuestas pueden tener bends diferentes. El
    /// formato MIDI tiene 16 canales, asi que como mucho 16 notas distintas
    /// pueden tener bend simultaneo.
    pub fn with_multiple_pitch_bends(mut self, enabled: bool) -> Self {
        self.multiple_pitch_bends = enabled;
        self
    }

    /// Convierte segundos en ticks.
    pub fn time_to_ticks(&self, seconds: f64) -> u32 {
        let ticks = seconds * self.tempo / 60.0 * self.resolution as f64;
        if ticks < 0.0 {
            0
        } else {
            ticks.round() as u32
        }
    }

    /// Construye el archivo MIDI a partir de los eventos.
    pub fn build(&self, notes: &[NoteEvent]) -> Result<Smf<'_>, MidiWriteError> {
        for note in notes {
            note.validate()?;
        }

        let header = Header::new(
            Format::Parallel,
            Timing::Metrical(u15::new(self.resolution)),
        );

        // Se agrupan las notas por canal.
        //
        // Con `multiple_pitch_bends`, cada nota MIDI tiene su propio canal
        // (como en la referencia). Sin ellos, todas las notas van al canal 0.
        let mut tracks: Vec<Vec<(u32, TrackEventKind<'static>)>> = vec![Vec::new()];

        // Evento inicial de tempo, en el primer track.
        let tempo_us_per_beat = (60_000_000.0 / self.tempo) as u32;
        tracks[0].push((
            0,
            TrackEventKind::Meta(MetaMessage::Tempo(u24::new(tempo_us_per_beat))),
        ));
        tracks[0].push((
            0,
            TrackEventKind::Meta(MetaMessage::TimeSignature(4, 2, 24, 8)),
        ));

        for note in notes {
            let channel = self.channel_for(note.pitch);
            let track_index = if self.multiple_pitch_bends {
                channel as usize
            } else {
                0
            };
            while tracks.len() <= track_index {
                tracks.push(Vec::new());
            }

            let start = self.time_to_ticks(note.start_time);
            let end = self.time_to_ticks(note.end_time);
            let ch = u4::new(channel % 16);
            let key = u7::new(note.pitch);
            let velocity = u7::new(note.velocity);

            let mut events = Vec::new();
            events.push((
                start,
                TrackEventKind::Midi {
                    channel: ch,
                    message: MidiMessage::NoteOn { key, vel: velocity },
                },
            ));
            events.push((
                end,
                TrackEventKind::Midi {
                    channel: ch,
                    message: MidiMessage::NoteOff {
                        key,
                        vel: u7::new(0),
                    },
                },
            ));

            // Pitch bend: un evento por punto de la secuencia.
            if let Some(bends) = &note.pitch_bends {
                let count = bends.len().max(1);
                for (i, bend) in bends.iter().enumerate() {
                    let t = note.start_time + (note.duration() * i as f64 / count as f64);
                    let value = PITCH_BEND_CENTER + *bend;
                    let value = value.clamp(0, 16383) as u16;
                    events.push((
                        self.time_to_ticks(t),
                        TrackEventKind::Midi {
                            channel: ch,
                            message: MidiMessage::PitchBend {
                                bend: PitchBend(u14::new(value)),
                            },
                        },
                    ));
                }
            }

            tracks[track_index].extend(events);
        }

        // Se ordenan los eventos por tiempo dentro de cada track y se
        // convierten a deltas relativos, que es lo que exige el formato MIDI.
        let mut tracks_out: Vec<Track<'static>> = Vec::new();
        for mut events in tracks {
            events.sort_by_key(|(tick, _)| *tick);
            let mut track = Track::new();
            let mut previous = 0u32;
            for (tick, event) in events {
                let delta = tick.saturating_sub(previous);
                previous = tick;
                track.push(TrackEvent {
                    delta: u28::new(delta),
                    kind: event,
                });
            }
            tracks_out.push(track);
        }

        let mut smf = Smf::new(header);
        smf.tracks = tracks_out;

        Ok(smf)
    }

    /// Canal asignado a una nota MIDI.
    fn channel_for(&self, pitch: u8) -> u8 {
        if self.multiple_pitch_bends {
            // Cada nota MIDI distinta recibe un canal distinto, hasta el
            // limite de 16 canales del formato MIDI.
            (pitch.saturating_sub(21)) % 16
        } else {
            0
        }
    }

    /// Escribe los eventos en un archivo.
    pub fn write(&self, notes: &[NoteEvent], path: &Path) -> Result<(), MidiWriteError> {
        let smf = self.build(notes)?;
        smf.save(path)?;
        Ok(())
    }
}

/// Reexportado por comodidad.
pub use crate::events::amplitude_to_velocity;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn los_segundos_se_convierten_en_ticks() {
        let writer = MidiWriter::new();
        // 1 segundo a 120 BPM con 220 ticks por negra son 440 ticks.
        assert_eq!(writer.time_to_ticks(1.0), 440);
        assert_eq!(writer.time_to_ticks(0.5), 220);
        assert_eq!(writer.time_to_ticks(-1.0), 0);
    }

    #[test]
    fn se_construye_un_midi_con_notas() {
        let notes = vec![
            NoteEvent::new(0.0, 1.0, 60, 100),
            NoteEvent::new(0.5, 1.5, 64, 90),
        ];
        let writer = MidiWriter::new();
        let smf = writer.build(&notes).unwrap();
        assert_eq!(smf.tracks.len(), 1);
    }

    #[test]
    fn un_evento_invalido_produce_error() {
        let notes = vec![NoteEvent::new(0.0, 1.0, 200, 100)];
        assert!(MidiWriter::new().build(&notes).is_err());
    }
}

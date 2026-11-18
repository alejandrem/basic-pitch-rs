//! Escritura de archivos MIDI.
//!
//! # Convenciones
//!
//! - Resolucion: 220 ticks por negra (la que usa `pretty_midi` por defecto).
//! - Tempo inicial: 120 BPM, igual que `DEFAULT_MINIMUM_MIDI_TEMPO`.
//! - Programa: `Acoustic Grand Piano` (programa GM **0**).
//!
//!   La referencia usa `Electric Piano 1` (GM 4), pero aqui se usa el piano
//!   acustico por decision del proyecto: es lo que suena mas natural en un
//!   archivo MIDIExportado, y es lo que usaba el MIDI de origen.
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

/// Valor de pitch bend que representa el centro.
///
/// Escala de conversion de los bends a ticks de MIDI.
pub const PITCH_BEND_SCALE: f64 = 4096.0;

/// Bins por semitono: los bends llegan en unidades de 1/3 de semitono.
pub const CONTOURS_BINS_PER_SEMITONE: usize = 3;

/// Limite de los ticks de pitch bend (unas dos semitonas).
pub const N_PITCH_BEND_TICKS: i64 = 8192;

/// Quita los pitch bends de las notas que se solapan en el tiempo.
///
/// El formato MIDI aplica el pitch bend **por canal**, no nota a nota. El
/// upstream resuelve el caso quitando los bends de las notas que se
/// solapan (salvo que se pida explicitamente `multiple_pitch_bends`, que
/// asigna un canal a cada nota).
///
/// El criterio es el del upstream: se ordenan las notas por
/// `(inicio, final, pitch)` y, para cada una, se marca sin bends a todas
/// las siguientes que empiezan antes de que acabe.
///
/// `notas` se devuelve MODIFICADA y en el mismo orden en que estaba: el
/// orden de las notas no cambia, solo sus bends.
pub fn drop_overlapping_pitch_bends(notas: &mut [NoteEvent]) {
    let mut orden: Vec<usize> = (0..notas.len()).collect();
    orden.sort_by(|a, b| {
        let x = &notas[*a];
        let y = &notas[*b];
        x.start_time
            .partial_cmp(&y.start_time)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(
                x.end_time
                    .partial_cmp(&y.end_time)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
            .then(x.pitch.cmp(&y.pitch))
    });

    for (pos, i) in orden.iter().enumerate() {
        for j in &orden[pos + 1..] {
            if notas[*j].start_time >= notas[*i].end_time {
                // La nota j empieza despues de que acaba la i.
                break;
            }
            notas[*i].pitch_bend = None;
            notas[*j].pitch_bend = None;
        }
    }
}
pub const PITCH_BEND_CENTER: i16 = 8192;

/// Programa General MIDI por defecto: `Acoustic Grand Piano` (GM 0).
///
/// Decision del proyecto. La referencia usa `Electric Piano 1` (GM 4).
pub const DEFAULT_PROGRAM: u8 = 0;

/// Escribe eventos musicales en un archivo MIDI.
#[derive(Debug, Clone)]
pub struct MidiWriter {
    /// Resolucion, en ticks por negra.
    pub resolution: u16,
    /// Tempo inicial, en pulsos por minuto.
    pub tempo: f64,
    /// Programa General MIDI del instrumento.
    ///
    /// Por defecto, `Acoustic Grand Piano` (programa GM **0**).
    ///
    /// La referencia usa `Electric Piano 1` (GM 4):
    ///
    /// ```python
    /// piano_program = pretty_midi.instrument_name_to_program("Electric Piano 1")
    /// ```
    ///
    /// Aqui se usa el piano acustico por decision del proyecto. Sin un
    /// evento de cambio de programa, el reproductor usaria el programa por
    /// defecto del canal 0, que es justamente `Acoustic Grand Piano` (GM 0):
    /// o se emite el cambio de programa, o el resultado depende de lo que
    /// haga el reproductor. Se emite, para que el MIDI sea explicito.
    pub program: u8,
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
            program: DEFAULT_PROGRAM,
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

        // El pitch bend del formato MIDI es POR CANAL, no nota a nota.
        // Salvo que se pidan bends multiples (que asignan un canal a cada
        // nota), dos notas que se solapan en el tiempo no pueden llevar
        // bends distintos: el ultimo que se escribe manda.
        //
        // El upstream hace esto en `note_events_to_midi`, que es el
        // ESCRITOR, y no en la deteccion: la lista de notas que devuelve
        // conserva todos sus bends, y es el MIDI el que los descarta.
        // Aqui igual: el drop se aplica aqui, no en la deteccion.
        let mut notas_con_bends: Vec<NoteEvent> = notes.to_vec();
        if !self.multiple_pitch_bends {
            drop_overlapping_pitch_bends(&mut notas_con_bends);
        }

        // Las notas se ordenan por tiempo de inicio, y a igualdad de
        // tiempo, por final y por pitch: es el criterio del upstream
        // (`sorted(nota_events_with_pitch_bends)`, que ordena por la tupla
        // completa).
        //
        // La deteccion de notas entrega las notas en el orden en que las
        // encuentra (los onsets, de mas reciente a mas antiguo, y luego el
        // truco de Melodia), no en orden temporal. Sin ordenar, el MIDI
        // tendria los eventos en un orden que no es el canonico.
        let mut ordenadas: Vec<&NoteEvent> = notas_con_bends.iter().collect();
        ordenadas.sort_by(|a, b| {
            a.start_time
                .partial_cmp(&b.start_time)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(
                    a.end_time
                        .partial_cmp(&b.end_time)
                        .unwrap_or(std::cmp::Ordering::Equal),
                )
                .then(a.pitch.cmp(&b.pitch))
        });

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

        // Cambio de programa: sin el, el reproductor usaria el programa por
        // defecto del canal (Acoustic Grand Piano) en vez del de la
        // referencia (Electric Piano 1).
        tracks[0].push((
            0,
            TrackEventKind::Midi {
                channel: u4::new(0),
                message: MidiMessage::ProgramChange {
                    program: u7::new(self.program),
                },
            },
        ));

        for note in &ordenadas {
            let note = *note;
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
            //
            // Los bends llegan en unidades de 1/3 de semitono (que es lo
            // que produce la deteccion) y se convierten a ticks de MIDI
            // aqui, igual que hace el upstream en `note_events_to_midi`:
            //
            //   ticks = round(shift * 4096 / 3), recortados a +-8192
            //
            // El recorte limita los bends a unas dos semitonas.
            if let Some(bends) = &note.pitch_bend {
                let count = bends.len().max(1);
                for (i, bend) in bends.iter().enumerate() {
                    // Los tiempos se reparten uniformemente entre el
                    // inicio y el final de la nota, como el
                    // `np.linspace(start_time, end_time, len(pitch_bend))`
                    // del upstream. OJO: el ultimo punto es
                    // EXACTAMENTE el final, asi que el divisor es
                    // `count - 1` y no `count`.
                    let t = if count > 1 {
                        note.start_time + (note.duration() * i as f64 / (count - 1) as f64)
                    } else {
                        note.start_time
                    };
                    let ticks = ((*bend as f64 * PITCH_BEND_SCALE
                        / CONTOURS_BINS_PER_SEMITONE as f64)
                        .round() as i64)
                        .clamp(-N_PITCH_BEND_TICKS, N_PITCH_BEND_TICKS - 1);
                    let value = (PITCH_BEND_CENTER as i64 + ticks).clamp(0, 16383) as u16;
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

    /// Construye el archivo MIDI y lo devuelve como bytes.
    pub fn to_bytes(&self, notes: &[NoteEvent]) -> Result<Vec<u8>, MidiWriteError> {
        let smf = self.build(notes)?;
        let mut bytes = Vec::new();
        smf.write_std(&mut bytes)?;
        Ok(bytes)
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

    #[test]
    fn el_programa_por_defecto_es_el_piano_acustico() {
        // Decision del proyecto: piano acustico (GM 0). La referencia usa
        // Electric Piano 1 (GM 4).
        assert_eq!(DEFAULT_PROGRAM, 0);
        assert_eq!(MidiWriter::new().program, 0);
    }

    #[test]
    fn el_midi_lleva_el_cambio_de_programa() {
        let notes = vec![NoteEvent::new(0.0, 1.0, 60, 100)];
        let writer = MidiWriter::new();
        let smf = writer.build(&notes).unwrap();

        // El cambio de programa tiene que estar en el primer track.
        let mut encontrado = false;
        for evento in smf.tracks[0].iter() {
            if let TrackEventKind::Midi {
                message: MidiMessage::ProgramChange { program },
                ..
            } = &evento.kind
            {
                assert_eq!(program.as_int(), DEFAULT_PROGRAM);
                encontrado = true;
            }
        }
        assert!(encontrado, "el MIDI no lleva cambio de programa");
    }

    #[test]
    fn el_programa_se_puede_cambiar() {
        let notes = vec![NoteEvent::new(0.0, 1.0, 60, 100)];
        let writer = MidiWriter {
            program: 0, // Acoustic Grand Piano
            ..MidiWriter::new()
        };
        let smf = writer.build(&notes).unwrap();

        let mut encontrado = false;
        for evento in smf.tracks[0].iter() {
            if let TrackEventKind::Midi {
                message: MidiMessage::ProgramChange { program },
                ..
            } = &evento.kind
            {
                assert_eq!(program.as_int(), 0);
                encontrado = true;
            }
        }
        assert!(encontrado);
    }
}

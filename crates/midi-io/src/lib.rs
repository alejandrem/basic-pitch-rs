//! Lectura y escritura de archivos MIDI para `basic-pitch-rs`.
//!
//! Los eventos musicales se representan con [`events::NoteEvent`], que es
//! independiente del formato MIDI y puede reutilizarse para visualizacion,
//! edicion o pruebas.

pub mod events;
pub mod writer;

pub use events::{amplitude_to_velocity, EventError, NoteEvent};
pub use writer::{MidiWriteError, MidiWriter};

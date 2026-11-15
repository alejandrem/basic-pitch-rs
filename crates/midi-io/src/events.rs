//! Eventos musicales, independientes del formato MIDI.
//!
//! # El tipo canonico
//!
//! `NoteEvent` es **el** tipo de evento musical del proyecto: lo produce la
//! transcripcion y lo consume el escritor MIDI. Vive aqui, y no en el crate
//! de transcripcion, para que no haya dos definiciones del mismo concepto
//! que puedan divergir.
//!
//! Es el mismo criterio que sigue el upstream: en `pretty_midi`, el tipo de
//! evento y el escritor de MIDI viven juntos.
//!
//! La fase 13 del plan pide exactamente estos campos:
//!
//! ```text
//! start_time, end_time, pitch, velocity, pitch_bend
//! ```
//!
//! Se anade `amplitude`, que es el dato **original** del modelo (la
//! activacion de la nota, entre 0 y 1). La velocidad MIDI es su proyeccion
//! (`round(127 * amplitud)`, como en la referencia), y tener las dos
//! permite saber que.nota salio de una activacion de 0.51 y no de 0.52.

/// Un evento de nota listo para exportar a MIDI.
#[derive(Debug, Clone, PartialEq)]
pub struct NoteEvent {
    /// Tiempo de inicio, en segundos.
    pub start_time: f64,
    /// Tiempo de finalizacion, en segundos.
    pub end_time: f64,
    /// Nota MIDI.
    pub pitch: u8,
    /// Velocidad MIDI (0..=127).
    pub velocity: u8,
    /// Amplitud original de la activacion del modelo (0..=1).
    ///
    /// La velocidad es su proyeccion; se conserva para no perder
    /// informacion en el camino de ida y vuelta.
    pub amplitude: f32,
    /// Secuencia de pitch bend, en unidades de **1/3 de semitono**.
    ///
    /// Es la misma unidad que usa el upstream en `get_pitch_bends`. La
    /// conversion a ticks de MIDI ocurre al escribir el fichero, que es
    /// donde la hace el upstream tambien.
    ///
    /// `None` si la nota no tiene bends.
    pub pitch_bend: Option<Vec<i16>>,
}

impl NoteEvent {
    /// Crea un evento a partir de su velocidad MIDI.
    pub fn new(start_time: f64, end_time: f64, pitch: u8, velocity: u8) -> Self {
        Self {
            start_time,
            end_time,
            pitch,
            velocity,
            amplitude: velocity as f32 / 127.0,
            pitch_bend: None,
        }
    }

    /// Crea un evento a partir de la amplitud de la activacion.
    ///
    /// La velocidad es `round(127 * amplitud)`, que es lo que hace la
    /// referencia: `int(np.round(MIDI_VELOCITY_SCALE * amplitude))`.
    pub fn from_amplitude(start_time: f64, end_time: f64, pitch: u8, amplitude: f32) -> Self {
        Self {
            start_time,
            end_time,
            pitch,
            velocity: amplitude_to_velocity(amplitude),
            amplitude,
            pitch_bend: None,
        }
    }

    /// Anade una secuencia de pitch bend.
    ///
    /// Lo usara la Fase 10 (estimacion de pitch bend); por ahora el
    /// detector de notas no lo rellena.
    pub fn with_pitch_bend(mut self, pitch_bend: Option<Vec<i16>>) -> Self {
        self.pitch_bend = pitch_bend;
        self
    }

    /// Duracion del evento, en segundos.
    pub fn duration(&self) -> f64 {
        self.end_time - self.start_time
    }

    /// Valida que el evento sea representable en MIDI.
    pub fn validate(&self) -> Result<(), EventError> {
        if self.pitch > 127 {
            return Err(EventError::InvalidPitch(self.pitch));
        }
        if self.velocity > 127 {
            return Err(EventError::InvalidVelocity(self.velocity));
        }
        if self.end_time < self.start_time {
            return Err(EventError::NegativeDuration);
        }
        Ok(())
    }
}

/// Error en un evento musical.
#[derive(Debug, thiserror::Error)]
pub enum EventError {
    /// La nota MIDI esta fuera de rango.
    #[error("nota MIDI invalida: {0}")]
    InvalidPitch(u8),
    /// La velocidad esta fuera de rango.
    #[error("velocidad MIDI invalida: {0}")]
    InvalidVelocity(u8),
    /// La duracion es negativa.
    #[error("la duracion del evento es negativa")]
    NegativeDuration,
}

/// Convierte una amplitud en [0, 1] a una velocidad MIDI.
///
/// La referencia usa `int(np.round(127 * amplitud))`.
pub fn amplitude_to_velocity(amplitude: f32) -> u8 {
    let scaled = (127.0f32 * amplitude).round();
    scaled.clamp(0.0, 127.0) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn la_amplitud_se_convierte_a_velocidad() {
        assert_eq!(amplitude_to_velocity(0.0), 0);
        assert_eq!(amplitude_to_velocity(1.0), 127);
        assert_eq!(amplitude_to_velocity(0.5), 64); // round(63.5) = 64
    }

    #[test]
    fn un_evento_valido_pasa_la_validacion() {
        let event = NoteEvent::new(0.0, 1.0, 60, 100);
        assert!(event.validate().is_ok());
        assert_eq!(event.duration(), 1.0);
    }

    #[test]
    fn una_nota_fuera_de_rango_se_rechaza() {
        let event = NoteEvent::new(0.0, 1.0, 200, 100);
        assert!(matches!(
            event.validate(),
            Err(EventError::InvalidPitch(200))
        ));
    }

    #[test]
    fn el_evento_conserva_la_amplitud_original() {
        let event = NoteEvent::from_amplitude(0.0, 1.0, 60, 0.5);
        assert_eq!(event.velocity, 64);
        assert!((event.amplitude - 0.5).abs() < 1e-6);
    }

    #[test]
    fn el_pitch_bend_se_puede_añadir() {
        let event = NoteEvent::new(0.0, 1.0, 60, 100).with_pitch_bend(Some(vec![1, 2]));
        assert_eq!(event.pitch_bend, Some(vec![1, 2]));
        assert!(NoteEvent::new(0.0, 1.0, 60, 100).pitch_bend.is_none());
    }
}

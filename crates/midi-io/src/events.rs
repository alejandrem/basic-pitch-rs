//! Eventos musicales, independientes del formato MIDI.

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
    /// Secuencia de pitch bend, en ticks de MIDI (0 = centro).
    pub pitch_bends: Option<Vec<i16>>,
}

impl NoteEvent {
    /// Crea un evento sin pitch bend.
    pub fn new(start_time: f64, end_time: f64, pitch: u8, velocity: u8) -> Self {
        Self {
            start_time,
            end_time,
            pitch,
            velocity,
            pitch_bends: None,
        }
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
}

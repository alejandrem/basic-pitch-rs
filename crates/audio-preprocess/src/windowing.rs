//! Division del audio en ventanas de longitud fija.
//!
//! Reproduce el comportamiento de `basic_pitch/inference.py`:
//!
//! ```text
//! get_audio_input:
//!   1. se anteponen INITIAL_PADDING (3840) ceros al audio
//!   2. se extraen ventanas de AUDIO_N_SAMPLES (43844) muestras
//!      avanzando HOP_SIZE (36164) muestras cada vez
//!   3. la ultima ventana se completa con ceros si es corta
//!
//! unwrap_output:
//!   4. de cada ventana se descartan 15 frames por delante y 15 por detras,
//!      quedan FRAMES_KEPT_PER_WINDOW (142) frames utiles
//! ```
//!
//! El desplazamiento entre ventanas y el relleno inicial son los que
//! determinan que las notas no se dupliquen ni se pierdan en los limites.

use crate::{AUDIO_N_SAMPLES, HOP_SIZE, INITIAL_PADDING};

/// Una ventana de audio con su posicion temporal original.
#[derive(Debug, Clone, PartialEq)]
pub struct Window {
    /// Muestras de la ventana, de `AUDIO_N_SAMPLES` muestras.
    pub samples: Vec<f32>,
    /// Indice de la primera muestra de la ventana dentro del audio
    /// original.
    ///
    /// Es **negativo** para la primera ventana, porque el relleno inicial
    /// antepone `INITIAL_PADDING` ceros: la ventana 0 empieza `INITIAL_PADDING`
    /// muestras *antes* del inicio del audio original.
    pub start_sample: i64,
    /// Instante de inicio en segundos, relativo al audio original.
    pub start_time_secs: f64,
}

/// Un flujo de ventanas extraidas de una senal.
#[derive(Debug, Clone)]
pub struct AudioWindows {
    /// Muestras del audio original, ya remuestreadas y en mono.
    samples: Vec<f32>,
    /// Frecuencia de muestreo.
    sample_rate: u32,
    /// Numero de ventanas que se produciran.
    count: usize,
}

impl AudioWindows {
    /// Prepara el flujo de ventanas para una senal ya remuestreada a mono.
    ///
    /// `sample_rate` debe ser 22.050 Hz.
    pub fn new(samples: Vec<f32>, sample_rate: u32) -> Self {
        // Se anteponen los ceros de relleno inicial. A partir de aqui, el
        // indice 0 de la senal rellenada corresponde a la muestra
        // `-INITIAL_PADDING` del audio original.
        let total = samples.len() + INITIAL_PADDING;
        // `window_audio_file` itera en `range(0, len, hop_size)`, de modo que
        // el numero de ventanas es `ceil(len / hop_size)`.
        let count = if total == 0 {
            0
        } else {
            total.div_ceil(HOP_SIZE)
        };
        Self {
            samples,
            sample_rate,
            count,
        }
    }

    /// Numero de ventanas que se produciran.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Indica si no se producira ninguna ventana.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Longitud del audio original, en muestras, **antes** del relleno.
    pub fn original_length(&self) -> usize {
        self.samples.len()
    }

    /// Iterador sobre las ventanas.
    pub fn iter(&self) -> impl Iterator<Item = Window> + '_ {
        (0..self.count).map(move |index| self.window(index))
    }

    /// Construye la ventana `index`.
    fn window(&self, index: usize) -> Window {
        let offset = index * HOP_SIZE;

        // La ventana se extrae de la senal rellenada. La posicion en el audio
        // original es `offset - INITIAL_PADDING`, que puede ser negativa.
        let mut samples = Vec::with_capacity(AUDIO_N_SAMPLES);
        for i in 0..AUDIO_N_SAMPLES {
            let position = offset + i;
            let value = if position < INITIAL_PADDING {
                // Relleno inicial.
                0.0
            } else {
                let original = position - INITIAL_PADDING;
                self.samples.get(original).copied().unwrap_or(0.0)
            };
            samples.push(value);
        }

        let start_sample = offset as i64 - INITIAL_PADDING as i64;
        let start_time_secs = start_sample as f64 / self.sample_rate as f64;

        Window {
            samples,
            start_sample,
            start_time_secs,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ANNOT_N_FRAMES, FRAMES_KEPT_PER_WINDOW, N_OVERLAPPING_FRAMES};

    /// Audio con un unico valor no nulo en la posicion `pos`.
    fn impulse(len: usize, pos: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; len];
        v[pos] = 1.0;
        v
    }

    #[test]
    fn una_senal_menor_que_una_ventana_produce_una_ventana() {
        let windows = AudioWindows::new(vec![0.0; 1000], 22_050);
        assert_eq!(windows.len(), 1);
        let w = windows.iter().next().unwrap();
        assert_eq!(w.samples.len(), AUDIO_N_SAMPLES);
    }

    #[test]
    fn la_ventana_se_rellena_con_ceros_al_final() {
        let windows = AudioWindows::new(vec![1.0; 100], 22_050);
        let w = windows.iter().next().unwrap();
        assert_eq!(w.samples.len(), AUDIO_N_SAMPLES);
        // La ventana 0 empieza ANTES del audio original: los primeros
        // INITIAL_PADDING muestras son el relleno inicial.
        assert!(w.samples[INITIAL_PADDING + 99].eq(&1.0));
        assert!(w.samples[INITIAL_PADDING + 100].eq(&0.0));
    }

    #[test]
    fn el_relleno_inicial_antepone_ceros() {
        // Un impulso en la posicion 0 del audio original cae, en la senal
        // rellenada, en la posicion INITIAL_PADDING.
        let windows = AudioWindows::new(impulse(1000, 0), 22_050);
        let w = windows.iter().next().unwrap();
        assert!(w.samples[INITIAL_PADDING].eq(&1.0));
        assert!(w.samples[INITIAL_PADDING - 1].eq(&0.0));
        // Y la ventana empieza antes del audio original.
        assert_eq!(w.start_sample, -(INITIAL_PADDING as i64));
    }

    #[test]
    fn las_ventanas_se_solapan_con_el_desplazamiento_correcto() {
        let total = 100_000;
        let windows = AudioWindows::new(vec![0.0; total], 22_050);
        let all: Vec<_> = windows.iter().collect();

        // Ventanas consecutivas: el desplazamiento es HOP_SIZE.
        assert_eq!(all[1].start_sample - all[0].start_sample, HOP_SIZE as i64);

        // Y el solapamiento real es AUDIO_N_SAMPLES - HOP_SIZE = 7680.
        assert_eq!(AUDIO_N_SAMPLES - HOP_SIZE, crate::OVERLAP_LEN);
    }

    #[test]
    fn el_numero_de_ventanas_usa_el_salto_de_la_referencia() {
        // La referencia itera en `range(0, len_rellenado, hop_size)`.
        let total = 100_000;
        let expected = (total + INITIAL_PADDING).div_ceil(HOP_SIZE);
        let windows = AudioWindows::new(vec![0.0; total], 22_050);
        assert_eq!(windows.len(), expected);
    }

    #[test]
    fn se_conservan_142_frames_por_ventana() {
        // Frames emitidos por ventana frente a frames conservados.
        assert_eq!(ANNOT_N_FRAMES, 172);
        assert_eq!(FRAMES_KEPT_PER_WINDOW, 142);
        // Se descartan 15 por delante y 15 por detras.
        assert_eq!(N_OVERLAPPING_FRAMES / 2 * 2, 30);
        assert_eq!(ANNOT_N_FRAMES - FRAMES_KEPT_PER_WINDOW, 30);
    }

    #[test]
    fn una_senal_vacia_produce_una_ventana_de_silencio() {
        // Comportamiento de la referencia: `get_audio_input` antepone el
        // relleno inicial y `window_audio_file` itera en
        // `range(0, len_rellenado, hop_size)`. Con el audio vacio, la senal
        // rellenada tiene INITIAL_PADDING muestras, por lo que se produce
        // exactamente una ventana, llena de ceros.
        let windows = AudioWindows::new(Vec::new(), 22_050);
        assert_eq!(windows.len(), 1);
        assert_eq!(windows.original_length(), 0);
        let w = windows.iter().next().unwrap();
        assert_eq!(w.samples.len(), AUDIO_N_SAMPLES);
        assert!(w.samples.iter().all(|v| v.abs() < f32::EPSILON));
    }
}

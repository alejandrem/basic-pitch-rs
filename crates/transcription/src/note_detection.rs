//! Deteccion de notas musicales a partir de las predicciones del modelo.
//!
//! # Que reproduce
//!
//! `basic_pitch/note_creation.py::output_to_notes_polyphonic`, que es el
//! algoritmo de deteccion polifonica del upstream. Es un procedimiento con
//! tres pasos:
//!
//! 1. **Onsets**: se buscan los picos de la activacion de onsets (maximos
//!    locales a lo largo del tiempo, para cada nota). Por cada onset se
//!    busca hasta donde la activacion de nota se mantiene por encima del
//!    umbral, y con eso se forma la nota.
//! 2. **Energia restante**: cada nota detectada consume la energia de su
//!    banda de frecuencia (y de las vecinas), para que no se detecte dos
//!    veces la misma nota.
//! 3. **Melodia Trick**: si queda energia sin explicar por encima del
//!    umbral, se busca la nota mas fuerte que queda y se extrae tambien.
//!    Esto es lo que permite recuperar notas que el modelo no marco con un
//!    onset claro.
//!
//! # Los onsets inferidos
//!
//! Antes de buscar los picos, se combinan los onsets predichos con unos
//! "onsets inferidos" que se calculan a partir de los saltos de la
//! activacion de nota: un cambio grande de activacion indica un ataque, even
//! que la red no haya marcado un onset.
//!
//! # Las entradas
//!
//! - `frames` es la salida **note** del modelo: 88 valores por frame (uno
//!   por tecla).
//! - `onsets` es la salida **onset** del modelo: 88 valores por frame.
//!
//! El `contour` (264 valores por frame, tres por semitono) NO interviene en
//! la deteccion: solo se usa para estimar el pitch bend (Fase 10) y para
//! saber cuantos frames hay.
//!
//! # Umbrales por defecto
//!
//! | Parametro | Valor | Origen |
//! |---|---|---|
//! | `onset_threshold` | 0.5 | `DEFAULT_ONSET_THRESHOLD` |
//! | `frame_threshold` | 0.3 | `DEFAULT_FRAME_THRESHOLD` |
//! | `minimum_note_length_ms` | 127.7 | `DEFAULT_MINIMUM_NOTE_LENGTH_MS` |
//! | `energy_tol` | 11 | `ENERGY_TOLERANCE` |
//! | `melodia_trick` | activado | `--no-melodia` lo desactiva |
//!
//! # Conversion de milisegundos a frames
//!
//! ```python
//! min_note_len = int(np.round(minimum_note_length / 1000 * (AUDIO_SAMPLE_RATE / FFT_HOP)))
//! ```
//!
//! Con los valores por defecto: `round(127.7 / 1000 * 86.1328125) = 11` frames.
//!
//! El fps es la division **flotante** `AUDIO_SAMPLE_RATE / FFT_HOP`, no el
//! entero `ANNOTATIONS_FPS` (= 86) de `constants.py`: con otros valores de
//! duracion minima los dos no coinciden.

use basic_pitch_audio_preprocess::{AUDIO_SAMPLE_RATE, FFT_HOP};

// El tipo de evento musical es el canonico de midi-io.
pub use basic_pitch_midi_io::NoteEvent;

/// Umbral de activacion de onset por defecto.
pub const DEFAULT_ONSET_THRESHOLD: f64 = 0.5;

/// Umbral de activacion de frame por defecto.
pub const DEFAULT_FRAME_THRESHOLD: f64 = 0.3;

/// Duracion minima de nota por defecto, en milisegundos.
pub const DEFAULT_MINIMUM_NOTE_LENGTH_MS: f64 = 127.7;

/// Tolerancia de energia por defecto, en frames.
pub const DEFAULT_ENERGY_TOL: usize = 11;

/// Desplazamiento entre el indice MIDI y el indice de la matriz de notas.
pub const MIDI_OFFSET: u8 = 21;

/// Numero de notas del modelo (88 teclas).
pub const N_NOTES: usize = 88;

/// Ultimo indice de frecuencia de las notas.
///
/// El upstream lo usa para no salirse al limpiar la energia de la banda
/// siguiente: `if freq_idx < MAX_FREQ_IDX`.
pub const MAX_FREQ_IDX: usize = 87;

/// Configuracion de la deteccion de notas.
#[derive(Debug, Clone, PartialEq)]
pub struct NoteDetectionConfig {
    /// Umbral de activacion de onset.
    pub onset_threshold: f64,
    /// Umbral de activacion de frame.
    pub frame_threshold: f64,
    /// Duracion minima de nota, en milisegundos.
    pub minimum_note_length_ms: f64,
    /// Tolerancia de energia, en frames.
    pub energy_tol: usize,
    /// Si se infieren onsets adicionales a partir de cambios de frame.
    pub infer_onsets: bool,
    /// Si se aplica el truco de Melodia.
    pub melodia_trick: bool,
    /// Frecuencia minima permitida, en Hz.
    pub min_freq: Option<f64>,
    /// Frecuencia maxima permitida, en Hz.
    pub max_freq: Option<f64>,
}

impl Default for NoteDetectionConfig {
    fn default() -> Self {
        Self {
            onset_threshold: DEFAULT_ONSET_THRESHOLD,
            frame_threshold: DEFAULT_FRAME_THRESHOLD,
            minimum_note_length_ms: DEFAULT_MINIMUM_NOTE_LENGTH_MS,
            energy_tol: DEFAULT_ENERGY_TOL,
            infer_onsets: true,
            melodia_trick: true,
            min_freq: None,
            max_freq: None,
        }
    }
}

impl NoteDetectionConfig {
    /// Duracion minima de nota convertida a frames.
    ///
    /// Reproduce `int(np.round(ms / 1000 * (22050 / 256)))`.
    ///
    /// # El fps es flotante, no entero
    ///
    /// El upstream usa `AUDIO_SAMPLE_RATE / FFT_HOP` = **86.1328125**, que es
    /// division flotante. No es lo mismo que `ANNOTATIONS_FPS`, que en
    /// `constants.py` es `AUDIO_SAMPLE_RATE // FFT_HOP` = 86 (division
    /// entera) y se usa para otras cosas.
    ///
    /// Con el valor por defecto (127.7 ms) ambos dan 11 frames, asi que el
    /// error pasaba desapercibido. Pero no siempre coinciden: con 5.81 ms,
    /// el upstream da 1 frame y el valor entero daria 0.
    ///
    /// El redondeo es `round_ties_even`, que es lo que hace `np.round`.
    pub fn min_note_len_frames(&self) -> usize {
        let fps = AUDIO_SAMPLE_RATE as f64 / FFT_HOP as f64;
        (self.minimum_note_length_ms / 1000.0 * fps).round_ties_even() as usize
    }
}

/// Una nota detectada, en frames (sin convertir a tiempo).
#[derive(Debug, Clone, PartialEq)]
pub struct RawNote {
    /// Frame inicial.
    pub start_frame: usize,
    /// Frame final (excluido).
    pub end_frame: usize,
    /// Nota MIDI (21 = A0, 108 = C8).
    pub pitch: u8,
    /// Amplitud media de la nota, entre 0 y 1.
    pub amplitude: f32,
}

/// Convierte una frecuencia a numero MIDI.
///
/// Equivale a `librosa.hz_to_midi`: `69 + 12 * log2(f / 440)`.
pub fn hz_to_midi(freq: f64) -> f64 {
    69.0 + 12.0 * (freq / 440.0).log2()
}

/// Pone a cero las activaciones fuera del rango de frecuencias permitido.
///
/// Reproduce `constrain_frequency` del upstream.
fn constrain_frequency(
    onsets: &mut [f32],
    frames: &mut [f32],
    n_frames: usize,
    n_freqs: usize,
    min_freq: Option<f64>,
    max_freq: Option<f64>,
) {
    let mut min_idx = 0usize;
    let mut max_idx = n_freqs;

    if let Some(freq) = min_freq {
        // `int(np.round(librosa.hz_to_midi(freq) - MIDI_OFFSET))`
        min_idx = (hz_to_midi(freq) - MIDI_OFFSET as f64)
            .round_ties_even()
            .max(0.0) as usize;
    }
    if let Some(freq) = max_freq {
        max_idx = (hz_to_midi(freq) - MIDI_OFFSET as f64)
            .round_ties_even()
            .max(0.0) as usize;
    }
    let min_idx = min_idx.min(n_freqs);
    let max_idx = max_idx.min(n_freqs);

    for frame in 0..n_frames {
        let base = frame * n_freqs;
        for f in 0..min_idx {
            onsets[base + f] = 0.0;
            frames[base + f] = 0.0;
        }
        for f in max_idx..n_freqs {
            onsets[base + f] = 0.0;
            frames[base + f] = 0.0;
        }
    }
}

/// Combina los onsets predichos con los inferidos de los cambios de frame.
///
/// Reproduce `get_infered_onsets` del upstream.
///
/// La diferencia entre frames consecutivos se calcula hacia atras (con ceros
/// al principio), se toma el **minimo** de las diferencias, se recorta por
/// debajo de cero, y se reescala para tener el mismo maximo que los onsets
/// predichos.
///
/// # Una diferencia con el upstream
///
/// El upstream divide por `max(frame_diff)`, que es cero cuando las
/// activaciones nunca aumentan (por ejemplo, en silencio). En NumPy eso
/// produce `NaN`, que se propaga y hace que **no se detecte ninguna nota**.
///
/// Aqui, cuando el maximo es cero, el resultado es cero, que produce el
/// mismo resultado (ninguna nota) sin propagar `NaN`.
fn get_infered_onsets(
    onsets: &[f32],
    frames: &[f32],
    n_frames: usize,
    n_freqs: usize,
    n_diff: usize,
) -> Vec<f32> {
    let mut frame_diff = vec![f32::INFINITY; n_frames * n_freqs];

    for n in 1..=n_diff {
        for t in 0..n_frames {
            for f in 0..n_freqs {
                let anterior = if t >= n {
                    frames[(t - n) * n_freqs + f]
                } else {
                    0.0
                };
                let d = frames[t * n_freqs + f] - anterior;
                let idx = t * n_freqs + f;
                if d < frame_diff[idx] {
                    frame_diff[idx] = d;
                }
            }
        }
    }

    // Los negativos se recortan a cero.
    for value in frame_diff.iter_mut() {
        if *value < 0.0 {
            *value = 0.0;
        }
    }
    // Los primeros `n_diff` frames se ponen a cero.
    for t in 0..n_diff.min(n_frames) {
        for f in 0..n_freqs {
            frame_diff[t * n_freqs + f] = 0.0;
        }
    }

    // Reescalar al maximo de los onsets predichos.
    let max_onsets = onsets.iter().copied().fold(0.0f32, f32::max);
    let max_diff = frame_diff.iter().copied().fold(0.0f32, f32::max);
    if max_diff > 0.0 {
        let factor = max_onsets / max_diff;
        for value in frame_diff.iter_mut() {
            *value *= factor;
        }
    } else {
        // El upstream produce NaN aqui; el resultado practical es el mismo
        // (no hay onsets inferidos) sin propagar NaN.
        for value in frame_diff.iter_mut() {
            *value = 0.0;
        }
    }

    // El maximo elemento a elemento.
    onsets
        .iter()
        .zip(frame_diff.iter())
        .map(|(a, b)| a.max(*b))
        .collect()
}

/// Decodifica las activaciones del modelo en notas polifonicas.
///
/// Reproduce `basic_pitch/note_creation.py::output_to_notes_polyphonic`.
///
/// # Argumentos
///
/// - `frames`: la salida **note** del modelo, `(n_frames, n_freqs)` en orden
///   row-major.
/// - `onsets`: la salida **onset** del modelo, con la misma forma.
/// - `config`: los umbrales y las opciones.
///
/// # Salida
///
/// Las notas detectadas, en frames.
pub fn output_to_notes_polyphonic(
    frames: &[f32],
    onsets: &[f32],
    n_frames: usize,
    n_freqs: usize,
    config: &NoteDetectionConfig,
) -> Vec<RawNote> {
    if n_frames == 0 || n_freqs == 0 {
        return Vec::new();
    }

    let mut frames = frames.to_vec();
    let mut onsets = onsets.to_vec();

    // 1. Restricciones de frecuencia.
    constrain_frequency(
        &mut onsets,
        &mut frames,
        n_frames,
        n_freqs,
        config.min_freq,
        config.max_freq,
    );

    // 2. Onsets inferidos de los cambios de frame.
    if config.infer_onsets {
        onsets = get_infered_onsets(&onsets, &frames, n_frames, n_freqs, 2);
    }

    // 3. Maximos locales de los onsets, a lo largo del tiempo.
    //
    // `scipy.signal.argrelmax(onsets, axis=0)` con `order=1`: un frame es
    // un pico si es **estrictamente mayor** que sus dos vecinos. Los
    // extremos no son picos.
    let mut peak_thresh_mat = vec![0.0f32; n_frames * n_freqs];
    for f in 0..n_freqs {
        for t in 1..n_frames.saturating_sub(1) {
            let v = onsets[t * n_freqs + f];
            if v > onsets[(t - 1) * n_freqs + f] && v > onsets[(t + 1) * n_freqs + f] {
                peak_thresh_mat[t * n_freqs + f] = v;
            }
        }
    }

    // Los pares (frame, frecuencia) que superan el umbral, en orden
    // row-major, y luego **invertidos**: el upstream recorre los onsets de
    // mas reciente a mas antiguo.
    let mut onset_idx: Vec<(usize, usize)> = (0..n_frames * n_freqs)
        .filter(|i| peak_thresh_mat[*i] >= config.onset_threshold as f32)
        .map(|i| (i / n_freqs, i % n_freqs))
        .collect();
    onset_idx.reverse();

    // 4. Energia restante: la que todavia no se ha atribuido a ninguna nota.
    let mut remaining_energy = frames.clone();
    let mut notes: Vec<RawNote> = Vec::new();
    let min_note_len = config.min_note_len_frames();

    // --- Los onsets ---
    for (start, freq) in onset_idx {
        // Si estamos demasiado cerca del final, no hay nota.
        if start >= n_frames - 1 {
            continue;
        }

        // Buscar hasta donde la activacion se mantiene sobre el umbral.
        let mut i = start + 1;
        let mut k = 0usize;
        while i < n_frames - 1 && k < config.energy_tol {
            if remaining_energy[i * n_freqs + freq] < config.frame_threshold as f32 {
                k += 1;
            } else {
                k = 0;
            }
            i += 1;
        }
        i -= k; // volver al ultimo frame por encima del umbral

        // Si la nota es demasiado corta, se descarta.
        if i - start <= min_note_len {
            continue;
        }

        // Consumir la energia de la banda y de las vecinas.
        consume_energy(&mut remaining_energy, n_freqs, start, i, freq);

        let amplitude = mean_column(&frames, n_freqs, start, i, freq);
        notes.push(RawNote {
            start_frame: start,
            end_frame: i,
            pitch: (freq + MIDI_OFFSET as usize) as u8,
            amplitude,
        });
    }

    // --- El truco de Melodia ---
    if config.melodia_trick {
        loop {
            // El maximo global de la energia restante. `np.argmax` devuelve
            // la **primera** ocurrencia en orden row-major.
            let mut best = 0.0f32;
            let mut best_idx = None;
            for (i, value) in remaining_energy.iter().enumerate() {
                if *value > best {
                    best = *value;
                    best_idx = Some(i);
                }
            }
            let Some(idx) = best_idx else { break };
            if best <= config.frame_threshold as f32 {
                break; // `while np.max(remaining_energy) > frame_thresh`
            }

            let i_mid = idx / n_freqs;
            let freq = idx % n_freqs;
            remaining_energy[idx] = 0.0;

            // Pasada hacia delante.
            let mut i = i_mid + 1;
            let mut k = 0usize;
            while i < n_frames - 1 && k < config.energy_tol {
                if remaining_energy[i * n_freqs + freq] < config.frame_threshold as f32 {
                    k += 1;
                } else {
                    k = 0;
                }
                remaining_energy[i * n_freqs + freq] = 0.0;
                if freq < MAX_FREQ_IDX {
                    remaining_energy[i * n_freqs + freq + 1] = 0.0;
                }
                if freq > 0 {
                    remaining_energy[i * n_freqs + freq - 1] = 0.0;
                }
                i += 1;
            }
            let i_end = i - 1 - k;

            // Pasada hacia atras.
            let mut i = i_mid as i64 - 1;
            let mut k = 0usize;
            while i > 0 && k < config.energy_tol {
                let ii = i as usize;
                if remaining_energy[ii * n_freqs + freq] < config.frame_threshold as f32 {
                    k += 1;
                } else {
                    k = 0;
                }
                remaining_energy[ii * n_freqs + freq] = 0.0;
                if freq < MAX_FREQ_IDX {
                    remaining_energy[ii * n_freqs + freq + 1] = 0.0;
                }
                if freq > 0 {
                    remaining_energy[ii * n_freqs + freq - 1] = 0.0;
                }
                i -= 1;
            }
            let i_start = (i + 1 + k as i64) as usize;

            if i_end <= i_start {
                continue;
            }
            // Si la nota es demasiado corta, se descarta.
            if i_end - i_start <= min_note_len {
                continue;
            }

            let amplitude = mean_column(&frames, n_freqs, i_start, i_end, freq);
            notes.push(RawNote {
                start_frame: i_start,
                end_frame: i_end,
                pitch: (freq + MIDI_OFFSET as usize) as u8,
                amplitude,
            });
        }
    }

    notes
}

/// Pone a cero la energia de una banda de frecuencia y de sus vecinas.
fn consume_energy(
    remaining_energy: &mut [f32],
    n_freqs: usize,
    start: usize,
    end: usize,
    freq: usize,
) {
    for t in start..end {
        remaining_energy[t * n_freqs + freq] = 0.0;
        if freq < MAX_FREQ_IDX {
            remaining_energy[t * n_freqs + freq + 1] = 0.0;
        }
        if freq > 0 {
            remaining_energy[t * n_freqs + freq - 1] = 0.0;
        }
    }
}

/// Media de una columna: los frames `[start, end)` de la frecuencia `freq`.
///
/// El upstream hace `np.mean(frames[start:end, freq])`, o sea la media de
/// una **columna** de la matriz. Un slice lineal del buffer incluiria otras
/// frecuencias, asi que esto no se puede escribir como un rango.
fn mean_column(values: &[f32], n_freqs: usize, start: usize, end: usize, freq: usize) -> f32 {
    if end <= start {
        return 0.0;
    }
    let suma: f64 = (start..end)
        .map(|t| values[t * n_freqs + freq] as f64)
        .sum();
    (suma / (end - start) as f64) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn la_duracion_minima_en_frames_es_11() {
        let config = NoteDetectionConfig::default();
        assert_eq!(config.min_note_len_frames(), 11);
    }

    #[test]
    fn el_fps_es_el_flotante_del_upstream() {
        // Con 5.81 ms, la division flotante (86.1328125) da 0.50043 y
        // redondea a 1 frame; la entera (86) daria 0.49966 y 0 frames.
        let config = NoteDetectionConfig {
            minimum_note_length_ms: 5.81,
            ..Default::default()
        };
        assert_eq!(config.min_note_len_frames(), 1);

        // El valor por defecto sigue dando 11.
        assert_eq!(NoteDetectionConfig::default().min_note_len_frames(), 11);
    }

    #[test]
    fn los_umbrales_por_defecto_son_los_de_la_referencia() {
        let config = NoteDetectionConfig::default();
        assert_eq!(config.onset_threshold, 0.5);
        assert_eq!(config.frame_threshold, 0.3);
        assert_eq!(config.energy_tol, 11);
        assert!(config.melodia_trick);
        assert!(config.infer_onsets);
    }

    #[test]
    fn hz_a_midi_reproduce_librosa() {
        // 440 Hz es el La4 (MIDI 69).
        assert!((hz_to_midi(440.0) - 69.0).abs() < 1e-9);
        // 27.5 Hz es el La0 (MIDI 21).
        assert!((hz_to_midi(27.5) - 21.0).abs() < 1e-9);
    }

    #[test]
    fn una_senal_vacia_no_produce_notas() {
        let config = NoteDetectionConfig::default();
        assert!(output_to_notes_polyphonic(&[], &[], 0, 88, &config).is_empty());
    }

    #[test]
    fn el_silencio_no_produce_notas() {
        // 100 frames de silencio: sin onsets ni energia, no hay notas.
        let n_frames = 100;
        let n_freqs = 88;
        let frames = vec![0.0f32; n_frames * n_freqs];
        let onsets = vec![0.0f32; n_frames * n_freqs];
        let config = NoteDetectionConfig::default();
        let notas = output_to_notes_polyphonic(&frames, &onsets, n_frames, n_freqs, &config);
        assert!(notas.is_empty(), "el silencio no deberia dar notas");
    }

    #[test]
    fn una_nota_clara_se_detecta_con_su_pitch_y_su_duracion() {
        // Una nota en el pitch 60 (indice 39) durante 20 frames, con un
        // onset marcado al principio.
        let n_frames = 60;
        let n_freqs = 88;
        let freq = 39usize; // 39 + MIDI_OFFSET = 60
        let mut frames = vec![0.0f32; n_frames * n_freqs];
        let mut onsets = vec![0.0f32; n_frames * n_freqs];
        for t in 10..30 {
            frames[t * n_freqs + freq] = 0.9;
        }
        onsets[10 * n_freqs + freq] = 0.9;

        let config = NoteDetectionConfig::default();
        let notas = output_to_notes_polyphonic(&frames, &onsets, n_frames, n_freqs, &config);

        assert_eq!(notas.len(), 1, "notas: {notas:?}");
        assert_eq!(notas[0].pitch, 60);
        assert_eq!(notas[0].start_frame, 10);
        assert!(notas[0].end_frame > 10);
        assert!(notas[0].amplitude > 0.8);
    }
}

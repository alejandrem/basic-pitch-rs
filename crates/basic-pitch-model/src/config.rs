//! Configuracion tecnica del modelo.
//!
//! Los valores proceden de `model_config.json`, generado en la Fase 1 a partir
//! del modelo oficial. Se validan al cargar, de modo que un manifiesto
//! incoherente falla pronto y con un mensaje claro.

use std::path::Path;

use serde::{Deserialize, Serialize};

/// Configuracion del modelo ICASSP 2022.
///
/// Refleja exactamente la estructura de `model_config.json`, que genera la
/// Fase 1. Si el JSON cambia, este tipo debe cambiar con el.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Identificacion del modelo y su procedencia.
    pub model: ModelIdentity,
    /// Parametros de audio.
    pub audio: AudioConfig,
    /// Parametros de frecuencia.
    pub frequency: FrequencyConfig,
    /// Parametros de la CQT.
    pub cqt: CqtConfig,
    /// Parametros de Harmonic Stacking.
    pub harmonic_stacking: HarmonicStackingConfig,
    /// Parametros de Batch Normalization.
    pub batch_norm: BatchNormConfig,
    /// Capas convolucionales activas.
    pub layers: Vec<LayerConfig>,
    /// Configuracion de la concatenacion de onsets.
    pub concat: ConcatConfig,
    /// Ventanas superpuestas.
    pub windowing: WindowingConfig,
}

/// Identidad y procedencia del modelo.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelIdentity {
    /// Nombre del modelo.
    pub name: String,
    /// Variante (`icassp-2022`).
    pub variant: String,
    /// Framework de origen.
    #[serde(default)]
    pub framework: String,
    /// Repositorio de referencia.
    #[serde(default)]
    pub reference_repo: String,
    /// Commit de referencia del repositorio original.
    pub reference_commit: String,
    /// Version del paquete original.
    #[serde(default)]
    pub reference_package_version: String,
    /// Tipo de dato de los tensores.
    pub dtype: String,
}

/// Parametros de audio.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioConfig {
    /// Frecuencia de muestreo.
    pub sample_rate: u32,
    /// Numero de canales de entrada.
    pub channels: u16,
    /// Longitud de la ventana, en segundos.
    pub window_length_s: u64,
    /// Desplazamiento entre frames, en muestras.
    pub fft_hop: usize,
    /// Frames de anotacion por segundo.
    pub annotations_fps: usize,
    /// Frames por ventana.
    pub annot_n_frames: usize,
    /// Muestras por ventana.
    pub audio_n_samples: usize,
}

/// Parametros de frecuencia.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrequencyConfig {
    /// Frecuencia del bin mas bajo.
    pub base_frequency: f64,
    /// Semitonos musicales.
    pub n_semitones: usize,
    /// Bins por semitono de la representacion de contornos.
    pub contours_bins_per_semitone: usize,
    /// Bins por semitono de la representacion de notas.
    #[serde(default)]
    pub notes_bins_per_semitone: usize,
    /// Semitonos cubiertos por la CQT.
    pub cqt_n_semitones: usize,
    /// Bins que produce la CQT.
    pub cqt_n_bins: usize,
    /// Bins de la representacion de contornos.
    pub n_freq_bins_contours: usize,
    /// Bins de la representacion de notas.
    pub n_freq_bins_notes: usize,
}

/// Parametros de la CQT.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CqtConfig {
    /// Frecuencia de muestreo.
    #[serde(default)]
    pub sr: u32,
    /// Frecuencia del bin mas bajo.
    #[serde(default)]
    pub fmin: f64,
    /// Numero de bins.
    pub n_bins: usize,
    /// Bins por octava.
    pub bins_per_octave: usize,
    /// Desplazamiento entre frames, en muestras.
    pub hop_length: usize,
    /// Factor Q.
    #[serde(default)]
    pub q_factor: f64,
    /// Numero de octavas.
    pub n_octaves: usize,
    /// Numero de filtros del kernel de la octava superior.
    pub n_filters: usize,
    /// Longitud del kernel.
    pub n_fft: usize,
    /// Si hay submuestreo temprano.
    pub early_downsample: bool,
    /// Factor de submuestreo temprano.
    pub downsample_factor: usize,
    /// Modo de relleno.
    pub pad_mode: String,
    /// Normalizacion de la base.
    pub basis_norm: u8,
    /// Nombre del asset con los kernels.
    #[serde(default)]
    pub kernels_asset: String,
}

/// Parametros de Harmonic Stacking.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HarmonicStackingConfig {
    /// Armonicos utilizados.
    pub harmonics: Vec<f64>,
    /// Desplazamiento resultante de cada armónico, en bins.
    pub shifts: Vec<i64>,
    /// Bins por semitono.
    pub bins_per_semitone: usize,
    /// Bins de frecuencia de la salida.
    pub n_output_freqs: usize,
    /// Bins de frecuencia de la entrada.
    pub input_n_bins: usize,
    /// Descripcion del relleno.
    #[serde(default)]
    pub padding: String,
}

/// Parametros de Batch Normalization.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BatchNormConfig {
    /// Epsilon de Keras.
    pub epsilon: f64,
    /// Origen del epsilon.
    pub epsilon_source: String,
    /// Nota sobre el epsilon.
    #[serde(default)]
    pub note: String,
    /// BatchNorm presentes en el modelo.
    pub layers: Vec<BatchNormLayerConfig>,
}

/// Una BatchNorm del modelo.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BatchNormLayerConfig {
    /// Nombre en TensorFlow.
    pub name: String,
    /// Numero de canales.
    pub channels: usize,
    /// Posicion en la arquitectura.
    pub position: String,
}

/// Configuracion de una capa convolucional.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayerConfig {
    /// Nombre en TensorFlow.
    pub name: String,
    /// Nombre en Rust.
    pub rust_name: String,
    /// Kernel, como (alto, ancho) en TensorFlow.
    pub kernel_tf: [usize; 2],
    /// Numero de filtros.
    pub filters: usize,
    /// Canales de entrada.
    pub in_channels: usize,
    /// Stride, como (tiempo, frecuencia).
    pub stride_tf: [usize; 2],
    /// Forma de entrada, como (tiempo, frecuencia, canales).
    pub in_shape_nhwc: [usize; 3],
    /// Forma de salida, como (tiempo, frecuencia, canales).
    pub out_shape_nhwc: [usize; 3],
    /// Padding SAME, como ((antes, despues) del tiempo, (antes, despues) de la frecuencia).
    pub pad_same_tf: [[usize; 2]; 2],
    /// Activacion.
    pub activation: String,
    /// Si la capa tiene BatchNorm.
    pub batch_norm: bool,
    /// Nota sobre la capa.
    #[serde(default)]
    pub note: String,
}

/// Configuracion de la concatenacion de onsets.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConcatConfig {
    /// Nombre de la operacion.
    #[serde(default)]
    pub name: String,
    /// Eje de concatenacion (3 en NHWC: los canales).
    pub axis: usize,
    /// Orden de los tensores concatenados.
    pub order: Vec<String>,
    /// Numero de canales resultantes.
    pub channels: usize,
    /// Nota sobre la concatenacion.
    #[serde(default)]
    pub note: String,
}

/// Configuracion de las ventanas superpuestas.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WindowingConfig {
    /// Frames superpuestos.
    pub n_overlapping_frames: usize,
    /// Longitud del solapamiento, en muestras.
    pub overlap_len: usize,
    /// Desplazamiento entre ventanas, en muestras.
    pub hop_size: usize,
    /// Relleno inicial, en muestras.
    pub initial_padding: usize,
    /// Frames emitidos por ventana.
    pub frames_per_window: usize,
    /// Frames conservados por ventana.
    pub frames_kept_per_window: usize,
    /// Desplazamiento de alineacion magico.
    pub magic_alignment_offset: f64,
    /// Nota de compatibilidad.
    #[serde(default)]
    pub compatibility_note: String,
}

/// Los valores del contrato tecnico del modelo ICASSP 2022.
///
/// Fuente: `docs/teoria/fase-03-espe-tecnicas.md`, verificados contra el
/// SavedModel oficial, el ONNX oficial y el codigo del upstream.
///
/// Son la **fuente de verdad** de la validacion: `ModelConfig::validate`
/// comprueba que la configuracion cargada los respete. Antes las
/// comprobaciones usaban literales sueltos y solo cubrian una parte de los
/// campos que el codigo lee.
mod contrato {
    /// Frecuencia de muestreo.
    pub const SAMPLE_RATE: u32 = 22_050;
    /// Longitud de la ventana de audio, en muestras.
    pub const AUDIO_N_SAMPLES: usize = 43_844;
    /// Desplazamiento entre frames de anotacion, en muestras.
    pub const FFT_HOP: usize = 256;
    /// Frames de anotacion por segundo.
    pub const ANNOTATIONS_FPS: usize = 86;
    /// Frames que el modelo emite por ventana.
    pub const ANNOT_N_FRAMES: usize = 172;
    /// Frecuencia del bin mas grave.
    pub const BASE_FREQUENCY: f64 = 27.5;
    /// Semitonos musicales (teclas de piano).
    pub const N_SEMITONES: usize = 88;
    /// Bins por semitono de los contornos.
    pub const CONTOURS_BINS_PER_SEMITONE: usize = 3;
    /// Bins por semitono de las notas.
    pub const NOTES_BINS_PER_SEMITONE: usize = 1;
    /// Semitonos que cubre la CQT.
    pub const CQT_N_SEMITONES: usize = 103;
    /// Bins que produce la CQT.
    pub const CQT_N_BINS: usize = 309;
    /// Bins de la representacion de contornos.
    pub const N_FREQ_BINS_CONTOURS: usize = 264;
    /// Valores de la prediccion de notas.
    pub const N_FREQ_BINS_NOTES: usize = 88;
    /// Bins por octava de la CQT.
    pub const BINS_PER_OCTAVE: usize = 36;
    /// Octavas que cubre la CQT.
    pub const CQT_N_OCTAVES: usize = 9;
    /// Filtros del kernel de la octava superior.
    pub const CQT_N_FILTERS: usize = 36;
    /// Longitud del kernel de la CQT, en muestras.
    pub const CQT_N_FFT: usize = 256;
    /// Epsilon de Keras para BatchNorm.
    pub const BATCH_NORM_EPSILON: f64 = 0.001;
    /// Armonicos del Harmonic Stacking.
    pub const HARMONICS: [f64; 8] = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    /// Frames superpuestos entre ventanas consecutivas.
    pub const N_OVERLAPPING_FRAMES: usize = 30;
    /// Longitud del solapamiento, en muestras.
    pub const OVERLAP_LEN: usize = 7_680;
    /// Desplazamiento entre ventanas consecutivas, en muestras.
    pub const HOP_SIZE: usize = 36_164;
    /// Zeros de relleno que se anteponen al audio.
    pub const INITIAL_PADDING: usize = 3_840;
    /// Frames conservados por ventana.
    pub const FRAMES_KEPT_PER_WINDOW: usize = 142;
    /// Canales resultantes de la concatenacion de onsets.
    pub const CONCAT_CHANNELS: usize = 33;
}

impl ModelConfig {
    /// Carga la configuracion desde `model_config.json`.
    pub fn load(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("no se pudo leer {}: {e}", path.display()))?;
        let config: ModelConfig = serde_json::from_str(&text)
            .map_err(|e| format!("no se pudo parsear {}: {e}", path.display()))?;
        config.validate()?;
        Ok(config)
    }

    /// Comprueba que la configuracion sea coherente.
    ///
    /// Estas comprobaciones son las que habrian detectado los errores
    /// silenciosos mas graves del proyecto: 264 bins de CQT en lugar de 309,
    /// o 32 canales de concatenacion en lugar de 33.
    ///
    /// # Que se comprueba y por que
    ///
    /// Se comprueba **todo campo que el codigo lee**, no solo un resumen.
    /// Antes se validaban siete cosas, y el frontend leia una decena mas
    /// sin ninguna comprobacion: por ejemplo, `cqt.n_bins` (que es el que
    /// usa el frontend) no se validaba, solo `frequency.cqt_n_bins`, que
    /// es otro campo con el mismo significado. Con `cqt.n_bins = 264` la
    /// configuracion pasaba la validacion y el modelo usaba 264 bins:
    /// exactamente el error que este proyecto lleva dos fases guardando.
    ///
    /// Los errores se acumulan y se devuelven todos juntos: fallar en el
    /// primero obligaba a reintentar N veces para ver los demas.
    pub fn validate(&self) -> Result<(), String> {
        use contrato::*;

        let mut errores: Vec<String> = Vec::new();
        let mut check = |ok: bool, mensaje: String| {
            if !ok {
                errores.push(mensaje);
            }
        };

        // --- Frecuencias ---
        check(
            self.frequency.base_frequency == BASE_FREQUENCY,
            format!(
                "la frecuencia base debe ser {BASE_FREQUENCY}, es {}",
                self.frequency.base_frequency
            ),
        );
        check(
            self.frequency.n_semitones == N_SEMITONES,
            format!(
                "deben cubrir {N_SEMITONES} semitonos, son {}",
                self.frequency.n_semitones
            ),
        );
        check(
            self.frequency.contours_bins_per_semitone == CONTOURS_BINS_PER_SEMITONE,
            format!(
                "los contornos usan {CONTOURS_BINS_PER_SEMITONE} bins por semitono, son {}",
                self.frequency.contours_bins_per_semitone
            ),
        );
        check(
            self.frequency.notes_bins_per_semitone == NOTES_BINS_PER_SEMITONE,
            format!(
                "las notas usan {NOTES_BINS_PER_SEMITONE} bin por semitono, son {}",
                self.frequency.notes_bins_per_semitone
            ),
        );
        check(
            self.frequency.cqt_n_semitones == CQT_N_SEMITONES,
            format!(
                "la CQT debe cubrir {CQT_N_SEMITONES} semitonos, son {}",
                self.frequency.cqt_n_semitones
            ),
        );
        check(
            self.frequency.cqt_n_bins == CQT_N_BINS,
            format!(
                "la CQT debe producir {CQT_N_BINS} bins, no {}",
                self.frequency.cqt_n_bins
            ),
        );
        check(
            self.frequency.n_freq_bins_contours == N_FREQ_BINS_CONTOURS,
            format!(
                "la representacion de contornos debe tener {N_FREQ_BINS_CONTOURS} bins, no {}",
                self.frequency.n_freq_bins_contours
            ),
        );
        check(
            self.frequency.n_freq_bins_notes == N_FREQ_BINS_NOTES,
            format!(
                "la prediccion de notas debe tener {N_FREQ_BINS_NOTES} valores, no {}",
                self.frequency.n_freq_bins_notes
            ),
        );

        // --- CQT ---
        //
        // Estos son los campos que lee el frontend acustico. Todos.
        check(
            self.cqt.sr == SAMPLE_RATE,
            format!("la CQT debe usar {SAMPLE_RATE} Hz, usa {}", self.cqt.sr),
        );
        check(
            self.cqt.fmin == BASE_FREQUENCY,
            format!(
                "la CQT debe empezar en {BASE_FREQUENCY} Hz, empieza en {}",
                self.cqt.fmin
            ),
        );
        check(
            self.cqt.n_bins == CQT_N_BINS,
            format!(
                "la CQT debe producir {CQT_N_BINS} bins, no {} (este es el campo que usa el frontend)",
                self.cqt.n_bins
            ),
        );
        check(
            self.cqt.n_bins == self.frequency.cqt_n_bins,
            format!(
                "`cqt.n_bins` ({}) y `frequency.cqt_n_bins` ({}) no coinciden",
                self.cqt.n_bins, self.frequency.cqt_n_bins
            ),
        );
        check(
            self.cqt.bins_per_octave == BINS_PER_OCTAVE,
            format!(
                "la CQT debe tener {BINS_PER_OCTAVE} bins por octava, tiene {}",
                self.cqt.bins_per_octave
            ),
        );
        check(
            self.cqt.hop_length == FFT_HOP,
            format!(
                "el hop de la CQT debe ser {FFT_HOP} muestras, es {}",
                self.cqt.hop_length
            ),
        );
        check(
            self.cqt.n_fft == CQT_N_FFT,
            format!(
                "el kernel de la CQT debe medir {CQT_N_FFT} muestras, mide {}",
                self.cqt.n_fft
            ),
        );
        check(
            self.cqt.n_octaves == CQT_N_OCTAVES,
            format!(
                "la CQT debe cubrir {CQT_N_OCTAVES} octavas, cubre {}",
                self.cqt.n_octaves
            ),
        );
        check(
            self.cqt.n_filters == CQT_N_FILTERS,
            format!(
                "el kernel de la CQT debe tener {CQT_N_FILTERS} filtros, tiene {}",
                self.cqt.n_filters
            ),
        );
        check(
            !self.cqt.early_downsample,
            "el modelo no usa submuestreo temprano en la CQT".to_string(),
        );
        check(
            self.cqt.downsample_factor == 1,
            format!(
                "el factor de submuestreo de la CQT debe ser 1, es {}",
                self.cqt.downsample_factor
            ),
        );
        check(
            self.cqt.pad_mode == "reflect",
            format!(
                "el relleno de la CQT debe ser 'reflect', es '{}'",
                self.cqt.pad_mode
            ),
        );
        check(
            self.cqt.basis_norm == 1,
            format!(
                "la normalizacion de la base de la CQT debe ser 1 (L1), es {}",
                self.cqt.basis_norm
            ),
        );

        // --- Harmonic Stacking ---
        check(
            self.harmonic_stacking.harmonics == HARMONICS.to_vec(),
            format!(
                "los armonicos deben ser {HARMONICS:?}, son {:?}",
                self.harmonic_stacking.harmonics
            ),
        );
        check(
            self.harmonic_stacking.bins_per_semitone == CONTOURS_BINS_PER_SEMITONE,
            format!(
                "Harmonic Stacking usa {CONTOURS_BINS_PER_SEMITONE} bins por semitono, usa {}",
                self.harmonic_stacking.bins_per_semitone
            ),
        );
        check(
            self.harmonic_stacking.n_output_freqs == N_FREQ_BINS_CONTOURS,
            format!(
                "Harmonic Stacking debe devolver {N_FREQ_BINS_CONTOURS} bins, devuelve {}",
                self.harmonic_stacking.n_output_freqs
            ),
        );
        check(
            self.harmonic_stacking.input_n_bins == CQT_N_BINS,
            format!(
                "Harmonic Stacking espera {CQT_N_BINS} bins de entrada, espera {}",
                self.harmonic_stacking.input_n_bins
            ),
        );
        // Los shifts declarados tienen que ser los que se calculan a partir
        // de los armonicos. El codigo los recalcula (no los lee del
        // JSON), asi que un `shifts` equivocado en el fichero seria una
        // mentira silenciosa.
        let shifts_calculados = crate::harmonic_stacking::harmonic_shifts(
            &self.harmonic_stacking.harmonics,
            self.harmonic_stacking.bins_per_semitone,
        );
        check(
            self.harmonic_stacking.shifts == shifts_calculados,
            format!(
                "los shifts declarados {:?} no son los que se calculan desde los armonicos {shifts_calculados:?}",
                self.harmonic_stacking.shifts
            ),
        );

        // --- BatchNorm ---
        check(
            self.batch_norm.epsilon == BATCH_NORM_EPSILON,
            format!(
                "el epsilon de BatchNorm debe ser {BATCH_NORM_EPSILON} (Keras), es {}",
                self.batch_norm.epsilon
            ),
        );
        for (nombre, canales) in [
            ("batch_normalization", 1usize),
            ("batch_normalization_2", 8),
            ("batch_normalization_3", 32),
        ] {
            let presente = self
                .batch_norm
                .layers
                .iter()
                .find(|capa| capa.name == nombre);
            match presente {
                None => check(false, format!("falta la BatchNorm '{nombre}'")),
                Some(capa) => check(
                    capa.channels == canales,
                    format!(
                        "la BatchNorm '{nombre}' debe tener {canales} canales, tiene {}",
                        capa.channels
                    ),
                ),
            }
        }

        // --- Concatenacion de onsets ---
        check(
            self.concat.channels == CONCAT_CHANNELS,
            format!(
                "la concatenacion de onsets debe producir {CONCAT_CHANNELS} canales, produce {}",
                self.concat.channels
            ),
        );
        check(
            self.concat.order == ["x_notes_pre", "x_onset"],
            format!(
                "el orden de la concatenacion debe ser [x_notes_pre, x_onset], no {:?}",
                self.concat.order
            ),
        );

        // --- Audio ---
        check(
            self.audio.sample_rate == SAMPLE_RATE,
            format!(
                "el modelo espera {SAMPLE_RATE} Hz, la configuracion dice {}",
                self.audio.sample_rate
            ),
        );
        check(
            self.audio.channels == 1,
            format!(
                "el modelo espera 1 canal, la configuracion dice {}",
                self.audio.channels
            ),
        );
        check(
            self.audio.fft_hop == FFT_HOP,
            format!(
                "el hop de anotacion debe ser {FFT_HOP} muestras, es {}",
                self.audio.fft_hop
            ),
        );
        check(
            self.audio.annotations_fps == ANNOTATIONS_FPS,
            format!(
                "la frecuencia de anotacion debe ser {ANNOTATIONS_FPS} fps, es {}",
                self.audio.annotations_fps
            ),
        );
        check(
            self.audio.annot_n_frames == ANNOT_N_FRAMES,
            format!(
                "el modelo debe emitir {ANNOT_N_FRAMES} frames por ventana, emite {}",
                self.audio.annot_n_frames
            ),
        );
        check(
            self.audio.audio_n_samples == AUDIO_N_SAMPLES,
            format!(
                "la ventana de audio debe medir {AUDIO_N_SAMPLES} muestras, mide {}",
                self.audio.audio_n_samples
            ),
        );

        // --- Ventaneo ---
        check(
            self.windowing.n_overlapping_frames == N_OVERLAPPING_FRAMES,
            format!(
                "deben solaparse {N_OVERLAPPING_FRAMES} frames, se solapan {}",
                self.windowing.n_overlapping_frames
            ),
        );
        check(
            self.windowing.overlap_len == OVERLAP_LEN,
            format!(
                "el solapamiento debe medir {OVERLAP_LEN} muestras, mide {}",
                self.windowing.overlap_len
            ),
        );
        check(
            self.windowing.hop_size == HOP_SIZE,
            format!(
                "el desplazamiento entre ventanas debe ser {HOP_SIZE} muestras, es {}",
                self.windowing.hop_size
            ),
        );
        check(
            self.windowing.initial_padding == INITIAL_PADDING,
            format!(
                "el relleno inicial debe ser de {INITIAL_PADDING} muestras, es de {}",
                self.windowing.initial_padding
            ),
        );
        check(
            self.windowing.frames_per_window == ANNOT_N_FRAMES,
            format!(
                "cada ventana debe emitir {ANNOT_N_FRAMES} frames, emite {}",
                self.windowing.frames_per_window
            ),
        );
        check(
            self.windowing.frames_kept_per_window == FRAMES_KEPT_PER_WINDOW,
            format!(
                "deben conservarse {FRAMES_KEPT_PER_WINDOW} frames por ventana, se conservan {}",
                self.windowing.frames_kept_per_window
            ),
        );

        // --- Capas activas ---
        //
        // Hay que ser precisos: la capa comentada `CONTOUR_KERNEL_SIZE_1`
        // tiene 32 filtros, pero `ONSET_KERNEL_SIZE_1` (rama de onsets) tiene
        // tambien kernel (5, 5) y **si** esta activa. La invariante real es
        // que el conjunto de capas activas sea exactamente el esperado, sin
        // la capa comentada.
        let mut expected_layers = vec![
            "conv2d_1",
            "contours-reduced",
            "conv2d_2",
            "conv2d_3",
            "conv2d_4",
            "conv2d_5",
        ];
        expected_layers.sort_unstable();
        let mut actual: Vec<&str> = self.layers.iter().map(|l| l.name.as_str()).collect();
        actual.sort_unstable();
        check(
            actual == expected_layers,
            format!(
                "el conjunto de capas activas no coincide con el esperado.\n  esperado: {expected_layers:?}\n  encontrado: {actual:?}"
            ),
        );

        // Y en concreto, la capa de contornos con 32 filtros y kernel (5, 5)
        // no debe existir.
        check(
            !self.layers.iter().any(|l| {
                l.kernel_tf == [5, 5] && l.filters == 32 && l.rust_name.starts_with("contour")
            }),
            "la capa comentada CONTOUR_KERNEL_SIZE_1 (32 filtros, kernel 5x5) \
             no debe estar en la configuracion"
                .to_string(),
        );

        if errores.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "la configuracion del modelo no cumple el contrato:\n{}",
                errores
                    .iter()
                    .map(|e| format!("  - {e}"))
                    .collect::<Vec<_>>()
                    .join("\n")
            ))
        }
    }
    /// Configuracion de ejemplo: los mismos valores que el asset
    /// `models/basic-pitch/icassp-2022/model_config.json`.
    ///
    /// Sirve para tests y para comprobar invariantes sin depender del
    /// sistema de archivos.
    pub fn example() -> ModelConfig {
        ModelConfig {
            model: ModelIdentity {
                name: "basic-pitch".into(),
                variant: "icassp-2022".into(),
                framework: "tensorflow".into(),
                reference_repo: "https://github.com/spotify/basic-pitch.git".into(),
                reference_commit: "fa5997af0a8210982619003269994a1be25eddf3".into(),
                reference_package_version: "0.4.0".into(),
                dtype: "float32".into(),
            },
            audio: AudioConfig {
                sample_rate: 22_050,
                channels: 1,
                window_length_s: 2,
                fft_hop: 256,
                annotations_fps: 86,
                annot_n_frames: 172,
                audio_n_samples: 43_844,
            },
            frequency: FrequencyConfig {
                base_frequency: 27.5,
                n_semitones: 88,
                contours_bins_per_semitone: 3,
                notes_bins_per_semitone: 1,
                cqt_n_semitones: 103,
                cqt_n_bins: 309,
                n_freq_bins_contours: 264,
                n_freq_bins_notes: 88,
            },
            cqt: CqtConfig {
                sr: 22_050,
                fmin: 27.5,
                n_bins: 309,
                bins_per_octave: 36,
                hop_length: 256,
                n_fft: 256,
                q_factor: 51.44,
                n_octaves: 9,
                n_filters: 36,
                early_downsample: false,
                downsample_factor: 1,
                pad_mode: "reflect".into(),
                basis_norm: 1,
                kernels_asset: "cqt_kernels.safetensors".into(),
            },
            harmonic_stacking: HarmonicStackingConfig {
                harmonics: vec![0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                shifts: vec![-36, 0, 36, 57, 72, 84, 93, 101],
                bins_per_semitone: 3,
                n_output_freqs: 264,
                input_n_bins: 309,
                padding: "asimetrico".into(),
            },
            batch_norm: BatchNormConfig {
                epsilon: 0.001,
                epsilon_source: "keras".into(),
                note: String::new(),
                layers: vec![
                    BatchNormLayerConfig {
                        name: "batch_normalization".into(),
                        channels: 1,
                        position: "inicial".into(),
                    },
                    BatchNormLayerConfig {
                        name: "batch_normalization_2".into(),
                        channels: 8,
                        position: "contornos".into(),
                    },
                    BatchNormLayerConfig {
                        name: "batch_normalization_3".into(),
                        channels: 32,
                        position: "onsets".into(),
                    },
                ],
            },
            layers: vec![
                LayerConfig {
                    name: "conv2d_1".into(),
                    rust_name: "contour.conv1".into(),
                    kernel_tf: [3, 39],
                    filters: 8,
                    in_channels: 8,
                    stride_tf: [1, 1],
                    in_shape_nhwc: [172, 264, 8],
                    out_shape_nhwc: [172, 264, 8],
                    pad_same_tf: [[1, 1], [19, 19]],
                    activation: "relu".into(),
                    batch_norm: true,
                    note: String::new(),
                },
                LayerConfig {
                    name: "contours-reduced".into(),
                    rust_name: "contour.reduced".into(),
                    kernel_tf: [5, 5],
                    filters: 1,
                    in_channels: 8,
                    stride_tf: [1, 1],
                    in_shape_nhwc: [172, 264, 8],
                    out_shape_nhwc: [172, 264, 1],
                    pad_same_tf: [[2, 2], [2, 2]],
                    activation: "sigmoid".into(),
                    batch_norm: false,
                    note: String::new(),
                },
                LayerConfig {
                    name: "conv2d_2".into(),
                    rust_name: "notes.conv1".into(),
                    kernel_tf: [7, 7],
                    filters: 32,
                    in_channels: 1,
                    stride_tf: [1, 3],
                    in_shape_nhwc: [172, 264, 1],
                    out_shape_nhwc: [172, 88, 32],
                    pad_same_tf: [[3, 3], [2, 2]],
                    activation: "relu".into(),
                    batch_norm: false,
                    note: String::new(),
                },
                LayerConfig {
                    name: "conv2d_3".into(),
                    rust_name: "notes.conv2".into(),
                    kernel_tf: [7, 3],
                    filters: 1,
                    in_channels: 32,
                    stride_tf: [1, 1],
                    in_shape_nhwc: [172, 88, 32],
                    out_shape_nhwc: [172, 88, 1],
                    pad_same_tf: [[3, 3], [1, 1]],
                    activation: "sigmoid".into(),
                    batch_norm: false,
                    note: String::new(),
                },
                LayerConfig {
                    name: "conv2d_4".into(),
                    rust_name: "onset.conv1".into(),
                    kernel_tf: [5, 5],
                    filters: 32,
                    in_channels: 8,
                    stride_tf: [1, 3],
                    in_shape_nhwc: [172, 264, 8],
                    out_shape_nhwc: [172, 88, 32],
                    pad_same_tf: [[2, 2], [1, 1]],
                    activation: "relu".into(),
                    batch_norm: true,
                    note: String::new(),
                },
                LayerConfig {
                    name: "conv2d_5".into(),
                    rust_name: "onset.conv2".into(),
                    kernel_tf: [3, 3],
                    filters: 1,
                    in_channels: 33,
                    stride_tf: [1, 1],
                    in_shape_nhwc: [172, 88, 33],
                    out_shape_nhwc: [172, 88, 1],
                    pad_same_tf: [[1, 1], [1, 1]],
                    activation: "sigmoid".into(),
                    batch_norm: false,
                    note: String::new(),
                },
            ],
            concat: ConcatConfig {
                name: "concat".into(),
                axis: 3,
                order: vec!["x_notes_pre".into(), "x_onset".into()],
                channels: 33,
                note: String::new(),
            },
            windowing: WindowingConfig {
                n_overlapping_frames: 30,
                overlap_len: 7_680,
                hop_size: 36_164,
                initial_padding: 3_840,
                frames_per_window: 172,
                frames_kept_per_window: 142,
                magic_alignment_offset: 0.0018,
                compatibility_note: String::new(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn una_configuracion_correcta_se_valida() {
        if let Err(e) = ModelConfig::example().validate() {
            panic!("{e}");
        }
    }

    #[test]
    fn se_rechazan_264_bins_en_el_campo_que_usa_el_frontend() {
        // Este es el hueco que dejo pasar el bug B-01: `validate()`
        // comprobaba `frequency.cqt_n_bins` (309) pero el frontend lee
        // `cqt.n_bins`. Con solo romper este segundo campo, la
        // configuracion pasaba la validacion y el modelo usaba 264 bins.
        let mut config = ModelConfig::example();
        config.cqt.n_bins = 264;
        config.frequency.cqt_n_bins = 309;
        let error = config.validate().expect_err("deberia rechazar 264 bins");
        assert!(error.contains("cqt.n_bins"), "mensaje: {error}");
    }

    #[test]
    fn se_rechaza_que_los_dos_campos_de_bins_no_coincidan() {
        let mut config = ModelConfig::example();
        config.cqt.n_bins = 300;
        let error = config
            .validate()
            .expect_err("deberia rechazar la discrepancia");
        assert!(error.contains("no coinciden"), "mensaje: {error}");
    }

    #[test]
    fn se_rechazan_shifts_que_no_cuadran_con_los_armonicos() {
        // El codigo recalcula los shifts a partir de los armonicos, asi
        // que un `shifts` equivocado en el JSON seria una mentira
        // silenciosa.
        let mut config = ModelConfig::example();
        config.harmonic_stacking.shifts = vec![0, 0, 0, 0, 0, 0, 0, 0];
        let error = config.validate().expect_err("deberia rechazar los shifts");
        assert!(error.contains("shifts"), "mensaje: {error}");
    }

    #[test]
    fn se_rechaza_un_numero_de_bins_de_salida_distinto_de_264() {
        let mut config = ModelConfig::example();
        config.harmonic_stacking.n_output_freqs = 200;
        assert!(config.validate().is_err());
    }

    #[test]
    fn se_rechazan_264_bins_de_cqt_en_el_resumen_de_frecuencia() {
        let mut config = ModelConfig::example();
        config.frequency.cqt_n_bins = 264;
        assert!(config.validate().is_err());
    }

    #[test]
    fn se_rechaza_un_epsilon_distinto_de_keras() {
        let mut config = ModelConfig::example();
        config.batch_norm.epsilon = 1e-5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn se_rechazan_32_canales_de_concatenacion() {
        let mut config = ModelConfig::example();
        config.concat.channels = 32;
        assert!(config.validate().is_err());
    }

    #[test]
    fn se_rechaza_el_orden_invertido_de_la_concatenacion() {
        let mut config = ModelConfig::example();
        config.concat.order = vec!["x_onset".into(), "x_notes_pre".into()];
        assert!(config.validate().is_err());
    }

    #[test]
    fn se_rechaza_la_capa_5x5_comentada_de_la_rama_de_contornos() {
        // CONTOUR_KERNEL_SIZE_1: 32 filtros, kernel (5, 5), en la rama de
        // contornos. Esta comentada en el upstream y no debe existir.
        let mut config = ModelConfig::example();
        config.layers.push(LayerConfig {
            name: "conv2d_comentado".into(),
            rust_name: "contour.comentado".into(),
            kernel_tf: [5, 5],
            filters: 32,
            in_channels: 8,
            stride_tf: [1, 1],
            in_shape_nhwc: [172, 264, 8],
            out_shape_nhwc: [172, 264, 32],
            pad_same_tf: [[2, 2], [2, 2]],
            activation: "linear".into(),
            batch_norm: false,
            note: String::new(),
        });
        assert!(config.validate().is_err());
    }

    #[test]
    fn la_capa_de_onsets_con_kernel_5x5_si_es_valida() {
        // ONSET_KERNEL_SIZE_1 tambien tiene kernel (5, 5), pero es ACTIVA:
        // la invariante no debe confundirla con la comentada.
        let config = ModelConfig::example();
        assert!(config.validate().is_ok());
        assert!(config
            .layers
            .iter()
            .any(|l| l.name == "conv2d_4" && l.kernel_tf == [5, 5]));
    }
}

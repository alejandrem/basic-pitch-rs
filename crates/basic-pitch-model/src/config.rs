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
    pub fn validate(&self) -> Result<(), String> {
        if self.frequency.cqt_n_bins != 309 {
            return Err(format!(
                "la CQT debe producir 309 bins, no {}",
                self.frequency.cqt_n_bins
            ));
        }
        if self.frequency.n_freq_bins_contours != 264 {
            return Err(format!(
                "la representacion de contornos debe tener 264 bins, no {}",
                self.frequency.n_freq_bins_contours
            ));
        }
        if self.concat.channels != 33 {
            return Err(format!(
                "la concatenacion de onsets debe producir 33 canales, no {}",
                self.concat.channels
            ));
        }
        if self.concat.order != ["x_notes_pre", "x_onset"] {
            return Err(format!(
                "el orden de la concatenacion debe ser [x_notes_pre, x_onset], no {:?}",
                self.concat.order
            ));
        }
        if self.batch_norm.epsilon != 0.001 {
            return Err(format!(
                "el epsilon de BatchNorm debe ser 0.001 (Keras), no {}",
                self.batch_norm.epsilon
            ));
        }
        if self.windowing.frames_kept_per_window != 142 {
            return Err(format!(
                "deben conservarse 142 frames por ventana, no {}",
                self.windowing.frames_kept_per_window
            ));
        }
        if self.windowing.frames_per_window != 172 {
            return Err(format!(
                "el modelo debe emitir 172 frames por ventana, no {}",
                self.windowing.frames_per_window
            ));
        }
        // Las dos capas con kernel (5, 5) son concretas: la comentada no debe
        // existir en la configuracion.
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
        if actual != expected_layers {
            return Err(format!(
                "el conjunto de capas activas no coincide con el esperado.\n\
                 esperado: {expected_layers:?}\n\
                 encontrado: {actual:?}"
            ));
        }

        // Y en concreto, la capa de contornos con 32 filtros y kernel (5, 5)
        // no debe existir.
        if self
            .layers
            .iter()
            .any(|l| l.kernel_tf == [5, 5] && l.filters == 32 && l.rust_name.starts_with("contour"))
        {
            return Err(
                "la capa comentada CONTOUR_KERNEL_SIZE_1 (32 filtros, kernel 5x5) \
                 no debe estar en la configuracion"
                    .into(),
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config_de_ejemplo() -> ModelConfig {
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
                layers: Vec::new(),
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

    #[test]
    fn una_configuracion_correcta_se_valida() {
        assert!(config_de_ejemplo().validate().is_ok());
    }

    #[test]
    fn se_rechazan_264_bins_de_cqt() {
        let mut config = config_de_ejemplo();
        config.frequency.cqt_n_bins = 264;
        assert!(config.validate().is_err());
    }

    #[test]
    fn se_rechazan_32_canales_de_concatenacion() {
        let mut config = config_de_ejemplo();
        config.concat.channels = 32;
        assert!(config.validate().is_err());
    }

    #[test]
    fn se_rechaza_el_orden_invertido_de_la_concatenacion() {
        let mut config = config_de_ejemplo();
        config.concat.order = vec!["x_onset".into(), "x_notes_pre".into()];
        assert!(config.validate().is_err());
    }

    #[test]
    fn se_rechaza_un_epsilon_distinto_de_keras() {
        let mut config = config_de_ejemplo();
        config.batch_norm.epsilon = 1e-5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn se_rechaza_la_capa_5x5_comentada_de_la_rama_de_contornos() {
        // CONTOUR_KERNEL_SIZE_1: 32 filtros, kernel (5, 5), en la rama de
        // contornos. Esta comentada en el upstream y no debe existir.
        let mut config = config_de_ejemplo();
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
        let config = config_de_ejemplo();
        assert!(config.validate().is_ok());
        assert!(config
            .layers
            .iter()
            .any(|l| l.name == "conv2d_4" && l.kernel_tf == [5, 5]));
    }
}

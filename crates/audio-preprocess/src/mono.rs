//! Conversion a mono por promedio de canales.

use basic_pitch_audio_io::AudioBuffer;

/// Convierte una senal multicanal a mono **promediando** los canales.
///
/// `librosa.to_mono` usa `np.mean` sobre el eje de los canales: no suma, no
/// elige un canal. El promedio se calcula en `f64` y se redondea a `f32` al
/// final, que es lo que hace NumPy con una entrada `float32`.
///
/// # Panicos
///
/// Si `channels` es 0, la funcion devuelve un buffer vacio en lugar de
/// dividir por cero.
pub fn to_mono(buffer: &AudioBuffer) -> AudioBuffer {
    if buffer.channels <= 1 {
        // Un buffer mono (o vacio) ya es mono.
        return buffer.clone();
    }

    let channels = buffer.channels as usize;
    let frames = buffer.frames();
    let mut out = Vec::with_capacity(frames);

    for frame in 0..frames {
        let base = frame * channels;
        let mut sum = 0.0f64;
        for ch in 0..channels {
            sum += buffer.samples[base + ch] as f64;
        }
        out.push((sum / channels as f64) as f32);
    }

    AudioBuffer::mono(out, buffer.sample_rate)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn el_promedio_de_tres_canales_es_el_esperado() {
        let buffer = AudioBuffer {
            samples: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            sample_rate: 22_050,
            channels: 3,
        };
        let mono = to_mono(&buffer);
        assert_eq!(mono.channels, 1);
        assert_eq!(mono.frames(), 2);
        // (1+2+3)/3 = 2 y (4+5+6)/3 = 5
        assert_eq!(mono.samples, vec![2.0, 5.0]);
    }

    #[test]
    fn un_buffer_mono_no_se_modifica() {
        let buffer = AudioBuffer::mono(vec![0.1, 0.2], 22_050);
        let mono = to_mono(&buffer);
        assert_eq!(mono.samples, buffer.samples);
        assert_eq!(mono.channels, 1);
    }

    #[test]
    fn dos_canales_en_contrafase_se_promedian_a_cero() {
        let buffer = AudioBuffer {
            samples: vec![1.0, -1.0, 0.5, -0.5],
            sample_rate: 22_050,
            channels: 2,
        };
        let mono = to_mono(&buffer);
        assert_eq!(mono.samples, vec![0.0, 0.0]);
    }

    #[test]
    fn un_canal_con_mas_amplitude_pondera_mas() {
        let buffer = AudioBuffer {
            samples: vec![0.0, 2.0],
            sample_rate: 22_050,
            channels: 2,
        };
        let mono = to_mono(&buffer);
        assert_eq!(mono.samples, vec![1.0]);
    }
}

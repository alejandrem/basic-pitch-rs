# Fase 8: inferencia sobre audios de duración arbitraria

**Objetivo:** procesar archivos completos, aunque sean considerablemente más largos que una ventana de dos segundos.

El modelo recibe ventanas de longitud fija, pero el usuario necesita enviar canciones completas.

## 8.1. Procesamiento por ventanas

El módulo `inference.rs` coordinará:

1. Recepción de audio ya preprocesado.
2. Generación de ventanas.
3. Ejecución del modelo por ventana.
4. Almacenamiento de predicciones.
5. Registro de la posición temporal de cada ventana.

No se deberá cargar innecesariamente todo el audio y todas las predicciones intermedias en memoria si el archivo es largo.

Se implementará un flujo por lotes de tamaño controlado.

## 8.2. Reconstrucción de predicciones

Las ventanas se superponen para reducir problemas en los límites.

Por eso las predicciones no se pueden concatenar directamente.

Se deberá reproducir la lógica oficial de reconstrucción o `unwrap_output`, considerando:

- Número de frames superpuestos.
- Frames que deben descartarse en cada límite.
- Desplazamiento entre ventanas.
- Longitud original del audio.
- Cantidad de predicciones válidas.
- Padding inicial y final.

La reconstrucción debe evitar duplicar notas o perder eventos en los límites.

## 8.3. Conversión de frames a tiempo

Se deberá implementar la conversión temporal utilizada por Basic Pitch.

No se utilizará simplemente:

```text
tiempo = índice_del_frame / 86
```

La referencia utiliza `FFT_HOP`, la frecuencia de muestreo y una corrección temporal que depende de la ventana.

Se deberán reproducir también los ajustes de alineación definidos en la implementación original, incluido el desplazamiento de alineación correspondiente.

### 8.3.1. Corrección: reproducir literalmente, incluida la inconsistencia 172 / 142

La numeración de ventanas que usa la referencia **no es coherente con los
frames que realmente se emiten**, y esa incoherencia forma parte del
comportamiento observable. Debe reproducirse tal cual, sin "corregirla".

En `basic_pitch/note_creation.py`:

```python
def model_frames_to_time(n_frames):
    original_times = librosa.core.frames_to_time(
        np.arange(n_frames), sr=AUDIO_SAMPLE_RATE, hop_length=FFT_HOP
    )
    window_numbers = np.floor(np.arange(n_frames) / ANNOT_N_FRAMES)   # <-- divide por 172
    window_offset = (FFT_HOP / AUDIO_SAMPLE_RATE) * (
        ANNOT_N_FRAMES - (AUDIO_N_SAMPLES / FFT_HOP)
    ) + MAGIC_ALIGNMENT_OFFSET
    times = original_times - (window_offset * window_numbers)
    return times
```

Mientras que en `basic_pitch/inference.py`, `unwrap_output` conserva:

```python
n_frames_per_window = (AUDIO_WINDOW_LENGTH * ANNOTATIONS_FPS) - n_overlapping_frames  # 172 - 30 = 142
```

Es decir: **la numeración de ventanas divide entre 172, pero solo sobreviven
142 frames por ventana**. El desplazamiento temporal aplicado por ventana se
acumula más lentamente de lo que "debería". Es un defecto del upstream, pero
corregirlo cambiaría los tiempos de las notas respecto a los de Spotify.

Contrato a reproducir literalmente:

| Elemento | Valor | Origen |
|---|---:|---|
| Frames emitidos por ventana | 172 | `ANNOT_N_FRAMES` |
| Frames conservados por ventana | 142 | `172 - 30` |
| Divisor de la numeración de ventanas | **172** | `np.floor(idx / ANNOT_N_FRAMES)` |
| `MAGIC_ALIGNMENT_OFFSET` | 0.0018 s | `note_creation.py` |
| Truncado final | `int(n_expected_windows * 142)` | `unwrap_output` |

Además, en `unwrap_output` el número de ventanas esperadas se calcula con
**división flotante**, no con `ceil`:

```python
n_expected_windows = audio_original_length / hop_size   # float, no ceil
```

donde `audio_original_length` es la longitud **original**, anterior al padding
inicial de 3,840 ceros, y `hop_size` es 36,164 muestras. El truncado final es
`int(n_expected_windows * 142)` frames.

Documentación requerida: esta inconsistencia debe quedar registrada en el
código y en la documentación del proyecto como **comportamiento de
compatibilidad**, con una nota que explique que no es un error de la
implementación en Rust.

## 8.4. Pruebas de reconstrucción

Se realizarán pruebas con:

- Una canción que no llegue a completar una ventana.
- Una canción con exactamente una ventana.
- Una canción que genere dos ventanas.
- Una canción con muchas ventanas.
- Una nota que comience antes de un límite.
- Una nota que termine después de un límite.
- Una nota sostenida durante varias ventanas.

**Entregable de la fase:** inferencia completa sobre archivos largos, con reconstrucción temporal correcta.

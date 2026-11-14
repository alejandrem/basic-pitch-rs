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

## 8.5. Implementación en Rust

`crates/transcription/src/stitching.rs`.

### 8.5.1. Las piezas

| Símbolo | Qué hace |
|---|---|
| `expected_frames(len, hop)` | Frames que sobreviven: `int(len / hop * 142)`. Se conoce **antes** de inferir |
| `unwrap_output(tensor, len, overlap, hop)` | La versión "por lotes": reproduce el `unwrap_output` del upstream |
| `Unwrapper` | La versión **incremental**: se le pasan las ventanas una a una |
| `ModelUnwrapper` | Lo mismo para las tres salidas del modelo a la vez |

`unwrap_output` está implementado **sobre** `Unwrapper`: una sola
implementación, dos puntos de entrada. No es una copia del algoritmo.

### 8.5.2. El flujo por lotes (apartado 8.1)

El requisito de no cargar todas las predicciones intermedias se cumple
con `ModelUnwrapper`: la predicción de cada ventana se incorpora a la
reconstrucción y se suelta en el acto.

```
Salidas neuronales: 296 KiB por ventana × 36.6 ventanas/min = 10.6 MiB/min
                                            → 634 MiB por hora de audio
```

Con la reconstrucción incremental, solo queda en memoria la matriz
reconstruida (que es el resultado final, y que el upstream también
construye entera).

El número de frames que sobreviven se calcula **antes** de inferir,
porque depende solo de la longitud del audio:

```rust
let n_frames = expected_frames(windows.original_length(), HOP_SIZE);
let mut reconstruccion = ModelUnwrapper::new(n_frames, N_OVERLAPPING_FRAMES);
```

### 8.5.3. Una sola fuente de verdad para las constantes

Las constantes del ventaneo viven en `basic-pitch-audio-preprocess` (que es
quien produce las ventanas) y `stitching.rs` las **reexporta**. Así, la
reconstrucción y el ventaneo no pueden desincronizarse: si el solapamiento
cambia, cambia en los dos sitios a la vez.

```rust
pub const N_OVERLAPPING_FRAMES: usize = basic_pitch_audio_preprocess::N_OVERLAPPING_FRAMES;
pub const ANNOT_N_FRAMES: usize = basic_pitch_audio_preprocess::ANNOT_N_FRAMES;
pub const FRAMES_KEPT_PER_WINDOW: usize = basic_pitch_audio_preprocess::FRAMES_KEPT_PER_WINDOW;
pub const HOP_SIZE: usize = basic_pitch_audio_preprocess::HOP_SIZE;
```

## 8.6. Validación

### 8.6.1. Los siete escenarios

`crates/transcription/tests/reconstruction.rs` (10 tests) prueba los siete
escenarios del apartado 8.4, **a nivel de frames**, que es donde vive la
reconstrucción.

La técnica: cada frame lleva codificado su índice global en el bin 0, y
una "nota" ficticia ocupa el bin 1 de un rango de frames.

- El **bin 0** demuestra que los frames salen en orden, **sin huecos y sin
  duplicados**.
- El **bin 1** demuestra que la nota cruza el límite sin perder frames ni
  duplicarse.
- Los frames del borde de cada ventana (los 15 que se descartan) se
  rellenan con un **centinela**: si la reconstrucción no los descarta, el
  test falla al instante.

| Escenario | Audio | Ventanas | Frames |
|---|---|---:|---:|
| Menos de una ventana | 1 s | 1 | 86 |
| Exactamente una ventana | 2 s | 2 | 172 |
| Dos ventanas | 2 × hop | 3 | 284 |
| Muchas ventanas | 1 min | 61 | 8.520 |
| Nota que empieza antes del límite | 2 × hop | 3 | 284 |
| Nota que termina después del límite | 2 × hop | 3 | 284 |
| Nota sostenida entre ventanas | 4 × hop | 5 | 568 |

**Los tests tienen dientes.** Se comprobó por mutación: introduciendo
errores deliberados en la implementación, los tests fallan así:

| Mutación | Tests que fallan |
|---|---:|
| No recortar los 15 frames del borde | 7 de 10 |
| Conservar 172 frames en vez de 142 | 7 de 10 |
| Truncar con la longitud rellenada | 5 de 10 |
| Usar `ceil` en vez de división flotante | 3 de 10 |
| Concatenar en el eje equivocado | 6 de 10 |

### 8.6.2. Contra el upstream, de extremo a extremo

`tools/weight-export/reference_reconstruction.py` genera el oráculo
usando la función **`basic_pitch.inference.unwrap_output` del upstream, sin
modificar**, sobre audio real. No es una reimplementación: es el juez.

`crates/transcription/tests/reconstruction_reference.rs` recorre la cadena
completa (archivo → ventanas → red → reconstrucción) y compara:

| Audio | Etapa | Diferencia máxima | MAE |
|---|---|---:|---:|
| `vocadito_10` | note | 1.45e-5 | 2.27e-7 |
| `vocadito_10` | onset | 6.49e-5 | 1.21e-6 |
| `vocadito_10` | contour | 2.95e-5 | 4.81e-7 |
| `vocadito_14` | note | 1.37e-5 | 1.45e-7 |
| `vocadito_14` | onset | 8.54e-5 | 7.94e-7 |
| `vocadito_14` | contour | 3.69e-5 | 3.27e-7 |

Y las formas coinciden exactamente:

| Audio | Longitud original | Ventanas | Frames |
|---|---:|---:|---:|
| `vocadito_10` | 200.607 | 6 | 787 |
| `vocadito_14` | 268.962 | 8 | 1.056 |

Que es exactamente `int(longitud / 36164 * 142)`, con la división
flotante del upstream.

### 8.6.3. La armonía entre P1, P2 y P4

Los tres bloques encajan porque:

1. **P1** produce las ventanas con el solapamiento de 30 frames.
2. **P2** devuelve las salidas en `(batch, 172, n_freqs)`, que es
   **exactamente** el layout que espera el `unwrap_output` del upstream
   (tiempo antes que frecuencia). Por eso la reconstrucción no necesita
   ninguna conversión.
3. **P4** recorta 15 frames por lado y concatena.

> **Nota histórica.** El punto 2 solo es cierto desde la corrección del
> bug B-32. Con el `transpose(1, 2)` de más, las salidas del modelo eran
> `(batch, 88, 172)` y la reconstrucción habría producido matrices
> invertidas sin ningún error visible.

## 8.7. Lo que queda de esta fase

- La **detección de notas** (Fase 9) sigue sin implementarse:
  `to_notes` devuelve una lista vacía. La reconstrucción ya entrega las
  matrices correctas, que es lo que necesita.
- La conversión frames→tiempo (`frame_to_time`) está implementada y
  probada, pero su uso en la detección de notas viene con la Fase 9.

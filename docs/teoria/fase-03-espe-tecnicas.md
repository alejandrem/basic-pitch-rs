# 3. Especificaciones técnicas de referencia

Antes de desarrollar las capas, se establecerá un contrato técnico basado en una versión concreta del código oficial.

No se deberá trabajar directamente contra una rama que pueda cambiar sin registrar el commit utilizado. Se deberá fijar la versión del repositorio de referencia y conservar los archivos de configuración correspondientes.

## 3.1. Parámetros principales del modelo

La implementación de referencia utiliza los siguientes valores:

| Parámetro | Valor |
|---|---:|
| Frecuencia de muestreo | 22,050 Hz |
| Canales de entrada | 1 |
| Duración nominal de ventana | 2 segundos |
| `FFT_HOP` | 256 muestras |
| `ANNOTATIONS_FPS` | 86 |
| `ANNOT_N_FRAMES` | 172 |
| `AUDIO_N_SAMPLES` | 43,844 |
| Frecuencia base | 27.5 Hz |
| Semitonos musicales | 88 |
| Bins de contorno por semitono | 3 |
| **Bins de la CQT (entrada del modelo)** | **309** |
| Bins de contorno finales (salida de Harmonic Stacking) | 264 |
| Bins de notas | 88 |

**Importante:** la longitud exacta de entrada es de 43,844 muestras, no 43,834 ni 44,100. La constante del modelo se calcula mediante:

```python
AUDIO_N_SAMPLES = AUDIO_SAMPLE_RATE * AUDIO_WINDOW_LENGTH - FFT_HOP
```

Por tanto:

```text
22050 × 2 - 256 = 43844
```

Este valor deberá conservarse en la configuración de Rust y utilizarse para crear las ventanas que recibe el modelo.

## 3.1.1. Corrección crítica: 309 bins en la entrada, 264 en la salida

La CQT **no** produce 264 bins. Produce **309**. La reducción a 264 ocurre
después, dentro de Harmonic Stacking.

El cálculo es el siguiente (`models.py`, función `get_cqt`):

```python
MAX_N_SEMITONES = int(np.floor(12.0 * np.log2(0.5 * AUDIO_SAMPLE_RATE / ANNOTATIONS_BASE_FREQUENCY)))
# -> 103

n_semitones = np.min([
    int(np.ceil(12.0 * np.log2(n_harmonics))) + ANNOTATIONS_N_SEMITONES,   # 36 + 88 = 124
    MAX_N_SEMITONES,                                                        # 103
])
# -> 103

n_bins = n_semitones * CONTOURS_BINS_PER_SEMITONE   # 103 * 3 = 309
```

Por tanto la cadena de dimensiones del frontend es:

```text
CQT                      -> (batch, 172, 309)
NormalizedLog            -> (batch, 172, 309)
BatchNorm inicial        -> (batch, 172, 309, 1)
Harmonic Stacking        -> (batch, 172, 264, 8)     <-- aquí se trunca a 264
```

Los 264 bins de salida son necesarios para que los armónicos superiores
(desplazamientos de hasta +101 bins) tengan contenido que desplazar. Una
implementación que genere directamente 264 bins producirá un modelo
**numéricamente incorrecto** y el error se propagará de forma silenciosa hasta
las notas detectadas.

El valor de 309 queda confirmado de forma independiente en el artefacto ONNX
oficial, en el inicializador de normalización
`model_1/cq_t2010v2_1/Sqrt` con dimensiones `[309, 1, 1]`.

## 3.1.2. Constantes verificadas de la CQT

Extraídas de `basic_pitch/layers/nnaudio.py` y confirmadas contra los
inicializadores del ONNX oficial (`saved_models/icassp_2022/nmp.onnx`):

| Parámetro | Valor | Origen |
|---|---:|---|
| `fmin` | 27.5 Hz | `ANNOTATIONS_BASE_FREQUENCY` |
| `bins_per_octave` | 36 | `12 * CONTOURS_BINS_PER_SEMITONE` |
| `n_bins` | 309 | derivado, ver 3.1.1 |
| `hop_length` | 256 | `FFT_HOP` |
| Factor Q | 51.44 | `1 / (2^(1/36) - 1)` |
| `n_octaves` | 9 | `ceil(309 / 36)` |
| `n_filters` (kernel top-octave) | 36 | `min(bins_per_octave, n_bins)` |
| `fftLen` (`n_fft`) | 256 | `2 ** ceil(log2(ceil(Q*fs/fmin_t)))` |
| Early downsampling | **no** (`downsample_factor = 1`) | `early_downsample_count() == 0` |
| `pad_mode` | `reflect` | `ReflectionPad1D(n_fft // 2)` |
| `basis_norm` | 1 (norma L1) | `np.linalg.norm(sig, 1)` |
| Normalización final | `* sqrt(lengths)` | `lengths = ceil(Q*sr/freqs)`, 309 valores |

Los kernels CQT (real e imaginario, forma `[36, 1, 1, 256]`) y el filtro
lowpass `firwin2` (forma `[1, 1, 1, 256]`) **ya están disponibles** como
inicializadores en el ONNX oficial local:

| Inicializador ONNX | Dimensiones | Contenido |
|---|---|---|
| `const_fold_opt__664` | `[36, 1, 1, 256]` | kernel CQT (parte real) |
| `const_fold_opt__655` | `[36, 1, 1, 256]` | kernel CQT (parte imaginaria) |
| `const_fold_opt__734` | `[1, 1, 1, 256]` | filtro lowpass `firwin2` |

Esto permite extraerlos como asset durante la Fase 1 en lugar de reimplementar
`create_cqt_kernels` y `scipy.signal.firwin2` en Rust, sin perder fidelidad
numérica.

## 3.1.3. Desplazamientos armónicos

El plan de la Fase 5 define el cálculo del desplazamiento. Los valores
resultantes para la configuración por defecto son:

| Armónico | `shift` (bins) |
|---:|---:|
| 0.5 | -36 |
| 1 | 0 |
| 2 | 36 |
| 3 | 57 |
| 4 | 72 |
| 5 | 84 |
| 6 | 93 |
| 7 | 101 |

El desplazamiento negativo del armónico 0.5 desplaza el contenido hacia bins
**superiores** rellenando con ceros al principio; los positivos lo desplazan
hacia bins **inferiores** rellenando con ceros al final. La asimetría es
obligatoria.

## 3.1.4. Parámetros de ventanas superpuestas

Validados contra `basic_pitch/inference.py`:

| Parámetro | Valor | Origen |
|---|---:|---|
| `n_overlapping_frames` | 30 | `DEFAULT_OVERLAPPING_FRAMES` |
| `overlap_len` (muestras) | 7,680 | `30 * FFT_HOP` |
| `hop_size` (muestras) | 36,164 | `AUDIO_N_SAMPLES - overlap_len` |
| Padding inicial | 3,840 ceros | `overlap_len / 2` |
| Frames por ventana (modelo) | 172 | `ANNOT_N_FRAMES` |
| Frames conservados por ventana | 142 | `172 - n_overlapping_frames` |
| `MAGIC_ALIGNMENT_OFFSET` | 0.0018 s | `note_creation.py` |


## 3.2. Dimensiones principales

Las salidas musicales esperadas son:

- `note`: 88 valores por instante, correspondientes a las notas del rango del modelo.
- `onset`: 88 valores por instante, correspondientes a los comienzos de las notas.
- `contour`: 264 valores por instante, con una resolución de tres bins por semitono.

Estas dimensiones son fundamentales para el procesamiento posterior y deberán comprobarse automáticamente durante las pruebas.

## 3.3. Configuración de armónicos

La configuración predeterminada utiliza ocho desplazamientos armónicos:

```text
[0.5, 1, 2, 3, 4, 5, 6, 7]
```

Estos valores generan ocho canales de información armónica a partir de la representación CQT.

No deben confundirse con ocho capas convolucionales ni con ocho notas. Son desplazamientos de frecuencia utilizados para enriquecer la entrada del modelo.

## 3.4. Parámetros que deberán extraerse de la referencia

Además de los valores anteriores, se deberán obtener y documentar:

- `AUDIO_N_SAMPLES`.
- `AUDIO_SAMPLE_RATE`.
- `FFT_HOP`.
- `ANNOT_N_FRAMES`.
- `ANNOTATIONS_BASE_FREQUENCY`.
- `ANNOTATIONS_N_SEMITONES`.
- `CONTOURS_BINS_PER_SEMITONE`.
- `N_FREQ_BINS_CONTOURS`.
- Parámetros exactos de CQT.
- Configuración de Batch Normalization.
- Parámetros de ventanas superpuestas.
- Umbrales predeterminados del posprocesamiento.
- Parámetros de estimación de pitch bend.
- Desplazamientos temporales utilizados en la reconstrucción de las predicciones.

La implementación no deberá inventar valores faltantes ni sustituirlos por aproximaciones sin validación.

---

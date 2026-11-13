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
| Bins de contorno finales | 264 |
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

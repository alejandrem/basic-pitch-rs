4. Fase 0: auditoría de la implementación oficial

Objetivo: comprender exactamente qué modelo se va a reproducir antes de escribir su implementación en Rust.

Esta será la primera fase y una condición obligatoria para comenzar las siguientes.

4.1. Identificar la versión de referencia

Se deberá descargar o consultar el repositorio oficial de Spotify Basic Pitch:

https://github.com/spotify/basic-pitch

Se registrarán:

URL del repositorio.

Commit exacto utilizado.

Versión del paquete, cuando corresponda.

Versión de Python.

Versión de TensorFlow.

Versión de las dependencias que intervienen en el procesamiento de audio.

Ubicación y procedencia del modelo preentrenado.

Licencias de código y recursos utilizados.

La finalidad es que los resultados de referencia puedan reproducirse más adelante, incluso si el repositorio original cambia.

4.1.1. Versión fijada (ya resuelta)

| Dato | Valor |
|---|---|
| Repositorio | `https://github.com/spotify/basic-pitch.git` |
| Commit | `fa5997af0a8210982619003269994a1be25eddf3` |
| Fecha del commit | 2025-11-13 |
| Versión del paquete | `0.4.0` |
| Copia local | `basic-pitch/` (en el repositorio, ignorada por git) |
| Python de referencia | 3.11.18 |
| TensorFlow de referencia | 2.15.0 |
| librosa de referencia | 0.10.2.post1 |
| numpy de referencia | 1.26.4 |
| soxr de referencia | el que resuelva `librosa==0.10.2.post1` (`res_type='soxr_hq'`) |

El modelo preentrenado usado es `ICASSP_2022_MODEL_PATH`, que corresponde a
`basic-pitch/basic_pitch/saved_models/icassp_2022/`. Ya están presentes en
disco cuatro serializaciones del mismo modelo:

| Artefacto | Uso en este proyecto |
|---|---|
| `nmp/` (SavedModel de TF) | **Fuente canónica** de pesos y estadísticas BN |
| `nmp.onnx` | Cross-check, y extracción de kernels de CQT |
| `nmp.tflite` | Verificación opcional |
| `nmp.mlpackage/` | No se usa (CoreML) |

No se deberán mezclar pesos de versiones diferentes del modelo.

4.2. Inspeccionar la arquitectura real

Se analizará principalmente:

models.py

nn.py

inference.py

note_creation.py

constants.py

layers/nnaudio.py

layers/signal.py

Se documentará el recorrido completo de los datos desde el audio hasta las salidas del modelo.

No se deberá utilizar únicamente un diagrama conceptual de la red: se deberá reconstruir el grafo de operaciones que efectivamente utiliza el modelo entrenado.

4.3. Identificar las capas activas

Un detalle especialmente importante es que el código oficial contiene una primera capa convolucional de contornos que está comentada porque fue omitida accidentalmente durante el entrenamiento del modelo presentado originalmente.

Por tanto, no se deberá implementar esa capa comentada como parte de la red entrenada.

La implementación deberá reproducir el grafo activo, que comienza la rama de contornos con una convolución de ocho filtros y kernel (3, 39).

4.3.1. Corrección: distinguir dos capas de kernel (5, 5)

En `basic_pitch/models.py` existen **dos** capas convolucionales con kernel
`(5, 5)`. Una está comentada y la otra es activa. Confundirlas produce una red
silenciosamente incorrecta:

| Constante | Kernel | Filtros | Activación | Estado |
|---|---|---:|---|---|
| `CONTOUR_KERNEL_SIZE_1` | (5, 5) | 32 (`n_filters_contour`) | ninguna | **COMENTADA - omitir** |
| `CONTOUR_KERNEL_SIZE_3` | (5, 5) | 1 | sigmoid | **ACTIVA - implementar** |

La capa activa `CONTOUR_KERNEL_SIZE_3` produce la salida `contours-reduced`, que
alimenta la rama de notas y se conserva para la estimación de pitch bend.

Regla: **omitir `CONTOUR_KERNEL_SIZE_1`; implementar `CONTOUR_KERNEL_SIZE_3`.**

4.3.2. Grafo activo completo (verificado)

```text
entrada (batch, 43844, 1)
  -> FlattenAudioCh            -> (batch, 43844)
  -> CQT2010v2                 -> (batch, 172, 309)
  -> NormalizedLog             -> (batch, 172, 309)
  -> expand_dims               -> (batch, 172, 309, 1)
  -> BatchNorm inicial (1 ch)  -> (batch, 172, 309, 1)
  -> HarmonicStacking          -> (batch, 172, 264, 8)     <-- aquí se trunca a 264
       |
       +--> conv contour (8, 3x39) -> BatchNorm(8) -> ReLU
       |        -> conv contours-reduced (1, 5x5) + sigmoid
       |        -> FlattenFreqCh  -> (batch, 172, 264)      == contour
       |        -> expand_dims    -> (batch, 172, 264, 1)
       |        -> conv notes1 (32, 7x7, stride 1x3) -> ReLU
       |        -> conv notes2 (1, 7x3) + sigmoid        -> (batch, 172, 88, 1)
       |             |
       |             +--> x_notes_pre
       |
       +--> conv onset1 (32, 5x5, stride 1x3) -> BatchNorm(32) -> ReLU
                -> Concat(axis=3)([x_notes_pre, x_onset])  -> (batch, 172, 88, 33)
                -> conv onset2 (1, 3x3) + sigmoid          -> (batch, 172, 88)
```

4.3.3. Corrección: la concatenación produce 33 canales, en un orden fijo

`basic_pitch/models.py`:

```python
x_onset = tfkl.Concatenate(axis=3, name="concat")([x_notes_pre, x_onset])
```

- `x_notes_pre` es la salida **sigmoid de la conv de notas**, con forma
  `(batch, 172, 88, 1)`. **No** es la salida final `note` (que aplanada tiene
  88 valores).
- `x_onset` es la salida de la rama de onsets, `(batch, 172, 88, 32)`.
- El concatenado va sobre el **eje de canales** (eje 3 en NHWC), por lo que el
  resultado es `(batch, 172, 88, 33)`.

El **orden es obligatorio**: primero `x_notes_pre` (canal 0), después `x_onset`
(canales 1..32). Invertirlo silenciosamente produce un modelo que difiere del
original.

Confirmación independiente en el ONNX oficial: el kernel de la convolución
final de onsets es `const_fold_opt__680` con dimensiones `[1, 33, 3, 3]`, donde
los 33 canales de entrada son la prueba directa de esta concatenación.

4.3.4. Cross-check TensorFlow <-> ONNX

Además del inventario del modelo en TensorFlow, la auditoría deberá generar un
**manifiesto de trazabilidad** que relacione, para cada tensor:

- su nombre en TensorFlow (SavedModel),
- su nombre en ONNX,
- su nombre definitivo en Rust (Safetensors),
- sus dimensiones en los tres formatos,
- y el resultado de una comparación numérica tensor a tensor.

El artefacto ONNX oficial está disponible en el repositorio local en
`basic-pitch/basic_pitch/saved_models/icassp_2022/nmp.onnx`. Este cross-check
es lo que evita errores silenciosos: si un tensor cambia de nombre o de
dimensiones entre formatos, la Fase 1 lo detecta antes de escribir una línea
de Rust.

Nota: el ONNX tiene la BatchNorm **fusionada** (`FusedBatchNormV3`), por lo que
las estadísticas `moving_mean` / `moving_variance` originales solo están
disponibles en TensorFlow. El ONNX se usa para verificación y para extraer los
kernels deterministas de la CQT, no como fuente de las estadísticas de
normalización.

También se deberán identificar las salidas que alimentan a otras ramas. En particular, la salida de contornos reducida se utiliza como entrada de la rama de notas, mientras que la rama de onsets utiliza la representación armónica inicial y se combina con la salida de notas.

4.4. Generar un inventario del modelo

Se creará un script de inspección en Python que cargue el modelo original y genere un inventario de:

Nombre de cada capa.

Tipo de capa.

Número de filtros.

Dimensiones de kernel.

Strides.

Padding.

Activación.

Dimensiones de entrada.

Dimensiones de salida.

Parámetros entrenables.

Estadísticas de Batch Normalization.

Nombres y dimensiones de los tensores.

El resultado deberá guardarse en un archivo como:

reference/
├── model_inventory.json
├── model_summary.txt
├── model_config.json
├── tensor_provenance.json      <-- manifiesto TF <-> ONNX <-> Rust
├── reference_metadata.json
└── reference_outputs/

4.5. Identificar los puntos de comparación

Se deberán establecer puntos de inspección en el modelo original para poder guardar las salidas intermedias.

Como mínimo:

Audio de entrada.

CQT.

CQT normalizada.

CQT después de Batch Normalization.

Harmonic Stacking.

Salida de la convolución de contornos.

Contorno final.

Características de la rama de notas.

Predicción de notas.

Características de la rama de onsets.

Predicción de onsets.

Entregable de la fase: un documento de auditoría que describa las dimensiones, operaciones, parámetros y conexiones exactas del modelo.

Criterio de aceptación: no comenzar la implementación de las capas hasta conocer el grafo activo y disponer de una configuración de referencia reproducible.
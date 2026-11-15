# Fase 7: validación de equivalencia entre TensorFlow y Candle

**Objetivo:** demostrar que la implementación en Rust reproduce el comportamiento de la implementación original.

Esta será una de las etapas de mayor importancia del proyecto.

Que la red compile y produzca números no significa que esté funcionando correctamente.

## 7.1. Crear el programa de referencia

Se implementará un programa Python que:

1. Cargue el modelo oficial.
2. Reciba una señal de prueba.
3. Ejecute el preprocesamiento original.
4. Ejecute el modelo completo.
5. Capture las salidas intermedias.
6. Guarde los tensores de referencia.
7. Exporte las dimensiones y configuraciones utilizadas.

Los datos se guardarán en un formato que pueda utilizarse fácilmente desde Rust, por ejemplo NPY, NPZ o Safetensors.

## 7.2. Comparación por etapas

No se comparará únicamente la salida final.

Se deberán comparar:

| Etapa | Comparación |
|---|---|
| Entrada | Muestras de audio |
| CQT | Representación frecuencial |
| NormalizedLog | Valores normalizados |
| BatchNorm inicial | Características normalizadas |
| Harmonic Stacking | Canales armónicos |
| Contour branch | Activaciones intermedias |
| Note branch | Activaciones y predicciones |
| Onset branch | Activaciones y predicciones |
| Salidas finales | Nota, onset y contour |

Esto permitirá localizar los errores sin tener que revisar toda la red cada vez.

## 7.3. Métricas de comparación

Se implementarán métricas como:

- Error absoluto medio (MAE).
- Error absoluto máximo.
- Error cuadrático medio (MSE).
- Raíz del error cuadrático medio (RMSE).
- Similitud coseno.
- Diferencia relativa.

Las comparaciones deberán realizarse entre tensores de las mismas dimensiones y con el mismo significado.

No se establecerá automáticamente que un MAE inferior a `1e-4` sea un requisito universal. La tolerancia dependerá de la etapa, del algoritmo de CQT y de las diferencias numéricas entre operaciones.

Se establecerán umbrales documentados después de medir la referencia con el mismo conjunto de pruebas.

## 7.4. Casos de prueba

Se deberán preparar audios y señales controladas para evaluar:

1. Silencio.
2. Una nota sostenida.
3. Una nota corta.
4. Varias notas simultáneas.
5. Acordes.
6. Notas graves.
7. Notas agudas.
8. Transitorios rápidos.
9. Vibrato.
10. Deslizamientos de afinación.
11. Audio con ruido.
12. Audio de larga duración.
13. Audio con notas cerca de los límites entre ventanas.

Los archivos de referencia deberán contar con permisos de uso adecuados y su procedencia deberá documentarse.

## 7.5. Validación de resultados musicales

Además de comparar valores neuronales, se compararán eventos musicales:

- Nota detectada.
- Tiempo de inicio.
- Tiempo de finalización.
- Velocidad MIDI.
- Duración.
- Pitch bend.
- Notas simultáneas.

Dos resultados pueden presentar pequeñas diferencias numéricas y, aun así, producir la misma transcripción. También puede ocurrir lo contrario: una diferencia pequeña cerca de un umbral puede cambiar un evento musical.

Por eso las pruebas numéricas y musicales deberán ser complementarias.

**Entregable de la fase:** conjunto de pruebas de equivalencia y reporte de diferencias entre TensorFlow y Candle.

**Criterio de aceptación:** las diferencias estarán medidas, explicadas y dentro de las tolerancias establecidas para cada etapa.

---

## 7.6. Implementación en Rust

### 7.6.1. Las seis metricas

Viven en el crate `basic-pitch-validation` (solo desarrollo), para que
los validadores y los tests usen **la misma** aritmética y no haya dos
implementaciones que puedan divergir.

| Métrica | Definición |
|---|---|
| MAE | `mean(\|a - b\|)` |
| MSE | `mean((a - b)²)` |
| RMSE | `sqrt(MSE)`, en las mismas unidades que la señal |
| Error absoluto máximo | `max(\|a - b\|)` |
| Diferencia relativa máxima | `max(\|a - b\|) / max(\|b\|)`: el error respecto a la **escala** de la referencia |
| Similitud coseno | `(a · b) / (\|a\| \|b\|)`: si la **dirección** coincide, aunque las magnitudes no |

La diferencia relativa se define respecto a la escala de la referencia y
no a cada valor, porque es la que permite comparar etapas con escalas
distintas (una activación entre 0 y 1, un contour entre 0 y 1) sin que
el número absoluto diga nada.

La similitud coseno vale 1 cuando los dos tensores son paralelos. Es la
métrica que el cross-check de la Fase 0 usó entre TensorFlow y ONNX, donde
dio 1.0000000000.

### 7.6.2. El validador

```bash
cargo run --release --example validar_model -p basic-pitch-model
```

Mide las **once** etapas del pipeline (las cuatro del frontend acustico y
las siete de la red) y escribe el informe en
`reference/validation_results.json`.

```
OK   CQT                                         MAE 1.762e-7  MSE 8.035e-14  RMSE 2.835e-7  max 5.722e-6  rel 3.186e-7  cos 1.0000000000
OK   NormalizedLog                               MAE 1.010e-5  MSE 8.921e-10  RMSE 2.987e-5  max 1.378e-3  rel 1.378e-3  cos 0.9999999973
OK   BatchNorm inicial                           MAE 2.505e-5  MSE 5.490e-9   RMSE 7.410e-5  max 3.419e-3  rel 2.132e-3  cos 0.9999999782
OK   HarmonicStacking                            MAE 1.782e-5  MSE 3.001e-9   RMSE 5.479e-5  max 3.419e-3  rel 2.132e-3  cos 0.9999999885
OK   contour_conv                                MAE 4.974e-5  MSE 6.883e-9   RMSE 8.296e-5  max 1.924e-3  rel 4.171e-4  cos 0.9999999907
OK   contour_reduced                             MAE 1.328e-6  MSE 6.088e-12  RMSE 2.467e-6  max 4.426e-5  rel 5.380e-5  cos 0.9999999998
OK   contour                                     MAE 1.328e-6  MSE 6.088e-12  RMSE 2.467e-6  max 4.426e-5  rel 5.380e-5  cos 0.9999999998
OK   notes_features                              MAE 2.063e-6  MSE 1.461e-11  RMSE 3.822e-6  max 6.860e-5  rel 5.707e-5  cos 0.9999999999
OK   note                                        MAE 5.864e-7  MSE 1.645e-12  RMSE 1.283e-6  max 1.663e-5  rel 1.921e-5  cos 1.0000000000
OK   onset_features                              MAE 3.663e-5  MSE 4.830e-9   RMSE 6.950e-5  max 2.042e-3  rel 4.573e-4  cos 0.9999999834
OK   onset                                       MAE 2.653e-6  MSE 2.007e-11  RMSE 4.480e-6  max 5.338e-5  rel 7.936e-5  cos 0.9999999990
```

### 7.6.3. Los diecisiete casos de prueba

Las señales las genera `tools/weight-export/make_cases.py`, y sus
oráculos (salidas finales y notas detectadas) los calcula el **upstream**
con `tools/weight-export/make_cases_oracle.py`.

| Caso | Qué prueba | Ventanas | Notas (Rust = upstream) |
|---|---|---:|---:|
| `silencio` | que no se invente nada donde no hay nada | 2 | 2 = 2 |
| `nota_sostenida` | una nota con ataque y cauda limpios | 2 | 2 = 2 |
| `nota_corta` | notas por debajo de la duración mínima | 2 | 2 = 2 |
| `notas_simultaneas` | varias notas a la vez | 2 | 4 = 4 |
| `acordes` | progresión de acordes | 2 | 14 = 14 |
| `notas_graves` | el registro grave, con menos resolución | 3 | 6 = 6 |
| `notas_agudas` | el registro agudo | 3 | 3 = 3 |
| `transitorios_rapidos` | ataques muy seguidos | 2 | 0 = 0 |
| `vibrato` | la afinación oscila dentro de la nota | 2 | 2 = 2 |
| `deslizamiento` | un glissando, que no es ninguna nota | 2 | 5 = 5 |
| `ruido` | una nota con ruido de fondo | 2 | 1 = 1 |
| `duracion_larga` | 30 segundos, unas 19 ventanas | 19 | 63 = 63 |
| `limites_ventana` | notas a ambos lados del límite entre ventanas | 2 | 8 = 8 |
| `vibrato_sutil` | variaciones de afinación mínimas | 2 | 2 = 2 |
| `vibrato_extremo` | variaciones por encima del rango del pitch bend | 2 | 4 = 4 |
| `acorde_vibrato` | varias notas simultáneas, cada una con su vibrato | 2 | 6 = 6 |
| `notas_superpuestas` | dos notas del mismo pitch solapadas | 2 | 3 = 3 |

Los cuatro últimos son los de la **Fase 10** (pitch bend), y se validan
además nota a nota en
[`crates/transcription/tests/pitch_bend.rs`](../../crates/transcription/tests/pitch_bend.rs).

**Los diecisiete coinciden nota a nota con el upstream**: mismo número,
mismos pitches, mismos inicios y finales (con una tolerancia de un frame,
que se explica más abajo).

Dos detalles que salen de los tests:

- **El "silencio" no es silencio digital.** Lleva un residuo de ruido de
  1e-4, porque un silencio absoluto es degenerado para el modelo. El
  upstream también detecta dos notas en él, y lo que se comprueba es que
  Rust detecte **exactamente** las mismas.

- **La tolerancia de tiempo es de un frame** (11,6 ms), no de un
  microsegundo. El motivo está medido: un onset puede venir de los
  **onsets inferidos** (los cambios de la activación de nota), que se
  reescalan dividiendo por `max(frame_diff)`. Cuando dos frames vecinos
  tienen valores casi iguales (en `notas_simultaneas`, 0.0866 y 0.1054), el
  redondeo de `float32` decide si el máximo local cae en uno o en otro.
  Rust y NumPy hacen la misma fórmula, pero acumulan en orden distinto, así
  que pueden discrepar en un frame. El número de notas y los pitches sí se
  comparan de forma exacta: ahí no hay ambigüedad.

### 7.6.4. El criterio de las tolerancias

El MAE es la comprobación **estricta** (mide el error típico) y el máximo
va con margen, porque es la cola.

Medido en los diecisiete casos, el MAE está entre 2e-8 y 7e-6 en
**todos** ellos: el error de la red **no se acumula** con las ventanas (el caso de
30 segundos, con 19 ventanas, tiene el mismo MAE que uno de 2). Lo que
varía es el máximo (de 4e-7 a 1.03e-3), que depende de los valores
concretos de cada caso.

| Comprobación | Tolerancia |
|---|---:|
| MAE de las activaciones | 1e-4 |
| Máximo de las activaciones | 5e-3 |
| Número de notas | exacto |
| Pitch de cada nota | exacto |
| Inicio y final de cada nota | un frame (11,6 ms) |

### 7.6.5. Qué queda de esta fase

La **validación musical** del apartado 7.5 (comparar eventos musicales:
nota, tiempo, velocidad, duración, pitch bend y notas simultáneas) está
hecha **en parte**: el test de notas compara, nota a nota, el pitch, el
tiempo de inicio, el tiempo de finalización y la amplitud contra el
upstream. Lo que **no** está es el **pitch bend**, que es la Fase 10.

El módulo `crates/transcription/src/pitch_bend.rs` ya tiene el cálculo
implementado y probado, pero **no está conectado**: el campo `pitch_bend`
del evento se rellena con `None`. Falta el cableado, no el cálculo.

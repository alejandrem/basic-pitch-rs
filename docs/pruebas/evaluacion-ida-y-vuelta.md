# Evaluación contra una referencia: ida y vuelta sintética

Primera medición de **exactitud** del proyecto (hasta ahora solo se validaba
la **fidelidad** frente al upstream de Spotify). Herramienta:
`tools/eval/evaluar.py`, con `mir_eval.transcription`.

## Montaje

- Referencia: un MIDI de 1972 notas ("peso del silencio"). Procede de una
  transcripción de Basic Pitch editada a mano; para este experimento da igual,
  porque el audio se **genera** a partir de él y por tanto es la verdad
  absoluta de ese audio. No sirve como verdad contra el MP3 original.
- Audio: el MIDI renderizado con `tools/weight-export/render_midi.py`, en dos
  variantes: seno puro (`--limpio`) y soundfont.
- Métricas: onset a ±50 ms, pitch a ±50 cents. "F1 onset" ignora el final de la
  nota; "F1 on+off" exige además que el final caiga dentro de
  max(20 % de la duración, 50 ms).

```bash
cargo build --release
.venv/bin/python tools/eval/evaluar.py idavuelta <referencia.mid> --limpio
.venv/bin/python tools/eval/evaluar.py idavuelta <referencia.mid> --barrido
```

## Resultados

Seno puro, umbrales por defecto:

| frame_thr | Melodia | ref | est | P onset | R onset | F1 onset | F1 on+off |
|---|---|---|---|---|---|---|---|
| defecto | sí | 1972 | 1325 | 0.942 | 0.633 | 0.757 | 0.577 |

Soundfont, barrido de `frame_threshold`:

| frame_thr | Melodia | est | P onset | R onset | F1 onset | F1 on+off |
|---|---|---|---|---|---|---|
| 0.20 | sí | 2824 | 0.627 | 0.898 | 0.739 | 0.299 |
| 0.30 | sí | 2158 | 0.800 | 0.876 | 0.836 | 0.358 |
| 0.40 | sí | 1891 | 0.864 | 0.829 | 0.846 | 0.394 |
| 0.50 | sí | 1571 | 0.913 | 0.728 | 0.810 | 0.417 |
| 0.20 | no | 2587 | 0.680 | 0.892 | 0.772 | 0.312 |
| 0.30 | no | 2113 | 0.812 | 0.870 | 0.840 | 0.358 |
| 0.40 | no | 1847 | 0.879 | 0.824 | 0.850 | 0.394 |
| 0.50 | no | 1544 | 0.927 | 0.726 | 0.814 | 0.419 |

Deriva temporal (error de onset estimado − referencia): mediana de −2.0 ms en
el primer tercio y −1.9 ms en el último con seno puro; +4.9 ms y +5.0 ms con
soundfont. Las pendientes de la primera ejecución (+8.4 y +15.8 ms/min) eran un
ajuste por mínimos cuadrados que arrastraban emparejamientos erróneos; el
script usa ahora una pendiente robusta, así que esas cifras están superadas.

## Qué se puede afirmar

- **La deriva por la numeración 142/172 es despreciable** en esta pieza
  (≈0 ms entre el principio y el final). No hace falta un modo "corregido".
- **El umbral desplaza la precisión y el recall** de forma apreciable
  (de 0.30 a 0.50: P 0.81 → 0.93, R 0.87 → 0.73), pero el F1 de onset es
  casi plano entre 0.30 y 0.40 (0.836–0.850).
- **Melodia cambia poco** salvo con umbrales bajos: a 0.30–0.50 la
  diferencia de F1 es de ~0.004, que con una sola pieza no es concluyente.
- **El F1 con finales de nota es mucho menor que el de onset** (≈0.36–0.42
  frente a ≈0.84 con soundfont).

## Qué NO se puede afirmar con estos datos

- **Que los armónicos del timbre sean la causa de las notas espurias.** La
  precisión es menor con soundfont que con seno puro, lo que es compatible con
  esa hipótesis, pero la comparación está confundida: el seno puro es un timbre
  fuera de la distribución con la que se entrenó el modelo, y el render
  normaliza el nivel global, con lo que las notas individuales quedan más
  bajas en pasajes densos. Un recall de 0.63 con seno no prueba que "el timbre
  ayude a detectar".
- **Que el F1 on+off bajo se deba a los armónicos.** Una explicación más
  simple y también compatible con los datos: con soundfont el sonido sigue
  después del note-off del MIDI (decaimiento y release), así que el modelo ve
  notas más largas que la referencia. Encaja con que subir el umbral (que
  corta antes cada nota) mejore el F1 on+off (0.30 → 0.42) mientras el de
  onset se estanca. Con seno puro, cuyo final es nítido, la caída es mucho
  menor (0.76 → 0.58). Sigue siendo una hipótesis.
- **Que estos números valgan para audio real.** Es un render limpio, sin
  ruido, reverberación ni mezcla: un techo optimista. Una sola pieza, un solo
  soundfont, y un MIDI cuyo contenido salió originalmente de Basic Pitch, lo
  que puede favorecerlo ligeramente.
- **Que los umbrales por defecto deban cambiar.** La mejora de 0.3 a 0.4 es de
  0.01 de F1 en una pieza.

## Barrido del umbral de onset

Soundfont, resto de umbrales por defecto (`frame_threshold` 0.3, con Melodia):

| onset_threshold | notas est. | P onset | R onset | F1 onset | F1 on+off |
|---|---|---|---|---|---|
| 0.3 | 2584 | 0.624 | 0.818 | 0.708 | 0.416 |
| 0.4 | 2360 | 0.711 | 0.850 | 0.774 | 0.402 |
| 0.5 (defecto) | 2158 | 0.800 | 0.876 | 0.836 | 0.358 |
| 0.6 | 2045 | 0.855 | 0.887 | 0.871 | 0.327 |
| 0.7 | 1970 | 0.887 | 0.886 | 0.887 | 0.302 |
| 0.8 | 1880 | 0.907 | 0.865 | 0.885 | 0.277 |
| 0.9 | 1680 | 0.885 | 0.754 | 0.814 | 0.228 |

- En los rangos probados, el onset mueve el F1 más que el `frame_threshold`
  (0.71→0.89 frente a 0.74→0.85).
- **El F1 de onset alcanza su máximo en una meseta, no en un punto:** 0.6, 0.7
  y 0.8 dan 0.871, 0.887 y 0.885. Con una sola pieza, 0.7 y 0.8 son
  indistinguibles; a 0.9 el F1 cae a 0.814 porque el recall se hunde.
- **El F1 que exige los finales de nota baja de forma continua al subir el
  umbral** (0.416 con 0.3, 0.358 con 0.5, 0.302 con 0.7). El "óptimo" de onset
  no lo es si importa la duración de las notas. Causa no investigada.
- **Resultado inesperado y sin explicar:** al subir el umbral sube también el
  recall hasta 0.6 (0.82→0.89), cuando lo normal sería que bajara. Hasta
  donde sé (no lo he verificado en su código), el emparejamiento de
  `mir_eval` es óptimo y uno a uno, así que las notas de más no le quitan
  aciertos a las reales: no explica este efecto. Una posibilidad, no
  comprobada, es que los onsets espurios de umbrales bajos partan notas
  sostenidas en fragmentos más cortos que `minimum_note_length_ms` y se
  descarte con ellos el fragmento que estaba en el onset verdadero. Se puede
  comprobar: con `--onset-threshold 0.3 --min-note-length-ms 30` el recall
  debería subir si el mecanismo es ese.
- Que a 0.7 el número de notas estimadas (1970) casi coincida con el de la
  referencia (1972) no significa nada especial: con P≈R es lo esperable.
- Una sola pieza, un solo soundfont: no es base suficiente para cambiar el
  valor por defecto.

## Coste del posprocesado (`cargo bench -- deteccion_de_notas`)

Predicciones reales de `vocadito` repetidas para simular duraciones mayores:

| duración | con Melodia | sin Melodia |
|---|---|---|
| 12 s | 3.05 ms | 1.14 ms |
| 72 s | 53.1 ms | 5.31 ms |
| 240 s | 535 ms | 20.1 ms |

- Sin Melodia el coste es casi lineal. Con Melodia crece más rápido que la
  duración: ×6 de duración costaron ×17 de tiempo, y ×3.3 costaron ×10, un
  exponente que se acerca a 2 (cuadrático).
- **Importancia práctica baja para canciones.** Con el pipeline completo en
  torno a 10 s para 3 minutos (`docs/baseline-rendimiento.md`), 535 ms son unos
  pocos por ciento. El crecimiento cuadrático solo importaría con audios de
  muchas decenas de minutos.
- El bench repite un mismo clip; la densidad de notas es real, pero no hay
  variedad musical.

## Falsos positivos por categoría (con control de azar)

Mejor configuración del barrido de onset (0.7): unas ~220 notas estimadas no
emparejan con la referencia. Cada una se clasifica por su relación con las
notas reales cuyo onset cae a ±100 ms. La columna "azar" repite la
clasificación de esas mismas notas con el pitch sorteado de entre los de la
referencia.

| categoría | real | azar |
|---|---|---|
| mismo pitch, onset a 50-100 ms o duplicado | 3.6 % | 9.0 % |
| mismo pitch, re-ataque dentro de una nota real | 4.1 % | 1.9 % |
| mismo pitch, ≤ 0.5 s después del final de una nota | 33.8 % | 12.6 % |
| armónico superior (+12, +19, +24, +28, +31) | 44.1 % | 11.1 % |
| una o dos octavas por debajo | 5.4 % | 3.9 % |
| semitono vecino (±1) | 0.0 % | 6.4 % |
| otra nota real estaba sonando | 5.9 % | 50.8 % |
| sin relación | 3.2 % | 4.4 % |

Lectura:

- **Los falsos positivos no son aleatorios en pitch.** Ninguno cae a un
  semitono de una nota real (0.0 % frente a 6.4 % por azar) y solo el 5.9 %
  cae en "otra nota sonando" (frente a 50.8 %).
- **Dos estructuras suman el 78 %:**
  - *Armónico superior*, 44.1 % frente a 11.1 % (unas 4 veces lo esperable).
    Es compatible con armónicos del timbre o con errores de octava; esta
    clasificación no distingue entre ambos.
  - *Mismo pitch poco después del final de una nota*, 33.8 % frente a 12.6 %
    (unas 2.7 veces). Son re-ataques sobre la cola de una nota que ya terminó
    en el MIDI (decaimiento o resonancia del soundfont) o fragmentación de
    notas sostenidas. Es coherente con la explicación de las colas que se dio
    para el F1 con finales de nota.
- **Limitaciones del control.** El pitch se sortea de la distribución *global*
  de la referencia, no de la local, y muchos pitches sorteados caen lejos de
  toda nota sonando (de ahí el 50.8 % en "otra nota sonando"). Eso tiende a
  subestimar el azar de "armónico superior": con un control local, el hueco
  de 33 puntos probablemente sería menor. Que siga habiendo hueco es
  plausible por la ausencia total de semitonos vecinos, pero no está probado.
- **La clasificación es heurística** (ventana de ±100 ms, orden fijo de
  categorías) y solo se ha mirado en la configuración 0.7, no en la que viene
  por defecto.
- **Filtrar armónicos rendiría poco y es arriesgado.** Son unas ~100 notas, el
  5 % de las estimadas: la precisión pasaría de 0.89 a ~0.93 como mucho, y un
  filtro por intervalos borraría también octavas y quintas legítimas. Los
  re-ataques sobre colas (~75 notas) pesan casi lo mismo. No se ha probado
  ningún filtro.

## Pendiente

- Clasificar los falsos positivos también en la configuración por defecto
  (onset 0.5), para ver cómo cambia la mezcla de categorías.
- Comprobar el mecanismo del recall: `--onset-threshold 0.3
  --min-note-length-ms 30`.
- Rejilla de `onset_threshold` × `frame_threshold`: los dos umbrales empujan el
  F1 con finales de nota en sentidos opuestos, y no se ha probado ninguna
  combinación.
- Un control de azar con pitch sorteado en el registro local, no en el global.
- Evaluar con una referencia independiente del modelo sobre audio real. El
  clon del upstream (`basic-pitch/tests/resources/`) solo trae salidas del
  propio modelo (`model_output.npz`, `note_events.npz`), no anotaciones; las de
  Vocadito habría que descargarlas del registro de Zenodo citado en `NOTICE`.

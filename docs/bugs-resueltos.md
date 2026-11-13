# Bugs resueltos en P1 (Fases 0, 1 y 2)

Registro de todos los errores encontrados y corregidos durante el primer
bloque de trabajo, con el sintoma observable, la causa raiz y como se
verifico la correccion.

Esta lista existe para dos cosas:

1. que nadie tenga que redescubrirlos;
2. que sirvan de aviso: varios son **fallos silenciosos**, es decir,
   el programa no falla, devuelve numeros plausibles y el error aparece
   tres fases mas tarde como "el modelo no funciona".

---

## Parte 1: Errores del plan de implementacion

Eran afirmaciones del plan que no coincidian con el codigo real de
`basic-pitch` (commit `fa5997af`, v0.4.0).

### B-01. La CQT produce 264 bins en lugar de 309

**Sintoma.** Ninguno en si mismo; el efecto es que toda la representacion
acustica sale mal y las notas detectadas no cuadran con las de Spotify.

**Causa.** El plan asumia que la frecuencia era 264 en todas las etapas.
En realidad:

```python
n_semitones = min(ceil(12*log2(8)) + 88, MAX_N_SEMITONES=103)   # 103
n_bins      = n_semitones * CONTOURS_BINS_PER_SEMITONE          # 309
```

La CQT produce 309 bins y el truncado a 264 ocurre despues, dentro de
Harmonic Stacking. Los 264 bins de salida existen para que los armonicos
superiores (hasta el desplazamiento +101) tengan contenido que desplazar.

**Evidencia.** El inicializador `model_1/cq_t2010v2_1/Sqrt` del ONNX
oficial tiene forma `[309, 1, 1]`: es el factor `sqrt(lengths)` de la
normalizacion, con una entrada por bin de la CQT.

**Verificacion.** El contrato tecnico declara 309, y `ModelConfig::validate`
rechaza cualquier configuracion que diga otra cosa.

---

### B-02. La concatenacion de onsets produce 33 canales, no 32

**Sintoma.** El modelo daria predicciones de onset distintas de las de
referencia, sin error alguno.

**Causa.** El plan describia la concatenacion sin fijar el orden ni el
numero de canales. El codigo real es:

```python
x_onset = tfkl.Concatenate(axis=3, name="concat")([x_notes_pre, x_onset])
```

`x_notes_pre` es la salida **sigmoid de la conv de notas**, con forma
`(batch, 172, 88, 1)`, **no** la salida final `note`. `x_onset` tiene 32
canales. Total: **33**, y el orden es obligatorio (`x_notes_pre` primero).

**Evidencia.** El kernel de la conv final de onsets en el ONNX es
`const_fold_opt__680` con forma `[1, 33, 3, 3]`.

**Verificacion.** `ModelConfig::validate` rechaza 32 canales y tambien
rechaza el orden invertido.

---

### B-03. `padding="same"` y el stride por eje no existen en Candle

**Sintoma.** La primera conv con stride `(1, 3)` daria 89 o 87 bins de
frecuencia en lugar de 88, y el modelo entero se descuadra.

**Causa.** `candle_nn::Conv2dConfig` solo admite:

```rust
pub struct Conv2dConfig {
    pub padding: usize,   // un valor, simetrico, en AMBOS ejes
    pub stride: usize,    // un valor, en AMBOS ejes
    ...
}
```

No hay `Padding::Same`, ni padding por eje, ni stride por eje. El modelo
usa `padding="same"` (asimetrico con stride > 1) y strides `(1, 3)`.

**Solucion.** Padding manual + conv stride-1 + submuestreo en frecuencia.
Es **algebraicamente exacto**:

```text
y_salida[o] = sum_j w[j] * x_pad[o*s + j]
y_conv1[m]  = sum_j w[j] * x_pad[m + j]
=> y_salida[o] = y_conv1[o*s]
```

**Verificacion.** Test de forma: una entrada `(1, 1, 172, 264)` con kernel
`(1, 1, 7, 7)` y stride de frecuencia 3 produce exactamente 88 bins.

**Nota.** La equivalencia es exacta en álgebra, pero **no se promete
igualdad bit a bit** entre backends: TensorFlow, Candle y el motor de
convolucion de Rust acumulan en orden distinto. La medicion corresponde a
la Fase 7.

---

### B-04. BatchNorm inicial: un canal, escalares, y epsilon de Keras

**Sintoma.** La normalizacion inicial daria un resultado distinto en cada
bin de frecuencia, y con un epsilon 100 veces menor.

**Causa.** Dos confusiones en el plan:

1. La BatchNorm inicial se aplica **despues** de `expand_dims`, sobre
   `(batch, 172, 309, 1)`. Con un solo canal, `gamma`, `beta`,
   `moving_mean` y `moving_variance` tienen longitud **1**: son
   escalares, no un vector por bin de frecuencia. La normalizacion es
   global por muestra, sobre `(batch, time, freq)`.
2. El `epsilon` correcto es el de Keras, **0.001**. El atributo
   `epsilon=1e-5` del nodo `FusedBatchNormV3` de ONNX **no** es el valor
   de este modelo, porque el ONNX fusiona la normalizacion dentro de la
   convolucion.

**Evidencia.** Los inicializadores `model_1/batch_normalization/FusedBatchNormV3`
del ONNX tienen forma `[1]`: un unico canal.

**Verificacion.** `ModelConfig::validate` rechaza un epsilon distinto de
0.001.

---

### B-05. Remuestreo: `rubato` no es `soxr_hq`, y el mono va antes

**Sintoma.** El audio remuestreado difiere del de la referencia y el CQT
se desplaza.

**Causa.** Tres errores encadenados en el plan:

1. `librosa.load(sr=22050)` usa `res_type='soxr_hq'` en librosa >= 0.10
   (en 0.8/0.9 era `kaiser_best`, de `resampy`). `rubato` no es
   bit-identico a ninguno de los dos.
2. El plan proponia `rubato` sin fijar la version de librosa de la
   referencia, con lo que el audio de partida no estaba determinado.
3. El orden de las operaciones: la referencia convierte a mono **antes**
   de remuestrear, no al reves.

**Solucion.** `soxr` en modo `HQ`, con librosa 0.10.2 fijada en el
entorno de referencia, y mono por **promedio** antes del remuestreo.

**Nota honesta.** Usar `soxr` garantiza el mismo algoritmo, pero no
igualdad muestra a muestra: el decodificador puede diferir del de la
referencia (`soundfile`/`libsndfile` frente a `symphonia`). Se asume
como tolerancia y se mide en la Fase 7.

---

### B-06. La numeracion de ventanas divide entre 172, pero quedan 142

**Sintoma.** Los tiempos de las notas se desplazan de forma creciente a
lo largo del audio.

**Causa.** El upstream es inconsistente consigo mismo:

```python
# inference.py: se conservan 142 frames por ventana
n_frames_per_window = (AUDIO_WINDOW_LENGTH * ANNOTATIONS_FPS) - n_overlapping_frames

# note_creation.py: pero la ventana se numera dividiendo entre 172
window_numbers = np.floor(np.arange(n_frames) / ANNOT_N_FRAMES)
```

El desplazamiento temporal por ventana se acumula mas lento de lo que
"deberia".

**Decision.** Reproducirlo **literalmente**. Corregirlo daria tiempos
distintos de los de Spotify, que es justo lo que se quiere replicar.
Queda documentado como comportamiento de compatibilidad.

---

### B-07. Dos capas con kernel (5, 5): una comentada y otra activa

**Sintoma.** Si se omite la capa equivocada, el modelo pierde la salida
de contornos y con ella el pitch bend.

**Causa.** En `models.py` hay dos capas de kernel `(5, 5)`:

| Constante | Kernel | Filtros | Estado |
|---|---|---:|---|
| `CONTOUR_KERNEL_SIZE_1` | (5, 5) | 32 | **comentada**, se omite |
| `CONTOUR_KERNEL_SIZE_3` | (5, 5) | 1 | **activa** (`contours-reduced`) |

La instruccion del plan ("no implementar la capa (5, 5)") es ambigua
justo en el punto donde mas dano hace.

**Verificacion.** `ModelConfig::validate` comprueba el conjunto exacto de
seis capas activas y que no aparezca la de 32 filtros en la rama de
contornos. Hay un test que verifica que `conv2d_4` (onsets), que tambien
tiene kernel `(5, 5)`, **si** es valido.

---

## Parte 2: Bugs del propio codigo

Aparecieron al implementar las Fases 0, 1 y 2.

### B-08. Los nombres de capa de Keras no son estables entre builds

**Sintoma.** `KeyError: la capa 'conv2d' espera el tensor 'conv2d/kernel'`.

**Causa.** El contador de nombres autoasignados de Keras es **global a la
sesion**. El SavedModel oficial tiene `conv2d_1`..`conv2d_5`; un build
fresco produce `conv2d`..`conv2d_4`. Los nombres no coinciden, aunque
el modelo sea identico.

**Solucion.** Resolver los tensores por **rol** (sufijo del nombre:
`/kernel`, `/bias`, `/gamma`, ...) y **forma**, nunca por nombre. El
bias de cada conv se empareja por el nombre `<conv>/bias`, que si es
determinista.

---

### B-09. Buscar el equivalente ONNX por forma produce falsos positivos

**Sintoma.** La validacion de pesos fallaba con una diferencia maxima de
**36.4** en un tensor que deberia coincidir bit a bit.

**Causa.** El emparejamiento por forma tomaba el primer inicializador
con la misma forma. Los biases de 1 canal y las BatchNorm de 1 canal
comparten forma, asi que un bias de conv se comparaba contra una escala
de BatchNorm.

**Solucion.** Identificar el equivalente ONNX **por contenido**: se
prueban las relaciones `direct`, `bn_folded` y `bn_folded_bias`, y se
elige la de menor diferencia. Es una comprobacion mas fuerte que una
simple igualdad de formas.

**Aprendizaje.** Comparar tensores por forma es una bomba de reloji en
cualquier port de pesos entre frameworks.

---

### B-10. `midi_pitch_to_contour_bin` devolvia `-inf` para la nota MIDI 21

**Sintoma.** `assertion left == right, left: -9223372036854775808`.

**Causa.** Error algebraico mio: use `log2(semitones)` en vez de la
frecuencia. La referencia calcula

```python
12 * 3 * log2(pitch_hz / 27.5)
```

que, como `440 / 27.5 = 16 = 2^4`, se simplifica **exactamente** a
`3 * (pitch_midi - 21)`. Con `log2(0)` el resultado es `-inf`.

**Verificacion.** Test que comprueba la equivalencia con la formula
literal de la referencia para **las 88 notas**.

---

### B-11. `soxr::process` puede no consumir toda la entrada

**Sintoma.** Audio truncado al final, de forma intermitente segun el
tamano del buffer de salida.

**Causa.** `soxr_process` consume solo la parte de la entrada que cabe
en el buffer de salida. Si el buffer se llena, devuelve
`input_frames == 0` y el resto de la entrada se pierde en silencio.

**Solucion.** Iterar hasta consumir la entrada entera, y crecer el buffer
si hace falta.

---

### B-12. `libsoxr` corrompe la salida con instancias concurrentes

**Sintoma.** El test de remuestreo de una seno de 100 Hz fallaba de forma
**intermitente**, con 174 cruces por cero en lugar de 99. Aislamiento: el
test pasaba siempre en solitario y fallaba dentro del suite en paralelo.

**Causa.** `libsoxr` mantiene un **descriptor de FFT compartido** entre
todas sus instancias. Dos instancias usadas a la vez desde hilos
distintos corrompen ese estado compartido. La salida resultante tiene
la **longitud correcta** pero la **amplitud y la fase equivocadas**
(amplitud maxima 0.069 en lugar de 1.0), lo que la hace parecer un
problema de la senal y no de la biblioteca.

Reproduccion: 8 hilos x 60 remuestreos = **480/480 anomalias**.

**Solucion.** Serializar el acceso a `soxr` con un cerrojo global. El
motor procesa las ventanas de forma secuencial, asi que el coste es nulo,
y de paso se garantiza que el resultado es **reproducible**.

**Verificacion.** Con el cerrojo: **0/480**. Test de regresion concurrente
incluido en el suite.

---

### B-13. `Tensor::i()` con un tensor de indices no indexa el ultimo eje

**Sintoma.** `index-select invalid index 3 with dim size 1`.

**Causa.** Se asumio que `x.i(&indices)` seleccionaba sobre la ultima
dimension. No es asi: indexa sobre la dimension 0.

**Solucion.** `x.index_select(&indices, dim)` con `dim = 3`.

---

### B-14. `soxr` en modo multihilo no es reproducible

**Causa.** El planificador de hilos de libsoxr decide como repartir el
filtro, y el reparto cambia el resultado numerico.

**Solucion.** `RuntimeSpec::new(1)`: un solo hilo. El determinismo es
imprescindible en un proyecto cuyo objetivo es comparar contra una
referencia.

---

### B-15. El padding SAME se calculaba mal

**Sintoma.** `the trait bound usize: std::ops::Neg is not satisfied`, y
un padding manual claramente incorrecto.

**Causa.** Se usaba el valor "antes" del padding SAME para los dos lados
y se descartaba el "despues". El codigo correcto es:

```text
out      = ceil(in / s)
pad_tot  = max((out - 1) * s + k - in, 0)
pad_ant  = pad_tot / 2
pad_post = pad_tot - pad_ant
```

**Solucion.** `same_padding2d(in, kernel, stride)` devuelve los dos
valores por eje, y el test comprueba los valores exactos de las seis
capas contra la formula de TensorFlow.

---

### B-16. El validador del contrato rechazaba el modelo correcto

**Sintoma.** `cargo run` fallaba con "solo debe existir una capa activa
con kernel (5, 5); se encontraron [contours-reduced, conv2d_4]".

**Causa.** **El validador estaba mal, no el modelo.** `ONSET_KERNEL_SIZE_1`
(rama de onsets) tambien tiene kernel `(5, 5)` y si esta activa.

**Solucion.** La invariante correcta no es "un unico kernel 5x5", sino el
**conjunto exacto** de seis capas activas, mas la ausencia explicita de
la capa comentada. Se anadio un test que verifica que `conv2d_4` con
kernel 5x5 **si** es valido, para que nadie "arregle" el validador al
reves.

---

### B-17. `repository` del manifiesto apuntaba al repositorio de Spotify

**Causa.** Copiado del upstream. `cargo publish` habria publicado este
port bajo la URL del proyecto original.

**Solucion.** Campo omitido, con un comentario que explica por que, mas
un `NOTICE` que acredita la derivacion.

---

### B-18. Diferencias de API entre versiones de las dependencias

Errores de compilacion y de comportamiento al actualizar a las versiones
actuales. Se documentan porque las|相关aciones volveran a tropezar:

| Dependencia | Cambio | Solucion |
|---|---|---|
| `candle` 0.9 -> 0.11 | `Tensor::from_slice` exige `&Device`; `i_min`/`i_max` -> `min`/`max`; `sigmoid` sale de `Tensor`; `Tensor::cat` pasa a metodo asociado; `flatten_all()` devuelve `Result`; `reshape` no admite `-1` | Ajuste de las llamadas; `sigmoid` via `candle_nn::ops`; ancho explicito en los `reshape` |
| `safetensors` 0.5 -> 0.8 | `Error` -> `SafeTensorError`; `TensorView` -> `View`; `float_data()` -> `data()` (bytes) | Conversion manual a `f32` con `from_le_bytes` |
| `symphonia` 0.5.0 -> 0.5.5 | `probe` exige `Hint`; `SampleBuffer::new(duration, spec)`; `decode` devuelve `AudioBufferRef`; `Decoder::try_new` en vez de `make` | Reescritura del decodificador |
| `symphonia` 0.6 | API incompatible con 0.5 | Fijado a **0.5.5** |
| `midly` 0.5 -> 0.10 | `Format::Standard` -> `Parallel`; `TrackEvent.event` -> `.kind`; `PitchBend { value }` -> `{ bend: PitchBend(u14) }`; `u4`/`u7`/`u14`/`u24`; `Track` es `Vec<TrackEvent>` | Reescritura del escritor MIDI |
| `soxr` | El crate es `soxr`, no `soxr-rs`; `Quality::HQ` -> `QualityRecipe::high()` (= 20 bits); `process` devuelve `Processed { output_frames }` | Ajuste de las llamadas |
| `protobuf` | `onnx` 1.23 exige >=6.31 y `tensorflow` 2.15 exige <5 | Fijado a **4.25.9** con `onnx` 1.18 |

---

### B-19. Errores menores de compilacion y de test

Cosas que no llegan a ser bugs del modelo, pero habrian costado tiempo:

- `Tensor::i()` con tensor de indices (B-13).
- `to_vec1`/`to_vec2` sobre tensores de rango 3 o 4: hay que aplanar
  antes (`flatten_all().unwrap().to_vec1()`).
- `flatten_all()` devuelve `Result` en candle 0.11.
- `reshape` no admite `-1`: hay que pasar el ancho explicito.
- Un bloque ` ```rust ` en un docstring con `...` dentro se compila como
  doctest y falla: hay que marcarlo como ` ```text `.
- `start_sample` con `saturating_sub` ocultaba que la ventana 0 empieza
  **antes** del audio original (`-INITIAL_PADDING`): ahora es `i64` y
  negativo.
- `__pycache__` se colaba en el primer commit de Python: anadido a
  `.gitignore`.

---

## Resumen

| Categoria | Cantidad | De ellos silenciosos |
|---|---:|---:|
| Errores del plan (documentales) | 7 | 6 |
| Bugs del codigo | 12 | 4 |
| Diferencias de API entre versiones | 1 (8 casos) | - |

**Los 10 silenciosos** (el programa no falla, devuelve numeros
plausibles) son los que justifican la Fase 7: comparar etapa a etapa
contra `reference/reference_outputs/`, en lugar de solo comparar el MIDI
final, que oculta donde se ha acumulado el error.

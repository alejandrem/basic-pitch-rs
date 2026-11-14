# Bugs resueltos

Registro de todos los errores encontrados y corregidos, en los
bloques P1 (Fases 0, 1 y 2) y P2 (Fases 3, 4 y 5).

Cada entrada tiene el sintoma observable, la causa raiz y como se
verifico la correccion.

Esta lista existe para dos cosas:

1. que nadie tenga que redescubrirlos;
2. que sirvan de aviso: varios son **fallos silenciosos**, es decir,
   el programa no falla, devuelve numeros plausibles y el error aparece
   tres fases mas tarde como "el modelo no funciona".

## P1 (Fases 0, 1 y 2)

### Parte 1: Errores del plan de implementacion

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

### Parte 2: Bugs del propio codigo

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

## P2 (Fases 3, 4 y 5)

El bloque del frontend acustico. Aqui los errores no estan en el codigo
(que no existia) sino en las **trampas de la referencia**, y en un
fenomeno que parece un bug y no lo es.

### B-20. El factor de normalizacion de la CQT solo admite batch 1

**Sintoma.** Ninguno: es una limitacion de la referencia que se
descubre al implementarla.

**Causa.** En `nnaudio.py`, la normalizacion final es:

```python
CQT *= tf.math.sqrt(tf.cast(self.lengths.reshape((-1, 1, 1)), self.dtype))
```

`CQT` tiene forma `(batch, 309, frames)` y `lengths` tiene forma
`(309, 1, 1)`. TensorFlow alinea los ejes **por la derecha**, asi que
`(309, 1, 1)` se interpreta como `(1, 309, 1)` frente a
`(batch, 309, frames)`, y la operacion **solo es valida con
`batch == 1`**. Para `batch == 2`, haria falta `batch == 309`.

**Solucion.** En Rust la operacion se hace en NCHW, con la forma
`(1, 309, 1)`, que es valida para cualquier batch. Es un superconjunto
del comportamiento de la referencia: con batch 1 el resultado es
identico.

**Por que importa.** Si alguien implementase la CQT en NHWC sin
caerse en esto, el motor solo funcionaria con un audio a la vez, y
fallaria de forma confusa al Procesar un lote.

### B-21. El relleno de la CQT es reflectante, no de ceros

**Sintoma.** Un CQT con la forma correcta pero con los valores
erroneos: la diferencia aparece en los **bordes** de cada frame, donde
el relleno pesa mas.

**Causa.** `CQT2010v2` usa `pad_mode="reflect"`, que en TensorFlow es
`tf.pad(..., "REFLECT")`: el reflejo **no repite el valor del borde**
(`[a b c] -> [b a b c b]`), a diferencia del modo `symmetric`.

**Trampa asociada.** Candle no tiene padding reflectante. Se
implementa con `index_select` sobre un vector de indices reflejados, y
el modo `REFLECT` de TensorFlow exige que el relleno sea **menor** que
la dimension de la senal. En la ultima octava eso significa que la
senal tiene 171 muestras y el relleno 128: justo por debajo del limite.

**Verificacion.** Test unitario del relleno y, sobre todo, la
comparacion con la referencia: la CQT coincide con una diferencia
maxima de 5.7e-6 absolutos, que es el epsilon de `float32`.

### B-22. El submuestreo usa relleno de ceros manual, no `SAME`

**Sintoma.** El numero de frames de la CQT no cuadra: 171 en lugar de
172, o un desplazamiento de una muestra en las octavas inferiores.

**Causa.** `downsampling_by_n` con `match_torch_exactly=True` hace el
relleno a mano:

```python
paddings = [[0, 0], [0, 0], [(filter_kernel.shape[-1] - 1) // 2, (filter_kernel.shape[-1] - 1) // 2]]
```

Es decir, **127 ceros a cada lado**, no el `SAME` de TensorFlow. La
diferencia es de una muestra, y se propaga a todos los frames de las
octavas inferiores.

**Verificacion.** El numero de frames es 172 en las nueve octavas, y
las longitudes de la senal submuestreada son exactamente
43844 -> 21922 -> 10961 -> 5480 -> 2740 -> 1370 -> 685 -> 342 -> 171.

### B-23. La normalizacion por `sqrt(lengths)` va ANTES de la magnitud

**Sintoma.** Un factor `sqrt(309)` de diferencia en toda la CQT: un
error que parece de pesos y no lo es.

**Causa.** En la referencia, la secuencia es:

```python
CQT = CQT * self.downsample_factor   # 1.0
CQT *= tf.math.sqrt(...)             # por bin de frecuencia
return tf.transpose(tf.math.sqrt(tf.math.reduce_sum(tf.math.pow(CQT, 2), axis=-1)), [0, 2, 1])
```

El factor por bin se aplica a la parte real **y** a la imaginaria
**antes** de tomar la magnitud. Aplicarlo despues (a la magnitud ya
calculada) es matematicamente equivalente, pero no numericamente igual,
y con `float32` se nota.

**Ademas**, `CQT_imag` es el **negativo** de la convolucion con el
kernel imaginario: `CQT_imag = -tf.nn.conv1d(...)`. Olvidar el signo
cambia la magnitud, no solo la fase.

### B-24. El error del CQT se amplifica ~2800 veces en el logaritmo

**Sintoma.** La CQT de Rust coincide con la de TensorFlow al nivel de
`float32` (5.7e-6), pero el NormalizedLog difiere en 1.4e-3. Parece un
bug en el logaritmo.

**Causa.** No es un bug. `NormalizedLog` calcula `10*log10(v² + 1e-10)`
y divide por el rango. En los bins silenciosos, `v ≈ 5e-6`, su cuadrado
es del mismo orden que el estabilizador `1e-10`, y la derivada respecto
de `v` vale **2819**.

El error del CQT es **absoluto** (no depende del valor del bin), asi que
en esos bins es un 9.3 % de error relativo, y el logaritmo lo convierte
en 1.4e-3 absoluto en la salida.

**Verificacion.** Modelo analitico: error previsto 1.344e-3, observado
1.378e-3, cociente 1.025. Y la prueba de que el logaritmo esta bien:
aplicandolo a la CQT **de la referencia**, el error cae a 1.3e-7.

**Por que importa.** Sin este analisis, el siguiente paso habria sido
"arreglar" el NormalizedLog, y se habria roto. El detalle esta en
[fase-08-frontend-cqt.md](practica/fase-08-frontend-cqt.md), apartado
4.7.2.

### B-25. El lector de `.npy` tropezo con la tupla de la forma

**Sintoma.** `InvalidHeader("forma invalida '(1, 172, 309)': invalid
digit found in string")` al leer un `.npy` perfectamente valido.

**Causa.** El extractor generico de campos de la cabecera corta en la
primera coma, y la forma **es** una tupla con comas: `(1, 172, 309)`
devolvia `(1`.

**Solucion.** Un extractor especifico para la forma, que busca el
parentesis que cierra. Y un segundo fallo en el mismo sitio: la forma
`(43844,)` tiene un elemento vacio despues de la coma, que hay que
filtrar.

**Reflexion.** Es el mismo tipo de error que B-09 (emparejar tensores
por forma en lugar de por contenido): los formatos de fichero tienen
casos limite que hay que tratar explicitamente.

### B-26. `ModelConfig` no tenia un constructor por defecto

**Sintoma.** `no associated function or constant named 'default' found
for struct 'config::ModelConfig'`.

**Causa.** La configuracion de ejemplo vivia como helper privado del
modulo de tests de `config.rs`, y hacia falta en los tests de
`frontend.rs`.

**Solucion.** Se sube a `ModelConfig::example()`, con los mismos valores
que el asset `model_config.json`. Los tests de los demas modulos ya no
dependen del sistema de archivos.

### B-27. Los enums de error se habian inflado hasta pasar de 128 bytes

**Sintoma.** Aviso de clippy `result_large_err` en el crate de
transcripcion, que aparecio justo despues de anadir el frontend.

**Causa.** `candle_core::Error` es grande (contiene `Shape` y `String`),
y al encadenar `ModelError -> FrontendError -> TransformError ->
candle_core::Error`, cada enum arrastraba el tamaño del anterior.

**Solucion.** Encuadrar (`Box`) las variantes grandes: `candle_core::Error`,
`WeightsError`, `KernelsError`, `FrontendError` y `ModelError`. Cada
`Result` que devuelve uno de esos enums copia ahora 8 bytes en lugar de
128.

**Nota.** No es un bug funcional: el lint avisa de rendimiento en la
ruta de error, que es fria. Pero el coste de arreglarlo es bajo y el
codigo queda mejor.

### B-28. La diferencia entre Symphonia y soundfile: era una incógnita abierta

**Sintoma.** Ninguno: era una incertidumbre declarada. El plan (fase 3,
apartado 3.3.1) avisaba de que usar `soxr` en Rust garantiza el mismo
*algoritmo* de remuestreo, pero no la igualdad muestra a muestra, porque
el decodificador puede diferir.

**Causa.** El oraculo sintetico (`input_audio.npy`) viene ya decodificado
y remuestreado, asi que no permetia medir esa diferencia.

**Solucion.** Se ha generado un segundo oraculo con audio real del
upstream (`reference/audio_reference/`), usando el SavedModel oficial y
el ventaneo literal de `basic_pitch/inference.py`.

**Resultado.** La diferencia entre Symphonia y soundfile es de **2.7e-8**
de media y **3.0e-7** de maximo: por debajo del epsilon de `float32`.
Es decir, los dos decodificadores producen **la misma senal**.

**Consecuencia.** La incertidumbre queda resuelta: el error que queda no
viene de una decision de diseno (decodificador distinto, remuestreador
distinto, orden mono/remuestreo invertido), sino del redondeo de coma
flotante. El apartado 3.3.1 del plan se actualiza con esta medida.

### B-29. El parser de `.npy` duplicado fallaba con la forma

**Sintoma.** `forma invalida: ParseIntError { kind: InvalidDigit }` al
leer un `.npy` con la forma `(200607,)`.

**Causa.** Se habia escrito un segundo lector de `.npy` dentro del test
de audio, en vez de reutilizar el del crate `basic-pitch-reference-npy`.
El error era el mismo que en B-25: el corte por comas rompia la tupla de
la forma, y aqui el problema era el `}` final del diccionario.

**Solucion.** El test usa el lector del crate compartido, y el
duplicado desaparece. Un parser de formato de fichero es codigo que no
debe estar duplicado.

### B-30. Medir el retardo de grupo con la fase de Goertzel da un falso positivo

**Sintoma.** El test del retardo de grupo fallaba: "la fase rota 1.71
radianes entre el principio y el final". El remuestreador parecia tener
un retardo que variaba con el tiempo, lo que deformaria la senal.

**Causa.** No era el remuestreador, era **el metodo de medida**. La fase
que devuelve el algoritmo de Goertzel tiene un sesgo que depende de la
posicion de la ventana cuando esta no contiene un numero entero de
periodos de la frecuencia medida. Ese sesgo es indistinguible de una
deriva de fase.

Se comprobo de dos maneras:

- Midiendo la fase de la senal **ideal, sin remuestrear**: derivaba
  exactamente lo mismo (1,71 radianes).
- Midiendo la salida de `librosa.resample(res_type="soxr_hq")` de la
  referencia: fase **identica** a la ideal, sample a sample. Es decir,
  `soxr_hq` no deforma la fase en absoluto.

Ademas, el primer intento en Python tenia un bug mas grave: el
coeficiente de la recurrencia era `2*omega*cos(omega)` en vez de
`2*cos(omega)`, lo que hacia que la magnitud saliera **cero** para un
seno perfecto. El de Rust era correcto.

**Solucion.** Medir el retardo de grupo con los **cruces de cero
ascendentes**: si el retardo es constante, el espaciado entre cruces es
exactamente el periodo, sin ambiguedad. Y el helper de Goertzel usa
ventana de Hann con normalizacion por la ganancia coherente, que ademas
hace la magnitud correcta para cualquier numero de periodos.

**Aprendizaje.** Cuando una medicion da un resultado raro, hay que
**medir el fenomeno con otro metodo** antes de culpar al codigo. Si la
senal ideal derivea igual, el metodo es el culpable.

### B-31. Los tests de la matriz de la fase 3 faltaban casi todos

**Sintoma.** Ninguno: era un hueco de cobertura, no un fallo.

**Causa.** La implementacion de la fase 3 se habia validado contra la
referencia (que es lo importante), pero la **matriz de pruebas** que
enuncia el apartado 3.5 del plan no estaba completa: solo se cubrian
dos de los nueve escenarios, y los formatos MP3, FLAC, OGG y M4A no se
habian probado nunca.

**Solucion.** Tres ficheros de test nuevos (`formats.rs`, `resampler.rs`,
`scenarios.rs`) con 25 tests, y los fixtures multiformato generados por
`tools/weight-export/make_audio_fixtures.py`.

**Lo que sale de ahi.** Los fixtures son el mismo contenido en cinco
formatos, y la comprobacion no es "ha devuelto samples" sino que el
contenido sea el correcto: se mide la magnitud en la frecuencia de cada
canal y el desfase entre canales. Ademas se descubre que con 1,5
segundos de audio la referencia produce **DOS** ventanas y no una,
porque el relleno inicial (3840) se pasa del salto (36164).
### B-32. Las tres salidas del modelo salian con el tiempo y la frecuencia invertidos

**Sintoma.** Ninguno en el momento: el programa no fallaba. Pero las
salidas del modelo tenian las formas equivocadas:

| Salida | Forma real | Forma correcta |
|---|---|---|
| `contour` | `(batch, 264, 172)` | `(batch, 172, 264)` |
| `note` | `(batch, 88, 172)` | `(batch, 172, 88)` |
| `onset` | `(batch, 88, 172)` | `(batch, 172, 88)` |

**Causa.** Un `transpose(1, 2)` de mas al final de cada rama:

```rust
// contour.rs, notes.rs, onsets.rs
let salida = conv.squeeze(1)?.transpose(1, 2)?.contiguous()?;
```

`squeeze(1)` ya da `(batch, 172, 264)`, que es la forma correcta. El
`transpose` la convertia en `(batch, 264, 172)`.

La referencia (`FlattenFreqCh`) **solo aplana el eje de canales**:
`(batch, 172, 264, 1) -> (batch, 172, 264)`, sin transponer nada. Como el
trabajo interno es NCHW, el equivalente es `squeeze(1)` y nada mas.

**Como se detecto.** Comparando las salidas de la red con la referencia de
TensorFlow: al transponerlas a mano, coincidian con 4.4e-5 (contour),
1.7e-5 (note) y 5.3e-5 (onset). Es decir, **la red era correcta**; lo
unico que estaba mal era el volteo final.

**Por que no lo detectaron los tests de P1.** Porque no existia ninguna
comparacion de las salidas de la red contra la referencia: eso es la
Fase 7 (P3). Los tests de P1 solo comprobaban formas *intermedias*.

**Impacto.** En la Fase 9, el postprocesamiento habria recibido el tiempo
de una nota como si fuera un pitch: el `note` de `(batch, 88, 172)` se
interpretaria como 88 "frames" con 172 valores de nota cada uno. Habria
sido un desastre silencioso, del tipo que hace que el modelo "no
suene".

**Solucion.** Quitar el `transpose` de las tres ramas, y documentar en
cada sitio por que no hay que transponer (para que nadie lo vuelva a
poner).

**Tests de regresion** (`crates/basic-pitch-model/tests/model_output.rs`):

- la forma de las tres salidas;
- una propiedad **estructural** que no depende de la forma: si la entrada
  es constante en el tiempo, la salida tiene que ser constante en el
  tiempo (las convoluciones son invariantes a la traslacion temporal y la
  BatchNorm en modo inferencia no calcula estadisticas sobre el tiempo);
- y la simetrica: si la entrada es constante en la frecuencia, la salida
  tiene que ser constante en la frecuencia dentro de cada frame.

### B-33. El validador del contrato no validaba los campos que el codigo usa

**Sintoma.** `ModelConfig::validate()` daba el visto bueno a una
configuracion que hacia que el modelo usara **264 bins de CQT en vez de
309**, que es exactamente el error B-01 que el proyecto lleva dos fases
guardando.

**Causa.** La configuracion tiene **dos campos con el mismo significado**:
`frequency.cqt_n_bins` (un resumen) y `cqt.n_bins` (el detalle). El
frontend lee el segundo; el validador comprobaba el primero. Con
`cqt.n_bins = 264` y `frequency.cqt_n_bins = 309`, todo pasaba.

Ademas, `validate()` comprobaba siete cosas y el codigo leia muchas mas:
`cqt.hop_length`, `n_fft`, `n_octaves`, `n_filters`, `sr`, `fmin`,
`early_downsample`, `downsample_factor`, `pad_mode`, `basis_norm`,
`harmonic_stacking.shifts`, `n_output_freqs`, `input_n_bins`,
`bins_per_semitone`, el contrato de audio y el de ventaneo.

**El caso de `shifts` es el masinteresting.** El codigo **recalcula** los
desplazamientos a partir de los armonicos y no lee el campo `shifts` del
JSON. Un `shifts` equivocado en el fichero nooria nada: el campo era
decorativo y podia mentir.

**Solucion.**

1. Un bloque de constantes `contrato` con los valores del modelo
   ICASSP 2022, tomados de `docs/teoria/fase-03-espe-tecnicas.md`, como
   **fuente de verdad** de la validacion.
2. `validate()` comprueba **todo** campo que el codigo lee, y ademas que
   `cqt.n_bins` y `frequency.cqt_n_bins` coincidan.
3. Los `shifts` declarados se comparan con los calculados.
4. Los errores se **acumulan** y se devuelven todos juntos: antes
   fallaba en el primero y habia que reintentar N veces para ver los
   demas.

**Efecto secundario.** `ModelConfig::example()` (el helper de test) estaba
incompleto: no tenia las tres capas de BatchNorm, asi que el validador
nuevo lo rechazo. Se completo, y ahora el ejemplo coincide con el asset.

### B-34. La duracion minima de nota usaba el fps entero en vez del flotante

**Sintoma.** Ninguno con los valores por defecto: 127.7 ms dan 11 frames
con las dos formulas.

**Causa.** El upstream calcula:

```python
min_note_len = int(np.round(minimum_note_length / 1000 * (AUDIO_SAMPLE_RATE / FFT_HOP)))
```

`AUDIO_SAMPLE_RATE / FFT_HOP` es **division flotante**: 86.1328125. Pero
el codigo usaba `ANNOTATIONS_FPS`, que en `constants.py` es
`AUDIO_SAMPLE_RATE // FFT_HOP` = **86** (division entera) y sirve para
otra cosa.

**Por que no se notaba.** Con 127.7 ms, `127.7/1000*86.1328 = 10.99915` y
`127.7/1000*86 = 10.9822`: los dos redondean a 11.

**Donde si se nota.** Con 5.81 ms: el upstream da 0.50043 -> 1 frame, y el
valor entero daria 0.49966 -> 0 frames.

**Solucion.** Usar la division flotante, y `round_ties_even` (que es lo que
hace `np.round`: redondea al par en los empates) en vez de `round`, que
redondea hacia arriba.

**Aprendizaje.** `ANNOTATIONS_FPS` y `AUDIO_SAMPLE_RATE / FFT_HOP` son
valores **distintos** que se parecen mucho. El primero es entero a
proposito en el upstream; el segundo es flotante. Copiar el nombre sin
mirar la operacion es un error facil de cometer.


## Resumen

| Bloque | Categoria | Cantidad | De ellos silenciosos |
|---|---|---:|---:|
| P1 | Errores del plan (documentales), B-01..B-07 | 7 | 6 |
| P1 | Bugs del codigo, B-08..B-19 | 12 | 4 |
| P2 | Trampas de la referencia (documentales), B-20..B-23 | 4 | 3 |
| P2 | Bugs del codigo, B-24..B-31 | 8 | 2 |
| P2 | Bugs de la auditoria del codigo, B-32..B-34 | 3 | 2 |
| **Total** | | **34** | **17** |

La diferencia de API entre versiones (B-18) son ocho casos de un mismo
problema, no ocho entradas.

Ademas, dos entradas documentan **huecos de cobertura** que no eran
fallos: B-31 (la matriz de pruebas de la fase 3 estaba casi vacia) y
B-29 (el parser de `.npy` duplicado).

**Los tres ultimos (B-32, B-33 y B-34) los encontro una auditoria del
codigo, no un test.** Dos de los tres son silenciosos: el programa no
falla, devuelve numeros plausibles y el error aparece tres fases mas
tarde. B-32 habria roto el postprocesamiento musical entero; B-33 habria
dejado pasar el bug B-01 (264 bins de CQT) por la puerta de atras.

Eso es el argumento de por que una auditoria manual del codigo tiene su
sitio al lado de los tests: un test compara lo que se le dice que
compara, y no puede comparar lo que nadie se le ocurre comprobar.

**Los 17 silenciosos** (el programa no falla, devuelve numeros
plausibles) son los que justifican la validacion etapa a etapa contra
`reference/reference_outputs/`, en lugar de comparar solo el MIDI final,
que oculta donde se ha acumulado el error.

Los tres silenciosos de P2 que mas costan adivinar son:

- **B-20**: el factor de normalizacion de la CQT solo admite `batch == 1`
  en TensorFlow. El port lo hace valido para cualquier batch, pero si
  alguien "corrige" el port para que sea igual que la referencia en
  NHWC, rompe el procesamiento por lotes.
- **B-22**: el submuestreo usa relleno de ceros manual de 127 muestras,
  no `SAME`. La diferencia es de una muestra y desplaza todas las
  octavas inferiores.
- **B-24**: el error del CQT se amplifica 2819 veces en el logaritmo.
  Sin el analisis, el "arreglo" habria sido romper el NormalizedLog.

# Fase 5: implementación de Harmonic Stacking

**Objetivo:** reproducir la expansión armónica utilizada como entrada de las ramas neuronales.

## 5.1. Funcionamiento

La capa toma la representación CQT y construye diferentes canales desplazando el eje de frecuencias.

**Entrada:** `(batch, 172, 309, 1)` — los 309 bins de la CQT.
**Salida:** `(batch, 172, 264, 8)` — trunca a 264 bins, 8 canales armónicos.

Para el modelo predeterminado se utilizarán:

```text
[0.5, 1, 2, 3, 4, 5, 6, 7]
```

Cada valor determina un desplazamiento expresado en bins de frecuencia.

La cantidad de bins por semitono es tres.

Los desplazamientos se calculan mediante:

```text
shift = round(
    12 × bins_per_semitone × log2(harmonic)
)
```

Se deberá reproducir el comportamiento de redondeo de TensorFlow (`tf.math.round`,
que redondea al par en los empates). Los valores resultantes son:

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

## 5.2. Padding y desplazamientos

El relleno con ceros es **asimétrico** y esa asimetría es obligatoria. Tomando
como referencia `basic_pitch/nn.py`, clase `HarmonicStacking`:

| Caso | Operación | Efecto |
|---|---|---|
| `shift == 0` | `padded = x` | canal idéntico a la entrada |
| `shift > 0` | `pad(x[:, :, shift:, :], [[0,0],[0,0],[0,shift],[0,0]])` | descarta los `shift` bins **bajos**, rellena con ceros los **altos**. `out[i] = x[i + shift]` |
| `shift < 0` | `pad(x[:, :, :shift, :], [[0,0],[0,0],[-shift,0],[0,0]])` | rellena con ceros los bins **bajos**, conserva el resto. `out[i] = x[i + shift]` |

En ambos casos no negativo el desplazamiento equivale a `out[i] = x[i + shift]`
con ceros donde el índice sale de rango; la diferencia está en **dónde** se
colocan esos ceros.

El algoritmo no debe:

- Repetir el primer o último bin.
- Envolver los valores de un extremo al otro.
- Cambiar el orden de los canales.
- Interpolar frecuencias.
- Sustituir los ceros por el mínimo de la señal.

Después de generar los canales, se conservará el rango de frecuencias requerido
por el modelo: se trunca el eje de frecuencias a los primeros `264` bins
(`x[:, :, :n_output_freqs, :]`). La truncación ocurre **después** de concatenar
los 8 canales, no antes de desplazar.

## 5.3. Dimensiones

La salida deberá representar:

```text
(batch, time, 264, 8)
```

En Candle podrá mantenerse internamente una disposición compatible con NCHW, siempre que el orden lógico de frecuencias y canales sea equivalente.

## 5.4. Pruebas

Se probará cada desplazamiento de manera independiente con entradas artificiales.

Por ejemplo, se utilizarán matrices con valores identificables por posición para comprobar que los canales se desplazan y rellenan correctamente.

**Entregable de la fase:** Harmonic Stacking implementado y validado.

### 5.4.1. Un test por desplazamiento

`crates/basic-pitch-model/tests/harmonic_shifts.rs` tiene un test
independiente para cada uno de los ocho armónicos (0.5, 1, 2, 3, 4, 5,
6 y 7). El oráculo de cada uno es la fórmula **literal** de
`basic_pitch/nn.py`:

```text
shift == 0  ->  out = x
shift > 0   ->  out = pad(x[:, :, shift:, :], [[0,0],[0,0],[0,shift],[0,0]])
shift < 0   ->  out = pad(x[:, :, :shift, :], [[0,0],[0,0],[-shift,0],[0,0]])
```

Dos detalles de la entrada artificial que hay que acertar, y que
condicionan el significado de los tests:

- **Los valores empiezan en 1, no en 0.** Si la entrada tuviera ceros, no
  se podría distinguir un cero *del relleno* de un cero *que viene de la
  entrada*, y las comprobaciones de asimetría no significarían nada.
- **Los límites de los ceros dependen del truncado.** Con `shift = +36`
  sobre 309 bins, el primer cero está en el bin 273, que queda **fuera**
  de la salida de 264 bins: no hay ningún cero visible. Con
  `shift = +101` el primer cero sí cae dentro, en el bin 208.

### 5.4.2. La asimetría se comprueba, no se supone

El test comprueba **dónde** van los ceros, que es la parte que distingue
una implementación de otra:

| Desplazamiento | Los ceros van |
|---|---|
| Positivo | al final del eje de frecuencia |
| Negativo | al principio del eje de frecuencia |
| Cero | no hay ceros |

Y se comprueba también que el truncado a 264 bins ocurre **después** de
desplazar: si ocurriera antes, el canal del armónico 7 (`shift = +101`)
se quedaría sin contenido, porque la entrada truncada a 264 bins solo
llega hasta el 263 y haría falta hasta el 364.

### 5.4.3. Lo que ya estaba

La fase 5 también pide que no se repita el primer o último bin, no se
envuelvan los valores, no se cambie el orden de los canales, no se
interpolen frecuencias y no se sustituyan los ceros por el mínimo de la
señal. Cada uno de esos puntos tiene su comprobación, y el test de los
ocho canales verifica el **orden** (el canal 0 es el armónico 0.5, el 1
es el 1.0, y así).

## 5.5. Implementación en Rust

`crates/basic-pitch-model/src/harmonic_stacking.rs`.

La operación se hace **con tensores de Candle**, no con vectores: `narrow`
para descartar los bins que se desplazan y `Tensor::cat` para concatenar
los ocho canales en el eje de canales. El truncado a 264 bins ocurre
después de concatenar, igual que en la referencia.

El trabajo en NCHW: la entrada es `(batch, 1, 172, 309)` y la salida
`(batch, 8, 172, 264)`. En NHWC (TensorFlow) la salida equivalente es
`(batch, 172, 264, 8)`.

El relleno es **asimétrico** y por eso no se puede implementar como un
`slice` uniforme: para `shift > 0` se descartan los bins bajos y se
rellenan los altos con ceros; para `shift < 0` al revés.

## 5.6. Validación

| Comprobación | Resultado |
|---|---|
| Desplazamientos calculados | `[-36, 0, 36, 57, 72, 84, 93, 101]`, los de la referencia |
| Forma de la salida | `(1, 8, 172, 264)` en NCHW |
| Coincidencia con la referencia | MAE 1.78e-5, RMSE 5.48e-5, máx. 3.42e-3 |
| Equivalencia tensores ↔ vectores | test que compara ambas implementaciones |

La diferencia con la referencia es la **misma** que la de la BatchNorm
(3.42e-3): Harmonic Stacking no introduce error nuevo, solo mueve los
valores de la entrada, que ya venían amplificados desde el logaritmo de
`NormalizedLog`. El análisis está en
[fase-08-frontend-cqt.md](fase-08-frontend-cqt.md), apartado 4.7.2.

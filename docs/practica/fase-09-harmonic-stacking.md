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

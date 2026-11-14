# Estado de P2 (Fases 3, 4 y 5)

Resumen de lo entregado en el segundo bloque de trabajo y los comandos
exactos para reproducirlo.

P2 es el bloque **numérico difícil** del proyecto: el frontend acústico
completo, desde el audio hasta la representación armónica que consume la
red neuronal.

---

## Qué se ha entregado

| Fase | Contenido | Dónde |
|---|---|---|
| 3 | Procesamiento de audio: decodificación, mono por promedio, remuestreo, ventaneo | `crates/audio-io`, `crates/audio-preprocess` |
| 4 | Frontend CQT: la transformada multirresolución, `NormalizedLog` y la BatchNorm inicial | `crates/cqt` |
| 5 | Harmonic Stacking, verificado contra la referencia | `crates/basic-pitch-model` |

### Lo que faltaba y se ha escrito

- **`crates/cqt/src/transform.rs`**: la aritmética de la CQT. Es la
  pieza central de P2: el crate tenía los *kernels* (extraídos del ONNX
  en la Fase 1) pero no el cálculo que los usa.
- **`crates/cqt/src/batch_norm.rs`**: la BatchNorm inicial en modo
  inferencia, con estadísticas escalares y epsilon de Keras.
- **`crates/basic-pitch-model/src/frontend.rs`**: la composición de las
  cuatro etapas, con `stages()` para poder compararlas una a una.
- **`crates/reference-npy/`**: lector de `.npy` (solo desarrollo).
- **`crates/basic-pitch-model/examples/validar_frontend.rs`**: el
  validador que imprime la tabla de métricas.
- **`crates/basic-pitch-model/tests/frontend_reference.rs`**: los tests
  de regresión contra la referencia.
- **`crates/audio-preprocess/tests/pipeline.rs`**: el test de la cadena
  de audio de punta a punta.

---

## Resultados de la validación

```bash
cargo run --release --example validar_frontend -p basic-pitch-model
```

```text
CQT                  MAE 1.762e-7  RMSE 2.835e-7  max 5.722e-6  rel 3.186e-7
NormalizedLog        MAE 1.010e-5  RMSE 2.987e-5  max 1.378e-3  rel 1.378e-3
  (diagnostico)      MAE 7.789e-9  RMSE 1.808e-8  max 1.341e-7  rel 1.341e-7
BatchNorm inicial    MAE 2.505e-5  RMSE 7.410e-5  max 3.419e-3  rel 2.132e-3
HarmonicStacking     MAE 1.782e-5  RMSE 5.479e-5  max 3.419e-3  rel 2.132e-3
```

**La CQT coincide con TensorFlow al nivel del epsilon de `float32`**
(3.2e-7 relativos): la convolución acumula 256 sumandos y TensorFlow y
Candle los acumulan en orden distinto. La diferencia está medida y
explicada, no supuesta.

**El 1.4e-3 del NormalizedLog no es un error del port**: es la
sensibilidad de la propia referencia. El logaritmo
`10*log10(v² + 1e-10)` tiene una derivada de **2819** en los bins donde
la señal roza el silencio, y el error del CQT es *absoluto*, así que ahí
es un 9.3 % relativo. El modelo analítico predice 1.344e-3 y se observan
1.378e-3 (cociente 1.025). La prueba de que `NormalizedLog` está bien es
la fila del diagnóstico: aplicándolo a la CQT **de la referencia**, el
error cae a 1.3e-7.

Análisis completo en
[`docs/practica/fase-08-frontend-cqt.md`](practica/fase-08-frontend-cqt.md),
apartado 4.7.

---

## Reproducirlo

```bash
# 1. El entorno de referencia (solo si se van a regenerar los .npy).
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements-ref.txt

# 2. La validacion numerica del frontend.
cargo run --release --example validar_frontend -p basic-pitch-model

# 3. Los tests de regresion contra la referencia.
cargo test -p basic-pitch-model --test frontend_reference

# 4. La cadena de audio.
cargo test -p basic-pitch-audio-preprocess --test pipeline

# 5. Todo.
cargo test --workspace
cargo clippy --workspace --all-targets
cargo fmt --all -- --check
```

### Volcar las etapas de Rust para analizarlas

```bash
cargo run --release --example validar_frontend -p basic-pitch-model -- --dump target/frontend_rust
```

Deja `cqt_rust.npy`, `normalized_log_rust.npy`, `batch_norm_rust.npy` y
`harmonic_stacking_rust.npy`, que se pueden abrir con NumPy para
analizar la sensibilidad o el espectro del error.

---

## Estado del repositorio

```text
tests:   139 en verde
clippy:  0 avisos
fmt:     sin diferencias
python:  82/82 comprobaciones superadas
git:     arbol limpio
```

---

## Decisiones que quedan fijadas

1. **El frontend vive en `basic-pitch-cqt`, la composición en
   `basic-pitch-model`.** El crate de la CQT no depende de nada del
   modelo; el del modelo depende del de la CQT. La flecha de las
   dependencias apunta hacia arriba.

2. **El trabajo interno es NCHW.** El CQT se ensambla en `(batch,
   frecuencia, frames)` porque es como lo hace la referencia, y se
   transpone al final. La normalización por `sqrt(lengths)` se aplica en
   NCHW con la forma `(1, 309, 1)`, lo que además de ser más natural hace
   que funcione con cualquier batch (la referencia solo admite `batch ==
   1`, ver B-20).

3. **El relleno reflectante se implementa con `index_select`.** Candle
   no lo tiene, y el modo `REFLECT` de TensorFlow no repite el valor del
   borde, así que no se puede aproximar con un `slice`.

4. **`stages()` es parte de la API pública.** Comparar etapa a etapa no
   es un capricho de debug: es la única forma de saber dónde se ha
   acumulado el error. Sin eso, la diferencia entre la CQT y el MIDI
   final sería una caja negra.

5. **Las tolerancias de los tests son cuatro veces el error medido**, y
   el motivo de cada una está escrito en el propio test y en la fase 4.

---

## Trampas para P3

Las que se han encontrado al implementar el frontend y que conviene
tener presentes:

- **La CQT es de 309 bins, no de 264.** El truncado ocurre en Harmonic
  Stacking.
- **El submuestreo de la CQT usa relleno de ceros manual de 127
  muestras**, no `SAME`. Una muestra de diferencia desplaza todas las
  octavas inferiores.
- **El relleno de la conv de la CQT es reflectante**, no de ceros.
- **El logaritmo amplifica el ruido 2819 veces** en los bins
  silenciosos. Cualquier comparación de una etapa con logaritmo debe
  hacerse contra el error *medido*, no contra una intuición de lo que
  "debería" ser.
- **La referencia solo admite `batch == 1`** en su normalización final.
- **El epsilon de la BatchNorm es 0.001** (Keras), no el 1e-5 del ONNX
  fusionado.

---

## Lo que queda para P3

P3 (Fases 6 y 7) es la red neuronal en Candle y la validación de
equivalencia etapa por etapa:

- la red ya está escrita (`crates/basic-pitch-model`), pero su
  validación contra la referencia **todavía no existe**: los `.npy` de
  `contour_conv`, `contour_reduced`, `contour`, `notes_features`,
  `note`, `onset_features` y `onset` son el oráculo pendiente;
- la estrategia de `padding="same"` + stride por eje (padding manual +
  conv stride-1 + submuestreo) está documentada e implementada en
  `layers.rs`, pero su equivalencia numérica con TensorFlow **está sin
  medir**;
- el validador `validar_frontend` se puede ampliar con las etapas de la
  red sin cambios de estructura.

# Estado del proyecto: P1 completado (Fases 0, 1 y 2)

> **Documento histórico (2026-11-13).** Las cifras (6 crates, 54 tests) y la
> fila "3-15 Pendientes" describen el proyecto en esa fecha, no el actual. El
> estado actual está en el `README.md` y en el CI.

**Fecha:** 2026-11-13
**Commit de referencia:** `fa5997af0a8210982619003269994a1be25eddf3` (basic-pitch v0.4.0)

Este documento resume lo entregado en el primer bloque de trabajo y como
reproducirlo. El plan completo vive en los archivos `docs/fase-*`.

## Que se ha completado

| Fase | Estado | Entregable |
|---|---|---|
| 0 - Auditoria | **Completada** | `reference/` con inventario, configuracion, manifiesto de trazabilidad y salidas de referencia |
| 1 - Conversion de pesos | **Completada** | `models/basic-pitch/icassp-2022/` con `model.safetensors`, `cqt_kernels.safetensors` y manifiestos |
| 2 - Entorno Rust | **Completada** | Workspace Cargo con 6 crates, 54 tests, `cargo run` operativo |
| 3-15 | Pendientes | Ver `docs/practica/` y `docs/pruebas/` |

## Reproduccion

```bash
# 1. Entorno Python de referencia (una sola vez)
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python \
  "tensorflow==2.15.0.post1" "librosa==0.10.2.post1" "pretty_midi>=0.2.9" \
  "mir_eval>=0.6.0" "resampy>=0.2.2,<0.4.3" "safetensors>=0.4" "numpy<2" \
  "onnx==1.18.0" "onnxruntime==1.20.1" "protobuf==4.25.9"

# 2. Auditoria (Fase 0): inventario + referencias + cross-check TF vs ONNX
.venv/bin/python tools/weight-export/inspect_model.py

# 3. Exportacion (Fase 1): Safetensors + manifiestos
.venv/bin/python tools/weight-export/export_weights.py

# 4. Validacion de los pesos (82 comprobaciones)
.venv/bin/python tools/weight-export/validate_weights.py

# 5. Verificacion de Rust
cargo test --workspace
cargo run
```

## Resultados de la verificacion

- **Cross-check TensorFlow vs ONNX** (pipeline completo, misma entrada):
  `contour` cos = 1.0000000000, `note` cos = 1.0000000000, `onset` cos = 0.9999998808.
  Las diferencias maxima (~1e-4) se concentran en bins de muy baja energia,
  donde el `log10` de la NormalizedLog amplifica las diferencias de coma
  flotante.
- **Fusion de BatchNorm en ONNX** verificada por la relacion
  `W_onnx = W_tf * scale`, con `scale = gamma / sqrt(var + 0.001)`:
  diferencias de 3e-8 y 4.8e-7.
- **Permutacion de kernels**: la permutacion inversa recupera **exactamente**
  (diferencia 0.0) los tensores de TensorFlow.
- **Pesos**: 24 tensores, 16.864 parametros, mas 5 tensores de kernels de CQT.
- **Rust**: `cargo build` sin errores, `cargo clippy` 0 avisos, `cargo fmt`
  sin diferencias, 54 tests en verde y deterministas.

## Artefactos

```text
reference/                          # Fase 0: auditoria (material de desarrollo)
├── model_inventory.json            # capas, tensores, formas, activaciones
├── model_config.json               # contrato tecnico verificado
├── model_summary.txt               # resumen legible
├── reference_metadata.json         # procedencia y hashes
├── tensor_provenance.json          # trazabilidad TF <-> ONNX <-> Rust
└── reference_outputs/              # 12 puntos de comparacion (.npy + index.json)

models/basic-pitch/icassp-2022/     # Fase 1: modelo desplegable
├── model.safetensors               # 24 tensores, layout de Candle
├── cqt_kernels.safetensors         # 5 tensores: kernels deterministas
├── model_config.json
├── tensor_inventory.json
├── tensor_provenance.json
├── source_metadata.json
└── SHA256SUMS

crates/                             # Fase 2: workspace Rust
├── audio-io/                       # decodificacion (symphonia)
├── audio-preprocess/               # mono, remuestreo (soxr), ventanas
├── cqt/                            # kernels de CQT + NormalizedLog
├── basic-pitch-model/              # arquitectura en Candle
├── transcription/                  # deteccion, reconstruccion, pitch bend
└── midi-io/                        # eventos y escritura MIDI
```

## Decisiones que quedan fijadas

1. **TensorFlow es la fuente canonica** de los pesos: el ONNX tiene la
   BatchNorm fusionada y no permite recuperar `moving_mean` / `moving_variance`.
2. **El ONNX si se usa** para extraer los kernels deterministas de la CQT y
   como verificacion cruzada.
3. **El equivalente ONNX de cada tensor se identifica por contenido**, no por
   forma: varios tensores comparten forma y emparejar solo por forma produce
   falsos positivos silenciosos.
4. **Remuestreo con `soxr` en modo HQ y un solo hilo**: el modo multihilo de
   libsoxr no es reproducible, y el determinismo es obligatorio para poder
   comparar contra la referencia.
5. **Las 7 correcciones del plan estan aplicadas** y documentadas en los
   archivos de fase correspondientes. La lista vive en `docs/implementacion.md`.

## Siguiente bloque: P2 (Fases 3, 4 y 5)

Audio -> CQT -> NormalizedLog -> BatchNorm inicial -> Harmonic Stacking.

Es el bloque numericamente mas dificil. Sus puntos de control son las salidas
intermedias ya guardadas en `reference/reference_outputs/`, que permiten
comparar etapa a etapa.

Bugs ya detectados y evitados de antemano:

- La CQT produce **309** bins, no 264.
- `Candle::Conv2d` no admite `padding="same"` ni stride por eje: la estrategia
  es **padding manual + conv stride-1 + submuestreo**, que es exacta.
- `Tensor::i()` con un tensor de indices **no** indexa el ultimo eje: hay que
  usar `index_select(&indices, dim)`.
- `soxr::process` puede no consumir toda la entrada si el buffer de salida es
  pequeno: hay que iterar hasta consumirla entera.

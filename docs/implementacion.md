# Plan de implementación: Basic Pitch en Rust con Candle y Safetensors

**Proyecto:** Implementación nativa en Rust del sistema de inferencia de Spotify Basic Pitch  
**Framework de inferencia:** Hugging Face Candle  
**Formato de pesos:** Safetensors  
**Lenguaje principal:** Rust  
**Lenguaje auxiliar:** Python, exclusivamente para extracción de pesos, generación de referencias y validación  
**Objetivo:** reproducir la inferencia del modelo preentrenado de Basic Pitch sin entrenarlo nuevamente, modificar sus pesos ni utilizar Python durante la ejecución final.

---

## Índice de fases

Cada fase se encuentra en su propio archivo dentro de esta carpeta `docs/`:

| # | Fase | Archivo |
|---|---|---|
| 1 | Fase 1: Objetivo | [teoria/fase-01-objetivo.md](teoria/fase-01-objetivo.md) |
| 2 | Fase 2: Arquitectura | [teoria/fase-02-arquitectura.md](teoria/fase-02-arquitectura.md) |
| 3 | Fase 3: Especificaciones técnicas | [teoria/fase-03-espe-técnicas.md](teoria/fase-03-espe-técnicas.md) |
| 4 | Fase 4: Plan de implementación (Fase 0) | [teoria/primer paso/fase-04-implementación.md](teoria/primer%20paso/fase-04-implementación.md) |
| 5 | Fase 5: Conversión de pesos (Fase 1) | [teoria/primer paso/fase-05-conversión-pesos.md](teoria/primer%20paso/fase-05-conversión-pesos.md) |
| 6 | Fase 6: Entorno Rust (Fase 2) | [teoria/primer paso/fase-06-entorno-rust.md](teoria/primer%20paso/fase-06-entorno-rust.md) |
| 7 | Fase 7: Procesamiento de audio (Fase 3) | [práctica/fase-07-procesamiento-audio.md](práctica/fase-07-procesamiento-audio.md) |
| 8 | Fase 8: Frontend CQT (Fase 4) | [práctica/fase-08-frontend-cqt.md](práctica/fase-08-frontend-cqt.md) |
| 9 | Fase 9: Harmonic Stacking (Fase 5) | [práctica/fase-09-harmonic-stacking.md](práctica/fase-09-harmonic-stacking.md) |
| 10 | Fase 10: Arquitectura neuronal en Candle (Fase 6) | [práctica/fase-10-arquitectura-neuronal.md](práctica/fase-10-arquitectura-neuronal.md) |
| 11 | Fase 11: Validación de equivalencia (Fase 7) | [práctica/fase-11-validacion-equivalencia.md](práctica/fase-11-validacion-equivalencia.md) |
| 12 | Fase 12: Inferencia por ventanas (Fase 8) | [práctica/fase-12-inferencia-duracion-arbitraria.md](práctica/fase-12-inferencia-duracion-arbitraria.md) |
| 13 | Fase 13: Posprocesamiento musical (Fase 9) | [práctica/fase-13-posprocesamiento-musical.md](práctica/fase-13-posprocesamiento-musical.md) |
| 14 | Fase 14: Pitch bend (Fase 10) | [pruebas/fase-14-pitch-bend.md](pruebas/fase-14-pitch-bend.md) |
| 15 | Fase 15: Exportación MIDI (Fase 11) | [pruebas/fase-15-exportacion-midi.md](pruebas/fase-15-exportacion-midi.md) |
| 16 | Fase 16: Integración del motor (Fase 12) | [pruebas/fase-16-integracion-motor.md](pruebas/fase-16-integracion-motor.md) |
| 17 | Fase 17: Pruebas automatizadas (Fase 13) | [pruebas/fase-17-pruebas-automatizadas.md](pruebas/fase-17-pruebas-automatizadas.md) |
| 18 | Fase 18: Rendimiento (Fase 14) | [pruebas/fase-18-rendimiento.md](pruebas/fase-18-rendimiento.md) |
| 19 | Fase 19: Documentación (Fase 15) | [fase-19-documentacion.md](fase-19-documentacion.md) |

---

## Correcciones v2 (aplicadas)

Verificadas contra el código de `basic-pitch` (commit
`fa5997af0a8210982619003269994a1be25eddf3`) y contra los inicializadores del
ONNX oficial.

| # | Corrección | Dónde |
|---|---|---|
| 1 | La CQT produce **309** bins; Harmonic Stacking reduce a 264 | [fase-03](teoria/fase-03-espe-técnicas.md), [fase-08](práctica/fase-08-frontend-cqt.md), [fase-09](práctica/fase-09-harmonic-stacking.md) |
| 2 | La concatenación de onsets es `[x_notes_pre, x_onset]` -> **33 canales** | [fase-04](teoria/primer%20paso/fase-04-implementación.md), [fase-10](práctica/fase-10-arquitectura-neuronal.md) |
| 3 | `padding="same"` y stride por eje no existen en Candle: padding manual + conv stride-1 + submuestreo | [fase-10](práctica/fase-10-arquitectura-neuronal.md) |
| 4 | BatchNorm inicial: 1 canal, estadísticas escalares, `epsilon = 0.001` (Keras, no el 1e-5 de ONNX) | [fase-08](práctica/fase-08-frontend-cqt.md) |
| 5 | Audio: mono por **promedio** y **antes** del remuestreo; `soxr_hq` (librosa 0.10.2), no `rubato` | [fase-07](práctica/fase-07-procesamiento-audio.md) |
| 6 | Temporización: reproducir literal los 142 frames y el divisor 172; documentar como compatibilidad | [fase-12](práctica/fase-12-inferencia-duracion-arbitraria.md) |
| 7 | Contornos: `CONTOUR_KERNEL_SIZE_1` (32 filtros) omitida vs `CONTOUR_KERNEL_SIZE_3` (1 filtro) activa | [fase-04](teoria/primer%20paso/fase-04-implementación.md), [fase-10](práctica/fase-10-arquitectura-neuronal.md) |
| 8 | Manifiesto de trazabilidad TF <-> ONNX <-> Rust | [fase-04](teoria/primer%20paso/fase-04-implementación.md), [fase-05](teoria/primer%20paso/fase-05-conversión-pesos.md) |
| 9 | Directorio canónico único `models/basic-pitch/icassp-2022/` + `reference/` | [fase-05](teoria/primer%20paso/fase-05-conversión-pesos.md), [fase-06](teoria/primer%20paso/fase-06-entorno-rust.md) |
| 10 | La normalización final de la CQT solo admite `batch == 1` en TensorFlow; en Rust (NCHW) vale para cualquier batch | [fase-08](práctica/fase-08-frontend-cqt.md), [bugs-resueltos](bugs-resueltos.md) |
| 11 | El relleno de la CQT es **reflectante** (no repite el borde) y el submuestreo usa relleno de ceros manual de 127 muestras | [fase-08](práctica/fase-08-frontend-cqt.md), [bugs-resueltos](bugs-resueltos.md) |
| 12 | El `NormalizedLog` amplifica el ruido del CQT ~2819 veces en los bins silenciosos: el error grande está explicado, no es un bug | [fase-08](práctica/fase-08-frontend-cqt.md), [bugs-resueltos](bugs-resueltos.md) |


---

## Bloques de trabajo

| Bloque | Fases | Por qué juntas | Riesgo | Estado |
| --- | --- | --- | --- | --- |
| **P1** (0 + 1 + 2) | Auditoría + pesos + workspace. Todo mecánico, todo con el modelo y el código ya en disco. Deja listo el oráculo de referencia. | Bajo | **Cerrado** ([estado](estado-p1.md)) |
| **P2** (3 + 4 + 5) | Audio → CQT → NormalizedLog → BN → HarmonicStacking. El bloque numérico difícil. Con kernels extraídos del ONNX es viable en un prompt. | Alto | **Cerrado** ([estado](estado-p2.md)) |
| **P3** (6 + 7) | Red Candle + validación de equivalencia etapa por etapa. | Medio | Pendiente |
| **P4** (8 + 9 + 10 + 11) | Ventanas/reconstrucción → notas → pitch bend → MIDI. Es port de numpy casi 1:1. | Bajo | Pendiente |
| **P5** (12 + 13 + 14 + 15) | API + tests + CI + perf + docs. | Bajo | Pendiente |


# Resultado esperado

Al finalizar este plan se tendrá una biblioteca de transcripción musical implementada en Rust, utilizando Candle para la inferencia neuronal y Safetensors para almacenar los pesos preentrenados.

El motor podrá recibir audio, reproducir el procesamiento acústico de Basic Pitch, ejecutar sus tres ramas neuronales, detectar notas y variaciones de afinación, reconstruir la secuencia temporal y exportar un archivo MIDI.

El resultado será un núcleo independiente que posteriormente podrá integrarse en una aplicación más amplia de edición y producción musical, sin tener que reescribir el motor de transcripción.
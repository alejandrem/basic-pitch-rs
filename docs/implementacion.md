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
| 1 | Fase 1: Objetivo | [fase-01-objetivo.md](fase-01-objetivo.md) |
| 2 | Fase 2: Arquitectura | [fase-02-arquitectura.md](fase-02-arquitectura.md) |
| 3 | Fase 3: Especificaciones técnicas | [fase-03-espe-tecnicas.md](fase-03-espe-tecnicas.md) |
| 4 | Fase 4: Plan de implementación | [fase-04-implementacion.md](fase-04-implementacion.md) |
| 5 | Fase 5: Conversión de pesos | [fase-05-conversion-pesos.md](fase-05-conversion-pesos.md) |
| 6 | Fase 6: Entorno Rust | [fase-06-entorno-rust.md](fase-06-entorno-rust.md) |
| 7 | Fase 7: Procesamiento de audio | [fase-07-procesamiento-audio.md](fase-07-procesamiento-audio.md) |
| 8 | Fase 8: Frontend CQT | [fase-08-frontend-cqt.md](fase-08-frontend-cqt.md) |
| 9 | Fase 9: Harmonic Stacking | [fase-09-harmonic-stacking.md](fase-09-harmonic-stacking.md) |
| 10 | Fase 10: Arquitectura neuronal en Candle | [fase-10-arquitectura-neuronal.md](fase-10-arquitectura-neuronal.md) |
| 11 | Fase 11: Validación de equivalencia | [fase-11-validacion-equivalencia.md](fase-11-validacion-equivalencia.md) |
| 12 | Fase 12: Inferencia por ventanas | [fase-12-inferencia-duracion-arbitraria.md](fase-12-inferencia-duracion-arbitraria.md) |
| 13 | Fase 13: Posprocesamiento musical | [fase-13-posprocesamiento-musical.md](fase-13-posprocesamiento-musical.md) |
| 14 | Fase 14: Pitch bend | [fase-14-pitch-bend.md](fase-14-pitch-bend.md) |
| 15 | Fase 15: Exportación MIDI | [fase-15-exportacion-midi.md](fase-15-exportacion-midi.md) |
| 16 | Fase 16: Integración del motor | [fase-16-integracion-motor.md](fase-16-integracion-motor.md) |
| 17 | Fase 17: Pruebas automatizadas | [fase-17-pruebas-automatizadas.md](fase-17-pruebas-automatizadas.md) |
| 18 | Fase 18: Rendimiento | [fase-18-rendimiento.md](fase-18-rendimiento.md) |
| 19 | Fase 19: Documentación | [fase-19-documentacion.md](fase-19-documentacion.md) |
| 20 | Estructura del flujo de trabajo | [estructura-flujo-trabajo.md](estructura-flujo-trabajo.md) |
| 21 | Criterios de terminación | [criterios-terminacion.md](criterios-terminacion.md) |
| 22 | Primera tarea | [primera-tarea.md](primera-tarea.md) |

---
Aquí tienes la información organizada en una tabla:

| Fases | Por qué juntas | Riesgo |
| --- | --- | --- |
| **P1** (0 + 1 + 2) | Auditoría + pesos + workspace. Todo mecánico, todo con el modelo y el código ya en disco. Deja listo el oráculo de referencia. | Bajo |
| **P2** (3 + 4 + 5) | Audio → CQT → NormalizedLog → BN → HarmonicStacking. El bloque numérico difícil. Con kernels extraídos del ONNX es viable en un prompt. | Alto |
| **P3** (6 + 7) | Red Candle + validación de equivalencia etapa por etapa. | Medio |
| **P4** (8 + 9 + 10 + 11) | Ventanas/reconstrucción → notas → pitch bend → MIDI. Es port de numpy casi 1:1. | Bajo |
| **P5** (12 + 13 + 14 + 15) | API + tests + CI + perf + docs. | Bajo |


# Resultado esperado

Al finalizar este plan se tendrá una biblioteca de transcripción musical implementada en Rust, utilizando Candle para la inferencia neuronal y Safetensors para almacenar los pesos preentrenados.

El motor podrá recibir audio, reproducir el procesamiento acústico de Basic Pitch, ejecutar sus tres ramas neuronales, detectar notas y variaciones de afinación, reconstruir la secuencia temporal y exportar un archivo MIDI.

El resultado será un núcleo independiente que posteriormente podrá integrarse en una aplicación más amplia de edición y producción musical, sin tener que reescribir el motor de transcripción.
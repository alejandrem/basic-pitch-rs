# Linea base de rendimiento (Fase 14)

Medido con `cargo bench -p basic-pitch-transcription` sobre
`basic-pitch/tests/resources/vocadito_10.wav` (9.1 s de audio), en CPU,
modo release, con el modelo `models/basic-pitch/icassp-2022`.

| Etapa | Tiempo (mediana) |
|---|---:|
| Decodificacion | 1.06 ms |
| Remuestreo | 1.09 ms |
| Ventanas + frontend + red neuronal | 584 ms |
| Pipeline completo | 585 ms |

El cuello de botella es el frontend acustico y la red neuronal. El audio
(decodificacion + remuestreo) es despreciable.

Las mediciones completas, con intervalos de confianza y deteccion de
outliers, quedan en `target/criterion/` tras ejecutar los benchmarks. Los
benchmarks viven en `crates/transcription/benches/transcripcion.rs` y usan
`criterion` como dev-dependency, por lo que no afectan al binario ni a las
dependencias de produccion.
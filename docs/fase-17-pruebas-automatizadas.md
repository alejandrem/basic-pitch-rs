# Fase 13: pruebas automatizadas y control de calidad

**Objetivo:** asegurar que los cambios futuros no rompan la transcripción.

## 13.1. Pruebas unitarias

Cada crate deberá incluir pruebas para sus operaciones principales:

- Decodificación.
- Conversión a mono.
- Remuestreo.
- Ventanas.
- CQT.
- NormalizedLog.
- Batch Normalization.
- Harmonic Stacking.
- Convoluciones.
- Concatenación.
- Aplanamiento de tensores.
- Detección de notas.
- Pitch bend.
- Exportación MIDI.

## 13.2. Pruebas de integración

Se probará el flujo completo:

```text
Audio → Preprocesamiento → CQT → Candle
      → Predicciones → Eventos → MIDI
```

Las pruebas deberán verificar que la salida puede ser abierta por un lector MIDI independiente.

## 13.3. Pruebas de regresión

Se conservará un conjunto de archivos de audio de referencia y sus resultados esperados.

Cada cambio en el código deberá ejecutar nuevamente las pruebas de regresión.

Se deberán registrar:

- Versiones de herramientas.
- Parámetros utilizados.
- Diferencias numéricas.
- Diferencias en eventos musicales.
- Tiempo de ejecución.

## 13.4. Integración continua

Se configurará CI para ejecutar:

- `cargo fmt --check`
- `cargo clippy`
- `cargo test`
- Pruebas de lectura de pesos.
- Pruebas del modelo.
- Pruebas de transcripción.
- Pruebas de exportación MIDI.

Las pruebas que dependan de recursos grandes podrán ejecutarse en un conjunto de CI separado.

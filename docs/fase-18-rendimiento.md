# Fase 14: medición y optimización del rendimiento

**Objetivo:** mejorar velocidad y uso de recursos una vez que los resultados sean equivalentes a la referencia.

La optimización no deberá comenzar antes de conseguir una implementación funcional y validada.

## 14.1. Establecer una línea base

Se medirán por separado:

- Tiempo de decodificación.
- Tiempo de remuestreo.
- Tiempo de CQT.
- Tiempo de normalización.
- Tiempo de ejecución neuronal.
- Tiempo de posprocesamiento.
- Tiempo de exportación MIDI.
- Memoria máxima utilizada.

Esto permitirá identificar qué partes del sistema realmente consumen más recursos.

## 14.2. Optimizaciones posibles

Después de validar la equivalencia se podrán evaluar:

- Procesamiento por lotes de ventanas.
- Reutilización de buffers.
- Reducción de copias entre tensores.
- Procesamiento por bloques.
- Paralelismo controlado.
- Optimización de kernels CQT.
- Backend acelerado de Candle.
- Carga eficiente de Safetensors.

No se deberán aplicar varias optimizaciones simultáneamente sin poder medir su efecto.

## 14.3. Precisión numérica

La primera versión se mantendrá en `float32`.

Se podrá investigar precisión reducida o cuantización en el futuro, pero no formará parte de la primera implementación.

Las optimizaciones deberán medirse en términos de:

- Velocidad.
- Memoria.
- Diferencia numérica.
- Impacto en eventos musicales.

No se considerará exitosa una optimización que acelere el modelo a costa de degradar considerablemente la transcripción.

# Fase 10: estimación de pitch bend

**Objetivo:** aprovechar la salida de contornos para detectar variaciones de afinación dentro de las notas.

La salida de contornos no representa únicamente qué notas existen. También aporta información de frecuencia con mayor resolución que una matriz de 88 notas.

El modelo trabaja con tres bins de contorno por semitono.

## 10.1. Conversión de nota a índice de contorno

Se deberá implementar la conversión entre una nota MIDI y su posición correspondiente en la matriz de contornos.

La referencia utiliza la frecuencia base de 27.5 Hz y la escala logarítmica de frecuencias.

La conversión deberá ser consistente con:

- Frecuencia base.
- Número de bins por semitono.
- Número de semitonos.
- Rango de frecuencias disponible.

## 10.2. Estimación por nota

Para cada evento detectado:

1. Obtener el índice de contorno correspondiente a su nota MIDI.
2. Crear una región de búsqueda alrededor de ese índice.
3. Aplicar el peso gaussiano utilizado por la referencia.
4. Extraer la región temporal correspondiente a la duración de la nota.
5. Encontrar el bin de mayor activación por frame.
6. Convertir el desplazamiento obtenido en unidades de contorno.
7. Asociar la secuencia al evento.

Se deberán conservar los parámetros originales de tolerancia y desviación de la ventana gaussiana.

## 10.3. Conversión a MIDI

La información de contorno se transformará en eventos MIDI Pitch Bend.

Se deberá reproducir la escala original, los límites y el rango de variación.

También se tendrá que considerar el comportamiento de notas simultáneas.

El formato MIDI tradicional aplica pitch bend por canal, no de manera independiente a cada nota dentro del mismo canal. La implementación original resuelve este caso mediante una organización específica de instrumentos cuando se permiten pitch bends múltiples.

La versión Rust deberá documentar y reproducir el mismo comportamiento.

## 10.4. Pruebas

Se probarán:

- Notas sin vibrato.
- Notas con vibrato.
- Notas con deslizamiento de afinación.
- Notas con variaciones pequeñas.
- Notas con variaciones superiores al rango permitido.
- Varias notas simultáneas con pitch bend.
- Notas superpuestas.

**Entregable de la fase:** estimación de pitch bend integrada en los eventos musicales.

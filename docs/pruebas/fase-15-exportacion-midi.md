# Fase 11: exportación MIDI

**Objetivo:** generar archivos MIDI válidos a partir de los eventos detectados.

Se utilizará `midly` para construir el archivo MIDI.

## 11.1. Estructura musical

La exportación deberá contemplar:

- Cabecera MIDI.
- Tracks.
- Eventos Note On.
- Eventos Note Off.
- Velocidad.
- Canal.
- Programa de instrumento.
- Tempo.
- Eventos Pitch Bend.
- Tiempos delta.

Los tiempos internos de los eventos se convertirán a ticks MIDI usando la resolución elegida para el archivo.

## 11.2. Conversión de notas

Cada `NoteEvent` se transformará en:

- Un evento Note On al comienzo.
- Un evento Note Off al finalizar.

La amplitud estimada se convertirá a velocidad MIDI siguiendo la escala definida en la referencia.

Se deberán validar los límites permitidos para el pitch y velocity.

## 11.3. Conversión de pitch bend

Los desplazamientos de contorno se transformarán en valores Pitch Bend de MIDI.

Se conservarán los límites del rango utilizado por la implementación original.

También deberá documentarse cómo se asignan los canales y tracks cuando existen notas superpuestas con distintas variaciones de afinación.

## 11.4. Validación

Los archivos generados se verificarán mediante:

- Lectura del MIDI generado.
- Comprobación de la estructura.
- Comprobación de eventos Note On y Note Off.
- Comprobación de tiempos.
- Comprobación de Pitch Bend.
- Prueba de reproducción con software MIDI independiente.

**Entregable de la fase:** exportador MIDI funcional.

2. Arquitectura general del sistema

El sistema se organizará como una cadena de procesamiento:

```text
                ARCHIVO DE AUDIO
                       |
                       v
               Decodificación de audio
                       |
                       v
                Conversión a mono
                       |
                       v
                Remuestreo a 22050 Hz
                       |
                       v
                 División en ventanas
                       |
                       v
               Transformación CQT
                       |
                       v
                NormalizedLog
                       |
                       v
             Batch Normalization inicial
                       |
                       v
                Harmonic Stacking
                       |
             +---------+---------+
             |                   |
             v                   v
       Contour Branch       Onset Branch
             |
             v
       Note Branch
             |
             +---------+---------+
                       |
                       v
              Predicciones del modelo
                       |
                       v
               Reconstrucción temporal
                       |
                       v
              Detección de eventos
                       |
                       v
               Estimación de pitch bend
                       |
                       v
                  MIDI Encoder
                       |
                       v
                  ARCHIVO MIDI
```

La arquitectura se dividirá en cinco áreas principales:

| Área | Responsabilidad |
|---|---|
| Audio | Decodificar, convertir, remuestrear y dividir el audio |
| Frontend acústico | Calcular CQT, normalización y representación armónica |
| Modelo neuronal | Ejecutar la arquitectura de Basic Pitch con Candle |
| Transcripción | Detectar notas, tiempos, amplitudes y variaciones de afinación |
| MIDI | Transformar los eventos musicales en un archivo MIDI válido |

La separación permitirá probar y reemplazar componentes sin modificar el resto del sistema.

---

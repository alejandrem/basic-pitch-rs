# Fase 7: validación de equivalencia entre TensorFlow y Candle

**Objetivo:** demostrar que la implementación en Rust reproduce el comportamiento de la implementación original.

Esta será una de las etapas de mayor importancia del proyecto.

Que la red compile y produzca números no significa que esté funcionando correctamente.

## 7.1. Crear el programa de referencia

Se implementará un programa Python que:

1. Cargue el modelo oficial.
2. Reciba una señal de prueba.
3. Ejecute el preprocesamiento original.
4. Ejecute el modelo completo.
5. Capture las salidas intermedias.
6. Guarde los tensores de referencia.
7. Exporte las dimensiones y configuraciones utilizadas.

Los datos se guardarán en un formato que pueda utilizarse fácilmente desde Rust, por ejemplo NPY, NPZ o Safetensors.

## 7.2. Comparación por etapas

No se comparará únicamente la salida final.

Se deberán comparar:

| Etapa | Comparación |
|---|---|
| Entrada | Muestras de audio |
| CQT | Representación frecuencial |
| NormalizedLog | Valores normalizados |
| BatchNorm inicial | Características normalizadas |
| Harmonic Stacking | Canales armónicos |
| Contour branch | Activaciones intermedias |
| Note branch | Activaciones y predicciones |
| Onset branch | Activaciones y predicciones |
| Salidas finales | Nota, onset y contour |

Esto permitirá localizar los errores sin tener que revisar toda la red cada vez.

## 7.3. Métricas de comparación

Se implementarán métricas como:

- Error absoluto medio (MAE).
- Error absoluto máximo.
- Error cuadrático medio (MSE).
- Raíz del error cuadrático medio (RMSE).
- Similitud coseno.
- Diferencia relativa.

Las comparaciones deberán realizarse entre tensores de las mismas dimensiones y con el mismo significado.

No se establecerá automáticamente que un MAE inferior a `1e-4` sea un requisito universal. La tolerancia dependerá de la etapa, del algoritmo de CQT y de las diferencias numéricas entre operaciones.

Se establecerán umbrales documentados después de medir la referencia con el mismo conjunto de pruebas.

## 7.4. Casos de prueba

Se deberán preparar audios y señales controladas para evaluar:

1. Silencio.
2. Una nota sostenida.
3. Una nota corta.
4. Varias notas simultáneas.
5. Acordes.
6. Notas graves.
7. Notas agudas.
8. Transitorios rápidos.
9. Vibrato.
10. Deslizamientos de afinación.
11. Audio con ruido.
12. Audio de larga duración.
13. Audio con notas cerca de los límites entre ventanas.

Los archivos de referencia deberán contar con permisos de uso adecuados y su procedencia deberá documentarse.

## 7.5. Validación de resultados musicales

Además de comparar valores neuronales, se compararán eventos musicales:

- Nota detectada.
- Tiempo de inicio.
- Tiempo de finalización.
- Velocidad MIDI.
- Duración.
- Pitch bend.
- Notas simultáneas.

Dos resultados pueden presentar pequeñas diferencias numéricas y, aun así, producir la misma transcripción. También puede ocurrir lo contrario: una diferencia pequeña cerca de un umbral puede cambiar un evento musical.

Por eso las pruebas numéricas y musicales deberán ser complementarias.

**Entregable de la fase:** conjunto de pruebas de equivalencia y reporte de diferencias entre TensorFlow y Candle.

**Criterio de aceptación:** las diferencias estarán medidas, explicadas y dentro de las tolerancias establecidas para cada etapa.

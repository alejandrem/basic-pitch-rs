# Fase 8: inferencia sobre audios de duración arbitraria

**Objetivo:** procesar archivos completos, aunque sean considerablemente más largos que una ventana de dos segundos.

El modelo recibe ventanas de longitud fija, pero el usuario necesita enviar canciones completas.

## 8.1. Procesamiento por ventanas

El módulo `inference.rs` coordinará:

1. Recepción de audio ya preprocesado.
2. Generación de ventanas.
3. Ejecución del modelo por ventana.
4. Almacenamiento de predicciones.
5. Registro de la posición temporal de cada ventana.

No se deberá cargar innecesariamente todo el audio y todas las predicciones intermedias en memoria si el archivo es largo.

Se implementará un flujo por lotes de tamaño controlado.

## 8.2. Reconstrucción de predicciones

Las ventanas se superponen para reducir problemas en los límites.

Por eso las predicciones no se pueden concatenar directamente.

Se deberá reproducir la lógica oficial de reconstrucción o `unwrap_output`, considerando:

- Número de frames superpuestos.
- Frames que deben descartarse en cada límite.
- Desplazamiento entre ventanas.
- Longitud original del audio.
- Cantidad de predicciones válidas.
- Padding inicial y final.

La reconstrucción debe evitar duplicar notas o perder eventos en los límites.

## 8.3. Conversión de frames a tiempo

Se deberá implementar la conversión temporal utilizada por Basic Pitch.

No se utilizará simplemente:

```text
tiempo = índice_del_frame / 86
```

La referencia utiliza `FFT_HOP`, la frecuencia de muestreo y una corrección temporal que depende de la ventana.

Se deberán reproducir también los ajustes de alineación definidos en la implementación original, incluido el desplazamiento de alineación correspondiente.

## 8.4. Pruebas de reconstrucción

Se realizarán pruebas con:

- Una canción que no llegue a completar una ventana.
- Una canción con exactamente una ventana.
- Una canción que genere dos ventanas.
- Una canción con muchas ventanas.
- Una nota que comience antes de un límite.
- Una nota que termine después de un límite.
- Una nota sostenida durante varias ventanas.

**Entregable de la fase:** inferencia completa sobre archivos largos, con reconstrucción temporal correcta.

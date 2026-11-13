# Fase 3: implementación del procesamiento de audio

**Objetivo:** convertir los archivos de entrada en muestras de audio que respeten el contrato del modelo original.

La red neuronal no debe recibir audio directamente desde el decodificador. Primero será necesario realizar las transformaciones correspondientes.

## 3.1. Decodificación

El módulo `audio-io` se encargará de cargar el archivo y convertirlo en muestras de audio de punto flotante.

Se contemplará inicialmente la compatibilidad con:

- WAV.
- MP3.
- FLAC.
- OGG.
- M4A, según los códecs habilitados y disponibles en la biblioteca seleccionada.

Se deberá probar cada formato por separado. La extensión del archivo no garantiza que el códec interno sea compatible.

La salida del decodificador deberá utilizar una representación de muestras consistente, preferentemente `f32`.

## 3.2. Conversión a mono

Basic Pitch analiza audio mono.

Cuando el archivo tenga varios canales, se deberá realizar la misma conversión a mono que utiliza la implementación de referencia.

No se deberá seleccionar arbitrariamente el canal izquierdo ni descartar el derecho.

Se deberá comprobar cómo se comporta la conversión con archivos estéreo, señales desfasadas y canales con amplitudes diferentes.

## 3.3. Remuestreo

Toda señal se convertirá a:

```text
22050 Hz
```

El remuestreador elegido será `rubato`, sujeto a validación contra el proceso empleado por la versión de referencia de `librosa`.

Se deberá prestar especial atención a:

- Calidad del filtro.
- Retardo de grupo.
- Tratamiento de los extremos.
- Número final de muestras.
- Diferencias de amplitud.
- Comportamiento con frecuencias de muestreo poco habituales.

No se aplicará normalización automática de volumen por pico ni compresión dinámica si la referencia no la realiza.

El objetivo es conseguir una señal numéricamente comparable, no simplemente una señal que tenga la frecuencia de muestreo correcta.

## 3.4. División en ventanas

El modelo trabaja con ventanas de longitud fija.

Cada ventana deberá contener exactamente:

```text
43844 muestras
```

El módulo `windowing.rs` se encargará de:

1. Recibir la señal completa.
2. Obtener la configuración de solapamiento de la versión oficial fijada.
3. Calcular el desplazamiento entre ventanas.
4. Agregar el padding inicial requerido.
5. Extraer ventanas consecutivas.
6. Completar con ceros la última ventana cuando corresponda.
7. Conservar la posición temporal original de cada ventana.
8. Registrar la longitud real del audio antes del padding.

El solapamiento deberá reproducir el comportamiento de la función de inferencia original.

No se debe aplicar un solapamiento arbitrario ni simplemente concatenar las predicciones de ventanas consecutivas.

## 3.5. Pruebas del procesamiento de audio

Se crearán pruebas para:

- Audio mono y estéreo.
- Audio de menos de dos segundos.
- Audio de exactamente dos segundos.
- Audio de varios minutos.
- Audio con silencio inicial y final.
- Archivos con diferentes frecuencias de muestreo.
- Archivos con diferentes formatos.
- Señales con valores cercanos a los límites de amplitud.
- Audio con transitorios al inicio y al final.

**Entregable de la fase:** un módulo que produzca ventanas de audio compatibles con la entrada oficial.

**Criterio de aceptación:** las muestras, dimensiones, padding y posiciones temporales coincidirán con las reglas de la referencia.

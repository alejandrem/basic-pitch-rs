# Fase 5: implementación de Harmonic Stacking

**Objetivo:** reproducir la expansión armónica utilizada como entrada de las ramas neuronales.

## 5.1. Funcionamiento

La capa toma la representación CQT y construye diferentes canales desplazando el eje de frecuencias.

Para el modelo predeterminado se utilizarán:

```text
[0.5, 1, 2, 3, 4, 5, 6, 7]
```

Cada valor determina un desplazamiento expresado en bins de frecuencia.

La cantidad de bins por semitono es tres.

Los desplazamientos se calculan mediante:

```text
shift = round(
    12 × bins_per_semitone × log2(harmonic)
)
```

Se deberá reproducir el comportamiento de redondeo de TensorFlow.

## 5.2. Padding y desplazamientos

Cuando un canal se desplaza hacia una dirección, las posiciones que quedan fuera de los límites deberán rellenarse con ceros.

El algoritmo no debe:

- Repetir el primer o último bin.
- Envolver los valores de un extremo al otro.
- Cambiar el orden de los canales.
- Interpolar frecuencias.
- Sustituir los ceros por el mínimo de la señal.

Después de generar los canales, se conservará el rango de frecuencias requerido por el modelo.

## 5.3. Dimensiones

La salida deberá representar:

```text
(batch, time, 264, 8)
```

En Candle podrá mantenerse internamente una disposición compatible con NCHW, siempre que el orden lógico de frecuencias y canales sea equivalente.

## 5.4. Pruebas

Se probará cada desplazamiento de manera independiente con entradas artificiales.

Por ejemplo, se utilizarán matrices con valores identificables por posición para comprobar que los canales se desplazan y rellenan correctamente.

**Entregable de la fase:** Harmonic Stacking implementado y validado.

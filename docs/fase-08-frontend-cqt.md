# Fase 4: implementación del frontend CQT

**Objetivo:** reproducir la representación acústica que la red neuronal espera recibir.

Esta es una de las fases más delicadas del proyecto.

El modelo no recibe directamente muestras de audio ni un espectrograma STFT convencional. Recibe una representación CQT calculada mediante la implementación de `nnaudio.CQT` utilizada por Basic Pitch.

Por lo tanto, sustituirla por una CQT genérica o una STFT con parámetros parecidos no garantiza resultados equivalentes.

## 4.1. Comprender la transformación CQT

La transformación Constant-Q representa el contenido frecuencial del audio usando bins distribuidos logarítmicamente.

A diferencia de una STFT convencional, la resolución frecuencial está relacionada con la frecuencia de cada bin.

Para esta implementación se deberán reproducir:

- Frecuencia de muestreo.
- Frecuencia mínima.
- Número de bins.
- Bins por octava.
- Longitud del kernel.
- Ventana.
- Normalización.
- Padding.
- Hop length.
- Algoritmo de submuestreo.
- Orden de las operaciones.

Los parámetros finales se extraerán de la configuración real del modelo.

## 4.2. Reproducir NNAudio CQT

La capa oficial utiliza una variante de CQT con procesamiento multirresolución y submuestreo.

La implementación deberá reproducir las operaciones de:

1. Cálculo del factor Q.
2. Determinación de las frecuencias de cada bin.
3. Construcción de kernels.
4. Generación de componentes reales e imaginarios.
5. Padding de la señal.
6. Convoluciones para calcular la transformada.
7. Obtención de la magnitud.
8. Submuestreo entre octavas cuando corresponda.
9. Ensamblado de la representación CQT final.

Los kernels CQT son parámetros deterministas generados a partir de una configuración; no deben confundirse con los pesos entrenados de la red neuronal.

Se podrán generar desde Rust o exportar como recursos calculados durante el desarrollo, siempre que la ejecución final no necesite Python.

La primera versión deberá priorizar la coincidencia numérica con la referencia. Las optimizaciones podrán implementarse posteriormente.

## 4.3. Implementación de NormalizedLog

Después del CQT se aplica una transformación logarítmica normalizada.

La implementación deberá reproducir el procedimiento original:

1. Obtener la potencia elevando la magnitud al cuadrado.
2. Agregar el valor de estabilidad numérica utilizado por la referencia.
3. Calcular el logaritmo en base 10.
4. Multiplicar por 10.
5. Obtener el mínimo de la representación correspondiente a cada muestra.
6. Restar ese mínimo.
7. Obtener el máximo del resultado desplazado.
8. Dividir entre ese máximo mediante una operación segura que evite divisiones inválidas.

La operación no debe sustituirse por una normalización estándar de imagen ni por una normalización de amplitud global.

La reducción debe efectuarse sobre los ejes correctos y conservar las dimensiones esperadas.

## 4.4. Batch Normalization inicial

Después de NormalizedLog se aplica Batch Normalization.

Se deberán recuperar del modelo los parámetros originales:

- `gamma`
- `beta`
- `moving_mean`
- `moving_variance`
- `epsilon`

La operación en Rust se ejecutará en modo de inferencia.

No se deberán calcular estadísticas nuevas a partir de cada audio ni actualizar los parámetros almacenados.

## 4.5. Pruebas del frontend

Se compararán los resultados de Python y Rust para:

- CQT sin normalizar.
- NormalizedLog.
- Batch Normalization inicial.

Cada resultado se guardará con sus dimensiones, tipo de datos y métricas de error.

**Entregable de la fase:** implementación CQT y normalización funcional en Rust.

**Criterio de aceptación:** la representación que sale del frontend deberá ser suficientemente equivalente a la referencia antes de conectarla con la red neuronal.

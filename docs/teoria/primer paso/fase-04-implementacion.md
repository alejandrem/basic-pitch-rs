4. Fase 0: auditoría de la implementación oficial

Objetivo: comprender exactamente qué modelo se va a reproducir antes de escribir su implementación en Rust.

Esta será la primera fase y una condición obligatoria para comenzar las siguientes.

4.1. Identificar la versión de referencia

Se deberá descargar o consultar el repositorio oficial de Spotify Basic Pitch:

https://github.com/spotify/basic-pitch

Se registrarán:

URL del repositorio.

Commit exacto utilizado.

Versión del paquete, cuando corresponda.

Versión de Python.

Versión de TensorFlow.

Versión de las dependencias que intervienen en el procesamiento de audio.

Ubicación y procedencia del modelo preentrenado.

Licencias de código y recursos utilizados.

La finalidad es que los resultados de referencia puedan reproducirse más adelante, incluso si el repositorio original cambia.

4.2. Inspeccionar la arquitectura real

Se analizará principalmente:

models.py

nn.py

inference.py

note_creation.py

constants.py

layers/nnaudio.py

layers/signal.py

Se documentará el recorrido completo de los datos desde el audio hasta las salidas del modelo.

No se deberá utilizar únicamente un diagrama conceptual de la red: se deberá reconstruir el grafo de operaciones que efectivamente utiliza el modelo entrenado.

4.3. Identificar las capas activas

Un detalle especialmente importante es que el código oficial contiene una primera capa convolucional de contornos que está comentada porque fue omitida accidentalmente durante el entrenamiento del modelo presentado originalmente.

Por tanto, no se deberá implementar esa capa comentada como parte de la red entrenada.

La implementación deberá reproducir el grafo activo, que comienza la rama de contornos con una convolución de ocho filtros y kernel (3, 39).

También se deberán identificar las salidas que alimentan a otras ramas. En particular, la salida de contornos reducida se utiliza como entrada de la rama de notas, mientras que la rama de onsets utiliza la representación armónica inicial y se combina con la salida de notas.

4.4. Generar un inventario del modelo

Se creará un script de inspección en Python que cargue el modelo original y genere un inventario de:

Nombre de cada capa.

Tipo de capa.

Número de filtros.

Dimensiones de kernel.

Strides.

Padding.

Activación.

Dimensiones de entrada.

Dimensiones de salida.

Parámetros entrenables.

Estadísticas de Batch Normalization.

Nombres y dimensiones de los tensores.

El resultado deberá guardarse en un archivo como:

reference/
├── model_inventory.json
├── model_summary.txt
├── model_config.json
└── reference_metadata.json

4.5. Identificar los puntos de comparación

Se deberán establecer puntos de inspección en el modelo original para poder guardar las salidas intermedias.

Como mínimo:

Audio de entrada.

CQT.

CQT normalizada.

CQT después de Batch Normalization.

Harmonic Stacking.

Salida de la convolución de contornos.

Contorno final.

Características de la rama de notas.

Predicción de notas.

Características de la rama de onsets.

Predicción de onsets.

Entregable de la fase: un documento de auditoría que describa las dimensiones, operaciones, parámetros y conexiones exactas del modelo.

Criterio de aceptación: no comenzar la implementación de las capas hasta conocer el grafo activo y disponer de una configuración de referencia reproducible.
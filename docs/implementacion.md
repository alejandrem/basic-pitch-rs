# Plan de implementación: Basic Pitch en Rust con Candle y Safetensors

**Proyecto:** Implementación nativa en Rust del sistema de inferencia de Spotify Basic Pitch  
**Framework de inferencia:** Hugging Face Candle  
**Formato de pesos:** Safetensors  
**Lenguaje principal:** Rust  
**Lenguaje auxiliar:** Python, exclusivamente para extracción de pesos, generación de referencias y validación  
**Objetivo:** reproducir la inferencia del modelo preentrenado de Basic Pitch sin entrenarlo nuevamente, modificar sus pesos ni utilizar Python durante la ejecución final.

---

# 1. Objetivo general

Desarrollar una implementación en Rust capaz de recibir archivos de audio, procesarlos mediante el mismo flujo de preprocesamiento que utiliza Basic Pitch, ejecutar su arquitectura neuronal con Candle y generar una transcripción musical exportable a MIDI.

El proyecto debe reproducir tanto la red neuronal como los componentes externos que hacen posible su funcionamiento. No basta con trasladar las capas convolucionales: también es necesario reproducir correctamente la transformación Constant-Q (CQT), la normalización logarítmica, la acumulación de armónicos, el procesamiento de ventanas de audio, la reconstrucción temporal de las predicciones y el algoritmo de detección de notas.

La implementación deberá conservar el comportamiento del modelo original en la medida que lo permitan las diferencias numéricas entre frameworks.

## 1.1. Alcance

El proyecto contempla:

- Implementar en Rust la arquitectura neuronal activa de Basic Pitch.
- Utilizar los pesos preentrenados originales.
- Convertir los pesos al formato Safetensors sin alterar sus valores.
- Cargar y ejecutar los pesos mediante Candle.
- Implementar el preprocesamiento de audio necesario.
- Reproducir el procesamiento CQT y Harmonic Stacking.
- Procesar audios de distintas duraciones mediante ventanas superpuestas.
- Obtener las salidas de notas, comienzos de notas y contornos de frecuencia.
- Reproducir el posprocesamiento que transforma las predicciones en eventos musicales.
- Calcular y conservar información de pitch bend.
- Generar archivos MIDI.
- Verificar los resultados comparándolos con la implementación oficial de referencia.

## 1.2. Fuera del alcance inicial

Para mantener el proyecto controlado, esta primera implementación no incluirá:

- Entrenamiento de la red neuronal.
- Fine-tuning del modelo.
- Modificación de los pesos originales.
- Una interfaz gráfica.
- Un editor de piano roll.
- Un editor de partituras.
- Síntesis de instrumentos virtuales.
- Generación musical mediante inteligencia artificial.
- Exportación de WAV, MP3 o MP4 a partir del MIDI.

Estas funcionalidades podrán formar parte del producto completo posteriormente. La prioridad inicial será obtener un motor de transcripción funcional, verificable y reutilizable.

## 1.3. Principios técnicos

1. **Fidelidad antes que optimización:** primero se reproducirá el comportamiento original; después se mejorará el rendimiento.
2. **Pesos originales:** los valores numéricos de los pesos no deberán modificarse.
3. **Sin Python en producción:** Python solo servirá como herramienta de desarrollo y validación.
4. **Separación de responsabilidades:** audio, CQT, red neuronal, posprocesamiento y MIDI tendrán módulos independientes.
5. **Pruebas por etapas:** cada componente deberá validarse antes de integrarse con el siguiente.
6. **Resultados reproducibles:** se registrarán las versiones, configuraciones, dimensiones y procedencia de los recursos utilizados.

---

# 2. Arquitectura general del sistema

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

# 3. Especificaciones técnicas de referencia

Antes de desarrollar las capas, se establecerá un contrato técnico basado en una versión concreta del código oficial.

No se deberá trabajar directamente contra una rama que pueda cambiar sin registrar el commit utilizado. Se deberá fijar la versión del repositorio de referencia y conservar los archivos de configuración correspondientes.

## 3.1. Parámetros principales del modelo

La implementación de referencia utiliza los siguientes valores:

| Parámetro | Valor |
|---|---:|
| Frecuencia de muestreo | 22,050 Hz |
| Canales de entrada | 1 |
| Duración nominal de ventana | 2 segundos |
| `FFT_HOP` | 256 muestras |
| `ANNOTATIONS_FPS` | 86 |
| `ANNOT_N_FRAMES` | 172 |
| `AUDIO_N_SAMPLES` | 43,844 |
| Frecuencia base | 27.5 Hz |
| Semitonos musicales | 88 |
| Bins de contorno por semitono | 3 |
| Bins de contorno finales | 264 |
| Bins de notas | 88 |

**Importante:** la longitud exacta de entrada es de 43,844 muestras, no 43,834 ni 44,100. La constante del modelo se calcula mediante:

```python
AUDIO_N_SAMPLES = AUDIO_SAMPLE_RATE * AUDIO_WINDOW_LENGTH - FFT_HOP
```

Por tanto:

```text
22050 × 2 - 256 = 43844
```

Este valor deberá conservarse en la configuración de Rust y utilizarse para crear las ventanas que recibe el modelo.

## 3.2. Dimensiones principales

Las salidas musicales esperadas son:

- `note`: 88 valores por instante, correspondientes a las notas del rango del modelo.
- `onset`: 88 valores por instante, correspondientes a los comienzos de las notas.
- `contour`: 264 valores por instante, con una resolución de tres bins por semitono.

Estas dimensiones son fundamentales para el procesamiento posterior y deberán comprobarse automáticamente durante las pruebas.

## 3.3. Configuración de armónicos

La configuración predeterminada utiliza ocho desplazamientos armónicos:

```text
[0.5, 1, 2, 3, 4, 5, 6, 7]
```

Estos valores generan ocho canales de información armónica a partir de la representación CQT.

No deben confundirse con ocho capas convolucionales ni con ocho notas. Son desplazamientos de frecuencia utilizados para enriquecer la entrada del modelo.

## 3.4. Parámetros que deberán extraerse de la referencia

Además de los valores anteriores, se deberán obtener y documentar:

- `AUDIO_N_SAMPLES`.
- `AUDIO_SAMPLE_RATE`.
- `FFT_HOP`.
- `ANNOT_N_FRAMES`.
- `ANNOTATIONS_BASE_FREQUENCY`.
- `ANNOTATIONS_N_SEMITONES`.
- `CONTOURS_BINS_PER_SEMITONE`.
- `N_FREQ_BINS_CONTOURS`.
- Parámetros exactos de CQT.
- Configuración de Batch Normalization.
- Parámetros de ventanas superpuestas.
- Umbrales predeterminados del posprocesamiento.
- Parámetros de estimación de pitch bend.
- Desplazamientos temporales utilizados en la reconstrucción de las predicciones.

La implementación no deberá inventar valores faltantes ni sustituirlos por aproximaciones sin validación.

---

# 4. Fase 0: auditoría de la implementación oficial

**Objetivo:** comprender exactamente qué modelo se va a reproducir antes de escribir su implementación en Rust.

Esta será la primera fase y una condición obligatoria para comenzar las siguientes.

## 4.1. Identificar la versión de referencia

Se deberá descargar o consultar el repositorio oficial de Spotify Basic Pitch:

`https://github.com/spotify/basic-pitch`

Se registrarán:

- URL del repositorio.
- Commit exacto utilizado.
- Versión del paquete, cuando corresponda.
- Versión de Python.
- Versión de TensorFlow.
- Versión de las dependencias que intervienen en el procesamiento de audio.
- Ubicación y procedencia del modelo preentrenado.
- Licencias de código y recursos utilizados.

La finalidad es que los resultados de referencia puedan reproducirse más adelante, incluso si el repositorio original cambia.

## 4.2. Inspeccionar la arquitectura real

Se analizará principalmente:

- `models.py`
- `nn.py`
- `inference.py`
- `note_creation.py`
- `constants.py`
- `layers/nnaudio.py`
- `layers/signal.py`

Se documentará el recorrido completo de los datos desde el audio hasta las salidas del modelo.

No se deberá utilizar únicamente un diagrama conceptual de la red: se deberá reconstruir el grafo de operaciones que efectivamente utiliza el modelo entrenado.

## 4.3. Identificar las capas activas

Un detalle especialmente importante es que el código oficial contiene una primera capa convolucional de contornos que está comentada porque fue omitida accidentalmente durante el entrenamiento del modelo presentado originalmente.

Por tanto, **no se deberá implementar esa capa comentada como parte de la red entrenada**.

La implementación deberá reproducir el grafo activo, que comienza la rama de contornos con una convolución de ocho filtros y kernel `(3, 39)`.

También se deberán identificar las salidas que alimentan a otras ramas. En particular, la salida de contornos reducida se utiliza como entrada de la rama de notas, mientras que la rama de onsets utiliza la representación armónica inicial y se combina con la salida de notas.

## 4.4. Generar un inventario del modelo

Se creará un script de inspección en Python que cargue el modelo original y genere un inventario de:

- Nombre de cada capa.
- Tipo de capa.
- Número de filtros.
- Dimensiones de kernel.
- Strides.
- Padding.
- Activación.
- Dimensiones de entrada.
- Dimensiones de salida.
- Parámetros entrenables.
- Estadísticas de Batch Normalization.
- Nombres y dimensiones de los tensores.

El resultado deberá guardarse en un archivo como:

```text
reference/
├── model_inventory.json
├── model_summary.txt
├── model_config.json
└── reference_metadata.json
```

## 4.5. Identificar los puntos de comparación

Se deberán establecer puntos de inspección en el modelo original para poder guardar las salidas intermedias.

Como mínimo:

1. Audio de entrada.
2. CQT.
3. CQT normalizada.
4. CQT después de Batch Normalization.
5. Harmonic Stacking.
6. Salida de la convolución de contornos.
7. Contorno final.
8. Características de la rama de notas.
9. Predicción de notas.
10. Características de la rama de onsets.
11. Predicción de onsets.

**Entregable de la fase:** un documento de auditoría que describa las dimensiones, operaciones, parámetros y conexiones exactas del modelo.

**Criterio de aceptación:** no comenzar la implementación de las capas hasta conocer el grafo activo y disponer de una configuración de referencia reproducible.

---

# 5. Fase 1: extracción y conversión de los pesos preentrenados

**Objetivo:** obtener todos los parámetros originales del modelo y prepararlos para cargarlos desde Candle.

La extracción se realizará con Python. El programa Rust no dependerá de TensorFlow ni de Python para ejecutar el modelo.

## 5.1. Seleccionar el origen de los pesos

Se utilizará como primera opción el modelo preentrenado oficial de TensorFlow.

ONNX podrá utilizarse como alternativa de inspección o como fuente auxiliar si existe alguna dificultad para acceder al artefacto de TensorFlow.

Se recomienda utilizar TensorFlow como fuente canónica porque el modelo oficial se define originalmente en ese framework y algunas conversiones ONNX pueden introducir operaciones fusionadas, cambios en nombres de tensores o transformaciones adicionales.

No se deberán mezclar pesos de versiones diferentes del modelo.

## 5.2. Identificar todos los parámetros necesarios

El extractor deberá recuperar los parámetros correspondientes a:

### Convoluciones

Para cada capa convolucional:

- Kernel.
- Bias, cuando exista.
- Dimensiones originales.
- Nombre de la capa.
- Orden de los canales.

### Batch Normalization

Para cada capa de normalización:

- `gamma`
- `beta`
- `moving_mean`
- `moving_variance`
- `epsilon`

Las estadísticas `moving_mean` y `moving_variance` son indispensables para reproducir correctamente la inferencia. No se deberán reemplazar por valores calculados a partir del audio de entrada.

En inferencia se utilizarán las estadísticas almacenadas en el modelo entrenado.

## 5.3. Conversión del orden de los kernels

TensorFlow utiliza normalmente el orden:

```text
[kernel_height, kernel_width, input_channels, output_channels]
```

Candle espera los pesos de convolución en un orden compatible con:

```text
[output_channels, input_channels, kernel_height, kernel_width]
```

Por eso será necesario realizar una permutación de ejes:

```python
weights_rust = np.transpose(weights_tf, (3, 2, 0, 1))
```

Esta operación cambia la organización del tensor, pero no debe alterar sus valores numéricos.

Durante la extracción se deberá comprobar que la transformación inversa recupera exactamente el tensor de origen.

No se aplicará cuantización, normalización, redondeo ni ningún procedimiento que modifique los pesos.

## 5.4. Conversión a Safetensors

Se creará un programa de exportación en Python que:

1. Cargue el modelo original.
2. Obtenga los tensores de todas las capas activas.
3. Identifique su función dentro de la arquitectura.
4. Convierta los kernels al orden esperado por Candle.
5. Preserve los valores en `float32`.
6. Asigne nombres de tensores estables.
7. Exporte los parámetros a Safetensors.
8. Genere un manifiesto de configuración.

Estructura esperada:

```text
model-assets/
└── basic-pitch/
    └── icassp-2022/
        ├── model.safetensors
        ├── model_config.json
        ├── tensor_inventory.json
        ├── source_metadata.json
        └── SHA256SUMS
```

## 5.5. Manifiesto de pesos

El archivo `model_config.json` deberá contener información como:

- Nombre del modelo.
- Versión de referencia.
- Commit de origen.
- Tipo de datos.
- Número de armónicos.
- Dimensiones de entrada.
- Parámetros de CQT.
- Arquitectura de las ramas.
- Parámetros de Batch Normalization.
- Nombres de los tensores.
- Dimensiones esperadas de los tensores.

El inventario de tensores deberá indicar también el nombre original y el nombre asignado para Candle.

## 5.6. Validaciones de los pesos

Antes de utilizarlos desde Rust, se ejecutarán las siguientes comprobaciones:

- Todos los tensores esperados están presentes.
- Ningún tensor activo fue omitido.
- No existen nombres duplicados.
- No existen dimensiones incompatibles.
- Todos los parámetros tienen el tipo de datos esperado.
- Los valores no contienen `NaN` ni infinitos.
- Los kernels recuperados mediante la transformación inversa coinciden con los originales.
- El archivo Safetensors puede leerse correctamente.

**Entregable de la fase:** `model.safetensors`, manifiesto y herramientas reproducibles de extracción.

**Criterio de aceptación:** todos los pesos del modelo activo estarán disponibles y validados antes de comenzar a construir la red en Candle.

---

# 6. Fase 2: preparación del entorno Rust

**Objetivo:** preparar un espacio de trabajo modular y establecer las dependencias del proyecto.

## 6.1. Crear el workspace

Se recomienda organizar el proyecto mediante un Cargo Workspace:

```text
basic-pitch-rs/
│
├── Cargo.toml
├── Cargo.lock
├── README.md
├── LICENSE
├── NOTICE
│
├── crates/
│   │
│   ├── audio-io/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── decoder.rs
│   │       └── formats.rs
│   │
│   ├── audio-preprocess/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── mono.rs
│   │       ├── resample.rs
│   │       └── windowing.rs
│   │
│   ├── cqt/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── kernels.rs
│   │       ├── downsample.rs
│   │       ├── transform.rs
│   │       └── normalization.rs
│   │
│   ├── basic-pitch-model/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── model.rs
│   │       ├── contour.rs
│   │       ├── notes.rs
│   │       ├── onsets.rs
│   │       ├── harmonic_stacking.rs
│   │       └── weights.rs
│   │
│   ├── transcription/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── inference.rs
│   │       ├── stitching.rs
│   │       ├── note_detection.rs
│   │       └── pitch_bend.rs
│   │
│   └── midi-io/
│       ├── Cargo.toml
│       └── src/
│           ├── lib.rs
│           ├── events.rs
│           └── writer.rs
│
├── tools/
│   └── weight-export/
│       ├── inspect_model.py
│       ├── export_weights.py
│       └── validate_weights.py
│
├── models/
│   └── basic-pitch/
│       └── icassp-2022/
│
├── tests/
│   ├── fixtures/
│   ├── audio/
│   ├── model/
│   ├── transcription/
│   └── midi/
│
└── examples/
    └── transcribe.rs
```

Cada crate tendrá una responsabilidad específica. No se deberá colocar todo el procesamiento en un único archivo `main.rs`.

## 6.2. Dependencias principales

Se evaluarán e incorporarán las siguientes bibliotecas:

| Biblioteca | Función |
|---|---|
| `candle-core` | Operaciones con tensores |
| `candle-nn` | Capas y operaciones neuronales |
| `safetensors` | Herramientas de lectura y escritura del formato de pesos, cuando sea necesario |
| `symphonia` | Decodificación de formatos de audio |
| `rubato` | Remuestreo de audio |
| `midly` | Lectura y escritura MIDI |
| `serde` | Serialización de configuraciones |
| `serde_json` | Configuración y manifiestos JSON |
| `anyhow` | Manejo contextual de errores en herramientas y aplicaciones |
| `thiserror` | Definición de errores de las bibliotecas |
| `tracing` | Registro de eventos y diagnóstico |
| `tempfile` | Pruebas con archivos temporales |

Las versiones se fijarán en `Cargo.toml` y `Cargo.lock` después de comprobar la compatibilidad entre ellas.

No se deberá fijar una versión antigua de Candle simplemente porque aparezca en un ejemplo. Se seleccionará una versión estable cuyas APIs, dependencias y características de backend hayan sido verificadas.

## 6.3. Configuración del dispositivo

La primera versión utilizará CPU y tensores `float32`.

El objetivo es tener una referencia reproducible que no dependa de GPU ni de diferencias entre backends.

Posteriormente podrán evaluarse CUDA u otros backends compatibles con Candle.

**Entregable de la fase:** workspace compilable, dependencias fijadas y estructura modular creada.

---

# 7. Fase 3: implementación del procesamiento de audio

**Objetivo:** convertir los archivos de entrada en muestras de audio que respeten el contrato del modelo original.

La red neuronal no debe recibir audio directamente desde el decodificador. Primero será necesario realizar las transformaciones correspondientes.

## 7.1. Decodificación

El módulo `audio-io` se encargará de cargar el archivo y convertirlo en muestras de audio de punto flotante.

Se contemplará inicialmente la compatibilidad con:

- WAV.
- MP3.
- FLAC.
- OGG.
- M4A, según los códecs habilitados y disponibles en la biblioteca seleccionada.

Se deberá probar cada formato por separado. La extensión del archivo no garantiza que el códec interno sea compatible.

La salida del decodificador deberá utilizar una representación de muestras consistente, preferentemente `f32`.

## 7.2. Conversión a mono

Basic Pitch analiza audio mono.

Cuando el archivo tenga varios canales, se deberá realizar la misma conversión a mono que utiliza la implementación de referencia.

No se deberá seleccionar arbitrariamente el canal izquierdo ni descartar el derecho.

Se deberá comprobar cómo se comporta la conversión con archivos estéreo, señales desfasadas y canales con amplitudes diferentes.

## 7.3. Remuestreo

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

## 7.4. División en ventanas

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

## 7.5. Pruebas del procesamiento de audio

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

---

# 8. Fase 4: implementación del frontend CQT

**Objetivo:** reproducir la representación acústica que la red neuronal espera recibir.

Esta es una de las fases más delicadas del proyecto.

El modelo no recibe directamente muestras de audio ni un espectrograma STFT convencional. Recibe una representación CQT calculada mediante la implementación de `nnaudio.CQT` utilizada por Basic Pitch.

Por lo tanto, sustituirla por una CQT genérica o una STFT con parámetros parecidos no garantiza resultados equivalentes.

## 8.1. Comprender la transformación CQT

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

## 8.2. Reproducir NNAudio CQT

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

## 8.3. Implementación de NormalizedLog

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

## 8.4. Batch Normalization inicial

Después de NormalizedLog se aplica Batch Normalization.

Se deberán recuperar del modelo los parámetros originales:

- `gamma`
- `beta`
- `moving_mean`
- `moving_variance`
- `epsilon`

La operación en Rust se ejecutará en modo de inferencia.

No se deberán calcular estadísticas nuevas a partir de cada audio ni actualizar los parámetros almacenados.

## 8.5. Pruebas del frontend

Se compararán los resultados de Python y Rust para:

- CQT sin normalizar.
- NormalizedLog.
- Batch Normalization inicial.

Cada resultado se guardará con sus dimensiones, tipo de datos y métricas de error.

**Entregable de la fase:** implementación CQT y normalización funcional en Rust.

**Criterio de aceptación:** la representación que sale del frontend deberá ser suficientemente equivalente a la referencia antes de conectarla con la red neuronal.

---

# 9. Fase 5: implementación de Harmonic Stacking

**Objetivo:** reproducir la expansión armónica utilizada como entrada de las ramas neuronales.

## 9.1. Funcionamiento

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

## 9.2. Padding y desplazamientos

Cuando un canal se desplaza hacia una dirección, las posiciones que quedan fuera de los límites deberán rellenarse con ceros.

El algoritmo no debe:

- Repetir el primer o último bin.
- Envolver los valores de un extremo al otro.
- Cambiar el orden de los canales.
- Interpolar frecuencias.
- Sustituir los ceros por el mínimo de la señal.

Después de generar los canales, se conservará el rango de frecuencias requerido por el modelo.

## 9.3. Dimensiones

La salida deberá representar:

```text
(batch, time, 264, 8)
```

En Candle podrá mantenerse internamente una disposición compatible con NCHW, siempre que el orden lógico de frecuencias y canales sea equivalente.

## 9.4. Pruebas

Se probará cada desplazamiento de manera independiente con entradas artificiales.

Por ejemplo, se utilizarán matrices con valores identificables por posición para comprobar que los canales se desplazan y rellenan correctamente.

**Entregable de la fase:** Harmonic Stacking implementado y validado.

---

# 10. Fase 6: implementación de la arquitectura neuronal en Candle

**Objetivo:** reproducir la red neuronal activa del modelo utilizando `candle-core` y `candle-nn`.

Esta fase deberá ejecutarse únicamente cuando los pesos y el frontend acústico ya hayan sido comprobados.

## 10.1. Convenciones de dimensiones

TensorFlow utiliza habitualmente una organización de datos NHWC:

```text
(batch, height, width, channels)
```

Para las convoluciones de Candle se deberá trabajar con la disposición que espera su API, normalmente NCHW:

```text
(batch, channels, height, width)
```

En este modelo:

- Height representa el eje temporal.
- Width representa el eje de frecuencia.
- Channels representa los canales de características.

Se deberán implementar las transposiciones necesarias en los límites entre módulos.

La conversión de dimensiones no debe alterar la interpretación de la información musical.

También se deberá prestar especial atención al aplanamiento de frecuencia y canales: la operación original conserva un orden específico al transformar la matriz en la salida de cada rama.

## 10.2. Estructura del modelo

Se recomienda definir una estructura principal:

```rust
pub struct BasicPitchModel {
    // Frontend acústico
    // Batch normalization inicial
    // Rama de contornos
    // Rama de notas
    // Rama de onsets
}
```

La construcción del modelo recibirá los pesos mediante `VarBuilder`.

Se deberán separar las estructuras internas de cada rama para mantener el código legible y permitir pruebas independientes.

## 10.3. Rama de contornos

La rama de contornos comienza con la salida de Harmonic Stacking.

### Capa 1: convolución de características

```text
Conv2D
Filtros: 8
Kernel: 3 × 39
Padding: same
```

Después:

```text
Batch Normalization
ReLU
```

Esta es la primera convolución activa de la rama de contornos.

No se implementará la capa `(5, 5)` comentada en el código original.

### Capa 2: reducción de contornos

```text
Conv2D
Filtros: 1
Kernel: 5 × 5
Padding: same
Activación: sigmoid
```

El resultado se reorganizará para obtener una salida de contorno con 264 bins de frecuencia por instante.

La salida deberá conservarse porque también alimenta la rama de notas y posteriormente se utiliza para estimar pitch bend.

## 10.4. Rama de notas

La rama de notas recibe los contornos reducidos.

Primero se volverá a introducir la dimensión de canal necesaria para procesar la salida como un mapa bidimensional.

### Capa 1

```text
Conv2D
Filtros: 32
Kernel: 7 × 7
Stride: 1 × 3
Padding: same
```

Después se aplicará ReLU.

### Capa 2

```text
Conv2D
Filtros: 1
Kernel: 7 × 3
Padding: same
Activación: sigmoid
```

La reducción del eje frecuencial permitirá obtener las 88 salidas correspondientes a las notas musicales del modelo.

La salida final será:

```text
note: (batch, time, 88)
```

Internamente deberá conservarse el tensor previo que se utiliza en la concatenación con la rama de onsets.

## 10.5. Rama de onsets

Esta rama recibe directamente la representación obtenida después de Harmonic Stacking.

### Capa 1

```text
Conv2D
Filtros: 32
Kernel: 5 × 5
Stride: 1 × 3
Padding: same
```

Después:

```text
Batch Normalization
ReLU
```

### Concatenación

Se combinará la representación de la rama de onsets con el tensor de predicción previa de notas.

La concatenación se realizará sobre el eje de canales, conservando las dimensiones temporales y frecuenciales.

Esta operación es especialmente sensible a errores de layout. Se deberán comprobar los tamaños y el orden de los datos antes de continuar.

### Capa final

```text
Conv2D
Filtros: 1
Kernel: 3 × 3
Padding: same
Activación: sigmoid
```

La salida final será:

```text
onset: (batch, time, 88)
```

## 10.6. Salidas del modelo

La función `forward` deberá entregar una estructura con las tres predicciones:

```rust
pub struct ModelOutput {
    pub note: Tensor,
    pub onset: Tensor,
    pub contour: Tensor,
}
```

Las tres salidas deberán mantener el orden y las dimensiones esperadas.

La predicción `note` será la representación utilizada como activación de notas en el algoritmo de detección de eventos.

No se deberá eliminar la salida de contornos aunque inicialmente solo se busque producir un MIDI básico.

## 10.7. Carga de parámetros

El módulo `weights.rs` se encargará de:

- Abrir el archivo Safetensors.
- Construir el `VarBuilder`.
- Comprobar la existencia de los tensores.
- Comprobar sus dimensiones.
- Construir las capas.
- Asociar los parámetros correctos con cada operación.
- Mantener las estadísticas de Batch Normalization.
- Detectar parámetros faltantes o incompatibles.

El programa deberá producir errores descriptivos si un tensor no coincide con el manifiesto.

## 10.8. Ejecución de inferencia

La red se ejecutará en modo de inferencia y sin cálculo de gradientes.

El primer objetivo será ejecutar correctamente una sola ventana en CPU con `float32`.

Después se podrá introducir el procesamiento por lotes de ventanas para mejorar el rendimiento.

**Entregable de la fase:** arquitectura neuronal completa en Candle, capaz de cargar los pesos y producir las tres salidas.

**Criterio de aceptación:** una ventana válida de audio deberá recorrer toda la red sin errores de dimensiones y generar salidas con formas correctas.

---

# 11. Fase 7: validación de equivalencia entre TensorFlow y Candle

**Objetivo:** demostrar que la implementación en Rust reproduce el comportamiento de la implementación original.

Esta será una de las etapas de mayor importancia del proyecto.

Que la red compile y produzca números no significa que esté funcionando correctamente.

## 11.1. Crear el programa de referencia

Se implementará un programa Python que:

1. Cargue el modelo oficial.
2. Reciba una señal de prueba.
3. Ejecute el preprocesamiento original.
4. Ejecute el modelo completo.
5. Capture las salidas intermedias.
6. Guarde los tensores de referencia.
7. Exporte las dimensiones y configuraciones utilizadas.

Los datos se guardarán en un formato que pueda utilizarse fácilmente desde Rust, por ejemplo NPY, NPZ o Safetensors.

## 11.2. Comparación por etapas

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

## 11.3. Métricas de comparación

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

## 11.4. Casos de prueba

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

## 11.5. Validación de resultados musicales

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

---

# 12. Fase 8: inferencia sobre audios de duración arbitraria

**Objetivo:** procesar archivos completos, aunque sean considerablemente más largos que una ventana de dos segundos.

El modelo recibe ventanas de longitud fija, pero el usuario necesita enviar canciones completas.

## 12.1. Procesamiento por ventanas

El módulo `inference.rs` coordinará:

1. Recepción de audio ya preprocesado.
2. Generación de ventanas.
3. Ejecución del modelo por ventana.
4. Almacenamiento de predicciones.
5. Registro de la posición temporal de cada ventana.

No se deberá cargar innecesariamente todo el audio y todas las predicciones intermedias en memoria si el archivo es largo.

Se implementará un flujo por lotes de tamaño controlado.

## 12.2. Reconstrucción de predicciones

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

## 12.3. Conversión de frames a tiempo

Se deberá implementar la conversión temporal utilizada por Basic Pitch.

No se utilizará simplemente:

```text
tiempo = índice_del_frame / 86
```

La referencia utiliza `FFT_HOP`, la frecuencia de muestreo y una corrección temporal que depende de la ventana.

Se deberán reproducir también los ajustes de alineación definidos en la implementación original, incluido el desplazamiento de alineación correspondiente.

## 12.4. Pruebas de reconstrucción

Se realizarán pruebas con:

- Una canción que no llegue a completar una ventana.
- Una canción con exactamente una ventana.
- Una canción que genere dos ventanas.
- Una canción con muchas ventanas.
- Una nota que comience antes de un límite.
- Una nota que termine después de un límite.
- Una nota sostenida durante varias ventanas.

**Entregable de la fase:** inferencia completa sobre archivos largos, con reconstrucción temporal correcta.

---

# 13. Fase 9: implementación del posprocesamiento musical

**Objetivo:** transformar las matrices de predicciones en notas musicales reales.

Esta etapa convierte la información neuronal en eventos que pueden representarse en MIDI.

No deberá implementarse como una simple búsqueda de valores superiores a `0.3`.

Basic Pitch utiliza un procedimiento de detección polifónica que considera las predicciones de notas, los comienzos de notas y, opcionalmente, información adicional para encontrar eventos.

## 13.1. Detección polifónica

Se implementará una versión en Rust del procedimiento equivalente a `output_to_notes_polyphonic`.

La entrada contendrá:

- Activaciones de notas.
- Activaciones de onsets.
- Umbral de onset.
- Umbral de activación de nota.
- Duración mínima.
- Restricciones de frecuencia.
- Configuración de inferencia de onsets.
- Configuración de Melodia Trick.
- Tolerancia de energía.

## 13.2. Umbrales

Se deberán conservar los parámetros predeterminados de la versión de referencia.

Los umbrales deberán poder configurarse desde una estructura de configuración, pero inicialmente utilizarán los mismos valores que el modelo original.

No se deberán asumir valores de `0.3` o `0.5` sin comprobarlos en la versión fijada.

## 13.3. Inferencia de onsets

Cuando esté habilitada, se deberá reproducir la lógica que permite detectar comienzos adicionales de notas a partir de cambios significativos en las activaciones de frames.

Esto evita depender exclusivamente de que la red produzca una activación de onset clara para cada nota.

## 13.4. Melodia Trick

Se deberá portar el procedimiento que busca notas adicionales utilizando la energía residual de las activaciones.

Deberá reproducirse el comportamiento de:

1. Identificar las activaciones restantes más altas.
2. Determinar su continuidad temporal.
3. Evaluar el umbral de frames.
4. Aplicar la tolerancia de energía.
5. Eliminar la energía correspondiente a la nota detectada.
6. Evitar confundir notas cercanas en frecuencia.

La implementación deberá conservar la lógica de eliminación de energía en las frecuencias vecinas.

## 13.5. Duración mínima

La duración mínima se convertirá desde milisegundos a frames utilizando los parámetros temporales del modelo.

No se deberá comparar directamente una duración en milisegundos con un índice de frame.

## 13.6. Restricciones de frecuencia

Se incorporará la posibilidad de limitar el rango de frecuencia de las notas detectadas.

Esto deberá hacerse utilizando la conversión correcta entre frecuencia, nota MIDI e índice de la matriz.

## 13.7. Eventos internos

Se recomienda definir un tipo de datos independiente del formato MIDI:

```rust
pub struct NoteEvent {
    pub start_time: f64,
    pub end_time: f64,
    pub pitch: u8,
    pub velocity: u8,
    pub pitch_bend: Option<Vec<i16>>,
}
```

El evento musical deberá poder utilizarse tanto para exportar MIDI como para pruebas, visualización y futuras funciones de edición.

**Entregable de la fase:** detector de notas polifónicas en Rust que reproduzca el posprocesamiento original.

---

# 14. Fase 10: estimación de pitch bend

**Objetivo:** aprovechar la salida de contornos para detectar variaciones de afinación dentro de las notas.

La salida de contornos no representa únicamente qué notas existen. También aporta información de frecuencia con mayor resolución que una matriz de 88 notas.

El modelo trabaja con tres bins de contorno por semitono.

## 14.1. Conversión de nota a índice de contorno

Se deberá implementar la conversión entre una nota MIDI y su posición correspondiente en la matriz de contornos.

La referencia utiliza la frecuencia base de 27.5 Hz y la escala logarítmica de frecuencias.

La conversión deberá ser consistente con:

- Frecuencia base.
- Número de bins por semitono.
- Número de semitonos.
- Rango de frecuencias disponible.

## 14.2. Estimación por nota

Para cada evento detectado:

1. Obtener el índice de contorno correspondiente a su nota MIDI.
2. Crear una región de búsqueda alrededor de ese índice.
3. Aplicar el peso gaussiano utilizado por la referencia.
4. Extraer la región temporal correspondiente a la duración de la nota.
5. Encontrar el bin de mayor activación por frame.
6. Convertir el desplazamiento obtenido en unidades de contorno.
7. Asociar la secuencia al evento.

Se deberán conservar los parámetros originales de tolerancia y desviación de la ventana gaussiana.

## 14.3. Conversión a MIDI

La información de contorno se transformará en eventos MIDI Pitch Bend.

Se deberá reproducir la escala original, los límites y el rango de variación.

También se tendrá que considerar el comportamiento de notas simultáneas.

El formato MIDI tradicional aplica pitch bend por canal, no de manera independiente a cada nota dentro del mismo canal. La implementación original resuelve este caso mediante una organización específica de instrumentos cuando se permiten pitch bends múltiples.

La versión Rust deberá documentar y reproducir el mismo comportamiento.

## 14.4. Pruebas

Se probarán:

- Notas sin vibrato.
- Notas con vibrato.
- Notas con deslizamiento de afinación.
- Notas con variaciones pequeñas.
- Notas con variaciones superiores al rango permitido.
- Varias notas simultáneas con pitch bend.
- Notas superpuestas.

**Entregable de la fase:** estimación de pitch bend integrada en los eventos musicales.

---

# 15. Fase 11: exportación MIDI

**Objetivo:** generar archivos MIDI válidos a partir de los eventos detectados.

Se utilizará `midly` para construir el archivo MIDI.

## 15.1. Estructura musical

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

## 15.2. Conversión de notas

Cada `NoteEvent` se transformará en:

- Un evento Note On al comienzo.
- Un evento Note Off al finalizar.

La amplitud estimada se convertirá a velocidad MIDI siguiendo la escala definida en la referencia.

Se deberán validar los límites permitidos para el pitch y velocity.

## 15.3. Conversión de pitch bend

Los desplazamientos de contorno se transformarán en valores Pitch Bend de MIDI.

Se conservarán los límites del rango utilizado por la implementación original.

También deberá documentarse cómo se asignan los canales y tracks cuando existen notas superpuestas con distintas variaciones de afinación.

## 15.4. Validación

Los archivos generados se verificarán mediante:

- Lectura del MIDI generado.
- Comprobación de la estructura.
- Comprobación de eventos Note On y Note Off.
- Comprobación de tiempos.
- Comprobación de Pitch Bend.
- Prueba de reproducción con software MIDI independiente.

**Entregable de la fase:** exportador MIDI funcional.

---

# 16. Fase 12: integración del motor de transcripción

**Objetivo:** reunir todos los módulos en una API sencilla para utilizar el motor desde otros programas.

La arquitectura deberá evitar que las capas de la red neuronal conozcan detalles del formato de audio o del archivo MIDI.

## 16.1. API pública

Se recomienda exponer una interfaz similar a:

```rust
pub struct Transcriber {
    // Modelo cargado
    // Configuración de inferencia
    // Dispositivo de ejecución
}

impl Transcriber {
    pub fn load(
        model_path: &str,
        device: Device,
    ) -> Result<Self>;

    pub fn transcribe(
        &self,
        audio_path: &str,
    ) -> Result<TranscriptionResult>;
}
```

La API definitiva podrá incorporar opciones adicionales, pero deberá conservar la simplicidad de uso.

## 16.2. Resultado

Se recomienda una estructura similar a:

```rust
pub struct TranscriptionResult {
    pub notes: Vec<NoteEvent>,
    pub midi_data: Vec<u8>,
    pub duration_seconds: f64,
}
```

También podrá existir un resultado de depuración que conserve las salidas originales del modelo.

## 16.3. Carga única del modelo

El modelo deberá cargarse una sola vez y reutilizarse para procesar múltiples archivos.

No se deberán volver a leer los pesos desde Safetensors cada vez que se solicite una transcripción.

Esto reducirá los tiempos de espera y evitará trabajo repetido.

## 16.4. Errores

Se deberán definir errores específicos para:

- Archivo inexistente.
- Formato no compatible.
- Error de decodificación.
- Error de remuestreo.
- Error de dimensiones.
- Peso faltante.
- Configuración incorrecta.
- Error del backend de Candle.
- Fallo de generación MIDI.

Los errores deberán tener mensajes útiles para diagnosticar el problema.

**Entregable de la fase:** biblioteca Rust que permita transcribir audio y obtener los eventos o el archivo MIDI sin conocer los detalles internos.

---

# 17. Fase 13: pruebas automatizadas y control de calidad

**Objetivo:** asegurar que los cambios futuros no rompan la transcripción.

## 17.1. Pruebas unitarias

Cada crate deberá incluir pruebas para sus operaciones principales:

- Decodificación.
- Conversión a mono.
- Remuestreo.
- Ventanas.
- CQT.
- NormalizedLog.
- Batch Normalization.
- Harmonic Stacking.
- Convoluciones.
- Concatenación.
- Aplanamiento de tensores.
- Detección de notas.
- Pitch bend.
- Exportación MIDI.

## 17.2. Pruebas de integración

Se probará el flujo completo:

```text
Audio → Preprocesamiento → CQT → Candle
      → Predicciones → Eventos → MIDI
```

Las pruebas deberán verificar que la salida puede ser abierta por un lector MIDI independiente.

## 17.3. Pruebas de regresión

Se conservará un conjunto de archivos de audio de referencia y sus resultados esperados.

Cada cambio en el código deberá ejecutar nuevamente las pruebas de regresión.

Se deberán registrar:

- Versiones de herramientas.
- Parámetros utilizados.
- Diferencias numéricas.
- Diferencias en eventos musicales.
- Tiempo de ejecución.

## 17.4. Integración continua

Se configurará CI para ejecutar:

- `cargo fmt --check`
- `cargo clippy`
- `cargo test`
- Pruebas de lectura de pesos.
- Pruebas del modelo.
- Pruebas de transcripción.
- Pruebas de exportación MIDI.

Las pruebas que dependan de recursos grandes podrán ejecutarse en un conjunto de CI separado.

---

# 18. Fase 14: medición y optimización del rendimiento

**Objetivo:** mejorar velocidad y uso de recursos una vez que los resultados sean equivalentes a la referencia.

La optimización no deberá comenzar antes de conseguir una implementación funcional y validada.

## 18.1. Establecer una línea base

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

## 18.2. Optimizaciones posibles

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

## 18.3. Precisión numérica

La primera versión se mantendrá en `float32`.

Se podrá investigar precisión reducida o cuantización en el futuro, pero no formará parte de la primera implementación.

Las optimizaciones deberán medirse en términos de:

- Velocidad.
- Memoria.
- Diferencia numérica.
- Impacto en eventos musicales.

No se considerará exitosa una optimización que acelere el modelo a costa de degradar considerablemente la transcripción.

---

# 19. Fase 15: documentación y preparación para reutilización

**Objetivo:** dejar el motor listo para integrarse posteriormente en aplicaciones de escritorio, web o móviles.

## 19.1. Documentación técnica

El repositorio deberá incluir:

- Arquitectura general.
- Estructura de carpetas.
- Explicación de los módulos.
- Procedimiento de extracción de pesos.
- Formato de los recursos del modelo.
- Instrucciones para compilar.
- Instrucciones para ejecutar.
- Parámetros de transcripción.
- Formato de resultados.
- Limitaciones conocidas.
- Reportes de validación.

## 19.2. Documentación del modelo

Se documentará:

- Procedencia de los pesos.
- Versión del modelo.
- Hashes.
- Dimensiones de entrada y salida.
- Arquitectura implementada.
- Transformaciones realizadas durante la exportación.
- Licencias y avisos aplicables.

El hecho de exportar los pesos a Safetensors no elimina la necesidad de revisar las condiciones de distribución del modelo original.

## 19.3. Preparación para interfaces futuras

El motor se mantendrá independiente de la interfaz gráfica.

Esto permitirá integrarlo después con:

- Tauri.
- Vue.
- TypeScript.
- Aplicaciones de escritorio.
- Una posible interfaz web.
- Una aplicación móvil compatible.

La biblioteca principal no deberá depender de componentes visuales ni de tecnologías exclusivas de Tauri.

---

# 20. Estructura definitiva del flujo de trabajo

El orden de desarrollo recomendado es el siguiente:

| Orden | Fase | Resultado principal |
|---:|---|---|
| 0 | Auditoría de Basic Pitch | Contrato técnico de referencia |
| 1 | Extracción de pesos | Safetensors y manifiesto |
| 2 | Preparación Rust | Workspace y dependencias |
| 3 | Procesamiento de audio | Audio mono y ventanas correctas |
| 4 | CQT y NormalizedLog | Representación acústica |
| 5 | Harmonic Stacking | Entrada armónica del modelo |
| 6 | Red neuronal Candle | Predicciones de las tres ramas |
| 7 | Validación de equivalencia | Comparaciones TensorFlow/Rust |
| 8 | Inferencia por ventanas | Audio de duración arbitraria |
| 9 | Posprocesamiento | Eventos de notas |
| 10 | Pitch bend | Variaciones de afinación |
| 11 | Exportación MIDI | Archivo MIDI válido |
| 12 | API de transcripción | Motor reutilizable |
| 13 | Pruebas automatizadas | Calidad y regresiones |
| 14 | Optimización | Mejor rendimiento |
| 15 | Documentación | Proyecto reproducible |

---

# 21. Criterios generales para considerar terminado el proyecto

La implementación se considerará funcionalmente completa cuando cumpla todos los siguientes puntos:

## Modelo

- [ ] Se identificó y fijó una versión concreta del modelo original.
- [ ] Se inspeccionó la arquitectura realmente entrenada.
- [ ] Se identificaron todas las capas activas.
- [ ] Se extrajeron todos los pesos necesarios.
- [ ] Los pesos se exportaron a Safetensors.
- [ ] Se validó que la conversión no alterara los valores.
- [ ] Candle puede cargar todos los parámetros.

## Procesamiento de audio

- [ ] Se decodifican los formatos contemplados.
- [ ] El audio estéreo se convierte correctamente a mono.
- [ ] El audio se remuestrea a 22,050 Hz.
- [ ] Se generan ventanas de 43,844 muestras.
- [ ] Se reproduce el padding y solapamiento de la referencia.

## Frontend acústico

- [ ] CQT reproduce la configuración original.
- [ ] NormalizedLog está implementado.
- [ ] Batch Normalization inicial utiliza los parámetros originales.
- [ ] Harmonic Stacking reproduce los desplazamientos y padding correctos.

## Red neuronal

- [ ] Rama de contornos implementada.
- [ ] Rama de notas implementada.
- [ ] Rama de onsets implementada.
- [ ] Concatenación y layouts verificados.
- [ ] Salidas con dimensiones correctas.
- [ ] Pesos y estadísticas cargados desde Safetensors.

## Equivalencia

- [ ] Se compararon las etapas intermedias.
- [ ] Se compararon las salidas finales.
- [ ] Se documentaron las tolerancias numéricas.
- [ ] Se compararon eventos musicales.
- [ ] Se probaron diferentes señales y audios.

## Transcripción y MIDI

- [ ] Se reconstruyen correctamente las predicciones de las ventanas.
- [ ] Se convierten frames a tiempo correctamente.
- [ ] Se detectan notas polifónicas.
- [ ] Se reproducen los parámetros de posprocesamiento.
- [ ] Se calcula pitch bend.
- [ ] Se generan archivos MIDI válidos.
- [ ] Se verifican los MIDI con herramientas externas.

## Entrega técnica

- [ ] No se necesita Python para ejecutar una transcripción.
- [ ] El modelo puede cargarse una sola vez y reutilizarse.
- [ ] El código está dividido en módulos independientes.
- [ ] Existe documentación de instalación y uso.
- [ ] Las pruebas automatizadas pasan.
- [ ] Se registraron versiones, hashes y licencias.

---

# 22. Primera tarea que debe ejecutarse

La implementación no debe comenzar creando directamente las capas de Candle.

La primera actividad concreta será construir el programa de auditoría y extracción en Python.

Este programa deberá cargar el modelo oficial y producir tres artefactos:

```text
reference/
├── model_inventory.json
├── model_config.json
└── reference_outputs/
```

**`model_inventory.json`** contendrá todos los parámetros, nombres, tipos y dimensiones de las capas.

**`model_config.json`** contendrá las constantes, parámetros de CQT, configuración de ventanas y estructura del modelo.

**`reference_outputs/`** contendrá las salidas intermedias de pruebas controladas que más adelante utilizará Rust para verificar su equivalencia.

Una vez generados y revisados estos tres artefactos, se continuará con la exportación de los pesos a Safetensors y la preparación del workspace.

Esta secuencia evita construir una implementación basada en dimensiones supuestas, nombres incorrectos o capas que no forman parte del modelo efectivamente entrenado.

---

# Resultado esperado

Al finalizar este plan se tendrá una biblioteca de transcripción musical implementada en Rust, utilizando Candle para la inferencia neuronal y Safetensors para almacenar los pesos preentrenados.

El motor podrá recibir audio, reproducir el procesamiento acústico de Basic Pitch, ejecutar sus tres ramas neuronales, detectar notas y variaciones de afinación, reconstruir la secuencia temporal y exportar un archivo MIDI.

El resultado será un núcleo independiente que posteriormente podrá integrarse en una aplicación más amplia de edición y producción musical, sin tener que reescribir el motor de transcripción.
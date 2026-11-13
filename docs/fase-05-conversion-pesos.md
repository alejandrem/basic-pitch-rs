
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

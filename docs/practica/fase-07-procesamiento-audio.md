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

### 3.2.1. Corrección: mono por promedio, y ANTES del remuestreo

`librosa.load(path, sr=22050, mono=True)` ejecuta, en este orden exacto:

```python
y, sr_native = soundfile_load(path, ...)   # decodifica multicanal
if mono:
    y = to_mono(y)                          # 1. PROMEDIO de canales
if sr is not None:
    y = resample(y, orig_sr=sr_native, target_sr=sr, res_type=res_type)  # 2. remuestreo
```

Consecuencias obligatorias:

- La conversión a mono es el **promedio aritmético** de los canales
  (`librosa.to_mono` usa `np.mean`), no la suma ni un canal concreto.
- La conversión a mono ocurre **antes** del remuestreo, no después. Cambiar el
  orden produce una señal distinta y, por tanto, un CQT distinto.
- El promedio se calcula sobre los canales de la señal decodificada, con la
  precisión del `dtype` de lectura (`float32`).

Se deberá comprobar cómo se comporta la conversión con archivos estéreo,
señales desfasadas y canales con amplitudes diferentes.

## 3.3. Remuestreo

Toda señal se convertirá a:

```text
22050 Hz
```

El remuestreador elegido será **`soxr` en modo `HQ`**, equivalente al
`res_type="soxr_hq"` que usa `librosa` por defecto en la versión de referencia
fijada.

### 3.3.1. Corrección: `soxr_hq`, no `rubato`, y versiones fijadas

El `res_type` por defecto de `librosa.load` **depende de la versión**:

| librosa | `res_type` por defecto | Motor |
|---|---|---|
| 0.8 / 0.9 | `kaiser_best` | `resampy` |
| >= 0.10 | `soxr_hq` | `soxr` |

La versión de referencia fijada es **librosa 0.10.2.post1**, por lo que la
referencia usa **`soxr_hq`**. `rubato` no es bit-idéntico a `soxr_hq`: produce
un audio ligeramente distinto, lo que desplaza el CQT y, por tanto, las notas
detectadas. Se descarta `rubato` para el remuestreo de producción.

**Aviso importante:** usar `soxr` en Rust garantiza el mismo *algoritmo* de
remuestreo, pero **no** garantiza igualdad muestra a muestra, porque el
decodificador de audio puede diferir (librosa usa `soundfile` / `libsndfile`;
Rust usará `symphonia`). La diferencia se medirá en la Fase 7 y se documentará
como tolerancia asumida, no como error.

En consecuencia, el plan de validación debe comparar:

1. La señal completa (decodificación + mono + remuestreo) contra la referencia,
   midiendo la diferencia y su efecto en el CQT.
2. El resto de etapas (CQT en adelante) con la **misma** entrada de audio para
   aislar el error del remuestreo del error del frontend.

### 3.3.2. Versiones a fijar

| Componente | Versión fijada | Motivo |
|---|---|---|
| Python | 3.11.18 | Requisito de TensorFlow 2.15 |
| TensorFlow | 2.15.0 | Versión de referencia |
| librosa | 0.10.2.post1 | Define `res_type='soxr_hq'` |
| numpy | 1.26.4 | `<2` requerido por TF 2.15 |
| soxr | la que resuelva librosa 0.10.2 | Motor de remuestreo de la referencia |

El entorno de referencia se crea con `uv`:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python \
  "tensorflow==2.15.0.post1" "librosa==0.10.2.post1" \
  "pretty_midi>=0.2.9" "mir_eval>=0.6.0" \
  "resampy>=0.2.2,<0.4.3" "safetensors>=0.4" "numpy<2"
```

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

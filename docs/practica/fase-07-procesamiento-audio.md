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

**Aviso importante (medido en P2):** usar `soxr` en Rust garantiza el mismo
*algoritmo* de remuestreo, pero **no** garantiza igualdad muestra a muestra,
porque el decodificador de audio puede diferir (librosa usa `soundfile` /
`libsndfile`; Rust usa `symphonia`).

Esta incertidumbre **se ha medido** en la Fase 3 (P2) con audio real del
upstream, y el resultado es que **la diferencia está por debajo del epsilon
de `float32`**: Symphonia y `soundfile` producen la misma señal. Ver el
apartado 3.6.2. La diferencia que queda no procede de una decisión de
diseño, sino del redondeo de coma flotante.

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

## 3.6. Validación

La validación de esta fase es un test de integración de punta a punta
(`crates/audio-preprocess/tests/pipeline.rs`): escribe un archivo WAV real
en un directorio temporal, lo decodifica y recorre la cadena completa.

Lo que comprueba:

| Punto | Comprobación |
|---|---|
| Decodificación | 2 canales, 44.100 Hz, número de frames esperado |
| Mono | el **promedio** de los canales (0.75 × seno con un canal al doble de amplitud), no la suma ni un canal |
| Remuestreo | a 22.050 Hz, con la amplitud conservada: el remuestreador no normaliza |
| Ventaneo | `ceil((len + 3840) / 36164)` ventanas de 43.844 muestras |
| Relleno inicial | la ventana 0 empieza **antes** del audio original (3.840 ceros) |
| Contenido | a partir del relleno, la ventana contiene el audio remuestreado |
| Solapamiento | ventanas consecutivas desplazadas 36.164 muestras (7.680 de solapamiento real) |

Hay además un test de audio ya mono a 22.050 Hz, que debe salir intacto
del remuestreo.

Este test es el que detecta que el orden mono/remuestreo se invierta: si
se invirtiera, el promedio se calcularía sobre una señal ya remuestreada
y el resultado sería distinto.

### 3.6.1. Lo que esta fase NO puede validar

El audio de la referencia (`input_audio.npy`) son 43.844 muestras **ya
decodificadas, ya en mono y ya remuestreadas**. Por tanto, la
comparación contra el oráculo del frontend no cubre la decodificación ni
el remuestreo: cubre la ventana.

Para cerrar ese hueco se ha generado un segundo oráculo, con **audio
real**: `reference/audio_reference/`.

| Artefacto | Contenido |
|---|---|
| `vocadito_10_mono22050.npy` | La señal mono a 22.050 Hz de `vocadito_10.wav`, tal y como la produce la referencia |
| `vocadito_14_mono22050.npy` | Lo mismo para `vocadito_14.wav` |
| `*_v0_*.npy`, `*_v1_*.npy` | Las cuatro etapas del frontend de TensorFlow sobre las ventanas 0 y 1 |

El generador es
`tools/weight-export/reference_audio_frontend.py`, que usa el SavedModel
oficial y el **ventaneo literal** de `basic_pitch/inference.py`
(`get_audio_input`: relleno inicial de 3.840 ceros y ventanas de 43.844
con salto 36.164).

### 3.6.2. Resultado: Symphonia y soundfile son equivalentes

El oráculo permite medir, por fin, la diferencia entre el decodificador
de Rust (**Symphonia**) y el de la referencia (**soundfile**/libsndfile),
que era la incógnita declarada en el apartado 3.3.1.

| Archivo | MAE | Diferencia máxima | RMS de la señal |
|---|---:|---:|---:|
| `vocadito_10` | 2.71e-8 | 2.98e-7 | 0.148 |
| `vocadito_14` | 9.56e-9 | 1.57e-7 | 0.061 |

La diferencia está **por debajo del epsilon de `float32`** (1.2e-7
relativo). Es decir: **Symphonia y soundfile producen la misma señal**, y
el remuestreador es el mismo (`soxr` en modo HQ). La incertidumbre que
dejaba abierta la Fase 3 queda resuelta: no hay una diferencia
estructural entre los decodificadores, solo redondeo de `float32`.

Esto **no** es una garantía de que el MIDI final sea idéntico: la
diferencia de 3e-7 se propaga y se amplifica en el logaritmo (ver
[fase-08-frontend-cqt.md](fase-08-frontend-cqt.md), apartado 4.7.2). Lo
que sí queda garantizado es que el error no viene de una decisión de
diseño (decodificador distinto, remuestreador distinto, orden
mono/remuestreo invertido), sino del redondeo de coma flotante.

### 3.6.3. El cadena completa, de extremo a extremo

`crates/basic-pitch-model/tests/audio_end_to_end.rs` recorre **todo** el
camino, desde el archivo `.wav` hasta la entrada de la red, y lo compara
contra TensorFlow etapa a etapa:

```text
archivo .wav
  -> decodificacion      (Symphonia)       }
  -> mono por promedio   }                 } contra el oraculo de
  -> remuestreo a 22050  (soxr HQ)          } TensorFlow
  -> ventana de 43844    (relleno de 3840) }
  -> CQT -> NormalizedLog -> BatchNorm -> Harmonic Stacking
```

Resultados (ventanas 0 y 1 de `vocadito_10`, ventana 0 de
`vocadito_14`):

| Audio / ventana | CQT (max) | NormalizedLog (max) | BatchNorm (max) | Harmonic Stacking (max) |
|---|---:|---:|---:|---:|
| `vocadito_10` v0 | 2.38e-6 | 3.90e-4 | 9.68e-4 | 9.68e-4 |
| `vocadito_10` v1 | 3.34e-6 | 6.10e-4 | 1.51e-3 | 1.51e-3 |
| `vocadito_14` v0 | 9.54e-7 | 1.14e-4 | 2.83e-4 | 2.83e-4 |

El error es **menor** que con la señal sintética del oráculo original
(5.7e-6 en la CQT), porque el audio real tiene más energía y menos bins
en el silencio absoluto, que es donde el logaritmo amplifica.

### 3.6.4. El ventaneo, comparado con el upstream

El ventaneo de Rust se contrastó con el de `basic_pitch/inference.py`:

```python
for i in range(0, audio_original.shape[0], hop_size):
    window = audio_original[i : i + AUDIO_N_SAMPLES]
    if len(window) < AUDIO_N_SAMPLES:
        window = np.pad(window, pad_width=[[0, AUDIO_N_SAMPLES - len(window)]])
```

Coinciden los tres puntos: el desplazamiento entre ventanas, el relleno
final con ceros y el número de ventanas (`ceil((len + 3840) / 36164)`).

### 3.6.5. La matriz de pruebas del apartado 3.5

Los nueve escenarios que enumera el apartado 3.5 tienen su test, y cada
uno escribe un archivo real, lo decodifica y comprueba el contrato del
modelo:

| Escenario | Test | Fichero |
|---|---|---|
| Audio mono | `audio_mono` | `tests/scenarios.rs` |
| Audio estéreo | `audio_estereo` | `tests/scenarios.rs` |
| Menos de dos segundos | `audio_de_menos_de_dos_segundos` | `tests/scenarios.rs` |
| Exactamente dos segundos | `audio_de_exactamente_dos_segundos` | `tests/scenarios.rs` |
| Varios minutos | `audio_de_varios_minutos` | `tests/scenarios.rs` |
| Silencio inicial y final | `audio_con_silencio_inicial_y_final` | `tests/scenarios.rs` |
| Distintas frecuencias de muestreo | `archivos_con_diferentes_frecuencias_de_muestreo` | `tests/scenarios.rs` |
| Distintos formatos | 8 tests | `tests/formats.rs` |
| Cerca de los límites de amplitud | `senales_cerca_de_los_limites_de_amplitud` | `tests/scenarios.rs` |
| Transitorios al inicio y al final | `audio_con_transitorios_al_inicio_y_al_final` | `tests/scenarios.rs` |

**Los formatos por separado.** Los fixtures son el mismo contenido en
cinco formatos (`tools/weight-export/make_audio_fixtures.py`):

| Formato | Codec | Pérdidas |
|---|---|---|
| WAV | PCM 16 bits | ninguna |
| FLAC | FLAC | ninguna |
| MP3 | LAME, 128 kbps | sí |
| OGG | Vorbis, 128 kbps | sí |
| M4A | AAC, 128 kbps | sí |

La comprobación no es «ha devuelto muestras», sino que el **contenido**
sea el correcto: se mide la magnitud en la frecuencia de cada canal con
Goertzel (220 Hz a la izquierda, 330 Hz a la derecha) y el desfase entre
canales, que es lo que verifica que no se han intercambiado. Los
formatos con pérdidas tienen tolerancias más anchas.

### 3.6.6. Los seis puntos delicados del remuestreador

El apartado 3.3 pide atención explícita a seis cosas. Las seis tienen su
test en `tests/resampler.rs`:

| Punto | Cómo se comprueba |
|---|---|
| Calidad del filtro | La magnitud en la fundamental se conserva y no hay energía espurio a 1000 Hz |
| Retardo de grupo | Con los cruces de cero ascendentes: si el retardo es constante, el espaciado es exactamente el periodo |
| Tratamiento de los extremos | Un impulso central no produce transitorios espurios en los bordes |
| Número final de muestras | `ceil(len * to / from)` en nueve pares de frecuencias |
| Diferencias de amplitud | El remuestreador **no** normaliza: se comprueba con tres escalas |
| Frecuencias poco habituales | 8000, 11025, 17000, 32000, 48000 y 96000 Hz, y razones de 2,75 y 8 |

**Por qué los cruces de cero y no la fase.** La fase de Goertzel tiene un
sesgo que depende de la posición de la ventana cuando esta no contiene
un número entero de periodos de la frecuencia medida, y ese sesgo es
**indistinguible** de una deriva de fase. Con los cruces de cero no hay
ambigüedad: si el retardo de grupo es constante, el espaciado entre
cruces es exactamente el periodo.

> **Trampa documentada:** medir el retardo de grupo con la fase de
> Goertzel produce un falso positivo. En este proyecto, la fase medida
> «derivaba» 1,7 radianes entre el principio y el final de la señal...
> y la señal **ideal, sin remuestrear**, derivaba exactamente lo mismo.
> El error era del método de medida, no del remuestreador.



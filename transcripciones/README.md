# Transcripciones

En esta carpeta queda el resultado de las transcripciones que se hayan
hecho, para poder escucharlas y compararlas.

## Qué hay en cada transcripción

| Fichero | Qué es |
|---|---|
| `<nombre>.mp3` | El audio original |
| `<nombre>.mid` | **La transcripción**: el MIDI que genera el motor |
| `<nombre>_transcrito.wav` | El MIDI renderizado a audio, para poder oírlo |

El `.wav` **no** es una transcripción: es un renderizado del MIDI, hecho
con `fluidsynth` y el soundfont General MIDI. Sirve para escuchar el
resultado; el MIDI es la transcripción en sí.

```bash
# Volver a hacerlo
cargo run --release -- transcribe transcripciones/amaris.mp3 -o transcripciones/amaris.mid

# Volver a oírlo
.venv/bin/python tools/weight-export/render_midi.py \
    transcripciones/amaris.mid transcripciones/amaris_transcrito.wav
```

---

## `amaris`

| | |
|---|---|
| Audio original | MP3, 44.100 Hz, estéreo, 164,2 s |
| Transcripción | **1842 notas**, 163,7 s, piano acústico (GM 0) |
| Pitches detectados | 28..98 (grave a agudo) |
| Velocidades | 31..103 (media 60) |
| Tonalidad detectada | Sol menor (G, D, B, C, E, A) |

**Las 1842 notas coinciden una a una con las que produce el upstream de
Spotify** (mismo inicio, mismo final, mismo pitch). Se puede comprobar:

```bash
.venv/bin/python tools/weight-export/reference_notes.py transcripciones/amaris.mp3
cargo test --release -p basic-pitch-transcription --test notes_reference
```

### Las dos versiones

| Fichero | Notas | Qué es |
|---|---:|---|
| `amaris.mid` | 1842 | El **umbral por defecto** (0.3), idéntico al upstream de Spotify |
| `amaris_limpio.mid` | 1499 | Con `--frame-threshold 0.4`: menos armonías, a costa de algunas notas reales |

Las dos suenan a **piano acústico** (programa GM 0).

El umbral decide cuántas activaciones se consideran "nota encendida". Con
0.3 (el del upstream) entran las armonías del timbre del piano, que el
modelo oye como notas propias. Con 0.4 se eliminan 517 de esas, pero
también se pierden 174 notas reales: queda el 72 % del contenido.

```bash
cargo run --release -- transcribe amaris.mp3 -o amaris_limpio.mid --frame-threshold 0.4
```

---

## La prueba de fidelidad: audio limpio contra audio real

Para saber cuánto de la diferencia es del **motor** y cuánto es del
**timbre**, se renderiza el MIDI original de dos maneras:

- **con soundfont** (piano real, con armónicos),
- **limpio** (un seno puro por nota, sin armónicos: `render_midi.py --limpio`).

Y se transcriben los dos:

| Audio transcrito | Notas | Notas del original recuperadas | Notas extra |
|---|---:|---:|---:|
| Limpio (seno puro) | 1325 | 1203 (61 %) | 122 |
| Piano real | 2158 | 1716 (**87 %**) | 442 |
| Piano real, sin Melodia | 2113 | 1706 (86.5 %) | 407 |

**El resultado contraintuitivo: el audio limpio se transcribe PEOR.**

La razón es que Basic Pitch no oye notas: oye **timbre**. Un seno puro
no tiene los parciales que el modelo usa para afinar y para detectar los
ataques, así que el motor tiene menos información, no más.

Y la consecuencia práctica: **las "notas extra" no son un fallo del
motor, son el comportamiento del upstream**. Spotify Basic Pitch
transcribe exactamente las mismas 2158 notas, con las mismas armonías.

Quitar el truco de Melodia apenas cambia nada (2158 → 2113): las
armonías las predice la red, no las añade el postprocesado. Por eso la
única forma de "limpiar" es subir el umbral de frame, y eso también se
carry el costo de perder notas reales.

---

## Nota sobre el tamaño

El `.wav` pesa unos 29 MB y el `.mp3` unos 3,8 MB, así que **no se
versionan**: están en el `.gitignore`. Los `.mid` sí (son unos 13 KB).

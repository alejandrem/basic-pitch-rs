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
con `fluidsynth` y el soundfont `TimGM6mb`. Sirve para escuchar el
resultado; el MIDI es la transcripción en sí.

```bash
# Volver a hacerlo
cargo run --release -- transcribe transcripciones/amaris.mp3 -o transcripciones/amaris.mid

# Volver a oírlo
fluidsynth -F transcripciones/amaris_transcrito.wav -r 44100 -g 0.7 \
    .venv/lib/python3.11/site-packages/pretty_midi/TimGM6mb.sf2 transcripciones/amaris.mid
```

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

## Nota sobre el tamaño

El `.wav` pesa unos 29 MB y el `.mp3` unos 3,8 MB, así que **no se
versionan**: están en el `.gitignore`. Los `.mid` sí (son unos 13 KB).

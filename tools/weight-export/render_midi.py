#!/usr/bin/env python
"""Renderiza un MIDI a audio, para poder transcribirlo y oírlo.

Sirve para hacer la validacion definitiva: tu MIDI es la verdad de campo
(notas, tiempos, pitches) y el audio renderizado es lo que se transcribe.

## Dos motores de render

- **soundfont** (por defecto): `fluidsynth` con un soundfont General MIDI.
  Suena a piano de verdad, con sus armonias.
- **limpio**: un seno puro por nota, generado aqui. **Sin armonias**, que
  es justo lo que hace que se pueda comprobar la fidelidad de forma
  aislada: si el motor recupera exactamente las notas del MIDI original
  sobre el render limpio, se sabe que lo que "suena de mas" en una
  transcripcion real son los parciales del timbre, no un fallo del
  detector.

    .venv/bin/python tools/weight-export/render_midi.py entrada.mid salida.wav
    .venv/bin/python tools/weight-export/render_midi.py entrada.mid limpio.wav --limpio

Un `.mid` **no es audio**: es una lista de notas. Esto es un renderizado,
no una transcripcion.
"""

import argparse
import pathlib
import shutil
import subprocess
import sys

import numpy as np

#: Sitios donde puede estar un soundfont General MIDI, por orden de
#: preferencia (el GM estandar suena mejor que los genicos).
SOUNDFONTS = [
    pathlib.Path("/home/ale/Descargas/alesito.mix/frontend/FluidR3_GM.sf2"),
    pathlib.Path(
        "/home/ale/Descargas/basic-pitch-rs/.venv/lib/python3.11/site-packages"
        "/pretty_midi/TimGM6mb.sf2"
    ),
    pathlib.Path("/usr/share/sounds/sf2/FluidR3_GM.sf2"),
    pathlib.Path("/usr/share/sounds/sf2/default-GM.sf2"),
]


def encontrar_soundfont() -> pathlib.Path:
    for candidato in SOUNDFONTS:
        if candidato.exists():
            return candidato
    raise SystemExit(
        "no se encuentra ningun soundfont. Instala fluidsynth con sus "
        "sonidos, o pon uno en las rutas de SOUNDFONTS."
    )


def render_con_soundfont(entrada: pathlib.Path, salida: pathlib.Path, sr: int) -> None:
    fluidsynth = shutil.which("fluidsynth")
    if fluidsynth is None:
        raise SystemExit("fluidsynth no esta instalado (apt-get install fluidsynth)")

    soundfont = encontrar_soundfont()
    subprocess.run(
        [
            fluidsynth,
            "-F", str(salida),
            "-r", str(sr),
            "-g", "0.8",
            "-q",
            str(soundfont),
            str(entrada),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def render_limpio(entrada: pathlib.Path, salida: pathlib.Path, sr: int) -> None:
    """Un seno puro por nota: sin armonias, sin timbre, solo la nota."""
    import pretty_midi
    import soundfile as sf

    midi = pretty_midi.PrettyMIDI(str(entrada))
    fin = midi.get_end_time() + 1.0
    total = int(fin * sr)
    buffer = np.zeros(total, dtype=np.float64)

    for instrumento in midi.instruments:
        for nota in instrumento.notes:
            freq = 440.0 * 2.0 ** ((nota.pitch - 69) / 12.0)
            inicio = int(nota.start * sr)
            final = int(nota.end * sr)
            n = final - inicio
            if n <= 0:
                continue

            t = np.arange(n) / sr
            # Envolvente: ataque de 10 ms, sostenido, caida de 30 ms.
            envolvente = np.ones(n)
            ataque = min(int(0.010 * sr), n // 2)
            if ataque > 0:
                envolvente[:ataque] = np.linspace(0.0, 1.0, ataque)
            caida = min(int(0.030 * sr), n // 2)
            if caida > 0:
                envolvente[-caida:] = np.linspace(1.0, 0.0, caida)

            onda = np.sin(2.0 * np.pi * freq * t) * envolvente
            # La velocidad se respeta, para que las notas suaves no dominen.
            buffer[inicio:final] += onda * (nota.velocity / 127.0)

    pico = np.abs(buffer).max()
    if pico > 0:
        buffer = buffer / pico * 0.9

    sf.write(str(salida), buffer.astype(np.float32), sr, subtype="PCM_16")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("entrada", help="el MIDI de entrada")
    ap.add_argument("salida", help="el WAV de salida")
    ap.add_argument(
        "--limpio",
        action="store_true",
        help="seno puro por nota, sin armonias (para medir la fidelidad)",
    )
    ap.add_argument("--sr", type=int, default=44100, help="frecuencia de muestreo")
    args = ap.parse_args()

    entrada = pathlib.Path(args.entrada)
    salida = pathlib.Path(args.salida)

    if not entrada.exists():
        print(f"no existe {entrada}")
        return 1

    salida.parent.mkdir(parents=True, exist_ok=True)
    if args.limpio:
        render_limpio(entrada, salida, args.sr)
    else:
        render_con_soundfont(entrada, salida, args.sr)

    print(f"renderizado: {salida} ({salida.stat().st_size / 1024 / 1024:.1f} MiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Renderiza un MIDI a audio, para poder transcribirlo.

Sirve para hacer la validacion definitiva: tu MIDI es la verdad de campo
(notas, tiempos, pitches) y el audio renderizado es lo que se transcribe.

Usa `fluidsynth` con el soundfont que trae `pretty_midi`, asi que no hace
falta instalar nada mas.

    .venv/bin/python tools/weight-export/render_midi.py entrada.mid salida.wav

Nota: el audio resultante **no** se commitea. Es material personal del
usuario y ocupa 30 MB por minuto.
"""

import pathlib
import shutil
import subprocess
import sys


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2

    entrada = pathlib.Path(sys.argv[1])
    salida = pathlib.Path(sys.argv[2])

    if not entrada.exists():
        print(f"no existe {entrada}")
        return 1

    fluidsynth = shutil.which("fluidsynth")
    if fluidsynth is None:
        print("fluidsynth no esta instalado (apt-get install fluidsynth)")
        return 1

    # El soundfont que trae pretty_midi (TimGM6mb).
    sf2 = pathlib.Path(
        __file__
    ).resolve().parents[2] / ".venv/lib/python3.11/site-packages/pretty_midi/TimGM6mb.sf2"
    if not sf2.exists():
        print(f"no se encuentra el soundfont en {sf2}")
        return 1

    salida.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            fluidsynth,
            "-F", str(salida),
            "-r", "44100",
            "-g", "0.8",
            "-q",
            str(sf2),
            str(entrada),
        ],
        check=True,
    )

    print(f"renderizado: {salida} ({salida.stat().st_size / 1024 / 1024:.1f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

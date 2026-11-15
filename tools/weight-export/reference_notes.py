#!/usr/bin/env python
"""Oraculo de la deteccion de notas: el upstream, sin tocar.

Usa `basic_pitch.inference.predict` (que a su vez usa
`note_creation.output_to_notes_polyphonic`) sobre un audio real, y guarda
las notas detectadas.

Es el comparador de la Fase 9 (posprocesamiento musical): si el port en
Rust detecta las mismas notas, con los mismos tiempos, la portada es
correcta.

Salida: `reference/notes/<nombre>_notes.json` y `.npy`
"""

import json
import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "basic-pitch"))

from basic_pitch.inference import predict  # noqa: E402

OUT_DIR = REPO_ROOT / "reference" / "notes"


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("audio", nargs="+", help="archivos de audio")
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    indice = {}
    for ruta in args.audio:
        ruta = pathlib.Path(ruta)
        nombre = ruta.stem
        print(f"[1/2] {nombre}: prediciendo con el upstream...")

        # `predict` devuelve (salida del modelo, midi, eventos de nota).
        _, midi, note_events = predict(str(ruta))

        # Los eventos son (inicio, fin, pitch, amplitud, pitch_bends).
        notas = np.array(
            [(e[0], e[1], e[2], e[3]) for e in note_events], dtype=np.float32
        ).reshape(-1, 4)
        np.save(out_dir / f"{nombre}_notes.npy", notas)

        # Los PITCH BENDS son una secuencia de enteros por nota (en ticks
        # de MIDI), y no caben en una matriz, asi que van en un JSON aparte
        # con la lista completa de notas.
        completas = [
            {
                "start_time": float(e[0]),
                "end_time": float(e[1]),
                "pitch": int(e[2]),
                "amplitude": float(e[3]),
                "pitch_bend": (
                    [int(v) for v in e[4]] if e[4] is not None else None
                ),
            }
            for e in note_events
        ]
        (out_dir / f"{nombre}_notas.json").write_text(
            json.dumps(completas, indent=1) + "\n", encoding="utf-8"
        )

        midi.write(str(out_dir / f"{nombre}_upstream.mid"))

        indice[nombre] = {
            "audio": str(ruta),
            "n_notas": int(len(note_events)),
            "notas_npy": f"{nombre}_notes.npy",
            "notas_json": f"{nombre}_notas.json",
            "midi_upstream": f"{nombre}_upstream.mid",
            "columnas": ["start_time", "end_time", "pitch", "amplitude"],
            "primera": [float(x) for x in notas[0]] if len(notas) else None,
            "ultima": [float(x) for x in notas[-1]] if len(notas) else None,
        }
        print(f"      {len(note_events)} notas -> {nombre}_notes.npy")

    print("[2/2] Escribiendo el indice...")
    (out_dir / "index.json").write_text(
        json.dumps(
            {
                "descripcion": (
                    "Notas detectadas por el upstream (basic_pitch.inference.predict) "
                    "sobre audio real. Oraculo de la Fase 9."
                ),
                "nota": (
                    "Las notas vienen del codigo del upstream sin modificar, no de una "
                    "reimplementacion."
                ),
                "puntos": indice,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

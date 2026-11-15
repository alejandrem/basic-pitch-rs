#!/usr/bin/env python
"""Los oraculos de los 13 casos de la Fase 7.

Para cada caso de `reference/casos/`, genera:

- los `.npy` con las tres salidas finales del modelo (note, onset, contour),
  calculadas por el UPSTREAM (`basic_pitch.inference.run_inference`),
- las notas detectadas por el upstream, en JSON.

No se guardan los intermedios (CQT, NormalizedLog, ...): para trece casos
serian unos 100 MB. Los intermedios ya estan validados en un caso, con las
doce etapas completas (`reference/reference_outputs/`).

Salida: `reference/casos/<caso>_{note,onset,contour}.npy` y
`reference/casos/<caso>_notas.json`
"""

import json
import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "basic-pitch"))

import basic_pitch.note_creation as infer  # noqa: E402
from basic_pitch import ICASSP_2022_MODEL_PATH  # noqa: E402
from basic_pitch.constants import AUDIO_SAMPLE_RATE, FFT_HOP  # noqa: E402
from basic_pitch.inference import (  # noqa: E402
    DEFAULT_FRAME_THRESHOLD,
    DEFAULT_MINIMUM_NOTE_LENGTH_MS,
    DEFAULT_ONSET_THRESHOLD,
    Model,
    run_inference,
)

CASOS_DIR = REPO_ROOT / "reference" / "casos"


def main() -> int:
    indice = {}
    print("Cargando el modelo oficial...")
    model = Model(ICASSP_2022_MODEL_PATH)
    casos = sorted(p for p in CASOS_DIR.glob("*.wav"))
    if not casos:
        print(f"no hay casos en {CASOS_DIR}; ejecuta antes make_cases.py")
        return 1

    for i, audio in enumerate(casos, 1):
        nombre = audio.stem
        print(f"[{i}/{len(casos)}] {nombre}: prediciendo con el upstream...")

        # `run_inference` es la funcion del upstream: carga el modelo,
        # ventanea, infiere y reconstruye.
        salida = run_inference(str(audio), model)

        for etapa in ("note", "onset", "contour"):
            matriz = np.asarray(salida[etapa], dtype=np.float32)
            np.save(CASOS_DIR / f"{nombre}_{etapa}.npy", matriz)

        # Las notas detectadas, con los mismos parametros que usa el
        # upstream por defecto.
        min_note_len = int(
            np.round(DEFAULT_MINIMUM_NOTE_LENGTH_MS / 1000 * (AUDIO_SAMPLE_RATE / FFT_HOP))
        )
        _, eventos = infer.model_output_to_notes(
            salida,
            onset_thresh=DEFAULT_ONSET_THRESHOLD,
            frame_thresh=DEFAULT_FRAME_THRESHOLD,
            min_note_len=min_note_len,
            melodia_trick=True,
        )
        notas = [
            {
                "start_time": float(e[0]),
                "end_time": float(e[1]),
                "pitch": int(e[2]),
                "amplitude": float(e[3]),
                "pitch_bend": (
                    [int(v) for v in e[4]] if e[4] is not None else None
                ),
            }
            for e in eventos
        ]
        (CASOS_DIR / f"{nombre}_notas.json").write_text(
            json.dumps(notas, indent=1) + "\n", encoding="utf-8"
        )
        print(f"      {len(notas)} notas detectadas")

        indice[nombre] = {
            "note": f"{nombre}_note.npy",
            "onset": f"{nombre}_onset.npy",
            "contour": f"{nombre}_contour.npy",
            "shape_note": list(np.asarray(salida["note"]).shape),
            "shape_onset": list(np.asarray(salida["onset"]).shape),
            "shape_contour": list(np.asarray(salida["contour"]).shape),
            "notas": f"{nombre}_notas.json",
            "n_notas": len(notas),
        }

    print("Escribiendo el indice...")
    (CASOS_DIR / "index_salidas.json").write_text(
        json.dumps(
            {
                "descripcion": (
                    "Salidas finales del modelo, calculadas por el upstream "
                    "(basic_pitch.inference.run_inference) sobre los 13 casos."
                ),
                "nota": "Las notas detectadas estan en <caso>_notas.json.",
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
    sys.exit(main())

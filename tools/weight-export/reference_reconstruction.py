#!/usr/bin/env python
"""Oraculo de la reconstruccion temporal: el `unwrap_output` DEL UPSTREAM.

Este script NO reimplementa la reconstruccion: usa la funcion
`basic_pitch.inference.unwrap_output` tal cual, sobre audio real, y guarda
las tres matrices resultantes.

Es el comparador de la Fase 8 (inferencia sobre audios de duracion
arbitraria): si la reconstruccion de Rust se comporta como la del upstream,
las matrices tienen que coincidir.

Salida: `reference/reconstruction/<nombre>_{note,onset,contour}.npy`
"""

import json
import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "basic-pitch"))

from basic_pitch.inference import (  # noqa: E402
    DEFAULT_OVERLAPPING_FRAMES,
    Model,
    get_audio_input,
    unwrap_output,
)
from basic_pitch import ICASSP_2022_MODEL_PATH  # noqa: E402
from basic_pitch.constants import AUDIO_N_SAMPLES  # noqa: E402

OUT_DIR = REPO_ROOT / "reference" / "reconstruction"
RESOURCES = REPO_ROOT / "basic-pitch" / "tests" / "resources"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/3] Cargando el modelo oficial...")
    model = Model(ICASSP_2022_MODEL_PATH)

    n_overlapping_frames = DEFAULT_OVERLAPPING_FRAMES
    overlap_len = n_overlapping_frames * 256
    hop_size = AUDIO_N_SAMPLES - overlap_len

    indice = {}
    for nombre in ["vocadito_10", "vocadito_14"]:
        ruta = RESOURCES / f"{nombre}.wav"
        print(f"[2/3] {nombre}: ventilando e infiriendo...")

        # El bucle es el del upstream (`run_inference`).
        salidas = {"note": [], "onset": [], "contour": []}
        audio_original_length = 0
        for audio_windowed, _, audio_original_length in get_audio_input(
            ruta, overlap_len, hop_size
        ):
            for k, v in model.predict(audio_windowed).items():
                salidas[k].append(v)

        # Y la reconstruccion es la del upstream, sin tocar.
        reconstruidas = {
            k: unwrap_output(
                np.concatenate(salidas[k]),
                audio_original_length,
                n_overlapping_frames,
                hop_size,
            )
            for k in salidas
        }

        for k, matriz in reconstruidas.items():
            archivo = f"{nombre}_{k}.npy"
            np.save(OUT_DIR / archivo, matriz.astype(np.float32))
            indice[archivo] = {
                "audio": nombre,
                "etapa": k,
                "shape": list(matriz.shape),
                "dtype": "float32",
                "min": float(matriz.min()),
                "max": float(matriz.max()),
                "mean": float(matriz.mean()),
                "audio_original_length": int(audio_original_length),
                "n_ventanas": len(salidas["note"]),
            }
            print(
                f"      {k:8s} {str(list(matriz.shape)):16s} -> {archivo}"
            )

    print("[3/3] Escribiendo el indice...")
    (OUT_DIR / "index.json").write_text(
        json.dumps(
            {
                "descripcion": (
                    "Reconstruccion temporal del upstream (basic_pitch.inference."
                    "unwrap_output) sobre audio real. Oraculo de la Fase 8."
                ),
                "nota": (
                    "Las matrices vienen de la funcion del upstream sin modificar, "
                    "no de una reimplementacion."
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

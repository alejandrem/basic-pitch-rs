#!/usr/bin/env python
"""Oraculo de audio real: el frontend completo de TensorFlow sobre audio de verdad.

Genera, para una ventana de un archivo de audio real, las mismas etapas
que `reference/reference_outputs/` pero sobre el audio **decodificado** por
la referencia (librosa + soundfile + soxr_hq), y con el **ventaneo** del
upstream (`basic_pitch/inference.py`).

Esto cierra el hueco que dejaba el oraculo sintetico: aquel audio ya
venia decodificado y remuestreado, asi que no permitia validar la
decodificacion ni el remuestreo de Rust.

Salida: `reference/audio_reference/<nombre>_ventana<k>_<etapa>.npy`
"""

import json
import pathlib
import sys

import librosa
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from inspect_model import build_keras_model, layer_by_name, load_saved_model, map_saved_model_variables  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "reference" / "audio_reference"
RESOURCES = REPO_ROOT / "basic-pitch" / "tests" / "resources"

AUDIO_SAMPLE_RATE = 22050
AUDIO_N_SAMPLES = 43844
FFT_HOP = 256
OVERLAP_LEN = 7680
HOP_SIZE = AUDIO_N_SAMPLES - OVERLAP_LEN


def get_audio_input(audio_path, overlap_len, hop_size):
    """El ventaneo del upstream, literalmente (basic_pitch/inference.py)."""
    audio_original, _ = librosa.load(str(audio_path), sr=AUDIO_SAMPLE_RATE, mono=True)
    original_length = audio_original.shape[0]
    audio_original = np.concatenate(
        [np.zeros((int(overlap_len / 2),), dtype=np.float32), audio_original]
    )
    for i in range(0, audio_original.shape[0], hop_size):
        window = audio_original[i : i + AUDIO_N_SAMPLES]
        if len(window) < AUDIO_N_SAMPLES:
            window = np.pad(window, pad_width=[[0, AUDIO_N_SAMPLES - len(window)]])
        yield np.expand_dims(window, axis=0), float(i) / AUDIO_SAMPLE_RATE, original_length


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/3] Cargando el SavedModel...")
    _, raw_variables = load_saved_model()
    variables, _ = map_saved_model_variables(raw_variables)
    keras_model = build_keras_model(variables)
    print(f"      {len(keras_model.layers)} capas.")

    etapas = {
        "cqt": "cqt",
        "normalized_log": "normalized_log",
        "batch_norm": "batch_norm.input",
        "harmonic_stacking": "harmonic_stacking",
    }
    subs = {
        clave: tf.keras.Model(inputs=keras_model.inputs, outputs=layer_by_name(keras_model, capa).output)
        for clave, capa in etapas.items()
    }

    print("[2/3] Calculando el frontend sobre audio real...")
    indice = {}
    for nombre in ["vocadito_10", "vocadito_14"]:
        ruta = RESOURCES / f"{nombre}.wav"
        ventanas = list(get_audio_input(ruta, OVERLAP_LEN, HOP_SIZE))
        print(f"      {nombre}: {len(ventanas)} ventanas, {ventanas[0][2]} muestras originales")

        # Se guardan la primera ventana (con relleno inicial) y la segunda
        # (ya dentro del audio), para cubrir los dos casos. Del segundo
        # archivo basta la primera ventana.
        for indice_ventana in [0, 1] if nombre == "vocadito_10" else [0]:
            if indice_ventana >= len(ventanas):
                continue
            ventana, t_inicio, _ = ventanas[indice_ventana]
            for clave, sub in subs.items():
                salida = np.asarray(sub.predict(ventana, verbose=0), dtype=np.float32)
                archivo = f"{nombre}_v{indice_ventana}_{clave}.npy"
                np.save(OUT_DIR / archivo, salida)
                indice[archivo] = {
                    "audio": nombre,
                    "ventana": indice_ventana,
                    "t_inicio_s": t_inicio,
                    "etapa": clave,
                    "shape": list(salida.shape),
                    "dtype": "float32",
                    "min": float(salida.min()),
                    "max": float(salida.max()),
                }
                print(f"        v{indice_ventana} {clave:18s} {str(list(salida.shape)):18s} -> {archivo}")

    print("[3/3] Escribiendo el indice...")
    (OUT_DIR / "index_frontend.json").write_text(
        json.dumps(
            {
                "descripcion": (
                    "Frontend completo de TensorFlow sobre audio real, con el ventaneo "
                    "del upstream. Oraculo del test de extremo a extremo."
                ),
                "cadena": (
                    "librosa.load(sr=22050, mono=True) -> relleno inicial de 3840 ceros -> "
                    "ventanas de 43844 con salto 36164 -> modelo"
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

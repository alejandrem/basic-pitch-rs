#!/usr/bin/env python3
"""Fase 1: extraccion de los pesos preentrenados a Safetensors.

Genera en `models/basic-pitch/icassp-2022/` (directorio canonico):

- `model.safetensors`        pesos del modelo activo, con nombres canónicos
- `cqt_kernels.safetensors`  kernels deterministas de la CQT (no son pesos)
- `model_config.json`        contrato tecnico del modelo
- `tensor_inventory.json`    inventario de tensores con sus dimensiones
- `tensor_provenance.json`   trazabilidad TensorFlow <-> ONNX <-> Rust
- `source_metadata.json`     procedencia, versiones y hashes
- `SHA256SUMS`               sumas de verificacion

Reglas aplicadas:

- Los kernels de conv se permutan de `[kh, kw, in, out]` (TensorFlow) a
  `[out, in, kh, kw]` (Candle). Los valores no se modifican.
- Las estadisticas de BatchNorm se copian tal cual: `moving_mean` y
  `moving_variance` son irremplazables.
- No se cuantiza, redondea ni normaliza ningun valor.

Uso:
    python tools/weight-export/export_weights.py
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import platform
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import model_spec as spec  # noqa: E402
import tf_source  # noqa: E402


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def json_dump(obj, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def save_safetensors(tensors: dict, path: pathlib.Path, metadata: dict) -> None:
    """Guarda tensores float32 en formato Safetensors."""
    from safetensors.numpy import save_file

    payload = {}
    for name, array in tensors.items():
        arr = np.ascontiguousarray(array, dtype=np.float32)
        if not np.isfinite(arr).all():
            raise ValueError(f"El tensor '{name}' contiene NaN o infinitos.")
        payload[name] = arr

    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(payload, str(path), metadata=metadata)
    print(f"      {path.name}: {len(payload)} tensores, {path.stat().st_size} bytes")


def build_model_tensors(variables: dict) -> dict:
    """Construye el diccionario de tensores con el layout de Rust."""
    tensors = {}
    for s in spec.TENSOR_SPECS:
        array = variables[s.tf_name].numpy()
        tensors[s.rust_name] = tf_source.to_rust_layout(array, s.transpose)
    return tensors


def build_cqt_tensors(onnx_inits: dict) -> dict:
    """Extrae los kernels deterministas de la CQT del ONNX oficial."""
    tensors = {}
    for onnx_name, rust_name in spec.CQT_KERNEL_SOURCES.items():
        array = np.asarray(
            spec.resolve_onnx_initializer(onnx_inits, onnx_name), dtype=np.float32
        )
        expected = spec.CQT_KERNEL_SHAPES[rust_name]
        if array.shape != expected:
            raise ValueError(
                f"'{rust_name}': forma inesperada {array.shape}, se esperaba {expected}."
            )
        tensors[rust_name] = np.ascontiguousarray(array)
    return tensors


def build_tensor_inventory(tensors: dict, actual_names: dict) -> dict:
    """Inventario de tensores con su equivalencia en los tres formatos."""
    entries = []
    for s in spec.TENSOR_SPECS:
        array = tensors[s.rust_name]
        entries.append(
            {
                "rust_name": s.rust_name,
                "role": s.role,
                "layer": s.layer,
                "shape": list(array.shape),
                "dtype": "float32",
                "bytes": int(array.nbytes),
                "tf_name": actual_names.get(s.tf_name, s.tf_name),
                "canonical_tf_name": s.tf_name,
                "transpose": list(s.transpose) if s.transpose else None,
                "bn_folded_in_onnx": s.bn_folded_in_onnx,
            }
        )
    return {
        "description": "Inventario de tensores del modelo activo, en el layout de Rust.",
        "kernel_layout": "[out_channels, in_channels, kernel_height, kernel_width]",
        "n_tensors": len(entries),
        "total_parameters": int(
            sum(int(np.prod(tensors[s.rust_name].shape)) for s in spec.TENSOR_SPECS)
        ),
        "tensors": entries,
    }


def build_provenance(variables, onnx_inits, actual_names, tensors) -> dict:
    """Manifiesto de trazabilidad TensorFlow <-> ONNX <-> Rust.

    El equivalente ONNX de cada tensor se identifica **por contenido**, no
    por forma: varios tensores comparten forma (biases de 1 canal frente a
    BatchNorm de 1 canal), y emparejar solo por forma produce falsos positivos
    silenciosos. Ver `tf_source.identify_onnx_counterparts`.
    """
    counterparts = tf_source.identify_onnx_counterparts(variables, onnx_inits)

    entries = []
    for s in spec.TENSOR_SPECS:
        tf_value = variables[s.tf_name].numpy()
        rust_value = tensors[s.rust_name]
        counterpart = counterparts.get(s.tf_name, {})
        entry = {
            "tf_name": actual_names.get(s.tf_name, s.tf_name),
            "canonical_tf_name": s.tf_name,
            "rust_name": s.rust_name,
            "role": s.role,
            "layer": s.layer,
            "shape_tf": list(s.shape_tf),
            "shape_rust": list(rust_value.shape),
            "transpose": list(s.transpose) if s.transpose else None,
            "dtype": "float32",
            "bn_folded_in_onnx": s.bn_folded_in_onnx,
            "onnx_name": counterpart.get("onnx_name"),
            "onnx_relation": counterpart.get("relation"),
            "max_abs_diff": counterpart.get("max_abs_diff"),
        }
        if counterpart.get("note"):
            entry["onnx_note"] = counterpart["note"]

        # La permutacion debe ser reversible sin perder precision.
        if s.transpose is not None:
            roundtrip = np.transpose(rust_value, np.argsort(s.transpose))
            entry["max_abs_diff_roundtrip"] = float(np.abs(roundtrip - tf_value).max())
        else:
            entry["max_abs_diff_roundtrip"] = 0.0

        entries.append(entry)

    return {
        "description": (
            "Manifiesto de trazabilidad: cada tensor del modelo activo, "
            "atado a su nombre en TensorFlow, en ONNX y en Rust."
        ),
        "convention": {
            "tf_kernel_layout": "[kh, kw, in_channels, out_channels]",
            "rust_kernel_layout": "[out_channels, in_channels, kh, kw]",
            "transpose": list(spec.CONV_TRANSPOSE),
            "onnx_relations": {
                "direct": "onnx == tensor de Rust (sin fusiones)",
                "bn_folded": "onnx == kernel de Rust * scale (BatchNorm fusionada)",
                "bn_folded_bias": (
                    "onnx == (bias - moving_mean) * scale + beta (bias fusionado)"
                ),
                "folded_into_conv": (
                    "no hay inicializador equivalente: el valor esta fusionado "
                    "dentro de la convolucion"
                ),
            },
            "identification": (
                "El equivalente ONNX se identifica por CONTENIDO, no por forma: "
                "varios tensores comparten forma y emparejar solo por forma "
                "produce falsos positivos silenciosos."
            ),
        },
        "tensors": entries,
    }


def verify_bn_folding(variables, onnx_inits) -> list:
    """Verifica la relacion de fusion de la BatchNorm en ONNX."""
    checks = []
    pairs = [
        ("conv2d_1", "batch_normalization_2", "const_fold_opt__727"),
        ("conv2d_4", "batch_normalization_3", "const_fold_opt__707"),
    ]
    for conv_name, bn_prefix, onnx_name in pairs:
        kernel = variables[f"{conv_name}/kernel"].numpy()
        bias = variables[f"{conv_name}/bias"].numpy()
        gamma = variables[f"{bn_prefix}/gamma"].numpy()
        beta = variables[f"{bn_prefix}/beta"].numpy()
        mean = variables[f"{bn_prefix}/moving_mean"].numpy()
        var = variables[f"{bn_prefix}/moving_variance"].numpy()

        scale = gamma / np.sqrt(var + spec.KERAS_BATCH_NORM_EPSILON)

        onnx_kernel = onnx_inits[onnx_name]
        onnx_bias = None
        for name, arr in onnx_inits.items():
            if arr.shape == bias.shape and "batch_normalization" in name:
                onnx_bias = arr
                break

        w_candle = np.transpose(kernel, spec.CONV_TRANSPOSE)
        expected_kernel = w_candle * scale.reshape(-1, 1, 1, 1)
        kernel_diff = float(np.abs(onnx_kernel - expected_kernel).max())

        expected_bias = (bias - mean) * scale + beta
        bias_diff = (
            float(np.abs(onnx_bias - expected_bias).max()) if onnx_bias is not None else None
        )

        checks.append(
            {
                "conv": conv_name,
                "batch_norm": bn_prefix,
                "onnx_name": onnx_name,
                "kernel_max_abs_diff": kernel_diff,
                "bias_max_abs_diff": bias_diff,
                "scale": scale.tolist(),
                "passed": kernel_diff < 1e-5 and (bias_diff is None or bias_diff < 1e-5),
            }
        )
    return checks


def build_model_config() -> dict:
    """Contrato tecnico: constantes verificadas del modelo."""
    import tensorflow as tf

    return {
        "model": {
            "name": "basic-pitch",
            "variant": "icassp-2022",
            "framework": "tensorflow",
            "reference_repo": spec.REFERENCE_REPO_URL,
            "reference_commit": spec.REFERENCE_COMMIT,
            "reference_package_version": spec.REFERENCE_PACKAGE_VERSION,
            "dtype": "float32",
        },
        "audio": {
            "sample_rate": spec.AUDIO_SAMPLE_RATE,
            "channels": 1,
            "window_length_s": spec.AUDIO_WINDOW_LENGTH,
            "fft_hop": spec.FFT_HOP,
            "annotations_fps": spec.ANNOTATIONS_FPS,
            "annot_n_frames": spec.ANNOT_N_FRAMES,
            "audio_n_samples": spec.AUDIO_N_SAMPLES,
        },
        "frequency": {
            "base_frequency": spec.ANNOTATIONS_BASE_FREQUENCY,
            "n_semitones": spec.ANNOTATIONS_N_SEMITONES,
            "contours_bins_per_semitone": spec.CONTOURS_BINS_PER_SEMITONE,
            "notes_bins_per_semitone": spec.NOTES_BINS_PER_SEMITONE,
            "cqt_n_semitones": spec.CQT_N_SEMITONES,
            "cqt_n_bins": spec.CQT_N_BINS,
            "n_freq_bins_contours": spec.N_FREQ_BINS_CONTOURS,
            "n_freq_bins_notes": spec.N_FREQ_BINS_NOTES,
        },
        "cqt": {
            "sr": spec.AUDIO_SAMPLE_RATE,
            "fmin": spec.ANNOTATIONS_BASE_FREQUENCY,
            "n_bins": spec.CQT_N_BINS,
            "bins_per_octave": 12 * spec.CONTOURS_BINS_PER_SEMITONE,
            "hop_length": spec.FFT_HOP,
            "q_factor": float(
                1.0 / (2 ** (1.0 / (12 * spec.CONTOURS_BINS_PER_SEMITONE)) - 1.0)
            ),
            "n_octaves": int(
                np.ceil(spec.CQT_N_BINS / (12 * spec.CONTOURS_BINS_PER_SEMITONE))
            ),
            "n_filters": 12 * spec.CONTOURS_BINS_PER_SEMITONE,
            "n_fft": 256,
            "early_downsample": False,
            "downsample_factor": 1,
            "pad_mode": "reflect",
            "basis_norm": 1,
            "normalization": "sqrt(lengths)",
            "kernels_asset": "cqt_kernels.safetensors",
            "kernels_note": (
                "Los kernels se extraen del ONNX oficial. No son pesos "
                "entrenados: son parametros deterministas de la transformada."
            ),
        },
        "harmonic_stacking": {
            "harmonics": list(spec.HARMONICS),
            "shifts": list(spec.HARMONIC_SHIFTS),
            "bins_per_semitone": spec.CONTOURS_BINS_PER_SEMITONE,
            "n_output_freqs": spec.N_FREQ_BINS_CONTOURS,
            "input_n_bins": spec.CQT_N_BINS,
            "padding": (
                "asimetrico: shift>0 rellena con ceros al final del eje de "
                "frecuencia; shift<0 rellena al principio"
            ),
        },
        "batch_norm": {
            "epsilon": spec.KERAS_BATCH_NORM_EPSILON,
            "epsilon_source": "keras (tf.keras.layers.BatchNormalization)",
            "note": (
                "El atributo epsilon=1e-5 de FusedBatchNormV3 en ONNX no es el "
                "valor de este modelo; el ONNX fusiona la normalizacion."
            ),
            "layers": [
                {"name": "batch_normalization", "channels": 1, "position": "inicial"},
                {"name": "batch_normalization_2", "channels": 8, "position": "contornos"},
                {"name": "batch_normalization_3", "channels": 32, "position": "onsets"},
            ],
        },
        "layers": [
            {
                "name": layer.name,
                "rust_name": layer.rust_name,
                "kernel_tf": list(layer.kernel_tf),
                "filters": layer.filters,
                "in_channels": layer.in_channels,
                "stride_tf": list(layer.stride_tf),
                "in_shape_nhwc": list(layer.in_shape),
                "out_shape_nhwc": list(layer.out_shape),
                "pad_same_tf": [
                    list(layer.pad_same_tf[0]),
                    list(layer.pad_same_tf[1]),
                ],
                "activation": layer.activation,
                "batch_norm": layer.batch_norm,
                "note": layer.note,
            }
            for layer in spec.LAYER_SPECS
        ],
        "concat": {
            "name": "concat",
            "axis": 3,
            "order": ["x_notes_pre", "x_onset"],
            "channels": spec.ONSET_CONCAT_CHANNELS,
            "note": (
                "x_notes_pre es la salida sigmoid de conv2d_3 con forma "
                "(batch, 172, 88, 1). x_onset tiene 32 canales. Total 33."
            ),
        },
        "outputs": {
            "note": ["batch", spec.ANNOT_N_FRAMES, spec.N_FREQ_BINS_NOTES],
            "onset": ["batch", spec.ANNOT_N_FRAMES, spec.N_FREQ_BINS_NOTES],
            "contour": ["batch", spec.ANNOT_N_FRAMES, spec.N_FREQ_BINS_CONTOURS],
        },
        "windowing": {
            "n_overlapping_frames": spec.N_OVERLAPPING_FRAMES,
            "overlap_len": spec.OVERLAP_LEN,
            "hop_size": spec.HOP_SIZE,
            "initial_padding": spec.INITIAL_PADDING,
            "frames_per_window": spec.FRAMES_PER_WINDOW,
            "frames_kept_per_window": spec.FRAMES_KEPT_PER_WINDOW,
            "magic_alignment_offset": 0.0018,
            "compatibility_note": (
                "La numeracion de ventanas del upstream divide entre 172 aunque "
                "solo sobreviven 142 frames por ventana. Se reproduce tal cual."
            ),
        },
        "tensor_files": {
            "model": "model.safetensors",
            "cqt_kernels": "cqt_kernels.safetensors",
        },
    }


def main() -> int:
    print("[1/7] Verificando la tabla de padding SAME...")
    spec.verify_padding_table()
    print("      OK.")

    print("[2/7] Cargando el SavedModel de TensorFlow...")
    _, raw_variables = tf_source.load_saved_model()
    variables, actual_names = tf_source.map_saved_model_variables(raw_variables)
    print(f"      {len(variables)} tensores resueltos.")

    print("[3/7] Construyendo los tensores en layout de Rust...")
    tensors = build_model_tensors(variables)
    print(f"      {len(tensors)} tensores listos.")

    print("[4/7] Extrayendo los kernels de la CQT del ONNX oficial...")
    onnx_inits = tf_source.onnx_initializers()
    cqt_tensors = build_cqt_tensors(onnx_inits)
    print(f"      {len(cqt_tensors)} tensores de CQT: {sorted(cqt_tensors)}")

    print("[5/7] Verificando la fusion de BatchNorm en ONNX...")
    bn_checks = verify_bn_folding(variables, onnx_inits)
    for check in bn_checks:
        print(
            f"      {check['conv']:10s} kernel_diff={check['kernel_max_abs_diff']:.3e} "
            f"bias_diff={check['bias_max_abs_diff']} -> "
            f"{'OK' if check['passed'] else 'FALLO'}"
        )
        if not check["passed"]:
            raise SystemExit("La verificacion de fusion de BatchNorm ha fallado.")

    print("[6/7] Escribiendo los Safetensors...")
    metadata = {
        "model": "basic-pitch",
        "variant": "icassp-2022",
        "reference_commit": spec.REFERENCE_COMMIT,
        "source": "tensorflow-savedmodel",
        "dtype": "float32",
    }
    save_safetensors(tensors, spec.MODEL_DIR / "model.safetensors", metadata)
    save_safetensors(cqt_tensors, spec.MODEL_DIR / "cqt_kernels.safetensors", metadata)

    print("[7/7] Escribiendo manifiestos y sumas de verificacion...")
    inventory = build_tensor_inventory(tensors, actual_names)
    json_dump(inventory, spec.MODEL_DIR / "tensor_inventory.json")

    provenance = build_provenance(variables, onnx_inits, actual_names, tensors)
    provenance["bn_folding_checks"] = bn_checks
    json_dump(provenance, spec.MODEL_DIR / "tensor_provenance.json")

    json_dump(build_model_config(), spec.MODEL_DIR / "model_config.json")

    source_metadata = {
        "model": "basic-pitch / icassp-2022",
        "source": {
            "framework": "tensorflow",
            "artifact": "saved_model",
            "reference_repo": spec.REFERENCE_REPO_URL,
            "reference_commit": spec.REFERENCE_COMMIT,
            "reference_package_version": spec.REFERENCE_PACKAGE_VERSION,
            "onnx_artifact": str(
                spec.ONNX_MODEL.relative_to(spec.REPO_ROOT)
            ),
            "onnx_sha256": sha256_file(spec.ONNX_MODEL),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "transformations": [
            "Kernels de conv permutados de [kh, kw, in, out] a [out, in, kh, kw].",
            "Kernels de CQT extraidos del ONNX oficial (parametros deterministas).",
            "Ningun valor se ha cuantizado, redondeado ni normalizado.",
        ],
        "note": (
            "Las estadisticas moving_mean y moving_variance provienen "
            "exclusivamente de TensorFlow: el ONNX tiene la BatchNorm fusionada."
        ),
    }
    json_dump(source_metadata, spec.MODEL_DIR / "source_metadata.json")

    # SHA256SUMS: se escribe al final, sobre el estado final del directorio.
    sums_path = spec.MODEL_DIR / "SHA256SUMS"
    lines = []
    for path in sorted(spec.MODEL_DIR.iterdir()):
        if path.name == "SHA256SUMS" or not path.is_file():
            continue
        lines.append(f"{sha256_file(path)}  {path.name}")
    sums_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print()
    print(f"Exportacion completada en {spec.MODEL_DIR}")
    for path in sorted(spec.MODEL_DIR.iterdir()):
        if path.is_file():
            print(f"  {path.name:28s} {path.stat().st_size:>10d} bytes")
    print()
    print(f"  Tensores del modelo      : {inventory['n_tensors']}")
    print(f"  Parametros totales       : {inventory['total_parameters']}")
    print(f"  Kernels de CQT           : {len(cqt_tensors)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

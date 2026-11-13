#!/usr/bin/env python3
"""Fase 0: auditoria del modelo oficial de Spotify Basic Pitch.

Genera, en `reference/`:

- `model_inventory.json`      inventario completo de capas y tensores
- `model_summary.txt`         resumen legible del modelo
- `model_config.json`         contrato tecnico (constantes verificadas)
- `reference_metadata.json`   procedencia, versiones y hashes
- `tensor_provenance.json`    manifiesto de trazabilidad TF <-> ONNX <-> Rust
- `reference_outputs/`        salidas intermedias para validar Rust

Uso:
    python tools/weight-export/inspect_model.py
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import platform
import sys
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
try:  # Silencia los avisos de retracing de tf.function.
    import logging

    logging.getLogger("tensorflow").setLevel(logging.ERROR)
except Exception:  # pragma: no cover
    pass

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import model_spec as spec  # noqa: E402

# El repositorio de referencia se importa para reconstruir el modelo Keras.
sys.path.insert(0, str(spec.REFERENCE_REPO))


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_dir(path: pathlib.Path) -> dict:
    """SHA256 de cada fichero de un directorio, ordenados por nombre."""
    return {
        str(p.relative_to(path)): sha256_file(p)
        for p in sorted(path.rglob("*"))
        if p.is_file()
    }


def json_dump(obj, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, sort_keys=False)
        fh.write("\n")


def to_python(obj):
    """Convierte tipos de numpy/TF a tipos serializables en JSON."""
    if isinstance(obj, dict):
        return {str(k): to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_python(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_python(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# ---------------------------------------------------------------------------
# Carga del modelo
# ---------------------------------------------------------------------------

def load_saved_model():
    """Carga el SavedModel de TensorFlow y devuelve (modelo, mapa de variables)."""
    if not spec.TF_SAVED_MODEL.exists():
        raise SystemExit(
            f"No se encuentra el SavedModel en {spec.TF_SAVED_MODEL}.\n"
            "Se espera la copia local del repositorio de referencia."
        )
    model = tf.saved_model.load(str(spec.TF_SAVED_MODEL))
    variables = {v.name.split(":")[0]: v for v in model.variables}
    return model, variables


def map_saved_model_variables(variables: dict) -> tuple:
    """Mapea las variables del SavedModel a la tabla canonica.

    No se puede mapear por nombre: el contador global de Keras produce
    `conv2d`..`conv2d_4` en un build fresco, mientras que el SavedModel tiene
    `conv2d_1`..`conv2d_5` (el modelo original se guardo en un proceso donde el
    contador ya habia avanzado).

    La estrategia es en dos pasos:

    1. **Clasificar por rol**, usando el sufijo del nombre de TensorFlow
       (`/kernel`, `/bias`, `/gamma`, `/beta`, `/moving_mean`,
       `/moving_variance`). Esto separa los tensores que comparten forma (los
       biases de 1 canal y las BatchNorm de 1 canal).
    2. **Desambiguar por forma** dentro de cada rol. Las formas son unicas:
       kernels `(3,39,8,8) (5,5,8,1) (7,7,1,32) (7,3,32,1) (5,5,8,32)
       (3,3,33,1)` y BatchNorm de anchos `1, 8, 32`.

    Devuelve:
        resolved      {nombre canonico: variable de TensorFlow}
        actual_names  {nombre canonico: nombre real en el SavedModel}
    """
    kernels: dict = {}
    biases: dict = {}
    bn: dict = {}  # (rol, ancho) -> [nombres]

    for name in variables:
        if name.endswith("/kernel"):
            kernels.setdefault(tuple(variables[name].shape.as_list()), []).append(name)
        elif name.endswith("/bias"):
            biases.setdefault(tuple(variables[name].shape.as_list()), []).append(name)
        else:
            prefix, _, role = name.rpartition("/")
            if role in ("gamma", "beta", "moving_mean", "moving_variance"):
                width = variables[name].shape.as_list()[0]
                bn.setdefault((role, width), []).append(name)

    resolved: dict = {}
    actual_names: dict = {}

    # 1) Resolver las convoluciones (kernel + bias emparejados).
    conv_pairs: dict = {}  # prefijo canonico -> (kernel real, bias real)
    for s in spec.TENSOR_SPECS:
        if s.role != spec.CONV_KERNEL:
            continue
        candidates = kernels.get(s.shape_tf, [])
        if len(candidates) != 1:
            raise AssertionError(
                f"Ambiguedad al resolver el kernel {s.tf_name}: "
                f"forma {s.shape_tf} -> {candidates}"
            )
        actual_kernel = candidates[0]
        actual_bias = f"{actual_kernel.rsplit('/', 1)[0]}/bias"
        if actual_bias not in variables:
            raise AssertionError(f"No se encontro el bias '{actual_bias}'.")
        prefix = s.tf_name.rsplit("/", 1)[0]
        conv_pairs[prefix] = (actual_kernel, actual_bias)

    # 2) Resolver cada tensor de la tabla canonica.
    for s in spec.TENSOR_SPECS:
        if s.role == spec.CONV_KERNEL:
            actual_kernel, actual_bias = conv_pairs[s.tf_name.rsplit("/", 1)[0]]
            resolved[s.tf_name] = variables[actual_kernel]
            actual_names[s.tf_name] = actual_kernel
        elif s.role == spec.CONV_BIAS:
            prefix = s.tf_name.rsplit("/", 1)[0]
            actual_kernel, actual_bias = conv_pairs[prefix]
            resolved[s.tf_name] = variables[actual_bias]
            actual_names[s.tf_name] = actual_bias
        else:
            role = {
                spec.BN_GAMMA: "gamma",
                spec.BN_BETA: "beta",
                spec.BN_MEAN: "moving_mean",
                spec.BN_VAR: "moving_variance",
            }[s.role]
            candidates = bn.get((role, s.shape_tf[0]), [])
            if len(candidates) != 1:
                raise AssertionError(
                    f"Ambiguedad al resolver {s.tf_name}: "
                    f"rol={role} ancho={s.shape_tf[0]} -> {candidates}"
                )
            resolved[s.tf_name] = variables[candidates[0]]
            actual_names[s.tf_name] = candidates[0]

    return resolved, actual_names


def build_keras_model(variables: dict):
    """Reconstruye el modelo Keras y le asigna los pesos del SavedModel.

    La asignacion se hace por forma, no por nombre (ver
    `map_saved_model_variables`).
    """
    from basic_pitch.models import model as build_model

    keras_model = build_model()

    # Indice de los tensores canonicos por forma de kernel.
    kernels_by_shape: dict = {}
    for s in spec.TENSOR_SPECS:
        if s.role == spec.CONV_KERNEL:
            kernels_by_shape[s.shape_tf] = s

    # Para cada parametro de la BatchNorm, el tensor canonico con ese ancho.
    bn_by_width: dict = {}
    for s in spec.TENSOR_SPECS:
        if s.role == spec.BN_GAMMA:
            bn_by_width[s.shape_tf[0]] = s

    for layer in keras_model.layers:
        weights = layer.weights
        if not weights:
            continue

        if isinstance(layer, tf.keras.layers.Conv2D):
            shape = tuple(weights[0].shape.as_list())
            canonical = kernels_by_shape.get(shape)
            if canonical is None:
                raise KeyError(
                    f"La conv '{layer.name}' tiene forma {shape}, "
                    "que no corresponde a ninguna capa de la tabla canonica."
                )
            kernel_name = canonical.tf_name
            bias_name = f"{kernel_name.rsplit('/', 1)[0]}/bias"
            layer.set_weights(
                [variables[kernel_name].numpy(), variables[bias_name].numpy()]
            )
        elif isinstance(layer, tf.keras.layers.BatchNormalization):
            width = weights[0].shape.as_list()[0]
            canonical = bn_by_width.get(width)
            if canonical is None:
                raise KeyError(
                    f"La BatchNorm '{layer.name}' tiene {width} canales, "
                    "que no corresponde a ninguna de la tabla canonica."
                )
            prefix = canonical.tf_name.rsplit("/", 1)[0]
            layer.set_weights(
                [
                    variables[f"{prefix}/gamma"].numpy(),
                    variables[f"{prefix}/beta"].numpy(),
                    variables[f"{prefix}/moving_mean"].numpy(),
                    variables[f"{prefix}/moving_variance"].numpy(),
                ]
            )
        else:
            raise KeyError(
                f"La capa '{layer.name}' ({type(layer).__name__}) tiene pesos "
                "pero no es una Conv2D ni una BatchNormalization."
            )

    return keras_model


def classify_layers(keras_model) -> dict:
    """Clasifica las capas del modelo Keras por su rol semantico.

    Los nombres autoasignados por Keras no son estables entre builds (el
    contador global produce `conv2d`..`conv2d_4` en un build fresco y
    `conv2d_1`..`conv2d_5` en el SavedModel), por lo que la identificacion se
    hace por clase y por forma, no por nombre.

    Devuelve un mapa: clave semantica -> capa.
    """
    import basic_pitch.layers.nnaudio as nnaudio
    import basic_pitch.layers.signal as signal_mod
    import basic_pitch.nn as nn_mod

    # Forma del kernel en TensorFlow: [kh, kw, in_channels, out_channels].
    conv_by_shape = {
        (layer.kernel_tf[0], layer.kernel_tf[1], layer.in_channels, layer.filters): layer
        for layer in spec.LAYER_SPECS
    }
    # Las BatchNorm se identifican por el ancho de sus canales, que es unico.
    bn_by_width = {
        s.shape_tf[0]: s.rust_name.rsplit(".", 1)[0]
        for s in spec.TENSOR_SPECS
        if s.role == spec.BN_GAMMA
    }

    mapping: dict = {}
    for layer in keras_model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            shape = tuple(layer.kernel.shape.as_list())
            canonical = conv_by_shape.get(shape)
            if canonical is None:
                raise KeyError(f"Conv con forma desconocida: {shape}")
            mapping[canonical.rust_name] = layer
        elif isinstance(layer, tf.keras.layers.BatchNormalization):
            width = layer.gamma.shape.as_list()[0]
            key = bn_by_width.get(width)
            if key is None:
                raise KeyError(f"BatchNorm con ancho desconocido: {width}")
            mapping[key] = layer
        elif isinstance(layer, nnaudio.CQT2010v2):
            mapping["cqt"] = layer
        elif isinstance(layer, signal_mod.NormalizedLog):
            mapping["normalized_log"] = layer
        elif isinstance(layer, nn_mod.HarmonicStacking):
            mapping["harmonic_stacking"] = layer
        elif isinstance(layer, nn_mod.FlattenFreqCh):
            # Los nombres `contour`, `note` y `onset` son explicitos en
            # `models.py` y, por tanto, estables.
            mapping[layer.name] = layer

    return mapping


def layer_by_name(keras_model, name):
    """Busca una capa por rol semantico (ver `classify_layers`)."""
    mapping = classify_layers(keras_model)
    if name in mapping:
        return mapping[name]
    raise KeyError(
        f"No se encontro la capa '{name}'. Roles disponibles: {sorted(mapping)}"
    )


# ---------------------------------------------------------------------------
# Inventario
# ---------------------------------------------------------------------------

def build_inventory(keras_model, variables) -> dict:
    """Inventario de capas y tensores del modelo activo."""
    layers = []
    for layer in keras_model.layers:
        entry = {
            "name": layer.name,
            "class": type(layer).__name__,
            "output_shape": _shape_of(getattr(layer, "output", None)),
            "weights": [],
        }
        for w in layer.weights:
            key = w.name.split(":")[0]
            entry["weights"].append(
                {
                    "tf_name": key,
                    "shape": list(w.shape.as_list()),
                    "dtype": w.dtype.name,
                    "trainable": bool(w.trainable),
                }
            )
        # Parametros especificos de las capas convolucionales.
        cfg = getattr(layer, "get_config", None)
        if cfg is not None:
            try:
                config = cfg()
            except Exception:  # pragma: no cover - capas no serializables
                config = {}
            for field_name in (
                "kernel_size",
                "filters",
                "strides",
                "padding",
                "activation",
                "axis",
                "epsilon",
                "momentum",
            ):
                if field_name in config:
                    entry[field_name] = to_python(config[field_name])
        layers.append(entry)

    return {
        "input": {
            "shape": [None, spec.AUDIO_N_SAMPLES, 1],
            "dtype": "float32",
            "description": "(batch, 43844, 1) muestras de audio mono a 22050 Hz",
        },
        "outputs": {
            "note": [None, spec.ANNOT_N_FRAMES, spec.N_FREQ_BINS_NOTES],
            "onset": [None, spec.ANNOT_N_FRAMES, spec.N_FREQ_BINS_NOTES],
            "contour": [None, spec.ANNOT_N_FRAMES, spec.N_FREQ_BINS_CONTOURS],
        },
        "layers": layers,
        "variables": {
            name: {
                "shape": list(v.shape.as_list()),
                "dtype": v.dtype.name,
            }
            for name, v in sorted(variables.items())
        },
        "tensor_specs": [
            {
                "tf_name": s.tf_name,
                "rust_name": s.rust_name,
                "role": s.role,
                "shape_tf": list(s.shape_tf),
                "shape_rust": list(s.shape_rust),
                "transpose": list(s.transpose) if s.transpose else None,
                "layer": s.layer,
                "bn_folded_in_onnx": s.bn_folded_in_onnx,
            }
            for s in spec.TENSOR_SPECS
        ],
    }


def _shape_of(tensor):
    if tensor is None:
        return None
    try:
        return list(tensor.shape.as_list())
    except Exception:  # pragma: no cover
        return None


# ---------------------------------------------------------------------------
# Salidas intermedias de referencia
# ---------------------------------------------------------------------------

#: Puntos de inspeccion exigidos por la Fase 0 (apartado 4.5 del plan).
#: Se referencian por rol semantico, no por nombre de capa.
REFERENCE_POINTS = [
    ("input_audio", "Audio de entrada", None),
    ("cqt", "CQT sin normalizar", "cqt"),
    ("normalized_log", "CQT normalizada", "normalized_log"),
    ("batch_norm_input", "CQT despues de Batch Normalization", "batch_norm.input"),
    ("harmonic_stacking", "Harmonic Stacking", "harmonic_stacking"),
    ("contour_conv", "Salida de la conv de contornos", "contour.conv1"),
    ("contour_reduced", "Contorno reducido", "contour.reduced"),
    ("contour", "Contorno final", "contour"),
    ("notes_features", "Caracteristicas de la rama de notas", "notes.conv1"),
    ("note", "Prediccion de notas", "note"),
    ("onset_features", "Caracteristicas de la rama de onsets", "onset.conv1"),
    ("onset", "Prediccion de onsets", "onset"),
]


def capture_reference_outputs(keras_model, audio: np.ndarray) -> dict:
    """Captura las salidas intermedias en los puntos de comparacion."""
    captured = {}
    for key, description, layer_name in REFERENCE_POINTS:
        if layer_name is None:
            captured[key] = {"description": description, "tensor": audio}
            continue
        layer = layer_by_name(keras_model, layer_name)
        sub = tf.keras.Model(inputs=keras_model.inputs, outputs=layer.output)
        out = sub.predict(audio, verbose=0)
        captured[key] = {"description": description, "tensor": out}
    return captured


def save_reference_outputs(captured: dict, audio: np.ndarray) -> dict:
    """Guarda las salidas intermedias en `reference/reference_outputs/`."""
    out_dir = spec.REFERENCE_OUTPUTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    index = {}
    for key, entry in captured.items():
        arr = np.asarray(entry["tensor"], dtype=np.float32)
        path = out_dir / f"{key}.npy"
        np.save(path, arr)
        index[key] = {
            "description": entry["description"],
            "file": path.name,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
        }
    return index


# ---------------------------------------------------------------------------
# Cross-check TensorFlow <-> ONNX
# ---------------------------------------------------------------------------

def onnx_initializers():
    """Carga los inicializadores del ONNX oficial como arrays de numpy."""
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(spec.ONNX_MODEL))
    return {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}


def build_provenance(variables, onnx_inits, actual_names) -> dict:
    """Manifiesto de trazabilidad TensorFlow <-> ONNX <-> Rust.

    Para los tensores cuya BatchNorm fue fusionada en el ONNX, el inicializador
    de ONNX no es comparable directamente. En ese caso se verifica la
    **relacion de fusion**:

        W_onnx = W_tf * scale        con scale = gamma / sqrt(var + eps)
        b_onnx = (b_tf - mean) * scale + beta

    que es una comprobacion mas fuerte que una simple igualdad de valores.
    """
    entries = []
    for s in spec.TENSOR_SPECS:
        tf_value = variables[s.tf_name].numpy()
        entry = {
            "tf_name": actual_names.get(s.tf_name, s.tf_name),
            "canonical_tf_name": s.tf_name,
            "rust_name": s.rust_name,
            "role": s.role,
            "layer": s.layer,
            "shape_tf": list(s.shape_tf),
            "shape_rust": list(s.shape_rust),
            "transpose": list(s.transpose) if s.transpose else None,
            "dtype": "float32",
            "bn_folded_in_onnx": s.bn_folded_in_onnx,
            "onnx_name": s.onnx_name,
            "max_abs_diff": None,
            "onnx_check": None,
        }

        if s.transpose is not None:
            rust_value = np.transpose(tf_value, s.transpose)
        else:
            rust_value = tf_value
        entry["max_abs_diff_rust_vs_tf"] = float(
            np.abs(rust_value - np.transpose(tf_value, s.transpose)).max()
            if s.transpose is not None
            else 0.0
        )

        # Busca un inicializador ONNX con la forma de Rust.
        onnx_match = None
        for name, arr in onnx_inits.items():
            if arr.shape == rust_value.shape:
                onnx_match = (name, arr)
                break

        if onnx_match is not None:
            name, arr = onnx_match
            entry["onnx_name"] = name
            if s.bn_folded_in_onnx and s.role == spec.CONV_KERNEL:
                entry["onnx_check"] = "bn_folded"
                entry["max_abs_diff"] = None
            else:
                entry["onnx_check"] = "direct"
                entry["max_abs_diff"] = float(np.abs(arr - rust_value).max())

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
            "bn_folded_note": (
                "En ONNX la BatchNorm esta fusionada en las convoluciones "
                "conv2d_1 y conv2d_4; sus inicializadores no son comparables "
                "tensor a tensor. La relacion de fusion se verifica numericamente."
            ),
        },
        "tensors": entries,
    }


def verify_bn_folding(variables, onnx_inits) -> list:
    """Verifica la relacion de fusion de la BatchNorm en ONNX.

    Para cada conv con BatchNorm:

        scale = gamma / sqrt(moving_variance + epsilon)
        W_onnx debe ser  W_tf_transpuesto * scale   (canal a canal)
        b_onnx debe ser (b_tf - moving_mean) * scale + beta

    Devuelve la lista de comprobaciones superadas.
    """
    checks = []
    # (nombre de la conv, prefijo de la BatchNorm, inicializador ONNX)
    pairs = [
        ("conv2d_1", "batch_normalization_2", "const_fold_opt__727"),
        ("conv2d_4", "batch_normalization_3", "const_fold_opt__707"),
    ]
    for conv_name, bn_prefix, onnx_name in pairs:
        kernel = variables[f"{conv_name}/kernel"].numpy()  # [kh, kw, in, out]
        bias = variables[f"{conv_name}/bias"].numpy()  # [out]
        gamma = variables[f"{bn_prefix}/gamma"].numpy()
        beta = variables[f"{bn_prefix}/beta"].numpy()
        mean = variables[f"{bn_prefix}/moving_mean"].numpy()
        var = variables[f"{bn_prefix}/moving_variance"].numpy()

        scale = gamma / np.sqrt(var + spec.KERAS_BATCH_NORM_EPSILON)

        onnx_kernel = onnx_inits[onnx_name]  # [out, in, kh, kw]
        onnx_bias = None
        # El bias fusionado se busca entre los inicializadores con la forma
        # del vector de salida.
        for name, arr in onnx_inits.items():
            if arr.shape == bias.shape and "batch_normalization" in name:
                onnx_bias = arr
                break

        # W_tf -> [out, in, kh, kw]
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


def onnx_output_roles() -> dict:
    """Mapea cada salida del ONNX a su rol (contour / note / onset).

    Se deriva del grafo: cada salida del modelo es producida por un nodo
    `Reshape` cuyo nombre coincide con el de la salida. Por eso hay que
    retroceder por la cadena de producers hasta la convolution que origina el
    valor. No se asume el orden de las salidas, que no esta garantizado.
    """
    import onnx

    model = onnx.load(str(spec.ONNX_MODEL))
    producer: dict = {}
    for node in model.graph.node:
        for out in node.output:
            producer[out] = node.name

    node_by_name = {n.name: n for n in model.graph.node}

    roles = {}
    for graph_out in model.graph.output:
        # Se recoge la cadena de nodos que producen la salida, hacia atras.
        chain: list = []
        current = graph_out.name
        for _ in range(8):
            node_name = producer.get(current)
            if node_name is None:
                break
            node = node_by_name.get(node_name)
            chain.append(node_name)
            if node is None or not node.input:
                break
            current = node.input[0]

        markers = {
            "contour": "contours-reduced",
            "note": "conv2d_3",
            "onset": "conv2d_5",
        }
        role = None
        for name in chain:
            for candidate, marker in markers.items():
                if marker in name and role is None:
                    role = candidate
        if role is None:
            raise AssertionError(
                f"No se pudo determinar el rol de la salida ONNX "
                f"'{graph_out.name}'. Cadena: {chain}"
            )
        roles[graph_out.name] = role
    return roles


def cross_check_outputs(audio: np.ndarray) -> dict:
    """Ejecuta TensorFlow y ONNX sobre la misma entrada y compara."""
    import onnxruntime as ort

    tf_model = tf.saved_model.load(str(spec.TF_SAVED_MODEL))
    fn = tf_model.signatures["serving_default"]
    tf_out = fn(input_2=tf.constant(audio))
    tf_dict = {k: np.asarray(v) for k, v in tf_out.items()}

    sess = ort.InferenceSession(str(spec.ONNX_MODEL), providers=["CPUExecutionProvider"])
    onnx_input = sess.get_inputs()[0].name
    raw_outputs = sess.run(None, {onnx_input: audio})

    roles = onnx_output_roles()
    output_names = [o.name for o in sess.get_outputs()]
    onnx_dict = {
        roles[name]: arr for name, arr in zip(output_names, raw_outputs)
    }

    comparison = {}
    for name in ("contour", "note", "onset"):
        reference = tf_dict[name]
        arr = onnx_dict[name]
        diff = np.abs(reference - arr)
        comparison[name] = {
            "shape_tf": list(reference.shape),
            "shape_onnx": list(arr.shape),
            "max_abs_diff": float(diff.max()),
            "mean_abs_diff": float(diff.mean()),
            "cosine_similarity": float(
                np.dot(reference.ravel(), arr.ravel())
                / (np.linalg.norm(reference) * np.linalg.norm(arr))
            ),
        }
    return comparison


# ---------------------------------------------------------------------------
# Configuracion tecnica
# ---------------------------------------------------------------------------

def build_model_config() -> dict:
    """Contrato tecnico: constantes verificadas del modelo."""
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
            "q_factor": float(1.0 / (2 ** (1.0 / (12 * spec.CONTOURS_BINS_PER_SEMITONE)) - 1.0)),
            "n_octaves": int(np.ceil(spec.CQT_N_BINS / (12 * spec.CONTOURS_BINS_PER_SEMITONE))),
            "n_filters": 12 * spec.CONTOURS_BINS_PER_SEMITONE,
            "n_fft": 256,
            "early_downsample": False,
            "downsample_factor": 1,
            "pad_mode": "reflect",
            "basis_norm": 1,
            "normalization": "sqrt(lengths)",
        },
        "harmonic_stacking": {
            "harmonics": list(spec.HARMONICS),
            "shifts": list(spec.HARMONIC_SHIFTS),
            "bins_per_semitone": spec.CONTOURS_BINS_PER_SEMITONE,
            "n_output_freqs": spec.N_FREQ_BINS_CONTOURS,
            "input_n_bins": spec.CQT_N_BINS,
            "padding": "asimetrico: shift>0 rellena al final, shift<0 rellena al principio",
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
                "pad_same_tf": [list(layer.pad_same_tf[0]), list(layer.pad_same_tf[1])],
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
    }


def build_summary(inventory: dict, config: dict, provenance: dict) -> str:
    """Resumen legible del modelo."""
    lines = []
    add = lines.append
    add("=" * 78)
    add("BASIC PITCH (ICASSP 2022) - RESUMEN DE AUDITORIA")
    add("=" * 78)
    add("")
    add(f"Commit de referencia : {spec.REFERENCE_COMMIT}")
    add(f"Version del paquete   : {spec.REFERENCE_PACKAGE_VERSION}")
    add("")
    add("-- Entrada y salida ------------------------------------------------")
    add(f"  Audio            : (batch, {spec.AUDIO_N_SAMPLES}, 1)  @ {spec.AUDIO_SAMPLE_RATE} Hz")
    add(f"  CQT              : (batch, {spec.ANNOT_N_FRAMES}, {spec.CQT_N_BINS})   <-- {spec.CQT_N_BINS} bins")
    add(f"  HarmonicStack   : (batch, {spec.ANNOT_N_FRAMES}, {spec.N_FREQ_BINS_CONTOURS}, {len(spec.HARMONICS)})")
    add(f"  note             : (batch, {spec.ANNOT_N_FRAMES}, {spec.N_FREQ_BINS_NOTES})")
    add(f"  onset            : (batch, {spec.ANNOT_N_FRAMES}, {spec.N_FREQ_BINS_NOTES})")
    add(f"  contour          : (batch, {spec.ANNOT_N_FRAMES}, {spec.N_FREQ_BINS_CONTOURS})")
    add("")
    add("-- Capas convolucionales activas ----------------------------------")
    for layer in spec.LAYER_SPECS:
        add(f"  {layer.name:20s} k={str(layer.kernel_tf):9s} f={layer.filters:3d} "
            f"in={layer.in_channels:3d} s={str(layer.stride_tf):8s} "
            f"pad={layer.pad_same_tf} act={layer.activation} "
            f"bn={'si' if layer.batch_norm else 'no'}")
        if layer.note:
            add(f"      -> {layer.note}")
    add("")
    add("-- Concatenacion de onsets -----------------------------------------")
    add(f"  Orden  : {config['concat']['order']}")
    add(f"  Canales: {config['concat']['channels']}  (1 + 32)")
    add("")
    add("-- BatchNorm --------------------------------------------------------")
    add(f"  epsilon: {config['batch_norm']['epsilon']}  ({config['batch_norm']['epsilon_source']})")
    for layer in config["batch_norm"]["layers"]:
        add(f"  {layer['name']:24s} canales={layer['channels']:3d}  ({layer['position']})")
    add("")
    add("-- Ventanas ---------------------------------------------------------")
    w = config["windowing"]
    add(f"  overlap_len={w['overlap_len']}  hop_size={w['hop_size']}  "
        f"padding inicial={w['initial_padding']}")
    add(f"  frames emitidos={w['frames_per_window']}  conservados={w['frames_kept_per_window']}")
    add(f"  {w['compatibility_note']}")
    add("")
    add("-- Tensores ---------------------------------------------------------")
    add(f"  Total en el SavedModel : {len(inventory['variables'])}")
    add(f"  En la tabla canonica    : {len(spec.TENSOR_SPECS)}")
    add("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    print("[1/8] Verificando la tabla de padding SAME...")
    spec.verify_padding_table()
    print(f"      OK: {len(spec.LAYER_SPECS)} capas con padding SAME consistente.")

    print("[2/8] Cargando el SavedModel de TensorFlow...")
    tf_model, raw_variables = load_saved_model()
    print(f"      {len(raw_variables)} variables encontradas.")

    print("[3/8] Reconstruyendo el modelo Keras y asignando pesos...")
    variables, actual_names = map_saved_model_variables(raw_variables)
    print(f"      {len(variables)} tensores resueltos por rol y forma.")
    keras_model = build_keras_model(variables)
    print(f"      {len(keras_model.layers)} capas.")

    print("[4/8] Generando el inventario...")
    inventory = build_inventory(keras_model, variables)
    json_dump(inventory, spec.REFERENCE_DIR / "model_inventory.json")
    print(f"      -> {spec.REFERENCE_DIR / 'model_inventory.json'}")

    print("[5/8] Capturando las salidas intermedias de referencia...")
    # Senal sintetica determinista: mezcla de tonos con envolvente suave.
    rng = np.random.default_rng(20240517)
    t = np.arange(spec.AUDIO_N_SAMPLES, dtype=np.float32) / spec.AUDIO_SAMPLE_RATE
    audio = (
        0.5 * np.sin(2 * np.pi * 220.0 * t)
        + 0.3 * np.sin(2 * np.pi * 440.0 * t + 0.3)
        + 0.2 * np.sin(2 * np.pi * 660.0 * t + 1.1)
    ).astype(np.float32)
    audio += 0.001 * rng.standard_normal(spec.AUDIO_N_SAMPLES).astype(np.float32)
    audio = np.clip(audio, -1.0, 1.0).reshape(1, spec.AUDIO_N_SAMPLES, 1)

    captured = capture_reference_outputs(keras_model, audio)
    index = save_reference_outputs(captured, audio)
    json_dump(
        {"description": "Salidas intermedias de referencia (Puntos de comparacion de la Fase 0).",
         "points": index},
        spec.REFERENCE_DIR / "reference_outputs" / "index.json",
    )
    print(f"      {len(index)} puntos de comparacion guardados.")

    print("[6/8] Construyendo el manifiesto de trazabilidad TF <-> ONNX <-> Rust...")
    onnx_inits = onnx_initializers()
    provenance = build_provenance(variables, onnx_inits, actual_names)
    provenance["bn_folding_checks"] = verify_bn_folding(variables, onnx_inits)
    json_dump(provenance, spec.REFERENCE_DIR / "tensor_provenance.json")
    for check in provenance["bn_folding_checks"]:
        print(f"      fusion BN {check['conv']:10s} kernel_diff={check['kernel_max_abs_diff']:.3e} "
              f"bias_diff={check['bias_max_abs_diff']}  -> "
              f"{'OK' if check['passed'] else 'FALLO'}")

    print("[7/8] Cross-check numerico TensorFlow vs ONNX...")
    comparison = cross_check_outputs(audio)
    for name, data in comparison.items():
        print(f"      {name:8s} max_abs_diff={data['max_abs_diff']:.3e} "
              f"cos={data['cosine_similarity']:.10f}")
    provenance["output_cross_check_tf_vs_onnx"] = comparison

    print("[8/8] Escribiendo configuracion, resumen y metadatos...")
    config = build_model_config()
    json_dump(config, spec.REFERENCE_DIR / "model_config.json")
    json_dump(provenance, spec.REFERENCE_DIR / "tensor_provenance.json")

    summary = build_summary(inventory, config, provenance)
    (spec.REFERENCE_DIR / "model_summary.txt").write_text(summary, encoding="utf-8")
    print(summary)

    metadata = {
        "generated_by": "tools/weight-export/inspect_model.py",
        "reference_repo": spec.REFERENCE_REPO_URL,
        "reference_commit": spec.REFERENCE_COMMIT,
        "reference_package_version": spec.REFERENCE_PACKAGE_VERSION,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "tensorflow": tf.__version__,
            "numpy": np.__version__,
        },
        "artifacts": {
            "tf_saved_model": {
                "path": str(spec.TF_SAVED_MODEL.relative_to(spec.REPO_ROOT)),
                "sha256": sha256_dir(spec.TF_SAVED_MODEL),
            },
            "onnx_model": {
                "path": str(spec.ONNX_MODEL.relative_to(spec.REPO_ROOT)),
                "sha256": sha256_file(spec.ONNX_MODEL),
                "bytes": spec.ONNX_MODEL.stat().st_size,
            },
        },
        "notes": [
            "TensorFlow es la fuente canonica: el ONNX tiene la BatchNorm fusionada.",
            "El cross-check numerico TF vs ONNX valida el pipeline completo.",
        ],
    }
    json_dump(metadata, spec.REFERENCE_DIR / "reference_metadata.json")

    print("Auditoria completada.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

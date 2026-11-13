"""Carga del modelo de referencia: TensorFlow (canonico) y ONNX (verificacion).

Modulo compartido por `inspect_model.py` (Fase 0) y `export_weights.py`
(Fase 1).

Punto clave: los nombres de capa autoasignados por Keras **no son estables**
entre builds, porque el contador de nombres es global a la sesion de
TensorFlow. El modelo guardado tiene `conv2d_1`..`conv2d_5`, mientras que un
build fresco produce `conv2d`..`conv2d_4`. Por eso la resolucion de tensores se
hace por **rol** (sufijo del nombre) y **forma**, nunca por nombre.
"""

from __future__ import annotations

import os
import pathlib
import sys
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import model_spec as spec  # noqa: E402

sys.path.insert(0, str(spec.REFERENCE_REPO))


def load_saved_model() -> tuple:
    """Carga el SavedModel de TensorFlow.

    Devuelve (modelo, {nombre: variable}).
    """
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

    Dos pasos:

    1. **Clasificar por rol**, usando el sufijo del nombre de TensorFlow
       (`/kernel`, `/bias`, `/gamma`, `/beta`, `/moving_mean`,
       `/moving_variance`). Esto separa tensores que comparten forma (los
       biases de 1 canal y las BatchNorm de 1 canal).
    2. **Desambiguar por forma** dentro de cada rol. Las formas son unicas:
       kernels `(3,39,8,8) (5,5,8,1) (7,7,1,32) (7,3,32,1) (5,5,8,32)
       (3,3,33,1)` y BatchNorm de anchos `1, 8, 32`.

    Devuelve:
        resolved      {nombre canonico: variable de TensorFlow}
        actual_names  {nombre canonico: nombre real en el SavedModel}
    """
    kernels: dict = {}
    bn: dict = {}  # (rol, ancho) -> [nombres]

    for name in variables:
        if name.endswith("/kernel"):
            kernels.setdefault(tuple(variables[name].shape.as_list()), []).append(name)
        elif name.endswith("/bias"):
            continue  # se empareja con su conv
        else:
            prefix, _, role = name.rpartition("/")
            if role in ("gamma", "beta", "moving_mean", "moving_variance"):
                width = variables[name].shape.as_list()[0]
                bn.setdefault((role, width), []).append(name)

    resolved: dict = {}
    actual_names: dict = {}

    # 1) Resolver las convoluciones (kernel + bias emparejados).
    conv_pairs: dict = {}
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
        conv_pairs[s.tf_name.rsplit("/", 1)[0]] = (actual_kernel, actual_bias)

    # 2) Resolver cada tensor de la tabla canonica.
    for s in spec.TENSOR_SPECS:
        if s.role == spec.CONV_KERNEL:
            actual_kernel, actual_bias = conv_pairs[s.tf_name.rsplit("/", 1)[0]]
            resolved[s.tf_name] = variables[actual_kernel]
            actual_names[s.tf_name] = actual_kernel
        elif s.role == spec.CONV_BIAS:
            actual_kernel, actual_bias = conv_pairs[s.tf_name.rsplit("/", 1)[0]]
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
    """Reconstruye el modelo Keras y le asigna los pesos del SavedModel."""
    from basic_pitch.models import model as build_model

    keras_model = build_model()

    kernels_by_shape = {
        s.shape_tf: s for s in spec.TENSOR_SPECS if s.role == spec.CONV_KERNEL
    }
    bn_by_width = {
        s.shape_tf[0]: s for s in spec.TENSOR_SPECS if s.role == spec.BN_GAMMA
    }

    for layer in keras_model.layers:
        weights = layer.weights
        if not weights:
            continue

        if isinstance(layer, tf.keras.layers.Conv2D):
            shape = tuple(weights[0].shape.as_list())
            canonical = kernels_by_shape.get(shape)
            if canonical is None:
                raise KeyError(
                    f"La conv '{layer.name}' tiene forma {shape}, que no "
                    "corresponde a ninguna capa de la tabla canonica."
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
                    f"La BatchNorm '{layer.name}' tiene {width} canales, que no "
                    "corresponde a ninguna de la tabla canonica."
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

    Devuelve un mapa: clave semantica -> capa. Las claves son:

        cqt, normalized_log, harmonic_stacking,
        batch_norm.input, contour.batch_norm, onset.batch_norm,
        contour.conv1, contour.reduced, notes.conv1, notes.conv2,
        onset.conv1, onset.conv2,
        contour, note, onset   (salidas aplanadas)
    """
    import basic_pitch.layers.nnaudio as nnaudio
    import basic_pitch.layers.signal as signal_mod
    import basic_pitch.nn as nn_mod

    conv_by_shape = {
        (layer.kernel_tf[0], layer.kernel_tf[1], layer.in_channels, layer.filters): layer
        for layer in spec.LAYER_SPECS
    }
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


def onnx_initializers() -> dict:
    """Carga los inicializadores del ONNX oficial como arrays de numpy."""
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(spec.ONNX_MODEL))
    return {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}


def to_rust_layout(array: np.ndarray, transpose) -> np.ndarray:
    """Convierte un tensor de TensorFlow al layout que espera Candle."""
    if transpose is None:
        return np.ascontiguousarray(array)
    return np.ascontiguousarray(np.transpose(array, transpose))


# ---------------------------------------------------------------------------
# Identificacion del equivalente ONNX por CONTENIDO
# ---------------------------------------------------------------------------

def _bn_scale_and_folded_bias(variables: dict, bn_prefix: str):
    """Devuelve (scale, bias_fusionado) de una BatchNorm.

    scale[c] = gamma[c] / sqrt(moving_variance[c] + epsilon)
    bias_fusionado[c] = (bias[c] - moving_mean[c]) * scale[c] + beta[c]
    """
    gamma = variables[f"{bn_prefix}/gamma"].numpy()
    beta = variables[f"{bn_prefix}/beta"].numpy()
    mean = variables[f"{bn_prefix}/moving_mean"].numpy()
    var = variables[f"{bn_prefix}/moving_variance"].numpy()
    scale = gamma / np.sqrt(var + spec.KERAS_BATCH_NORM_EPSILON)
    return scale, beta


def identify_onnx_counterparts(variables: dict, onnx_inits: dict) -> dict:
    """Identifica el inicializador ONNX equivalente de cada tensor canonico.

    La identificacion es **por contenido**, no por forma: varios tensores
    comparten forma (los biases de 1 canal y las BatchNorm de 1 canal, o los
    biases de 32 canales y las escalas de 32 canales), y emparejar solo por
    forma produce falsos positivos silenciosos.

    Para cada tensor canonico se prueban tres relaciones, en este orden:

    1. `direct`      : onnx == tensor de Rust
    2. `bn_folded`   : onnx == tensor de Rust * scale   (kernel con BN fusionada)
    3. `bn_folded_bias`: onnx == (bias - mean) * scale + beta  (bias fusionado)

    Se elige la relacion con menor diferencia absoluta maxima. Un tensor sin
    equivalente en ONNX (los parametros de BatchNorm, que estan fusionados)
    se reporta con `relation = "folded_into_conv"`.

    Devuelve: {nombre canonico: {onnx_name, relation, max_abs_diff}}
    """
    # BatchNorm asociada a cada conv, por nombre de la conv.
    bn_for_conv = {
        "conv2d_1": "batch_normalization_2",
        "conv2d_4": "batch_normalization_3",
    }

    results: dict = {}
    for s in spec.TENSOR_SPECS:
        tf_value = variables[s.tf_name].numpy()
        rust_value = to_rust_layout(tf_value, s.transpose)

        # Los parametros de BatchNorm no tienen equivalente en ONNX: estan
        # fusionados dentro de las convoluciones. No existe una comparacion
        # numerica posible, asi que no se busca ninguna.
        if s.role in (spec.BN_GAMMA, spec.BN_BETA, spec.BN_MEAN, spec.BN_VAR):
            results[s.tf_name] = {
                "onnx_name": None,
                "relation": "folded_into_conv",
                "max_abs_diff": None,
                "note": (
                    "Parametro de BatchNorm: en ONNX esta fusionado dentro de "
                    "la convolucion y no existe un inicializador equivalente. "
                    "Su unica fuente es TensorFlow."
                ),
            }
            continue

        candidates = [
            (name, arr)
            for name, arr in onnx_inits.items()
            if tuple(arr.shape) == tuple(rust_value.shape)
        ]

        # Relaciones candidatas: (nombre, valor esperado).
        relations: list = [("direct", rust_value)]
        if s.role in (spec.CONV_KERNEL, spec.CONV_BIAS):
            conv_name = s.layer
            bn_prefix = bn_for_conv.get(conv_name)
            if bn_prefix is not None:
                scale, beta = _bn_scale_and_folded_bias(variables, bn_prefix)
                if s.role == spec.CONV_KERNEL:
                    relations.append(
                        ("bn_folded", rust_value * scale.reshape(-1, 1, 1, 1))
                    )
                else:
                    mean = variables[f"{bn_prefix}/moving_mean"].numpy()
                    relations.append(
                        ("bn_folded_bias", (tf_value - mean) * scale + beta)
                    )

        best = None
        for relation, expected in relations:
            for name, arr in candidates:
                diff = float(np.abs(arr - expected).max())
                if best is None or diff < best["max_abs_diff"]:
                    best = {
                        "onnx_name": name,
                        "relation": relation,
                        "max_abs_diff": diff,
                    }

        if best is None:
            results[s.tf_name] = {
                "onnx_name": None,
                "relation": "no_equivalent",
                "max_abs_diff": None,
                "note": "Sin inicializador ONNX con la misma forma.",
            }
        elif best["max_abs_diff"] > 1e-5:
            results[s.tf_name] = {
                "onnx_name": None,
                "relation": "folded_into_conv",
                "max_abs_diff": best["max_abs_diff"],
                "note": (
                    "El valor esta fusionado dentro de la convolucion en ONNX; "
                    "no existe un inicializador equivalente directo."
                ),
            }
        else:
            results[s.tf_name] = best

    return results

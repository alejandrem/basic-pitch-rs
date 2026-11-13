"""Especificacion canonica del modelo ICASSP 2022 de Spotify Basic Pitch.

Este modulo es la **unica fuente de verdad** sobre:

- los nombres definitivos de los tensores en Rust,
- el rol semantico de cada tensor,
- la correspondencia con los nombres de TensorFlow y de ONNX,
- las constantes verificadas del modelo.

Las tres correcciones criticas del proyecto estan codificadas aqui y deben
mantenerse sincronizadas con `docs/`:

1. La CQT produce **309** bins; Harmonic Stacking reduce a 264.
2. La concatenacion de onsets es ``[x_notes_pre, x_onset]`` -> **33 canales**.
3. ``CONTOUR_KERNEL_SIZE_1`` (32 filtros) esta comentada y se omite;
   ``CONTOUR_KERNEL_SIZE_3`` (1 filtro) es activa.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

#: Copia local del repositorio de referencia (ignorada por git).
REFERENCE_REPO = REPO_ROOT / "basic-pitch"

#: SavedModel de TensorFlow: fuente canonica de pesos y de estadisticas BN.
TF_SAVED_MODEL = REFERENCE_REPO / "basic_pitch" / "saved_models" / "icassp_2022" / "nmp"

#: ONNX oficial: verificacion cruzada y extraccion de kernels de CQT.
ONNX_MODEL = REFERENCE_REPO / "basic_pitch" / "saved_models" / "icassp_2022" / "nmp.onnx"

#: Directorio canonico del modelo desplegable.
MODEL_DIR = REPO_ROOT / "models" / "basic-pitch" / "icassp-2022"

#: Directorio canonico de los artefactos de auditoria.
REFERENCE_DIR = REPO_ROOT / "reference"

#: Directorio canonico de las salidas de referencia.
REFERENCE_OUTPUTS_DIR = REFERENCE_DIR / "reference_outputs"

# ---------------------------------------------------------------------------
# Constantes verificadas del modelo
# ---------------------------------------------------------------------------

AUDIO_SAMPLE_RATE = 22050
AUDIO_WINDOW_LENGTH = 2
FFT_HOP = 256
ANNOTATIONS_FPS = AUDIO_SAMPLE_RATE // FFT_HOP  # 86
ANNOT_N_FRAMES = ANNOTATIONS_FPS * AUDIO_WINDOW_LENGTH  # 172
AUDIO_N_SAMPLES = AUDIO_SAMPLE_RATE * AUDIO_WINDOW_LENGTH - FFT_HOP  # 43844

ANNOTATIONS_BASE_FREQUENCY = 27.5
ANNOTATIONS_N_SEMITONES = 88
CONTOURS_BINS_PER_SEMITONE = 3
NOTES_BINS_PER_SEMITONE = 1

#: Bins de frecuencia que produce la CQT. NO son 264.
#: 264 es el resultado despues de Harmonic Stacking.
CQT_N_SEMITONES = 103  # min(ceil(12*log2(8)) + 88, MAX_N_SEMITONES=103)
CQT_N_BINS = CQT_N_SEMITONES * CONTOURS_BINS_PER_SEMITONE  # 309

#: Bins por semitono de la representacion de contornos (salida del modelo).
N_FREQ_BINS_CONTOURS = ANNOTATIONS_N_SEMITONES * CONTOURS_BINS_PER_SEMITONE  # 264
N_FREQ_BINS_NOTES = ANNOTATIONS_N_SEMITONES * NOTES_BINS_PER_SEMITONE  # 88

#: Armonicos y desplazamientos resultantes.
HARMONICS: Tuple[float, ...] = (0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
HARMONIC_SHIFTS: Tuple[int, ...] = (-36, 0, 36, 57, 72, 84, 93, 101)

#: Ventanas superpuestas.
N_OVERLAPPING_FRAMES = 30
OVERLAP_LEN = N_OVERLAPPING_FRAMES * FFT_HOP  # 7680 muestras
HOP_SIZE = AUDIO_N_SAMPLES - OVERLAP_LEN  # 36164 muestras
INITIAL_PADDING = OVERLAP_LEN // 2  # 3840 ceros
FRAMES_PER_WINDOW = ANNOT_N_FRAMES  # 172 emitidos
FRAMES_KEPT_PER_WINDOW = ANNOT_N_FRAMES - N_OVERLAPPING_FRAMES  # 142 conservados

#: `epsilon` de Keras. El atributo de ONNX `FusedBatchNormV3` (1e-5) NO es el
#: valor correcto para este modelo: el ONNX fusiona la normalizacion.
KERAS_BATCH_NORM_EPSILON = 0.001

#: Numero de canales tras la concatenacion de la rama de onsets.
#: x_notes_pre (1) + x_onset (32) = 33
ONSET_CONCAT_CHANNELS = 33

#: Referencia de version fijada.
REFERENCE_COMMIT = "fa5997af0a8210982619003269994a1be25eddf3"
REFERENCE_REPO_URL = "https://github.com/spotify/basic-pitch.git"
REFERENCE_PACKAGE_VERSION = "0.4.0"

# ---------------------------------------------------------------------------
# Tabla canonica de tensores
# ---------------------------------------------------------------------------

#: Rol semantico de un tensor.
Role = str

CONV_KERNEL: Role = "conv_kernel"
CONV_BIAS: Role = "conv_bias"
BN_GAMMA: Role = "bn_gamma"
BN_BETA: Role = "bn_beta"
BN_MEAN: Role = "bn_moving_mean"
BN_VAR: Role = "bn_moving_variance"


@dataclass(frozen=True)
class TensorSpec:
    """Descripcion canonica de un tensor del modelo."""

    #: Nombre del tensor en el SavedModel de TensorFlow (sin sufijo `:0`).
    tf_name: str
    #: Nombre definitivo en Rust y en el Safetensors.
    rust_name: str
    #: Rol semantico.
    role: Role
    #: Forma esperada en TensorFlow.
    shape_tf: Tuple[int, ...]
    #: Si el kernel de conv requiere permutacion para Candle.
    transpose: Optional[Tuple[int, ...]] = None
    #: Si la BatchNorm fue fusionada en este tensor en la exportacion ONNX.
    bn_folded_in_onnx: bool = False
    #: Inicializador equivalente en ONNX (si se conoce).
    onnx_name: Optional[str] = None
    #: Descripcion de la capa a la que pertenece.
    layer: str = ""

    @property
    def shape_rust(self) -> Tuple[int, ...]:
        """Forma en Rust, tras la permutacion si aplica."""
        if self.transpose is not None:
            return tuple(self.shape_tf[i] for i in self.transpose)
        return self.shape_tf


#: La permutacion de los kernels de conv de TensorFlow a Candle.
#: TF: [kh, kw, in_ch, out_ch] -> Candle: [out_ch, in_ch, kh, kw]
CONV_TRANSPOSE: Tuple[int, ...] = (3, 2, 0, 1)


def _conv(tf_name: str, rust_name: str, shape: Tuple[int, ...], layer: str) -> TensorSpec:
    return TensorSpec(
        tf_name=tf_name,
        rust_name=rust_name,
        role=CONV_KERNEL if len(shape) == 4 else CONV_BIAS,
        shape_tf=shape,
        transpose=CONV_TRANSPOSE if len(shape) == 4 else None,
        layer=layer,
    )


def _bn(prefix: str, rust_prefix: str, width: int, layer: str, folded: bool) -> List[TensorSpec]:
    """Parametros de una BatchNorm de Keras."""
    specs = [
        TensorSpec(f"{prefix}/gamma", f"{rust_prefix}.gamma", BN_GAMMA, (width,), layer=layer),
        TensorSpec(f"{prefix}/beta", f"{rust_prefix}.beta", BN_BETA, (width,), layer=layer),
        TensorSpec(f"{prefix}/moving_mean", f"{rust_prefix}.moving_mean", BN_MEAN, (width,), layer=layer),
        TensorSpec(
            f"{prefix}/moving_variance",
            f"{rust_prefix}.moving_variance",
            BN_VAR,
            (width,),
            layer=layer,
        ),
    ]
    if folded:
        for s in specs:
            object.__setattr__(s, "bn_folded_in_onnx", True)
    return specs


#: Tabla completa y ordenada de los tensores del modelo activo.
#:
#: 28 tensores: 6 convoluciones (kernel + bias) y 3 BatchNorm (4 tensores cada
#: una). Verifica contra las variables del SavedModel oficial.
TENSOR_SPECS: List[TensorSpec] = (
    # --- BatchNorm inicial: 1 canal, estadisticas escalares -------------
    _bn("batch_normalization", "batch_norm.input", 1, "batch_normalization", folded=True)
    # --- Rama de contornos ----------------------------------------------
    + [
        _conv("conv2d_1/kernel", "contour.conv1.weight", (3, 39, 8, 8), "conv2d_1"),
        _conv("conv2d_1/bias", "contour.conv1.bias", (8,), "conv2d_1"),
    ]
    + _bn("batch_normalization_2", "contour.batch_norm", 8, "batch_normalization_2", folded=True)
    + [
        # CONTOUR_KERNEL_SIZE_3: 1 filtro, ACTIVA. (CONTOUR_KERNEL_SIZE_1,
        # con 32 filtros, esta comentada en el upstream y NO existe aqui.)
        _conv("contours-reduced/kernel", "contour.reduced.weight", (5, 5, 8, 1), "contours-reduced"),
        _conv("contours-reduced/bias", "contour.reduced.bias", (1,), "contours-reduced"),
    ]
    # --- Rama de notas (sin BatchNorm) ----------------------------------
    + [
        _conv("conv2d_2/kernel", "notes.conv1.weight", (7, 7, 1, 32), "conv2d_2"),
        _conv("conv2d_2/bias", "notes.conv1.bias", (32,), "conv2d_2"),
        _conv("conv2d_3/kernel", "notes.conv2.weight", (7, 3, 32, 1), "conv2d_3"),
        _conv("conv2d_3/bias", "notes.conv2.bias", (1,), "conv2d_3"),
    ]
    # --- Rama de onsets --------------------------------------------------
    + [
        _conv("conv2d_4/kernel", "onset.conv1.weight", (5, 5, 8, 32), "conv2d_4"),
        _conv("conv2d_4/bias", "onset.conv1.bias", (32,), "conv2d_4"),
    ]
    + _bn("batch_normalization_3", "onset.batch_norm", 32, "batch_normalization_3", folded=True)
    + [
        # 33 canales de entrada: x_notes_pre (1) + x_onset (32).
        _conv("conv2d_5/kernel", "onset.conv2.weight", (3, 3, 33, 1), "conv2d_5"),
        _conv("conv2d_5/bias", "onset.conv2.bias", (1,), "conv2d_5"),
    ]
)

#: Correccion: el kernel de conv2d_5 tiene 33 canales de entrada.
assert TENSOR_SPECS[-2].shape_tf == (3, 3, ONSET_CONCAT_CHANNELS, 1), (
    "El concatenado de onsets debe producir 33 canales (correccion #2)."
)

#: Mapa de busqueda por nombre en TensorFlow.
SPEC_BY_TF_NAME: Dict[str, TensorSpec] = {s.tf_name: s for s in TENSOR_SPECS}

#: Mapa de busqueda por nombre en Rust.
SPEC_BY_RUST_NAME: Dict[str, TensorSpec] = {s.rust_name: s for s in TENSOR_SPECS}

# ---------------------------------------------------------------------------
# Kernels deterministas de la CQT (extraidos del ONNX oficial)
# ---------------------------------------------------------------------------

#: Nombre del inicializador ONNX -> nombre en el asset de kernels de CQT.
#:
#: Los nombres reales del ONNX exportado por tf2onnx vienen "mangled" con un
#: sufijo `;<nombre del nodo>`. El resolutor (`resolve_onnx_initializer`)
#: acepta tanto el nombre completo como el prefijo antes del `;`.
CQT_KERNEL_SOURCES: Dict[str, str] = {
    "const_fold_opt__664": "cqt.kernels.real",
    "const_fold_opt__655": "cqt.kernels.imag",
    "const_fold_opt__734": "cqt.lowpass_filter",
    "model_1/cq_t2010v2_1/Sqrt": "cqt.lengths_sqrt",
    "model_1/cq_t2010v2_1/conv1d_25": "cqt.conv_bias",
}

#: Formas esperadas de esos tensores.
CQT_KERNEL_SHAPES: Dict[str, Tuple[int, ...]] = {
    "cqt.kernels.real": (36, 1, 1, 256),
    "cqt.kernels.imag": (36, 1, 1, 256),
    "cqt.lowpass_filter": (1, 1, 1, 256),
    "cqt.lengths_sqrt": (309, 1, 1),
    "cqt.conv_bias": (36,),
}


def resolve_onnx_initializer(onnx_inits: dict, name: str):
    """Busca un inicializador en ONNX tolerando el 'mangling' de tf2onnx.

    Los nombres exportados tienen el formato `nombre;nombre_del_nodo`. Se
    acepta el nombre exacto o el prefijo antes del `;`.
    """
    if name in onnx_inits:
        return onnx_inits[name]
    matches = [key for key in onnx_inits if key.split(";")[0] == name]
    if len(matches) == 1:
        return onnx_inits[matches[0]]
    if len(matches) > 1:
        raise KeyError(
            f"El inicializador '{name}' es ambiguo en ONNX: {matches}"
        )
    raise KeyError(
        f"El inicializador '{name}' no existe en el ONNX oficial. "
        f"Disponibles: {sorted(k for k in onnx_inits if 'const_fold' in k or 'Sqrt' in k)}"
    )


# ---------------------------------------------------------------------------
# Tabla de la arquitectura (contrato tecnico)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LayerSpec:
    """Descripcion de una capa convolucional activa del modelo."""

    name: str
    rust_name: str
    kernel_tf: Tuple[int, int]
    filters: int
    in_channels: int
    stride_tf: Tuple[int, int]
    #: Forma de la entrada en NHWC: (batch, time, freq, channels).
    in_shape: Tuple[int, int, int]
    #: Padding SAME de TensorFlow por eje (antes, despues).
    pad_same_tf: Tuple[Tuple[int, int], Tuple[int, int]]
    activation: str
    batch_norm: bool
    note: str = ""

    @property
    def out_shape(self) -> Tuple[int, int, int]:
        """Forma de la salida en NHWC: (batch, time, freq, channels)."""
        t, f, _ = self.in_shape
        kt, kf = self.kernel_tf
        st, sf = self.stride_tf
        out_t = -(-t // st)
        out_f = -(-f // sf)
        return (out_t, out_f, self.filters)


LAYER_SPECS: List[LayerSpec] = [
    LayerSpec(
        name="conv2d_1",
        rust_name="contour.conv1",
        kernel_tf=(3, 39),
        filters=8,
        in_channels=8,
        stride_tf=(1, 1),
        in_shape=(ANNOT_N_FRAMES, N_FREQ_BINS_CONTOURS, 8),
        pad_same_tf=((1, 1), (19, 19)),
        activation="relu",
        batch_norm=True,
        note="Primera conv ACTIVA de la rama de contornos.",
    ),
    LayerSpec(
        name="contours-reduced",
        rust_name="contour.reduced",
        kernel_tf=(5, 5),
        filters=1,
        in_channels=8,
        stride_tf=(1, 1),
        in_shape=(ANNOT_N_FRAMES, N_FREQ_BINS_CONTOURS, 8),
        pad_same_tf=((2, 2), (2, 2)),
        activation="sigmoid",
        batch_norm=False,
        note=(
            "CONTOUR_KERNEL_SIZE_3: ACTIVA. CONTOUR_KERNEL_SIZE_1 (32 filtros, "
            "kernel 5x5) esta comentada en el upstream y se omite."
        ),
    ),
    LayerSpec(
        name="conv2d_2",
        rust_name="notes.conv1",
        kernel_tf=(7, 7),
        filters=32,
        in_channels=1,
        stride_tf=(1, 3),
        in_shape=(ANNOT_N_FRAMES, N_FREQ_BINS_CONTOURS, 1),
        pad_same_tf=((3, 3), (2, 2)),
        activation="relu",
        batch_norm=False,
        note="La rama de notas NO tiene BatchNorm.",
    ),
    LayerSpec(
        name="conv2d_3",
        rust_name="notes.conv2",
        kernel_tf=(7, 3),
        filters=1,
        in_channels=32,
        stride_tf=(1, 1),
        in_shape=(ANNOT_N_FRAMES, N_FREQ_BINS_NOTES, 32),
        pad_same_tf=((3, 3), (1, 1)),
        activation="sigmoid",
        batch_norm=False,
        note="Produce `note` tras aplanar, y `x_notes_pre` para la concatenacion.",
    ),
    LayerSpec(
        name="conv2d_4",
        rust_name="onset.conv1",
        kernel_tf=(5, 5),
        filters=32,
        in_channels=8,
        stride_tf=(1, 3),
        in_shape=(ANNOT_N_FRAMES, N_FREQ_BINS_CONTOURS, 8),
        pad_same_tf=((2, 2), (1, 1)),
        activation="relu",
        batch_norm=True,
        note="Recibe directamente la salida de Harmonic Stacking.",
    ),
    LayerSpec(
        name="conv2d_5",
        rust_name="onset.conv2",
        kernel_tf=(3, 3),
        filters=1,
        in_channels=33,
        stride_tf=(1, 1),
        in_shape=(ANNOT_N_FRAMES, N_FREQ_BINS_NOTES, 33),
        pad_same_tf=((1, 1), (1, 1)),
        activation="sigmoid",
        batch_norm=False,
        note="Entrada de 33 canales: x_notes_pre (1) + x_onset (32).",
    ),
]


def verify_padding_table() -> None:
    """Comprueba que la tabla de padding SAME sea consistente.

    Reproduce la formula de TensorFlow para `padding="same"`:

        out      = ceil(in / s)
        pad_tot  = max((out - 1) * s + k - in, 0)
        pad_ant  = pad_tot // 2
        pad_post = pad_tot - pad_ant
    """
    for layer in LAYER_SPECS:
        in_t, in_f, _ = layer.in_shape
        for axis, (in_size, k, s) in enumerate(
            zip((in_t, in_f), layer.kernel_tf, layer.stride_tf)
        ):
            out = -(-in_size // s)
            pad_tot = max((out - 1) * s + k - in_size, 0)
            pad_ant = pad_tot // 2
            pad_post = pad_tot - pad_ant
            expected = layer.pad_same_tf[axis]
            actual = (pad_ant, pad_post)
            if actual != expected:
                raise AssertionError(
                    f"Padding SAME incorrecto en {layer.name} eje {axis}: "
                    f"esperado {expected}, calculado {actual} "
                    f"(in={in_size}, k={k}, s={s})"
                )


#!/usr/bin/env python3
"""Fase 1: validacion de los pesos exportados.

Comprueba el `model.safetensors` y el `cqt_kernels.safetensors` contra la
tabla canonica, contra TensorFlow y contra ONNX.

Las comprobaciones son las del apartado 5.6 del plan:

- Todos los tensores esperados estan presentes.
- Ningun tensor activo fue omitido.
- No existen nombres duplicados.
- No existen dimensiones incompatibles.
- Todos los parametros tienen el tipo de dato esperado.
- Los valores no contienen NaN ni infinitos.
- Los kernels recuperados mediante la transformacion inversa coinciden con los
  originales.
- El archivo Safetensors puede leerse correctamente.
- Cada tensor coincide numericamente con su equivalente ONNX.
- Cada tensor declarado en `tensor_provenance.json` existe en los tres formatos.
- Los kernels de CQT tienen las dimensiones esperadas.

Uso:
    python tools/weight-export/validate_weights.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import model_spec as spec  # noqa: E402
import tf_source  # noqa: E402


class Report:
    """Acumula resultados de comprobaciones."""

    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0
        self.failures: list = []

    def check(self, condition: bool, label: str, detail: str = "") -> bool:
        if condition:
            self.passed += 1
            print(f"  [ OK ] {label}")
        else:
            self.failed += 1
            self.failures.append(f"{label}: {detail}")
            print(f"  [FAIL] {label}  {detail}")
        return bool(condition)

    def section(self, title: str) -> None:
        print()
        print(title)
        print("-" * len(title))


def main() -> int:
    from safetensors.numpy import load_file

    report = Report()

    print("=" * 70)
    print("VALIDACION DE LOS PESOS - basic-pitch / icassp-2022")
    print("=" * 70)

    model_path = spec.MODEL_DIR / "model.safetensors"
    cqt_path = spec.MODEL_DIR / "cqt_kernels.safetensors"

    # ------------------------------------------------------------------ #
    report.section("1. Lectura de los Safetensors")
    report.check(model_path.exists(), "model.safetensors existe", str(model_path))
    report.check(cqt_path.exists(), "cqt_kernels.safetensors existe", str(cqt_path))
    if not (model_path.exists() and cqt_path.exists()):
        print("\nFaltan artefactos. Ejecuta antes export_weights.py.")
        return 1

    tensors = load_file(str(model_path))
    cqt_tensors = load_file(str(cqt_path))
    report.check(len(tensors) == len(spec.TENSOR_SPECS), "model.safetensors tiene todos los tensores",
                 f"{len(tensors)} != {len(spec.TENSOR_SPECS)}")
    report.check(len(cqt_tensors) == len(spec.CQT_KERNEL_SOURCES), "cqt_kernels.safetensors tiene todos los tensores",
                 f"{len(cqt_tensors)} != {len(spec.CQT_KERNEL_SOURCES)}")

    # ------------------------------------------------------------------ #
    report.section("2. Presencia, unicidad y dimensiones")
    expected = {s.rust_name for s in spec.TENSOR_SPECS}
    missing = expected - set(tensors)
    extra = set(tensors) - expected
    report.check(not missing, "No falta ningun tensor activo", f"faltan: {sorted(missing)}")
    report.check(not extra, "No hay tensores sobrantes", f"sobran: {sorted(extra)}")
    report.check(
        len(tensors) == len(set(tensors)), "No hay nombres duplicados"
    )

    for s in spec.TENSOR_SPECS:
        arr = tensors.get(s.rust_name)
        if arr is None:
            report.check(False, f"dims de {s.rust_name}", "tensor ausente")
            continue
        report.check(
            tuple(arr.shape) == s.shape_rust,
            f"dims de {s.rust_name}",
            f"esperado {s.shape_rust}, real {tuple(arr.shape)}",
        )
        report.check(arr.dtype == np.float32, f"dtype de {s.rust_name}", str(arr.dtype))

    for rust_name, shape in spec.CQT_KERNEL_SHAPES.items():
        arr = cqt_tensors.get(rust_name)
        report.check(
            arr is not None and tuple(arr.shape) == shape,
            f"dims de {rust_name}",
            f"esperado {shape}, real {None if arr is None else tuple(arr.shape)}",
        )

    # ------------------------------------------------------------------ #
    report.section("3. Valores finitos")
    bad = [n for n, a in tensors.items() if not np.isfinite(a).all()]
    report.check(not bad, "Todos los tensores del modelo son finitos", str(bad))
    bad_cqt = [n for n, a in cqt_tensors.items() if not np.isfinite(a).all()]
    report.check(not bad_cqt, "Todos los tensores de CQT son finitos", str(bad_cqt))

    # ------------------------------------------------------------------ #
    report.section("4. Reversibilidad de la permutacion")
    _, raw_variables = tf_source.load_saved_model()
    variables, actual_names = tf_source.map_saved_model_variables(raw_variables)

    worst = 0.0
    for s in spec.TENSOR_SPECS:
        tf_value = variables[s.tf_name].numpy()
        rust_value = tensors[s.rust_name]
        if s.transpose is None:
            diff = float(np.abs(rust_value - tf_value).max())
        else:
            # Permutacion inversa: deshacer la permutacion debe recuperar
            # exactamente el tensor original de TensorFlow.
            inverse = np.transpose(rust_value, np.argsort(s.transpose))
            diff = float(np.abs(inverse - tf_value).max())
        worst = max(worst, diff)
    report.check(
        worst == 0.0,
        "La permutacion inversa recupera exactamente los tensores de TensorFlow",
        f"diferencia maxima {worst}",
    )

    # ------------------------------------------------------------------ #
    report.section("5. Equivalencia numerica con ONNX")
    onnx_inits = tf_source.onnx_initializers()
    counterparts = tf_source.identify_onnx_counterparts(variables, onnx_inits)

    worst_direct = 0.0
    worst_folded = 0.0
    n_direct = 0
    n_folded = 0
    for s in spec.TENSOR_SPECS:
        info = counterparts.get(s.tf_name, {})
        relation = info.get("relation")
        diff = info.get("max_abs_diff")
        if relation == "direct":
            n_direct += 1
            if diff is not None:
                worst_direct = max(worst_direct, diff)
        elif relation in ("bn_folded", "bn_folded_bias"):
            n_folded += 1
            if diff is not None:
                worst_folded = max(worst_folded, diff)

    report.check(
        worst_direct == 0.0,
        f"Los {n_direct} tensores con equivalente directo en ONNX coinciden bit a bit",
        f"diferencia maxima {worst_direct}",
    )
    report.check(
        worst_folded < 1e-5,
        f"Los {n_folded} tensores con BN fusionada en ONNX son coherentes",
        f"diferencia maxima {worst_folded:.3e} (redondeo float32 de la fusion)",
    )

    # Toda BatchNorm debe tener sus cuatro parametros individualizados en
    # TensorFlow: en ONNX estan fusionados y no se pueden recuperar.
    n_bn_params = sum(
        1
        for s in spec.TENSOR_SPECS
        if s.role in (spec.BN_GAMMA, spec.BN_BETA, spec.BN_MEAN, spec.BN_VAR)
    )
    n_folded_params = sum(
        1
        for s in spec.TENSOR_SPECS
        if counterparts.get(s.tf_name, {}).get("relation") == "folded_into_conv"
    )
    report.check(
        n_folded_params == n_bn_params,
        "Los 12 parametros de BatchNorm estan fusionados en ONNX",
        f"{n_folded_params} de {n_bn_params}",
    )

    # ------------------------------------------------------------------ #
    report.section("6. Manifiesto de trazabilidad")
    provenance_path = spec.MODEL_DIR / "tensor_provenance.json"
    report.check(provenance_path.exists(), "tensor_provenance.json existe")
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        entries = provenance.get("tensors", [])
        report.check(
            len(entries) == len(spec.TENSOR_SPECS),
            "El manifiesto lista todos los tensores",
            f"{len(entries)} != {len(spec.TENSOR_SPECS)}",
        )
        for entry in entries:
            rust_name = entry["rust_name"]
            ok = (
                rust_name in tensors
                and entry["tf_name"] in raw_variables
                and entry["shape_rust"] == list(tensors[rust_name].shape)
            )
            if not ok:
                report.check(
                    False,
                    f"trazabilidad de {rust_name}",
                    f"tf={entry['tf_name']} rust={rust_name}",
                )
        report.check(
            all(e["rust_name"] in tensors for e in entries),
            "Todos los tensores del manifiesto existen en el Safetensors",
        )
        report.check(
            all(e["tf_name"] in raw_variables for e in entries),
            "Todos los tensores del manifiesto existen en TensorFlow",
        )

    inventory_path = spec.MODEL_DIR / "tensor_inventory.json"
    report.check(inventory_path.exists(), "tensor_inventory.json existe")

    config_path = spec.MODEL_DIR / "model_config.json"
    report.check(config_path.exists(), "model_config.json existe")
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        report.check(
            config["frequency"]["cqt_n_bins"] == spec.CQT_N_BINS,
            "model_config.json declara 309 bins de CQT",
            str(config["frequency"]["cqt_n_bins"]),
        )
        report.check(
            config["concat"]["channels"] == spec.ONSET_CONCAT_CHANNELS,
            "model_config.json declara 33 canales de concatenacion",
            str(config["concat"]["channels"]),
        )
        report.check(
            config["batch_norm"]["epsilon"] == spec.KERAS_BATCH_NORM_EPSILON,
            "model_config.json declara epsilon = 0.001 (Keras)",
            str(config["batch_norm"]["epsilon"]),
        )

    # ------------------------------------------------------------------ #
    report.section("7. Sumas de verificacion")
    sums_path = spec.MODEL_DIR / "SHA256SUMS"
    report.check(sums_path.exists(), "SHA256SUMS existe")
    if sums_path.exists():
        import hashlib

        for line in sums_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            digest, name = line.split("  ", 1)
            target = spec.MODEL_DIR / name
            if not target.exists():
                report.check(False, f"SHA256SUMS: {name}", "no existe")
                continue
            h = hashlib.sha256()
            with open(target, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            report.check(
                h.hexdigest() == digest, f"SHA256 de {name}", "no coincide"
            )

    # ------------------------------------------------------------------ #
    print()
    print("=" * 70)
    if report.failed:
        print(f"RESULTADO: {report.failed} comprobacion(es) fallida(s)")
        for failure in report.failures:
            print(f"  - {failure}")
        return 1
    print(f"RESULTADO: todas las {report.passed} comprobaciones superadas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

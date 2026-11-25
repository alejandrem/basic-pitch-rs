#!/usr/bin/env python
"""Evalua una transcripcion contra un MIDI de referencia, con mir_eval.

Responde a la pregunta que la validacion contra Spotify NO responde: no si el
resultado es igual al del upstream, sino si es **correcto**.

## Tres modos

    # 1. Comparar dos MIDI que ya tienes.
    .venv/bin/python tools/eval/evaluar.py comparar referencia.mid estimado.mid

    # 2. Ida y vuelta: renderiza el MIDI a audio, lo transcribe y compara.
    #    Aqui la referencia SI es la verdad absoluta del audio, venga de donde
    #    venga, porque el audio se genero a partir de ella.
    .venv/bin/python tools/eval/evaluar.py idavuelta referencia.mid
    .venv/bin/python tools/eval/evaluar.py idavuelta referencia.mid --barrido
    .venv/bin/python tools/eval/evaluar.py idavuelta referencia.mid --barrido-onset

    # 3. Audio real contra un MIDI de referencia.
    .venv/bin/python tools/eval/evaluar.py audio cancion.mp3 referencia.mid

Los modos 2 y 3 necesitan el binario compilado: `cargo build --release`.

## Barridos

- `--barrido`: `frame_threshold` 0.2-0.5, con y sin Melodia.
- `--barrido-onset`: `onset_threshold` 0.3-0.9 (necesita un binario con
  `--onset-threshold`).
- Tambien se pueden fijar `--frame-threshold`, `--onset-threshold`,
  `--min-note-length-ms` y `--no-melodia` para una sola configuracion.

## Como interpretar los resultados

- **El modo 3 solo mide exactitud si la referencia es independiente del
  modelo.** Si el MIDI salio de Basic Pitch y luego lo editaste a mano, el
  numero mide cuanto trabajo de limpieza te dejo el modelo, no su exactitud:
  los errores que no notaste cuentan como aciertos.
- **`--limpio` (seno puro) NO aisla el detector del timbre.** El seno es un
  timbre fuera de la distribucion de entrenamiento y el render normaliza el
  nivel global, asi que mezcla varios efectos. Usalo como un caso mas, no como
  una prueba de pureza.
- **Las metricas son las de `mir_eval.transcription`**: onset a +-50 ms y
  pitch a +-50 cents. `F1 onset` ignora el final de la nota; `F1 on+off`
  ademas exige que el final caiga dentro de max(20 % de la duracion, 50 ms).
- **Deriva temporal.** Se emparejan notas con una tolerancia amplia (300 ms)
  y se compara el error de onset (estimado - referencia) del primer y del
  ultimo tercio de la pieza. Un desplazamiento *constante* es normal (el
  upstream aplica un offset de alineacion fijo); lo que delata una deriva es
  la **pendiente**. Positivo = la nota estimada llega mas tarde que la real.
- **Falsos positivos.** Cada nota estimada que no empareja con ninguna de la
  referencia se clasifica segun su relacion con las notas reales cercanas en
  el tiempo (+-100 ms): mismo pitch, armonico superior (+12, +19, +24, +28,
  +31 semitonos), octava por debajo, semitono vecino, etc. Es lo que contrasta
  la hipotesis de que las notas espurias son armonicos del timbre. Las
  categorias se evaluan **en orden** y la primera que cumple gana.
  La columna `azar` repite la clasificacion de los mismos falsos positivos
  con el pitch sorteado de entre los de la referencia. Sin ese control, un
  porcentaje alto de 'armonico superior' no dice nada: en musica real las
  octavas y las quintas abundan y caen a +12 o +19 de alguna nota cercana
  por pura coincidencia.
"""

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

RAIZ = pathlib.Path(__file__).resolve().parents[2]
BINARIO = RAIZ / "target" / "release" / "basic-pitch"

TOLERANCIA_ONSET = 0.05  # 50 ms, el estandar de mir_eval
TOLERANCIA_PITCH = 50.0  # cents
TOLERANCIA_DERIVA = 0.30  # para emparejar notas al medir la deriva
UMBRALES_BARRIDO = (0.2, 0.3, 0.4, 0.5)
UMBRALES_ONSET = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
REPETICIONES_CONTROL = 50  # sorteos del control de falsos positivos

# Falsos positivos: una nota estimada se relaciona con las de referencia cuyo
# onset cae a menos de VENTANA_RELACION segundos.
VENTANA_RELACION = 0.10
# Semitonos sobre la fundamental: octava, octava+quinta, 2 octavas, ...
ARMONICOS = (12, 19, 24, 28, 31)
ETIQUETAS_FP = {
    "mismo_pitch_cerca": "mismo pitch, onset a 50-100 ms o duplicado",
    "reataque_en_nota": "mismo pitch, re-ataque dentro de una nota real",
    "tras_el_final": "mismo pitch, <= 0.5 s despues del final de una nota",
    "armonico_superior": "armonico superior (+12, +19, +24, +28, +31)",
    "octava_abajo": "una o dos octavas por debajo",
    "semitono_vecino": "semitono vecino (+-1)",
    "otra_nota_sonando": "otra nota real estaba sonando",
    "sin_relacion": "sin relacion con ninguna nota real",
}


def a_hz(pitches):
    """De numero de nota MIDI a Hz, que es lo que pide mir_eval."""
    return 440.0 * 2.0 ** ((np.asarray(pitches, dtype=float) - 69.0) / 12.0)


def leer_notas(ruta):
    """Lee un MIDI y devuelve (intervalos [n, 2], pitches [n]) ordenados."""
    import pretty_midi

    midi = pretty_midi.PrettyMIDI(str(ruta))
    notas = [n for inst in midi.instruments if not inst.is_drum for n in inst.notes]
    notas.sort(key=lambda n: (n.start, n.pitch))
    intervalos = np.array(
        [[n.start, max(n.end, n.start + 1e-3)] for n in notas], dtype=float
    ).reshape(-1, 2)
    pitches = np.array([n.pitch for n in notas], dtype=float)
    return intervalos, pitches


def metricas(ref, est):
    """Precision, recall y F1 de mir_eval, solo con onset y con onset+offset."""
    import mir_eval

    ri, rp = ref
    ei, ep = est
    if len(ri) == 0 or len(ei) == 0:
        return {
            "notas_ref": len(ri), "notas_est": len(ei),
            "p_onset": 0.0, "r_onset": 0.0, "f1_onset": 0.0,
            "p_nota": 0.0, "r_nota": 0.0, "f1_nota": 0.0,
        }

    p_on, r_on, f_on, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ri, a_hz(rp), ei, a_hz(ep),
        onset_tolerance=TOLERANCIA_ONSET,
        pitch_tolerance=TOLERANCIA_PITCH,
        offset_ratio=None,
    )
    p_no, r_no, f_no, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ri, a_hz(rp), ei, a_hz(ep),
        onset_tolerance=TOLERANCIA_ONSET,
        pitch_tolerance=TOLERANCIA_PITCH,
        offset_ratio=0.2,
        offset_min_tolerance=0.05,
    )
    return {
        "notas_ref": len(ri), "notas_est": len(ei),
        "p_onset": float(p_on), "r_onset": float(r_on), "f1_onset": float(f_on),
        "p_nota": float(p_no), "r_nota": float(r_no), "f1_nota": float(f_no),
    }


def deriva(ref, est):
    """Error de onset (estimado - referencia) al principio y al final."""
    import mir_eval

    ri, rp = ref
    ei, ep = est
    if len(ri) == 0 or len(ei) == 0:
        return None

    pares = mir_eval.transcription.match_notes(
        ri, a_hz(rp), ei, a_hz(ep),
        onset_tolerance=TOLERANCIA_DERIVA,
        pitch_tolerance=TOLERANCIA_PITCH,
        offset_ratio=None,
    )
    if len(pares) < 10:
        return None

    r_idx = np.array([p[0] for p in pares])
    e_idx = np.array([p[1] for p in pares])
    t = ri[r_idx, 0]
    error = ei[e_idx, 0] - t
    orden = np.argsort(t)
    t, error = t[orden], error[orden]

    tercio = len(t) // 3
    # Pendiente robusta: diferencia de medianas entre el ultimo y el primer
    # tercio, dividida por la distancia entre sus tiempos medianos. Un ajuste
    # por minimos cuadrados se deja arrastrar por los pocos emparejamientos
    # erroneos que admite una tolerancia amplia (300 ms).
    dt = np.median(t[-tercio:]) - np.median(t[:tercio])
    if dt > 0:
        pendiente = (np.median(error[-tercio:]) - np.median(error[:tercio])) / dt
    else:
        pendiente = 0.0  # segundos de error por segundo

    return {
        "emparejadas": len(pares),
        "de": len(ri),
        "primer_tercio_ms": float(np.median(error[:tercio])) * 1000.0,
        "ultimo_tercio_ms": float(np.median(error[-tercio:])) * 1000.0,
        "pendiente_ms_por_min": float(pendiente) * 60.0 * 1000.0,
    }


def categoria_fp(ri, rp, e0, p):
    """Categoria de una nota estimada (onset e0, pitch p) frente a la referencia."""
    cerca = np.abs(ri[:, 0] - e0) <= VENTANA_RELACION
    dif = p - rp[cerca]  # semitonos sobre las notas reales cercanas
    mismo = rp == p

    if np.any(dif == 0):
        return "mismo_pitch_cerca"
    if np.any(mismo & (ri[:, 0] < e0) & (e0 < ri[:, 1])):
        return "reataque_en_nota"
    if np.any(mismo & (e0 >= ri[:, 1]) & (e0 - ri[:, 1] <= 0.5)):
        return "tras_el_final"
    if np.any(np.isin(dif, ARMONICOS)):
        return "armonico_superior"
    if np.any(np.isin(dif, (-12, -24))):
        return "octava_abajo"
    if np.any(np.abs(dif) == 1):
        return "semitono_vecino"
    if np.any((ri[:, 0] <= e0) & (e0 <= ri[:, 1])):
        return "otra_nota_sonando"
    return "sin_relacion"


def clasificar_falsos_positivos(ref, est):
    """Clasifica cada nota estimada que no empareja con ninguna real.

    Las categorias se evaluan en el orden de `ETIQUETAS_FP`; gana la primera.
    """
    import mir_eval

    ri, rp = ref
    ei, ep = est
    if len(ri) == 0 or len(ei) == 0:
        return None

    pares = mir_eval.transcription.match_notes(
        ri, a_hz(rp), ei, a_hz(ep),
        onset_tolerance=TOLERANCIA_ONSET,
        pitch_tolerance=TOLERANCIA_PITCH,
        offset_ratio=None,
    )
    emparejadas = {e for _, e in pares}
    falsos = [j for j in range(len(ei)) if j not in emparejadas]

    cuentas = {clave: 0 for clave in ETIQUETAS_FP}
    for j in falsos:
        cuentas[categoria_fp(ri, rp, ei[j, 0], ep[j])] += 1

    # Control: los mismos falsos positivos, con el pitch sorteado de la
    # distribucion de pitches de la referencia. Responde a "cuanto saldria por
    # azar?" para cada categoria.
    rng = np.random.default_rng(0)
    control = {clave: 0.0 for clave in ETIQUETAS_FP}
    if falsos:
        for _ in range(REPETICIONES_CONTROL):
            for j in falsos:
                p_azar = float(rng.choice(rp))
                control[categoria_fp(ri, rp, ei[j, 0], p_azar)] += 1
        for clave in control:
            control[clave] /= REPETICIONES_CONTROL

    return {"total": len(falsos), "de": len(ei), "categorias": cuentas, "control": control}


def resultado(ref, est, cfg):
    return {
        "config": cfg,
        "metricas": metricas(ref, est),
        "deriva": deriva(ref, est),
        "falsos_positivos": clasificar_falsos_positivos(ref, est),
    }


def transcribir(audio, salida, cfg, binario):
    """Llama al CLI de Rust. Un valor None en `cfg` deja el defecto."""
    binario = pathlib.Path(binario)
    if not binario.exists():
        raise SystemExit(
            f"no existe {binario}. Compila primero con: cargo build --release"
        )
    orden = [str(binario), "transcribe", str(audio), "-o", str(salida)]
    for clave, flag in (
        ("frame_threshold", "--frame-threshold"),
        ("onset_threshold", "--onset-threshold"),
        ("min_note_length_ms", "--min-note-length-ms"),
    ):
        if cfg[clave] is not None:
            orden += [flag, str(cfg[clave])]
    if not cfg["melodia"]:
        orden.append("--no-melodia")

    ejecucion = subprocess.run(orden, cwd=RAIZ, capture_output=True, text=True)
    if ejecucion.returncode != 0:
        raise SystemExit(f"fallo la transcripcion:\n{ejecucion.stderr}")


def configuraciones(args):
    base = {
        "frame_threshold": args.frame_threshold,
        "onset_threshold": args.onset_threshold,
        "min_note_length_ms": args.min_note_length_ms,
        "melodia": not args.no_melodia,
    }
    if args.barrido:
        return [
            dict(base, frame_threshold=t, melodia=mel)
            for mel in (True, False)
            for t in UMBRALES_BARRIDO
        ]
    if args.barrido_onset:
        return [dict(base, onset_threshold=t) for t in UMBRALES_ONSET]
    return [base]


def etiqueta(cfg):
    def f(v):
        return "defecto" if v is None else f"{v:g}"

    return (
        f"frame={f(cfg['frame_threshold'])} onset={f(cfg['onset_threshold'])} "
        f"melodia={'si' if cfg['melodia'] else 'no'}"
    )


def evaluar(audio, ref_mid, args, trabajo):
    ref = leer_notas(ref_mid)
    resultados = []
    for i, cfg in enumerate(configuraciones(args)):
        salida = pathlib.Path(trabajo) / f"est_{i}.mid"
        print(f"transcribiendo ({etiqueta(cfg)})...", flush=True)
        transcribir(audio, salida, cfg, args.bin)
        resultados.append(resultado(ref, leer_notas(salida), cfg))
    return resultados


def tabla(resultados):
    def f(v):
        return "defecto" if v is None else f"{v:g}"

    cabecera = (
        "frame", "onset", "melodia", "ref", "est",
        "P onset", "R onset", "F1 onset", "F1 on+off",
    )
    filas = [cabecera]
    for r in resultados:
        m, c = r["metricas"], r["config"]
        filas.append((
            f(c["frame_threshold"]),
            f(c["onset_threshold"]),
            "si" if c["melodia"] else "no",
            str(m["notas_ref"]),
            str(m["notas_est"]),
            f"{m['p_onset']:.3f}",
            f"{m['r_onset']:.3f}",
            f"{m['f1_onset']:.3f}",
            f"{m['f1_nota']:.3f}",
        ))
    anchos = [max(len(fila[i]) for fila in filas) for i in range(len(cabecera))]
    for n, fila in enumerate(filas):
        print("| " + " | ".join(c.ljust(w) for c, w in zip(fila, anchos)) + " |")
        if n == 0:
            print("|" + "|".join("-" * (w + 2) for w in anchos) + "|")


def mostrar_falsos_positivos(r):
    fp = r["falsos_positivos"]
    if fp is None:
        return
    total = fp["total"]
    print()
    print(
        f"Falsos positivos ({etiqueta(r['config'])}): "
        f"{total} de {fp['de']} notas estimadas"
    )
    if total == 0:
        return
    print(f"  {'n':>5}  {'real':>6}  {'azar':>6}")
    for clave, texto in ETIQUETAS_FP.items():
        n = fp["categorias"][clave]
        azar = 100.0 * fp["control"][clave] / total
        print(f"  {n:5d}  {100.0 * n / total:5.1f}%  {azar:5.1f}%  {texto}")
    armonicos = 100.0 * fp["categorias"]["armonico_superior"] / total
    armonicos_azar = 100.0 * fp["control"]["armonico_superior"] / total
    print(
        f"  -> armonicos superiores: {armonicos:.1f} % reales frente a "
        f"{armonicos_azar:.1f} % esperable por azar"
    )
    print("     (azar = los mismos falsos positivos con el pitch sorteado de los de la referencia)")


def mostrar(resultados):
    print()
    tabla(resultados)

    mejor = max(resultados, key=lambda r: r["metricas"]["f1_onset"])
    mostrar_falsos_positivos(mejor)

    d = mejor["deriva"]
    print()
    if d is None:
        print("Deriva temporal: no hay notas emparejadas suficientes para medirla.")
        return
    print(
        f"Deriva temporal ({etiqueta(mejor['config'])}; {d['emparejadas']} de {d['de']} "
        f"notas emparejadas a {int(TOLERANCIA_DERIVA * 1000)} ms):"
    )
    print(f"  error mediano de onset, primer tercio: {d['primer_tercio_ms']:+7.1f} ms")
    print(f"  error mediano de onset, ultimo tercio: {d['ultimo_tercio_ms']:+7.1f} ms")
    print(f"  pendiente (robusta):                   {d['pendiente_ms_por_min']:+7.1f} ms por minuto")


def guardar_json(ruta, resultados):
    if ruta:
        pathlib.Path(ruta).write_text(json.dumps(resultados, indent=2), encoding="utf-8")
        print(f"\nresultados guardados en {ruta}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="modo", required=True)

    def comunes(p):
        p.add_argument("--barrido", action="store_true",
                       help="frame_threshold 0.2-0.5, con y sin Melodia")
        p.add_argument("--barrido-onset", action="store_true",
                       help="onset_threshold 0.3-0.9 (necesita --onset-threshold en el binario)")
        p.add_argument("--frame-threshold", type=float, default=None)
        p.add_argument("--onset-threshold", type=float, default=None)
        p.add_argument("--min-note-length-ms", type=float, default=None)
        p.add_argument("--no-melodia", action="store_true")
        p.add_argument("--bin", default=str(BINARIO), help="ruta del binario basic-pitch")
        p.add_argument("--json", help="guarda los resultados en este archivo")

    c = sub.add_parser("comparar", help="compara dos MIDI")
    c.add_argument("referencia")
    c.add_argument("estimado")
    c.add_argument("--json")

    i = sub.add_parser("idavuelta", help="render -> transcribe -> compara")
    i.add_argument("referencia", help="MIDI que se renderiza y sirve de verdad")
    i.add_argument("--limpio", action="store_true", help="seno puro, sin armonicos")
    comunes(i)

    a = sub.add_parser("audio", help="audio real contra un MIDI de referencia")
    a.add_argument("audio")
    a.add_argument("referencia")
    comunes(a)

    args = ap.parse_args()

    if args.modo == "comparar":
        cfg = {
            "frame_threshold": None, "onset_threshold": None,
            "min_note_length_ms": None, "melodia": True,
        }
        resultados = [resultado(leer_notas(args.referencia), leer_notas(args.estimado), cfg)]
        mostrar(resultados)
        guardar_json(args.json, resultados)
        return 0

    if args.modo == "idavuelta":
        sys.path.insert(0, str(RAIZ / "tools" / "weight-export"))
        import render_midi

        with tempfile.TemporaryDirectory() as trabajo:
            wav = pathlib.Path(trabajo) / "render.wav"
            if args.limpio:
                render_midi.render_limpio(pathlib.Path(args.referencia), wav, 44100)
            else:
                render_midi.render_con_soundfont(pathlib.Path(args.referencia), wav, 44100)
            resultados = evaluar(wav, args.referencia, args, trabajo)
        mostrar(resultados)
        guardar_json(args.json, resultados)
        return 0

    # modo "audio"
    print(
        "AVISO: si la referencia salio del propio modelo (aunque la hayas "
        "editado), estas cifras miden cuanto trabajo de limpieza dejo, no la "
        "exactitud del modelo.\n"
    )
    with tempfile.TemporaryDirectory() as trabajo:
        resultados = evaluar(args.audio, args.referencia, args, trabajo)
    mostrar(resultados)
    guardar_json(args.json, resultados)
    return 0


if __name__ == "__main__":
    sys.exit(main())

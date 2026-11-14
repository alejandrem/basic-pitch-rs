#!/usr/bin/env python
"""Los 13 casos de prueba de la Fase 7 (apartado 7.4).

Genera, para cada caso:

- el audio de prueba (una senal controlada y determinista),
- los `.npy` con las tres salidas finales del modelo (note, onset, contour),
- las notas detectadas por el UPSTREAM, en JSON.

Y con eso se puede comprobar que Rust y TensorFlow coinciden en los trece
casos, no solo en uno.

Los casos son:

    1.  silencio
    2.  nota sostenida
    3.  nota corta
    4.  varias notas simultaneas
    5.  acordes
    6.  notas graves
    7.  notas agudas
    8.  transitorios rapidos
    9.  vibrato
    10. deslizamientos de afinacion
    11. audio con ruido
    12. audio de larga duracion
    13. notas cerca de los limites entre ventanas

Salida: `reference/casos/`
"""

import json
import pathlib
import sys

import numpy as np
import soundfile as sf

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "reference" / "casos"

#: Frecuencia de muestreo: la del modelo, para no remuestrear.
SR = 22050


# ---------------------------------------------------------------------------
# Utilidades de sintesis
# ---------------------------------------------------------------------------


def envolvente(t: np.ndarray, ataque: float = 0.005, caida: float = 0.08) -> np.ndarray:
    """Envolvente de tipo piano: ataque rapido y caida exponencial."""
    env = np.exp(-t / caida)
    n_ataque = max(int(ataque * SR), 1)
    rampa = np.linspace(0.0, 1.0, min(n_ataque, len(env)))
    env[: len(rampa)] *= rampa
    return env


def tono(
    freq: float,
    duracion: float,
    amp: float = 0.5,
    armonicos: int = 4,
    fase: float = 0.0,
) -> np.ndarray:
    """Un tono con armónicos, que es lo que el modelo reconoce como nota.

    Un seno puro se transcribe peor (el modelo oye timbre, no notas), asi
    que los casos usan un timbre con armónicos, como un piano.
    """
    n = int(duracion * SR)
    t = np.arange(n) / SR
    onda = np.zeros(n)
    for k in range(1, armonicos + 1):
        onda += (0.5 ** (k - 1)) * np.sin(2 * np.pi * freq * k * t + fase * k)
    return amp * onda * envolvente(t)


def mix(clips: list[tuple[float, np.ndarray]]) -> np.ndarray:
    """Mezcla clips con su tiempo de inicio."""
    # `inicio` va en segundos y `clip` en muestras: hay que convertir.
    total = int(max(inicio * SR + len(clip) for inicio, clip in clips) + 0.5 * SR)
    buffer = np.zeros(total)
    for inicio, clip in clips:
        i = int(inicio * SR)
        buffer[i : i + len(clip)] += clip
    pico = np.abs(buffer).max()
    if pico > 0.95:
        buffer *= 0.95 / pico
    return buffer


def silencio(duracion: float, nivel: float = 1e-4) -> np.ndarray:
    """Silencio con un residuo minimo (un silencio digital absoluto es
    degenerado para el modelo)."""
    rng = np.random.default_rng(20240517)
    return rng.normal(0.0, nivel, int(duracion * SR)).astype(np.float64)


def vibrato(duracion: float, freq: float, cents: float = 30.0, hz: float = 5.0) -> np.ndarray:
    """Una nota con vibrato: la frecuencia oscila alrededor de `freq`."""
    n = int(duracion * SR)
    t = np.arange(n) / SR
    modulacion = cents / 100.0 * np.sin(2 * np.pi * hz * t)
    freq_inst = freq * 2 ** (modulacion / 12.0)
    fase = 2 * np.pi * np.cumsum(freq_inst) / SR
    onda = np.zeros(n)
    for k in range(1, 5):
        onda += (0.5 ** (k - 1)) * np.sin(k * fase)
    return 0.5 * onda * envolvente(t)


def glissando(duracion: float, f0: float, f1: float) -> np.ndarray:
    """Un deslizamiento de afinacion de `f0` a `f1`."""
    n = int(duracion * SR)
    t = np.arange(n) / SR
    freq = np.linspace(f0, f1, n)
    fase = 2 * np.pi * np.cumsum(freq) / SR
    onda = np.zeros(n)
    for k in range(1, 5):
        onda += (0.5 ** (k - 1)) * np.sin(k * fase)
    return 0.5 * onda * envolvente(t, ataque=0.01, caida=0.2)


#: Las 88 notas del modelo, en Hz.
def midi_a_hz(nota: int) -> float:
    return 440.0 * 2 ** ((nota - 69) / 12.0)


# ---------------------------------------------------------------------------
# Los 13 casos
# ---------------------------------------------------------------------------


def caso_silencio() -> np.ndarray:
    return silencio(2.0)


def caso_nota_sostenida() -> np.ndarray:
    return mix([(0.2, tono(midi_a_hz(69), 1.6))])  # La4


def caso_nota_corta() -> np.ndarray:
    return mix([(0.5, tono(midi_a_hz(69), 0.1)), (1.2, tono(midi_a_hz(72), 0.1))])


def caso_notas_simultaneas() -> np.ndarray:
    return mix([(0.3, tono(midi_a_hz(60), 1.2)), (0.3, tono(midi_a_hz(64), 1.2)), (0.3, tono(midi_a_hz(67), 1.2))])


def caso_acordes() -> np.ndarray:
    clips = []
    for i, (inicio, acorde) in enumerate(
        [(0.0, [48, 52, 55]), (0.7, [50, 53, 57]), (1.4, [52, 55, 59])]
    ):
        for nota in acorde:
            clips.append((inicio, tono(midi_a_hz(nota), 0.65, amp=0.35)))
    return mix(clips)


def caso_notas_graves() -> np.ndarray:
    return mix(
        [
            (0.2, tono(midi_a_hz(36), 0.8, amp=0.5)),   # C2
            (1.1, tono(midi_a_hz(40), 0.8, amp=0.5)),   # E2
            (2.0, tono(midi_a_hz(43), 0.8, amp=0.5)),   # G2
        ]
    )


def caso_notas_agudas() -> np.ndarray:
    return mix(
        [
            (0.2, tono(midi_a_hz(96), 0.8, amp=0.4)),   # C7
            (1.1, tono(midi_a_hz(98), 0.8, amp=0.4)),   # D7
            (2.0, tono(midi_a_hz(100), 0.8, amp=0.4)),  # E7
        ]
    )


def caso_transitorios_rapidos() -> np.ndarray:
    clips = []
    for i in range(16):
        clips.append((0.1 + i * 0.11, tono(midi_a_hz(60 + (i % 5)), 0.06, amp=0.45)))
    return mix(clips)


def caso_vibrato() -> np.ndarray:
    return mix([(0.3, vibrato(1.8, midi_a_hz(69)))])


def caso_deslizamiento() -> np.ndarray:
    return mix([(0.2, glissando(1.8, midi_a_hz(60), midi_a_hz(72)))])


def caso_ruido() -> np.ndarray:
    rng = np.random.default_rng(20240518)
    nota = tono(midi_a_hz(69), 1.8, amp=0.4)
    ruido = rng.normal(0.0, 0.03, len(nota))
    return mix([(0.2, nota + ruido)])


def caso_duracion_larga() -> np.ndarray:
    # 30 segundos: varias ventanas de una vez.
    clips = []
    rng = np.random.default_rng(20240519)
    for i in range(60):
        clips.append((0.5 + i * 0.48, tono(midi_a_hz(55 + rng.integers(0, 12)), 0.42, amp=0.4)))
    return mix(clips)


def caso_limites_ventana() -> np.ndarray:
    """Notas justo antes y despues del limite entre ventanas.

    El salto entre ventanas es de 36.164 muestras = 1.6402 s. Se colocan
    notas a ambos lados para que la reconstruccion tenga que decidir si
    las une o las separa.
    """
    limite = 36_164 / SR
    clips = []
    for i, nota in enumerate([60, 62, 64, 65]):
        clips.append((limite - 0.35 + i * 0.05, tono(midi_a_hz(nota), 0.3, amp=0.45)))
        clips.append((limite + 0.15 + i * 0.05, tono(midi_a_hz(nota + 4), 0.3, amp=0.45)))
    return mix(clips)


CASOS = {
    "silencio": caso_silencio,
    "nota_sostenida": caso_nota_sostenida,
    "nota_corta": caso_nota_corta,
    "notas_simultaneas": caso_notas_simultaneas,
    "acordes": caso_acordes,
    "notas_graves": caso_notas_graves,
    "notas_agudas": caso_notas_agudas,
    "transitorios_rapidos": caso_transitorios_rapidos,
    "vibrato": caso_vibrato,
    "deslizamiento": caso_deslizamiento,
    "ruido": caso_ruido,
    "duracion_larga": caso_duracion_larga,
    "limites_ventana": caso_limites_ventana,
}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    indice = {}
    for nombre, constructor in CASOS.items():
        audio = constructor()
        ruta = OUT_DIR / f"{nombre}.wav"
        sf.write(str(ruta), audio.astype(np.float32), SR, subtype="PCM_16")
        indice[nombre] = {
            "audio": ruta.name,
            "muestras": int(len(audio)),
            "duracion_s": len(audio) / SR,
            "sample_rate": SR,
        }
        print(f"  {nombre:22s} {len(audio) / SR:6.2f} s  {ruta.name}")

    (OUT_DIR / "index.json").write_text(
        json.dumps(
            {
                "descripcion": (
                    "Las 13 senales controladas del apartado 7.4 de la Fase 7. "
                    "Se generan con `make_cases.py`."
                ),
                "nota": (
                    "Tienen un timbre conarmonicos (como un piano) porque el modelo "
                    "oye timbre, no notas: un seno puro se transcribe peor."
                ),
                "puntos": indice,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\n  {len(CASOS)} casos en {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

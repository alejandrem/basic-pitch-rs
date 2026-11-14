#!/usr/bin/env python
"""Genera los fixtures de audio multiformato de los tests.

Crea el MISMO contenido en WAV, FLAC, MP3, OGG y M4A, para que los tests
puedan comprobar que el decodificador de Rust maneja cada formato.

La senal es deliberadamente adversarial para el resto de la matriz de
pruebas:

- estereo con canales distintos (izquierdo: 220 Hz, derecho: 330 Hz con
  amplitud distinta), para que el promedio a mono tenga sentido;
- un transitorio (impulso) en la primera muestra, para probar el
  tratamiento de los extremos;
- silencio al final, para probar el relleno final de las ventanas;
- amplitud cercana al limite, sin llegar a clipping.

Uso:
    .venv/bin/python tools/weight-export/make_audio_fixtures.py
"""

import pathlib
import subprocess
import sys
import wave

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "crates" / "audio-preprocess" / "tests" / "resources"
SAMPLE_RATE = 44100
DURATION_S = 1.0


def build_signal() -> np.ndarray:
    """Senal stereo de 1 segundo, con impulso inicial y silencio final."""
    n = int(SAMPLE_RATE * DURATION_S)
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE

    # Canal izquierdo: 220 Hz con envolvente suave.
    left = 0.55 * np.sin(2 * np.pi * 220.0 * t) * np.minimum(1.0, t / 0.05)
    # Canal derecho: 330 Hz, con menos amplitud y un poco desfasado.
    right = 0.35 * np.sin(2 * np.pi * 330.0 * t + 0.7) * np.minimum(1.0, t / 0.05)

    stereo = np.stack([left, right], axis=1)

    # Impulso en la primera muestra (transitorio al inicio).
    stereo[0, 0] += 0.9
    stereo[0, 1] -= 0.9

    # Silencio en el ultimo 10 % (silencio al final).
    stereo[int(n * 0.9) :, :] = 0.0

    # Se queda cerca del limite de amplitud, sin pasarse.
    peak = np.abs(stereo).max()
    if peak > 0.99:
        stereo *= 0.99 / peak

    return stereo.astype(np.float64)


def write_wav(path: pathlib.Path, stereo: np.ndarray) -> None:
    """Escribe un WAV PCM de 16 bits."""
    datos = np.clip(stereo, -1.0, 1.0)
    pcm = (datos * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())


def ffmpeg(source: pathlib.Path, destino: pathlib.Path, codec: list[str]) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(source), *codec, str(destino)],
        check=True,
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stereo = build_signal()

    wav = OUT_DIR / "tono.wav"
    write_wav(wav, stereo)
    print(f"  {wav.name}  ({wav.stat().st_size} bytes)")

    # FLAC: sin perdidas.
    flac = OUT_DIR / "tono.flac"
    ffmpeg(wav, flac, ["-c:a", "flac"])
    print(f"  {flac.name}  ({flac.stat().st_size} bytes)")

    # MP3: con perdidas, 128 kbps.
    mp3 = OUT_DIR / "tono.mp3"
    ffmpeg(wav, mp3, ["-c:a", "libmp3lame", "-b:a", "128k"])
    print(f"  {mp3.name}  ({mp3.stat().st_size} bytes)")

    # OGG Vorbis: con perdidas.
    ogg = OUT_DIR / "tono.ogg"
    ffmpeg(wav, ogg, ["-c:a", "libvorbis", "-b:a", "128k"])
    print(f"  {ogg.name}  ({ogg.stat().st_size} bytes)")

    # M4A: AAC, con perdidas.
    m4a = OUT_DIR / "tono.m4a"
    ffmpeg(wav, m4a, ["-c:a", "aac", "-b:a", "128k"])
    print(f"  {m4a.name}  ({m4a.stat().st_size} bytes)")

    # La extension equivocada NO se guarda como fixture: son 176 KB
    # duplicados. El test la crea en un directorio temporal, que es donde
    # tiene sentido: es un caso raro, no un recurso del proyecto.

    return 0


if __name__ == "__main__":
    sys.exit(main())

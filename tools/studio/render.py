"""Editing page render: one auto-flow result, re-voiced and re-mixed on this box.

    ~/.local/share/remifasol-autoflow/venv/bin/python tools/studio/render.py <out folder> <carpeta> '<params json>'

Input is final/<carpeta>/fuente.wav (her cleaned voice, before conversion).
params: voz (model in logs/), mezcla 0-100 (0 = only Quetzaly, 100 = only the
synthetic voice), afinacion 0-100 (pitch snap on the synthetic voice; 100 is
the obvious auto-tune), tono -6..6 semitones, pulido (a pipeline.CHAINS key),
lufs (loudness target).

1. Voice   RVC with autotune; cached per (voz, afinacion, tono) so moving
           only the blend or the polish does not convert again.
2. Blend   both voices at 48 kHz, the synthetic one aligned to hers by
           cross-correlation, mixed at the ratio with equal-power gains.
3. Polish  SoundCraft headless chain, bounced 24-bit at the target loudness.
4. Deliver final/<carpeta>/ediciones/<id>/: voz.wav (24-bit 48 kHz), voz.flac,
           escenas/*.wav and a zip of everything. Lossless only.
"""

import datetime
import json
import math
import os
import sys
import tempfile
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as pl  # noqa: E402
from pipeline import RATE, run, run_bytes, say  # noqa: E402


def read_f32(path):
    import numpy as np
    return np.frombuffer(run_bytes(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(RATE),
                                    "-f", "f32le", "-"]), np.float32).copy()


def write_f32(path, audio):
    import numpy as np
    raw = os.path.join(os.path.dirname(path), os.path.basename(path) + ".f32")
    np.asarray(audio, np.float32).tofile(raw)
    run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(RATE), "-ac", "1", "-i", raw,
         "-c:a", "pcm_f32le", path])
    os.remove(raw)


def align(ref, other, max_shift_ms=120):
    """Shift `other` so it lines up with `ref`. RVC keeps her timing, so this is
    normally 0; it guards against a model that adds latency. Matched on the
    loudness envelope at 1 ms steps: two different timbres have unrelated
    waveforms (their raw cross-correlation picks a random few-ms shift, which
    smears the blend), but they rise and fall on the same syllables."""
    import numpy as np
    hop = RATE // 1000

    def env(x):
        n = len(x) // hop
        return np.log(np.sqrt((x[:n * hop].reshape(n, hop) ** 2).mean(1)) + 1e-4)

    eo, es = env(ref), env(other)
    m = max_shift_ms
    n = min(len(eo), len(es)) - 2 * m
    if n < 2000:
        return other
    a = eo[m:m + n]
    best_lag, best_c = 0, -1.0
    for lag in range(-m, m + 1):
        c = float(np.corrcoef(a, es[m + lag:m + lag + n])[0, 1])
        if c > best_c:
            best_lag, best_c = lag, c
    if best_c < 0.6 or best_lag == 0:
        return other
    shift = -best_lag * hop  # other is late by best_lag ms -> drop its head
    if shift > 0:
        return np.concatenate([np.zeros(shift, np.float32), other])
    return other[-shift:]


def main(folder, carpeta, params):
    import numpy as np

    src_dir = os.path.join(folder, "final", carpeta)
    fuente = os.path.join(src_dir, "fuente.wav")
    if not os.path.isfile(fuente):
        raise SystemExit("ese resultado no tiene fuente.wav (procésalo de nuevo en la cabina)")
    voz = params.get("voz", "US_Female_1")
    mezcla = max(0, min(100, int(params.get("mezcla", 50)))) / 100
    afin = max(0, min(100, int(params.get("afinacion", 0)))) / 100
    semis = max(-6, min(6, int(params.get("tono", 0))))
    pulido = params.get("pulido", "natural")
    lufs = float(params.get("lufs", pl.TARGET_LUFS))
    if pulido not in pl.CHAINS:
        raise SystemExit("pulido desconocido")
    if not -30 <= lufs <= -8:
        raise SystemExit("loudness fuera de rango")

    rid = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = os.path.join(src_dir, "ediciones", rid)
    os.makedirs(out)
    work = tempfile.mkdtemp(prefix="studio-render-")

    original = read_f32(fuente)
    mix = original
    if mezcla > 0:
        if voz not in pl.voices():
            raise SystemExit("no existe ese modelo de voz")
        cache = os.path.join(src_dir, "cache", f"{voz}-a{int(afin * 100)}-t{semis}.wav")
        if not os.path.isfile(cache):
            say("voz", msg=f"Convirtiendo con {voz} (afinación {int(afin * 100)} %, tono {semis:+d})… 1–3 min")
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            tmp = os.path.join(work, "rvc.wav")
            pl.convert(fuente, tmp, voz, autotune=afin, pitch=semis)
            run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-ac", "1", "-ar", str(RATE), "-c:a", "pcm_f32le", cache])
        else:
            say("voz", msg="Voz sintética ya convertida con esos ajustes (caché).")
        synth = align(original, read_f32(cache))
        n = max(len(original), len(synth))
        o = np.pad(original, (0, n - len(original)))
        s = np.pad(synth, (0, n - len(synth)))
        # match loudness before mixing so the slider means what it says
        ro, rs = float(np.sqrt(np.mean(o ** 2)) + 1e-9), float(np.sqrt(np.mean(s ** 2)) + 1e-9)
        s = s * (ro / rs)
        go, gs = math.cos(mezcla * math.pi / 2), math.sin(mezcla * math.pi / 2)
        mix = go * o + gs * s
        peak = float(np.max(np.abs(mix)) + 1e-9)
        if peak > 0.98:
            mix = mix * (0.98 / peak)
    say("mezclar", msg=f"Mezcla: {int((1 - mezcla) * 100)} % Quetzaly · {int(mezcla * 100)} % sintética")
    mixed = os.path.join(work, "mezcla.wav")
    write_f32(mixed, mix)

    say("pulir", msg=f"Puliendo en SoundCraft ({pulido}, {lufs:g} LUFS)…")
    full = os.path.join(out, "voz.wav")
    got, gain = pl.polish(mixed, full, chain=pulido, target=lufs)

    say("entregar", msg="Preparando descargas sin pérdida (WAV 24-bit, FLAC, escenas)…")
    run(["ffmpeg", "-v", "error", "-y", "-i", full, "-c:a", "flac", "-sample_fmt", "s32", "-bits_per_raw_sample", "24",
         os.path.join(out, "voz.flac")])
    cortes_path = os.path.join(src_dir, "cortes.json")
    escenas = []
    if os.path.isfile(cortes_path):
        escenas = pl.cut_scenes(full, json.load(open(cortes_path, encoding="utf-8")), os.path.join(out, "escenas"))
    info = {"id": rid, "carpeta": carpeta, "voz": voz if mezcla > 0 else "original", "mezcla": int(mezcla * 100),
            "afinacion": int(afin * 100), "tono": semis, "pulido": pulido, "lufs_objetivo": lufs,
            "lufs": round(got, 1), "duracion": round(pl.duration(full), 1),
            "escenas": [os.path.basename(e) for e in escenas]}
    json.dump(info, open(os.path.join(out, "parametros.json"), "w"), ensure_ascii=False, indent=1)
    with zipfile.ZipFile(os.path.join(out, f"BeautyCita-edicion-{rid}.zip"), "w", zipfile.ZIP_STORED) as z:
        z.write(full, "voz.wav")
        z.write(os.path.join(out, "voz.flac"), "voz.flac")
        z.write(os.path.join(out, "parametros.json"), "parametros.json")
        for e in escenas:
            z.write(e, "escenas/" + os.path.basename(e))
    say("listo", msg="Listo", result=info)


if __name__ == "__main__":
    try:
        main(os.path.abspath(sys.argv[1]), sys.argv[2], json.loads(sys.argv[3]))
    except BaseException as exc:
        if not isinstance(exc, KeyboardInterrupt):
            say("error", msg=str(exc)[-500:])
        raise

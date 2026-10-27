"""Auto-flow: turn the booth's raw takes into a finished voiceover, one tone at a time.

    ~/.local/share/remifasol-autoflow/venv/bin/python tools/studio/pipeline.py assets/recordings/comercial --tono casual

1. Clean    DeepFilterNet removes room noise from every take of that tone.
2. Find     speech-to-text locates each scene of guion.json in the takes. A take
            picked with "★ usar esta" is the one used for its scene; otherwise
            the latest reading wins (a single re-take replaces just that scene).
3. Voice    ReMiFaSol converts the voice through a voice model (default
            US_Female_1, the female pop voice; --voz none keeps her own voice).
4. Polish   SoundCraft (headless) runs the voice chain and bounces at -16 LUFS.
5. Deliver  final/<stamp>-<tono>/: full WAV + MP3, one WAV per scene, a zip,
            resultado.json, and the editor's inputs: fuente.wav (her cleaned,
            levelled voice before any conversion) and cortes.json (where each
            scene sits in it).

The editing page (render.py) reuses convert() and polish() from here.
Progress lines go to stdout as `@@ <json>` so the pages can show them.
"""

import datetime
import difflib
import json
import os
import re
import subprocess
import sys
import tempfile
import unicodedata
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import guion  # noqa: E402

HOME = os.path.expanduser("~/.local/share/remifasol-autoflow")
APPLIO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEEP_FILTER = os.path.join(HOME, "deep-filter")
RATE = 48000
TARGET_LUFS = -16.0
PAD_BEFORE, PAD_AFTER, GAP = 0.20, 0.35, 0.6  # seconds

# SoundCraft voice chains. "natural" is the hand-made BeautyCita-voz-pulida chain.
CHAINS = {
    "natural": [
        ("eq_7band", {"hpf_on": 1, "hpf_freq": 90, "low_mid_freq": 300, "low_mid_gain": -2.5, "low_mid_q": 1.2,
                      "mid_freq": 3000, "mid_gain": 2, "mid_q": 0.9, "high_shelf_freq": 10000, "high_shelf_gain": 2.5}),
        ("de_esser", {"freq": 6500, "threshold": -28, "range": 6}),
        ("compressor", {"threshold": -24, "ratio": 3, "attack": 5, "release": 80, "knee": 6, "makeup": 3}),
        ("maximizer", {"threshold": -3, "ceiling": -1}),
    ],
    # Radio: more presence and air, firmer compression, the voice sits in front.
    "radio": [
        ("eq_7band", {"hpf_on": 1, "hpf_freq": 110, "low_mid_freq": 350, "low_mid_gain": -3.5, "low_mid_q": 1.0,
                      "mid_freq": 3500, "mid_gain": 4, "mid_q": 0.8, "high_shelf_freq": 9000, "high_shelf_gain": 4}),
        ("de_esser", {"freq": 6500, "threshold": -30, "range": 8}),
        ("compressor", {"threshold": -28, "ratio": 5, "attack": 3, "release": 60, "knee": 4, "makeup": 5}),
        ("maximizer", {"threshold": -5, "ceiling": -1}),
    ],
    # Cálido: body kept, top softened, gentle compression.
    "calido": [
        ("eq_7band", {"hpf_on": 1, "hpf_freq": 70, "low_mid_freq": 250, "low_mid_gain": 1, "low_mid_q": 1.0,
                      "mid_freq": 2500, "mid_gain": 0.5, "mid_q": 1.0, "high_shelf_freq": 10000, "high_shelf_gain": -1}),
        ("de_esser", {"freq": 6000, "threshold": -26, "range": 5}),
        ("compressor", {"threshold": -22, "ratio": 2.5, "attack": 10, "release": 120, "knee": 8, "makeup": 2}),
        ("maximizer", {"threshold": -2, "ceiling": -1}),
    ],
    # Sin color: no EQ or compression, only the loudness ceiling.
    "limpio": [
        ("maximizer", {"threshold": -1, "ceiling": -1}),
    ],
}


def say(step, **kw):
    print("@@ " + json.dumps({"step": step, **kw}, ensure_ascii=False), flush=True)


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def run_bytes(cmd):
    return subprocess.run(cmd, check=True, capture_output=True).stdout


def norm(text):
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.findall(r"[a-z0-9]+", text)


def duration(path):
    return float(run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path]).stdout)


def clean(take, work):
    """DeepFilterNet at 48 kHz float; -D keeps it sample-aligned with the take."""
    src = os.path.join(work, "in", os.path.basename(take))
    os.makedirs(os.path.dirname(src), exist_ok=True)
    run(["ffmpeg", "-v", "error", "-y", "-i", take, "-ac", "1", "-ar", str(RATE), "-c:a", "pcm_f32le", src])
    out_dir = os.path.join(work, "clean")
    run([DEEP_FILTER, "-D", "-a", "40", "-o", out_dir, src])
    return os.path.join(out_dir, os.path.basename(take))


def words_of(model, path):
    import numpy as np

    raw = run_bytes(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", "16000", "-f", "f32le", "-"])
    audio = np.frombuffer(raw, np.float32)
    segments, _ = model.transcribe(audio, language="es", word_timestamps=True, vad_filter=True,
                                   condition_on_previous_text=False)
    out = []
    for seg in segments:
        for w in seg.words:
            for tok in norm(w.word):
                out.append((w.start, w.end, tok))
    return out


def find_readings(words, window_tokens):
    """Every place the scene was read: [(score, first_word_idx, last_word_idx)]."""
    n = len(window_tokens)
    if not n:
        return []
    toks = [w[2] for w in words]
    hits = []
    for i in range(len(toks)):
        if toks[i] not in window_tokens[:4]:
            continue
        chunk = toks[i:i + int(n * 1.4) + 3]
        sm = difflib.SequenceMatcher(None, window_tokens, chunk, autojunk=False)
        blocks = [b for b in sm.get_matching_blocks() if b.size]
        matched = sum(b.size for b in blocks)
        if not blocks or matched / n < 0.6:
            continue
        first = i + blocks[0].b
        last = i + blocks[-1].b + blocks[-1].size - 1
        # Script words left unmatched at either edge (usually "BeautyCita", which
        # speech-to-text never spells right) are still spoken: take as many
        # neighbouring words, as long as they run on without a pause.
        for _ in range(blocks[0].a):
            if first == 0 or words[first][0] - words[first - 1][1] > 0.6:
                break
            first -= 1
        for _ in range(n - (blocks[-1].a + blocks[-1].size)):
            if last + 1 >= len(words) or words[last + 1][0] - words[last][1] > 0.6:
                break
            last += 1
        hits.append((matched / n, first, last))
    # keep the best hit of each cluster of overlapping hits
    hits.sort(key=lambda h: -h[0])
    kept = []
    for h in hits:
        if all(h[2] < k[1] or h[1] > k[2] for k in kept):
            kept.append(h)
    return sorted(kept, key=lambda h: h[1])


def polish(src, dst, chain="natural", target=TARGET_LUFS):
    """Two headless SoundCraft passes: measure, then bounce 24-bit at the target loudness."""
    plugins = CHAINS[chain]

    def bounce(gain):
        track = os.path.splitext(os.path.basename(src))[0]  # an imported file's track takes its name
        args = ["soundcraft-cli", "run", "--cmd", "file.import_audio=" + json.dumps({"path": src})]
        for slot, (plugin, params) in enumerate(plugins):
            args += ["--cmd", "mix.insert=" + json.dumps({"track": track, "slot": slot, "plugin": plugin, "params": params})]
        args += ["--cmd", "mix.volume=" + json.dumps({"tracks": [track], "db": gain}),
                 "--cmd", "file.bounce_mix=" + json.dumps({"path": dst, "format": "wav", "bit_depth": 24})]
        out = run(args).stdout
        m = re.findall(r'"lufs":\s*(-?[\d.]+)', out)
        if not m:
            raise RuntimeError("SoundCraft did not report a bounce: " + out[-400:])
        return float(m[-1])

    lufs = bounce(0.0)
    gain = max(-12.0, min(12.0, target - lufs))
    return bounce(round(gain, 2)), gain


def _weights(model_dir):
    """(pth, index, embedder) for a voice model folder. A model trained on the
    training page names its winner in studio.json; otherwise the folder's one
    weights file — never Applio's G_*/D_* training checkpoints."""
    info = {}
    try:
        info = json.load(open(os.path.join(model_dir, "studio.json"), encoding="utf-8"))
    except (OSError, ValueError):
        pass
    files = os.listdir(model_dir)
    pth = info.get("pth") if info.get("pth") in files else next(
        (f for f in sorted(files) if f.endswith(".pth") and not f.startswith(("G_", "D_"))), None)
    index = info.get("index") if info.get("index") in files else next((f for f in files if f.endswith(".index")), "")
    return pth, index, info.get("codificador", "contentvec")


def voices():
    """Voice models in logs/ ready to convert with (a training still running has
    no studio.json and only checkpoints, so it is not listed until it is done)."""
    root = os.path.join(APPLIO, "logs")
    out = []
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d)
        if not os.path.isdir(p):
            continue
        has_studio = os.path.exists(os.path.join(p, "studio.json"))
        trained_here = any(f.startswith(("G_", "D_")) for f in os.listdir(p))
        if trained_here and not has_studio:
            continue
        if _weights(p)[0]:
            out.append(d)
    return out


def convert(src, dst, voice, autotune=0.0, pitch=0):
    """RVC through logs/<voice>/: her timing and delivery, the model's timbre.
    autotune 0..1 snaps the pitch toward the nearest note (1 = the obvious effect);
    pitch shifts in semitones. Consonants protected. Uses the content encoder the
    model was trained with (studio.json)."""
    model_dir = os.path.join(APPLIO, "logs", voice)
    pth, index, embedder = _weights(model_dir)
    if not pth:
        raise RuntimeError(f"el modelo {voice} no tiene pesos")
    cmd = [os.path.join(APPLIO, ".venv", "bin", "python"), "core.py", "infer", "--input-path", src,
           "--output-path", dst, "--pth-path", os.path.join(model_dir, pth),
           "--index-path", os.path.join(model_dir, index) if index else "", "--pitch", str(int(pitch)),
           "--index-rate", "0.6", "--protect", "0.4", "--volume-envelope", "1", "--f0-method", "rmvpe",
           "--embedder-model", embedder, "--split-audio", "--export-format", "WAV"]
    if autotune > 0:
        cmd += ["--f0-autotune", "--f0-autotune-strength", f"{min(1.0, autotune):.2f}"]
    run(cmd, cwd=APPLIO)


def cut_scenes(full, cortes, out_dir):
    """One 24-bit WAV per scene, cut from a full-length render at cortes.json's times."""
    os.makedirs(out_dir, exist_ok=True)
    files = []
    for c in cortes:
        f = os.path.join(out_dir, f"{c['id']}.wav")
        run(["ffmpeg", "-v", "error", "-y", "-ss", f"{max(0.0, c['desde'] - 0.05):.3f}", "-t", f"{c['dur'] + 0.15:.3f}",
             "-i", full, "-c:a", "pcm_s24le", f])
        files.append(f)
    return files


def main(folder, voice="US_Female_1", tono="casual"):
    folder = os.path.abspath(folder)
    scenes = guion.load(folder)
    takes = [t for t in guion.takes(folder) if t[2] == tono]
    if not takes:
        raise SystemExit(f"no hay tomas en tono {tono}")
    picked = guion.elegidas(folder)[tono]
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = os.path.join(folder, "final", f"{stamp}-{tono}")
    os.makedirs(out)
    work = tempfile.mkdtemp(prefix="autoflow-")

    from faster_whisper import WhisperModel
    say("modelo", msg="Cargando reconocimiento de voz…")
    model = WhisperModel("small", device="cpu", compute_type="int8")

    best = {}  # scene id -> (start, end, clean path, take name, score, picked)
    for order, (take, _, _) in enumerate(takes):
        name = os.path.basename(take)
        say("limpiar", msg=f"Limpiando ruido: {name} ({order + 1}/{len(takes)})")
        cleaned = clean(take, work)
        say("buscar", msg=f"Buscando las escenas en {name}…")
        words = words_of(model, cleaned)
        for s in scenes:
            want = picked.get(s["id"])
            if want and want != name:
                continue  # a picked take owns its scene
            for score, a, b in find_readings(words, norm(s[tono]["texto"])):
                start = max(0.0, words[a][0] - PAD_BEFORE)
                if a > 0:
                    start = max(start, words[a - 1][1] + 0.05)
                end = words[b][1] + PAD_AFTER
                if b + 1 < len(words):
                    end = min(end, words[b + 1][0] - 0.05)
                # later takes, and later readings inside a take, replace earlier ones
                best[s["id"]] = (start, end, cleaned, name, round(score, 2), bool(want))

    say("armar", msg="Armando la voz completa…")
    pieces, found, missing = [], [], []
    silence = os.path.join(work, "gap.wav")
    run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"anullsrc=r={RATE}:cl=mono", "-t", str(GAP),
         "-c:a", "pcm_f32le", silence])
    for s in scenes:
        if s["id"] not in best:
            missing.append(s["id"])
            continue
        start, end, cleaned, take, score, was_picked = best[s["id"]]
        piece = os.path.join(work, f"{s['id']}.wav")
        # even out each piece before the chain so the compressor sees similar levels
        run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", cleaned,
             "-af", "loudnorm=I=-18:TP=-3:LRA=9,aresample=48000", "-ac", "1", "-c:a", "pcm_f32le", piece])
        pieces.append((s, piece, duration(piece)))
        found.append({"id": s["id"], "titulo": s["titulo"], "toma": take, "desde": round(start, 2),
                      "hasta": round(end, 2), "coincidencia": score, "elegida": was_picked})
    if not pieces:
        raise SystemExit("ninguna escena apareció en las tomas")

    listing = os.path.join(work, "list.txt")
    with open(listing, "w") as fh:
        for i, (_, piece, _) in enumerate(pieces):
            if i:
                fh.write(f"file '{silence}'\n")
            fh.write(f"file '{piece}'\n")
    joined = os.path.join(work, "joined.wav")
    run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", listing, "-c", "copy", joined])

    # The editor's inputs: her voice before any conversion, and where each scene sits.
    run(["ffmpeg", "-v", "error", "-y", "-i", joined, "-c:a", "pcm_f32le", os.path.join(out, "fuente.wav")])
    cortes, t = [], 0.0
    for s, _, dur in pieces:
        cortes.append({"id": s["id"], "titulo": s["titulo"], "desde": round(t, 3), "dur": round(dur, 3)})
        t += dur + GAP
    json.dump(cortes, open(os.path.join(out, "cortes.json"), "w"), ensure_ascii=False, indent=1)

    if voice and voice != "none":
        say("voz", msg=f"Convirtiendo la voz con el modelo {voice}…")
        converted = os.path.join(work, "converted.wav")
        convert(joined, converted, voice)
        joined = converted

    say("pulir", msg="Puliendo en SoundCraft (EQ, de-esser, compresor, maximizador)…")
    full = os.path.join(out, "voz-completa.wav")
    lufs, gain = polish(joined, full)

    say("entregar", msg="Cortando escenas y preparando descargas…")
    cut_scenes(full, cortes, out)
    run(["ffmpeg", "-v", "error", "-y", "-i", full, "-c:a", "libmp3lame", "-b:a", "192k",
         os.path.join(out, "voz-completa.mp3")])
    result = {"carpeta": f"{stamp}-{tono}", "tono": tono, "lufs": round(lufs, 1), "ganancia_db": round(gain, 1),
              "duracion": round(duration(full), 1), "voz": voice, "encontradas": found, "faltan": missing,
              "tomas": [os.path.basename(t[0]) for t in takes]}
    json.dump(result, open(os.path.join(out, "resultado.json"), "w"), ensure_ascii=False, indent=1)
    with zipfile.ZipFile(os.path.join(out, f"BeautyCita-voz-{stamp}-{tono}.zip"), "w") as z:
        for f in sorted(os.listdir(out)):
            if f.endswith((".wav", ".mp3", ".json")) and f != "fuente.wav":
                z.write(os.path.join(out, f), f)
    say("listo", msg="Listo", result=result)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folder", nargs="?", default=os.path.join("assets", "recordings", "voz"))
    ap.add_argument("--voz", default="US_Female_1", help="voice model in logs/, or 'none'")
    ap.add_argument("--tono", default="casual", choices=guion.TONOS)
    a = ap.parse_args()
    try:
        main(a.folder, a.voz, a.tono)
    except BaseException as exc:  # the pages show this line
        if not isinstance(exc, KeyboardInterrupt):
            say("error", msg=str(exc)[-500:])
        raise

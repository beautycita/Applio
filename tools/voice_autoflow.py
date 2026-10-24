"""Auto-flow: turn the booth's raw takes into a finished voiceover.

    ~/.local/share/remifasol-autoflow/venv/bin/python tools/voice_autoflow.py assets/recordings/comercial

1. Clean    DeepFilterNet removes room noise from every take.
2. Find     speech-to-text locates each guion.json window in the takes; when a
            window was read more than once, the latest reading wins (so a single
            re-take replaces just that window).
3. Voice    ReMiFaSol converts the voice through a voice model (default
            US_Female_1, the female pop voice; --voz none keeps her own voice).
4. Polish   SoundCraft (headless) runs the voice chain - EQ, de-esser,
            compressor, maximizer - and bounces at -16 LUFS.
5. Deliver  final/<stamp>/: full WAV + MP3, one WAV per window, a zip, and
            resultado.json (what was found, from which take, what is missing).

Progress lines go to stdout as `@@ <json>` so the booth can show them.
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

HOME = os.path.expanduser("~/.local/share/remifasol-autoflow")
APPLIO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEEP_FILTER = os.path.join(HOME, "deep-filter")
RATE = 48000
TARGET_LUFS = -16.0
TAKE_RE = re.compile(r"^[\w-]+-\d{6}\.wav$")
PAD_BEFORE, PAD_AFTER, GAP = 0.20, 0.35, 0.6  # seconds

# SoundCraft voice chain (same settings as the hand-made BeautyCita-voz-pulida)
CHAIN = [
    ("eq_7band", {"hpf_on": 1, "hpf_freq": 90, "low_mid_freq": 300, "low_mid_gain": -2.5, "low_mid_q": 1.2,
                  "mid_freq": 3000, "mid_gain": 2, "mid_q": 0.9, "high_shelf_freq": 10000, "high_shelf_gain": 2.5}),
    ("de_esser", {"freq": 6500, "threshold": -28, "range": 6}),
    ("compressor", {"threshold": -24, "ratio": 3, "attack": 5, "release": 80, "knee": 6, "makeup": 3}),
    ("maximizer", {"threshold": -3, "ceiling": -1}),
]


def say(step, **kw):
    print("@@ " + json.dumps({"step": step, **kw}, ensure_ascii=False), flush=True)


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


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


def run_bytes(cmd):
    return subprocess.run(cmd, check=True, capture_output=True).stdout


def find_readings(words, window_tokens):
    """Every place the window was read: [(score, first_word_idx, last_word_idx)]."""
    n = len(window_tokens)
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


def polish(src, dst):
    """Two headless SoundCraft passes: measure, then bounce at the target loudness."""
    def bounce(gain):
        track = os.path.splitext(os.path.basename(src))[0]  # an imported file's track takes its name
        args = ["soundcraft-cli", "run", "--cmd", "file.import_audio=" + json.dumps({"path": src})]
        for slot, (plugin, params) in enumerate(CHAIN):
            args += ["--cmd", "mix.insert=" + json.dumps({"track": track, "slot": slot, "plugin": plugin, "params": params})]
        args += ["--cmd", "mix.volume=" + json.dumps({"tracks": [track], "db": gain}),
                 "--cmd", "file.bounce_mix=" + json.dumps({"path": dst, "format": "wav", "bit_depth": 24})]
        out = run(args).stdout
        m = re.findall(r'"lufs":\s*(-?[\d.]+)', out)
        if not m:
            raise RuntimeError("SoundCraft did not report a bounce: " + out[-400:])
        return float(m[-1])

    lufs = bounce(0.0)
    gain = max(-12.0, min(12.0, TARGET_LUFS - lufs))
    return bounce(round(gain, 2)), gain


def convert(src, dst, voice):
    """RVC through logs/<voice>/ - same pitch (female to female), consonants protected."""
    model_dir = os.path.join(APPLIO, "logs", voice)
    pth = next(f for f in os.listdir(model_dir) if f.endswith(".pth"))
    index = next((f for f in os.listdir(model_dir) if f.endswith(".index")), "")
    run([os.path.join(APPLIO, ".venv", "bin", "python"), "core.py", "infer", "--input-path", src,
         "--output-path", dst, "--pth-path", os.path.join(model_dir, pth),
         "--index-path", os.path.join(model_dir, index) if index else "", "--pitch", "0",
         "--index-rate", "0.6", "--protect", "0.4", "--volume-envelope", "1", "--f0-method", "rmvpe",
         "--split-audio", "--export-format", "WAV"], cwd=APPLIO)


def main(folder, voice="US_Female_1"):
    folder = os.path.abspath(folder)
    windows = json.load(open(os.path.join(folder, "guion.json"), encoding="utf-8"))
    takes = sorted((os.path.join(folder, f) for f in os.listdir(folder) if TAKE_RE.match(f)), key=os.path.getmtime)
    if not takes:
        raise SystemExit("no takes")
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = os.path.join(folder, "final", stamp)
    os.makedirs(out)
    work = tempfile.mkdtemp(prefix="autoflow-")

    from faster_whisper import WhisperModel
    say("modelo", msg="Cargando reconocimiento de voz…")
    model = WhisperModel("small", device="cpu", compute_type="int8")

    best = {}  # window id -> (take order, start, end, clean path, take name, score)
    for order, take in enumerate(takes):
        name = os.path.basename(take)
        say("limpiar", msg=f"Limpiando ruido: {name} ({order + 1}/{len(takes)})")
        cleaned = clean(take, work)
        say("buscar", msg=f"Buscando las ventanas en {name}…")
        words = words_of(model, cleaned)
        for wi, w in enumerate(windows):
            for score, a, b in find_readings(words, norm(w["texto"])):
                start = max(0.0, words[a][0] - PAD_BEFORE)
                if a > 0:
                    start = max(start, words[a - 1][1] + 0.05)
                end = words[b][1] + PAD_AFTER
                if b + 1 < len(words):
                    end = min(end, words[b + 1][0] - 0.05)
                # later takes, and later readings inside a take, replace earlier ones
                best[w["id"]] = (order, start, end, cleaned, name, round(score, 2))

    say("armar", msg="Armando la voz completa…")
    pieces, found, missing = [], [], []
    silence = os.path.join(work, "gap.wav")
    run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"anullsrc=r={RATE}:cl=mono", "-t", str(GAP),
         "-c:a", "pcm_f32le", silence])
    for w in windows:
        if w["id"] not in best:
            missing.append(w["id"])
            continue
        _, start, end, cleaned, take, score = best[w["id"]]
        piece = os.path.join(work, f"{w['id']}.wav")
        # even out each piece before the chain so the compressor sees similar levels
        run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", cleaned,
             "-af", "loudnorm=I=-18:TP=-3:LRA=9,aresample=48000", "-ac", "1", "-c:a", "pcm_f32le", piece])
        pieces.append((w, piece, duration(piece)))
        found.append({"id": w["id"], "titulo": w["titulo"], "toma": take, "desde": round(start, 2),
                      "hasta": round(end, 2), "coincidencia": score})
    if not pieces:
        raise SystemExit("no window was found in the takes")

    listing = os.path.join(work, "list.txt")
    with open(listing, "w") as fh:
        for i, (_, piece, _) in enumerate(pieces):
            if i:
                fh.write(f"file '{silence}'\n")
            fh.write(f"file '{piece}'\n")
    joined = os.path.join(work, "joined.wav")
    run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", listing, "-c", "copy", joined])

    if voice and voice != "none":
        say("voz", msg=f"Convirtiendo la voz con el modelo {voice}…")
        converted = os.path.join(work, "converted.wav")
        convert(joined, converted, voice)
        joined = converted

    say("pulir", msg="Puliendo en SoundCraft (EQ, de-esser, compresor, maximizador)…")
    full = os.path.join(out, "voz-completa.wav")
    lufs, gain = polish(joined, full)

    say("entregar", msg="Cortando ventanas y preparando descargas…")
    t = 0.0
    for i, (w, _, dur) in enumerate(pieces):
        run(["ffmpeg", "-v", "error", "-y", "-ss", f"{max(0.0, t - 0.05):.3f}", "-t", f"{dur + 0.15:.3f}", "-i", full,
             "-c:a", "pcm_s24le", os.path.join(out, f"{w['id']}.wav")])
        t += dur + GAP
    run(["ffmpeg", "-v", "error", "-y", "-i", full, "-c:a", "libmp3lame", "-b:a", "192k",
         os.path.join(out, "voz-completa.mp3")])
    result = {"carpeta": stamp, "lufs": round(lufs, 1), "ganancia_db": round(gain, 1),
              "duracion": round(duration(full), 1), "voz": voice, "encontradas": found, "faltan": missing,
              "tomas": [os.path.basename(t) for t in takes]}
    json.dump(result, open(os.path.join(out, "resultado.json"), "w"), ensure_ascii=False, indent=1)
    with zipfile.ZipFile(os.path.join(out, f"BeautyCita-voz-{stamp}.zip"), "w") as z:
        for f in sorted(os.listdir(out)):
            if f.endswith((".wav", ".mp3", ".json")):
                z.write(os.path.join(out, f), f)
    say("listo", msg="Listo", result=result)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folder", nargs="?", default=os.path.join("assets", "recordings", "voz"))
    ap.add_argument("--voz", default="US_Female_1", help="voice model in logs/, or 'none'")
    a = ap.parse_args()
    try:
        main(a.folder, a.voz)
    except Exception as exc:  # the booth shows this line
        say("error", msg=str(exc)[-500:])
        raise

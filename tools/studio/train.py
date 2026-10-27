"""Entrenar su voz: a Spanish voice model of Quetzaly from the booth's training reading.

    ~/.local/share/remifasol-autoflow/venv/bin/python tools/studio/train.py <out folder> '<params json>'

Started by the training page (/voz/entrenar/) as its own systemd --user unit, so
hours of training survive booth restarts and can be cancelled. Progress goes to
<out>/entrenamiento/modelos/<name>/estado.json; the page polls it.

SPANISH, end to end (BC 2026-10-26: "this will all be in Spanish"):
  * Data: every take is cleaned and its language checked by speech-to-text.
    Anything not confidently Spanish is left out and listed.
  * Content encoder: Applio has no Spanish one. The only multilingual drop-ins
    found (mHuBERT-147, MMS) are licensed NON-COMMERCIAL, so they are not
    offered: this voice is for paid ads. Offered: contentvec (MIT, matches
    Applio's pretrained base, default) and spin-v2 — or both, compared.
  * Exam: held-out Spanish takes are converted with each saved epoch and
    transcribed in Spanish. The score is the word error rate, overall and on
    words carrying rr / ñ / ll-y / j-x, against the same transcription of her
    real voice. The best epoch wins, which also guards against over-training.

Stages: datos -> preparar (Applio preprocess) -> rasgos (extract, rmvpe pitch)
-> entrenar -> indice -> examen -> instalar (logs/<name>/ with studio.json, so
the editing page lists it). Refuses to start without the signed authorization.
"""

import datetime
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as pl  # noqa: E402

APPLIO = pl.APPLIO
APY = os.path.join(APPLIO, ".venv", "bin", "python")
ENT_TAKE = re.compile(r"^ent-(?P<id>[\w-]{1,40}?)-r(?P<r>[123])-\d{6}(?:-\d+)?\.wav$")
ENCODERS = {"contentvec": "contentvec (MIT · recomendado)", "spin-v2": "spin-v2"}
# Held out of training and used for the Spanish exam: the texts built on the
# hardest Spanish sounds, round 3 (falls back to any round that exists).
EXAM_IDS = ["07-sonidos-r", "08-sonidos-n-ll", "09-sonidos-j-x"]
SOUNDS = {"rr / r inicial": r"(rr|^r)", "ñ": r"ñ", "ll / y": r"(ll|y)", "j / g / x": r"(j|g[ei]|x)"}
MIN_MINUTES = 5.0


class Job:
    def __init__(self, folder, name):
        self.dir = os.path.join(folder, "entrenamiento", "modelos", name)
        os.makedirs(self.dir, exist_ok=True)
        self.path = os.path.join(self.dir, "estado.json")
        try:
            self.state = json.load(open(self.path, encoding="utf-8"))
        except (OSError, ValueError):
            self.state = {}

    def update(self, **kw):
        self.state.update(kw, actualizado=datetime.datetime.now().astimezone().isoformat(timespec="seconds"))
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.state, fh, ensure_ascii=False, indent=1)
        os.replace(tmp, self.path)

    def step(self, etapa, msg, **kw):
        print(f"[{etapa}] {msg}", flush=True)
        self.update(etapa=etapa, msg=msg, **kw)


def applio(args, job=None, parse=None):
    """Run core.py; stream its output so training progress can be read."""
    proc = subprocess.Popen([APY, "core.py", *args], cwd=APPLIO, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    tail = []
    for line in proc.stdout:
        line = line.rstrip()
        tail = (tail + [line])[-12:]
        print(line, flush=True)
        if parse:
            parse(line)
    if proc.wait() != 0:
        raise RuntimeError(f"core.py {args[0]} falló: " + " | ".join(tail[-4:])[-400:])
    return tail


def norm_words(text):
    import unicodedata
    text = text.lower()
    # keep ñ, drop other accents, so «año» stays distinct from «ano»
    text = "".join(c if c == "ñ" else unicodedata.normalize("NFD", c)[0] for c in text)
    return re.findall(r"[a-zñ0-9]+", text)


def wer(ref, hyp):
    """Word error rate and the reference words that were missed."""
    import difflib
    sm = difflib.SequenceMatcher(None, ref, hyp, autojunk=False)
    errors, missed = 0, set()
    for tag, a1, a2, b1, b2 in sm.get_opcodes():
        if tag != "equal":
            errors += max(a2 - a1, b2 - b1)
            missed.update(range(a1, a2))
    return errors / max(1, len(ref)), missed


def exam_scores(ref_words, hyp_words):
    total, missed = wer(ref_words, hyp_words)
    out = {"total": round(total * 100, 1)}
    for label, rx in SOUNDS.items():
        idx = [i for i, w in enumerate(ref_words) if re.search(rx, w)]
        if idx:
            out[label] = round(100 * sum(1 for i in idx if i in missed) / len(idx), 1)
    return out


def main(folder, params):
    folder = os.path.abspath(folder)
    name = re.sub(r"[^\w-]", "", params.get("nombre", "quetzaly-v1"))[:40] or "quetzaly-v1"
    encoders = ["contentvec", "spin-v2"] if params.get("codificador") == "comparar" else [params.get("codificador", "contentvec")]
    if any(e not in ENCODERS for e in encoders):
        raise SystemExit("codificador no permitido")
    epochs = max(5, min(1000, int(params.get("epocas", 300))))
    every = max(5, min(100, int(params.get("cada", 25))))
    batch = max(1, min(16, int(params.get("lote", 6))))
    job = Job(folder, name)
    job.update(nombre=name, codificadores=encoders, epocas=epochs, cada=every, lote=batch, estado="corriendo",
               inicio=datetime.datetime.now().astimezone().isoformat(timespec="seconds"), error=None, resultados={})

    ent = os.path.join(folder, "entrenamiento")
    if not os.path.isfile(os.path.join(ent, "consentimiento.json")):
        raise SystemExit("falta la autorización firmada de la voz (cabina › Entrenar su voz › texto 1)")
    textos = {t["id"]: t["texto"] for t in json.load(open(os.path.join(ent, "textos.json"), encoding="utf-8"))}

    # ---- 1. datos: clean, check Spanish, hold out the exam
    takes = sorted((f for f in os.listdir(ent) if ENT_TAKE.match(f)), key=lambda f: os.path.getmtime(os.path.join(ent, f)))
    latest = {}
    for f in takes:
        m = ENT_TAKE.match(f)
        if m["id"] != "00-consentimiento" and m["id"] in textos:
            latest[(m["id"], int(m["r"]))] = f  # the latest take of each text and round
    if not latest:
        raise SystemExit("no hay lecturas de entrenamiento todavía")
    exam = {}
    for eid in EXAM_IDS:
        for r in (3, 2, 1):
            if (eid, r) in latest and sum(1 for k in latest if k[0] == eid) > 1:
                exam[eid] = latest[(eid, r)]  # only hold out when another round of it stays in training
                break
    from faster_whisper import WhisperModel
    job.step("datos", "Cargando el reconocimiento de voz en español…")
    asr = WhisperModel("small", device="cpu", compute_type="int8")
    work = os.path.join(job.dir, "trabajo")
    shutil.rmtree(work, ignore_errors=True)
    dataset = os.path.join(job.dir, "dataset")
    shutil.rmtree(dataset, ignore_errors=True)
    os.makedirs(dataset)
    examdir = os.path.join(job.dir, "examen")
    shutil.rmtree(examdir, ignore_errors=True)
    os.makedirs(examdir)
    accepted, rejected, seconds = [], [], 0.0
    for i, ((tid, r), f) in enumerate(sorted(latest.items())):
        job.step("datos", f"Revisando que esté en español: {f} ({i + 1}/{len(latest)})")
        cleaned = pl.clean(os.path.join(ent, f), work)
        import numpy as np
        audio = np.frombuffer(pl.run_bytes(["ffmpeg", "-v", "error", "-i", cleaned, "-ac", "1", "-ar", "16000",
                                             "-f", "f32le", "-"]), np.float32)
        _, info = asr.transcribe(audio[:16000 * 30], beam_size=1)
        # Spanish must be the language heard (a real voice scores 90 %+; the bar
        # is lower so a short or flat reading is not thrown away for nothing)
        if info.language != "es" or info.language_probability < 0.5:
            rejected.append({"toma": f, "motivo": f"no parece español ({info.language} {info.language_probability:.0%})"})
            continue
        dest = examdir if exam.get(tid) == f else dataset
        out = os.path.join(dest, f)
        pl.run(["ffmpeg", "-v", "error", "-y", "-i", cleaned, "-af", "loudnorm=I=-20:TP=-3:LRA=11",
                "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", out])
        if dest == dataset:
            seconds += pl.duration(out)
            accepted.append(f)
    minutes = round(seconds / 60, 1)
    job.update(datos={"minutos": minutes, "aceptadas": len(accepted), "rechazadas": rejected,
                      "examen": sorted(os.listdir(examdir))})
    if minutes < MIN_MINUTES:
        raise SystemExit(f"solo hay {minutes} min en español para entrenar; hacen falta al menos {MIN_MINUTES:g} (ideal 20)")

    # her real voice through the same Spanish transcription: the bar the model is measured against
    exam_refs = {}
    for f in sorted(os.listdir(examdir)):
        tid = ENT_TAKE.match(f)["id"]
        exam_refs[f] = norm_words(textos[tid])
    def transcribe(path):
        import numpy as np  # decode with ffmpeg: faster-whisper's own decoder breaks on this PyAV
        audio = np.frombuffer(pl.run_bytes(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", "16000",
                                             "-f", "f32le", "-"]), np.float32)
        segs, _ = asr.transcribe(audio, language="es", beam_size=5, vad_filter=True)
        return norm_words(" ".join(s.text for s in segs))
    baseline = {}
    if exam_refs:
        job.step("datos", "Midiendo su voz real en el examen de español…")
        ref_all, hyp_all = [], []
        for f, ref in exam_refs.items():
            ref_all += ref
            hyp_all += transcribe(os.path.join(examdir, f))
        baseline = exam_scores(ref_all, hyp_all)
    job.update(examen_base=baseline)

    for enc in encoders:
        model = f"{name}-{enc}"
        res = {"codificador": enc, "etapa": "preparar"}
        job.state.setdefault("resultados", {})[enc] = res
        logs = os.path.join(APPLIO, "logs", model)
        shutil.rmtree(logs, ignore_errors=True)

        # ---- 2-3. Applio preprocess + extract (rmvpe pitch, the chosen encoder)
        job.step("preparar", f"[{enc}] Cortando y preparando {minutes} min de audio…", resultados=job.state["resultados"])
        applio(["preprocess", "--model-name", model, "--dataset-path", dataset, "--sample-rate", "48000",
                "--cpu-cores", "6", "--cut-preprocess", "Automatic", "--chunk-len", "3.0", "--overlap-len", "0.3",
                "--normalization-mode", "post"])
        job.step("rasgos", f"[{enc}] Extrayendo tono (rmvpe) y rasgos de voz ({enc})…")
        applio(["extract", "--model-name", model, "--f0-method", "rmvpe", "--cpu-cores", "6", "--gpu", "0",
                "--sample-rate", "48000", "--embedder-model", enc, "--include-mutes", "2"])

        # ---- 4. train
        t0 = time.time()

        def progress(line, enc=enc, t0=t0):
            m = re.search(r"\| epoch=(\d+) \|", line)
            if m:
                e = int(m[1])
                rate = (time.time() - t0) / max(1, e)
                job.step("entrenar", f"[{enc}] Época {e} de {epochs}", epoca=e,
                         restante_min=round(rate * (epochs - e) / 60))

        job.step("entrenar", f"[{enc}] Entrenando {epochs} épocas (guarda cada {every})…", epoca=0)
        applio(["train", "--model-name", model, "--vocoder", "HiFi-GAN", "--save-every-epoch", str(every),
                "--save-only-latest", "--total-epoch", str(epochs), "--sample-rate", "48000",
                "--batch-size", str(batch), "--gpu", "0", "--pretrained", "--index-algorithm", "Auto"],
               parse=progress)

        # ---- 5. index
        job.step("indice", f"[{enc}] Construyendo el índice…")
        applio(["index", "--model-name", model, "--index-algorithm", "Auto"])
        index = next(iter(glob.glob(os.path.join(logs, "*.index"))), "")

        # ---- 6. the Spanish exam, every saved epoch
        weights = sorted(glob.glob(os.path.join(logs, f"{model}_*e_*s.pth")),
                         key=lambda p: int(re.search(r"_(\d+)e_", p)[1]))
        if not weights:
            raise RuntimeError("el entrenamiento no guardó ningún modelo")
        scores = []
        samples = os.path.join(job.dir, f"muestras-{enc}")
        shutil.rmtree(samples, ignore_errors=True)
        os.makedirs(samples)
        for w in weights:
            e = int(re.search(r"_(\d+)e_", w)[1])
            job.step("examen", f"[{enc}] Examen de español, época {e}…")
            ref_all, hyp_all = [], []
            for f, ref in exam_refs.items():
                dst = os.path.join(samples, f"e{e}-{f}")
                applio(["infer", "--input-path", os.path.join(examdir, f), "--output-path", dst, "--pth-path", w,
                        "--index-path", index, "--pitch", "0", "--index-rate", "0.6", "--protect", "0.4",
                        "--volume-envelope", "1", "--f0-method", "rmvpe", "--embedder-model", enc,
                        "--export-format", "WAV"])
                ref_all += ref
                hyp_all += transcribe(dst)
            s = exam_scores(ref_all, hyp_all) if exam_refs else {"total": None}
            scores.append({"epoca": e, "pth": os.path.basename(w), **s})
            res["examen"] = scores
            job.update(resultados=job.state["resultados"])
        best = min(scores, key=lambda s: (s["total"] if s["total"] is not None else 0, -s["epoca"]))
        if not exam_refs:
            best = scores[-1]  # no held-out text: the last epoch
        # keep only the winner's samples
        for f in os.listdir(samples):
            if not f.startswith(f"e{best['epoca']}-"):
                os.remove(os.path.join(samples, f))

        # ---- 7. install: the winner only, with what the editor needs to use it
        for p in glob.glob(os.path.join(logs, "*.pth")):
            if os.path.basename(p) != best["pth"]:
                os.remove(p)  # other epochs, and Applio's G_/D_ training checkpoints (~1.3 GB)
        studio = {"nombre": model, "pth": best["pth"], "index": os.path.basename(index), "codificador": enc,
                  "idioma": "es", "epoca": best["epoca"], "examen": best, "examen_base": baseline,
                  "minutos": minutes, "entrenado": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                  "consentimiento": json.load(open(os.path.join(ent, "consentimiento.json"), encoding="utf-8"))}
        json.dump(studio, open(os.path.join(logs, "studio.json"), "w"), ensure_ascii=False, indent=1)
        res.update(etapa="listo", ganador=best, modelo=model, horas=round((time.time() - t0) / 3600, 2))
        job.update(resultados=job.state["resultados"])

    done = [r for r in job.state["resultados"].values() if r.get("etapa") == "listo"]
    winner = min(done, key=lambda r: r["ganador"]["total"] if r["ganador"]["total"] is not None else 0)
    job.step("listo", f"Listo. Mejor modelo: {winner['modelo']} (época {winner['ganador']['epoca']}). "
             f"Ya aparece en la mesa de edición.", estado="listo", ganador=winner["modelo"])
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    import signal

    def _stop(*_):  # "Cancelar" on the page stops the unit with SIGTERM
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _stop)
    folder, params = os.path.abspath(sys.argv[1]), json.loads(sys.argv[2])
    name = re.sub(r"[^\w-]", "", params.get("nombre", "quetzaly-v1"))[:40] or "quetzaly-v1"
    try:
        main(folder, params)
    except BaseException as exc:
        j = Job(folder, name)
        j.update(estado="cancelado" if isinstance(exc, KeyboardInterrupt) else "error",
                 error=str(exc)[-600:], etapa=j.state.get("etapa"))
        raise

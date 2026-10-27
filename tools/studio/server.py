"""The studio: record voiceover takes in a browser on another tailnet machine
(the booth, /voz/), and edit the results on this box (/voz/editar/).

The talent opens the booth on her own laptop (mic plugged in there), records,
and every take lands here as a WAV. The browser only allows the mic over
HTTPS, so the pages are published through `tailscale serve`:

    systemctl --user restart voice-booth      # runs this with --out assets/recordings/comercial
    tailscale serve --bg --set-path /voz http://127.0.0.1:8765
    -> https://<this-machine>.<tailnet>.ts.net/voz/   and   …/voz/editar/

Training reading (/entreno*): <out>/entrenamiento/ holds textos.json (Spanish
passages written to cover every Spanish sound and an ad's range of delivery),
the takes ent-<id>-r<round>-HHMMSS.wav, and consentimiento.json. Nothing there is
a commercial take. Recording is refused until her spoken authorization
(passage 00-consentimiento) is recorded and her written acceptance is saved.

Voice training (/voz/entrenar/, its own page): train.py turns the training
reading into a Spanish voice model, run as the systemd --user unit voz-entreno
(survives restarts, one at a time; renders wait for it, they share the GPU).

Modules: guion.py (script, tones, takes, picks), pipeline.py (auto-flow, run in
the autoflow venv), render.py (editing page renders), mictest.py (the booth's
"Prueba de micrófono"; clips kept in <out>/pruebas/, never used as takes),
booth.html, editor.html.
One job runs at a time: conversion and SoundCraft share the CPU and GPU.
Tailnet-only: nothing here is public.
"""

import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import guion  # noqa: E402

APPLIO = os.path.dirname(os.path.dirname(HERE))
AUTOFLOW_PY = os.path.expanduser("~/.local/share/remifasol-autoflow/venv/bin/python")
CARPETA = r"\d{8}-\d{6}(?:-(?:casual|profesional))?"
RID = r"\d{8}-\d{6}"
PULIDOS = {
    "natural": "Natural (EQ, de-esser, compresor)",
    "radio": "Radio (presencia y compresión firme)",
    "calido": "Cálido (cuerpo, agudos suaves)",
    "limpio": "Sin color (solo volumen)",
}
CTYPES = {
    "wav": "audio/wav",
    "flac": "audio/flac",
    "mp3": "audio/mpeg",
    "zip": "application/zip",
    "json": "application/json",
}

RONDAS = [
    {"n": 1, "intencion": "Natural, como platicando"},
    {"n": 2, "intencion": "Más expresiva: más sonrisa, más energía"},
    {"n": 3, "intencion": "Más suave y cercana, como al oído"},
]
ENT_TAKE = re.compile(r"^ent-(?P<id>[\w-]{1,40}?)-r(?P<r>[123])-\d{6}(?:-\d+)?\.wav$")
ENT_META_MIN = 20


def ent_dir(out):
    return os.path.join(out, "entrenamiento")


def ent_state(out):
    d = ent_dir(out)
    os.makedirs(d, exist_ok=True)
    try:
        textos = json.load(open(os.path.join(d, "textos.json"), encoding="utf-8"))
    except (OSError, ValueError):
        textos = []
    tomas = []
    for f in sorted(os.listdir(d), key=lambda f: os.path.getmtime(os.path.join(d, f))):
        m = ENT_TAKE.match(f)
        if m:
            tomas.append(
                {
                    "file": f,
                    "id": m["id"],
                    "ronda": int(m["r"]),
                    "seconds": round(
                        (os.path.getsize(os.path.join(d, f)) - 44) / (48000 * 3), 1
                    ),
                }
            )
    try:
        consent = json.load(
            open(os.path.join(d, "consentimiento.json"), encoding="utf-8")
        )
    except (OSError, ValueError):
        consent = None
    # minutes that train the model: the latest take of each passage and round, authorization excluded
    latest = {}
    for t in tomas:
        if t["id"] != "00-consentimiento":
            latest[(t["id"], t["ronda"])] = t["seconds"]
    return {
        "textos": textos,
        "rondas": RONDAS,
        "tomas": tomas[::-1],
        "consentimiento": consent,
        "minutos": round(sum(latest.values()) / 60, 1),
        "meta_min": ENT_META_MIN,
    }


ALLOWED_ORIGINS = {
    "https://bc-dev.tail897a37.ts.net",
    "https://bc-dev.tail897a37.ts.net:8443",
    "http://127.0.0.1:8765",
    "http://127.0.0.1:8766",
}
LOCK = threading.Lock()
FLOW = {"running": False, "msg": "", "result": None, "error": None}
EDIT = {"running": False, "msg": "", "result": None, "error": None}


TRAIN_UNIT = "voz-entreno"


def training_active():
    r = subprocess.run(
        ["systemctl", "--user", "is-active", TRAIN_UNIT], capture_output=True, text=True
    )
    return r.stdout.strip() in ("active", "activating")


def busy():
    return FLOW["running"] or EDIT["running"] or training_active()


def train_models(out):
    root = os.path.join(ent_dir(out), "modelos")
    items = []
    if os.path.isdir(root):
        for d in os.listdir(root):
            p = os.path.join(root, d, "estado.json")
            if os.path.isfile(p):
                try:
                    items.append(json.load(open(p, encoding="utf-8")))
                except ValueError:
                    pass
    return sorted(items, key=lambda m: m.get("inicio", ""), reverse=True)


def run_job(state, cmd):
    """Runs a pipeline/render script and mirrors its `@@` progress lines into state."""
    state.update(running=True, msg="Iniciando…", result=None, error=None)
    try:
        proc = subprocess.Popen(
            cmd, cwd=APPLIO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        tail = []
        for line in proc.stdout:
            tail = (tail + [line.strip()])[-5:]
            if line.startswith("@@ "):
                ev = json.loads(line[3:])
                state["msg"] = ev.get("msg", "")
                if ev.get("step") == "listo":
                    state["result"] = ev["result"]
                if ev.get("step") == "error":
                    state["error"] = ev.get("msg")
        if proc.wait() != 0 and not state["error"]:
            state["error"] = " | ".join(tail)[-400:]
    except Exception as exc:
        state["error"] = str(exc)
    finally:
        state["running"] = False


def results(out):
    """Auto-flow results, newest first, with whether the editor can use them."""
    final = os.path.join(out, "final")
    items = []
    if not os.path.isdir(final):
        return items
    for d in sorted(os.listdir(final), reverse=True):
        p = os.path.join(final, d, "resultado.json")
        if re.fullmatch(CARPETA, d) and os.path.exists(p):
            r = json.load(open(p, encoding="utf-8"))
            r["carpeta"] = d
            r.setdefault("tono", "casual")
            r["editable"] = os.path.isfile(os.path.join(final, d, "fuente.wav"))
            items.append(r)
    return items


def renders(out):
    final = os.path.join(out, "final")
    items = []
    if not os.path.isdir(final):
        return items
    for d in os.listdir(final):
        ed = os.path.join(final, d, "ediciones")
        if not (re.fullmatch(CARPETA, d) and os.path.isdir(ed)):
            continue
        for rid in os.listdir(ed):
            p = os.path.join(ed, rid, "parametros.json")
            if os.path.exists(p):
                items.append(json.load(open(p, encoding="utf-8")))
    return sorted(items, key=lambda r: r["id"], reverse=True)


class Studio(BaseHTTPRequestHandler):
    out = "."

    def log_message(self, fmt, *args):  # keep the journal to saves and errors
        if not (args and str(args[1]).startswith("2")):
            super().log_message(fmt, *args)

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False))

    def _file(self, path, download=False):
        ext = path.rsplit(".", 1)[-1]
        self.send_response(200)
        self.send_header("Content-Type", CTYPES.get(ext, "application/octet-stream"))
        self.send_header("Content-Length", str(os.path.getsize(path)))
        disp = "attachment" if download else "inline"
        self.send_header(
            "Content-Disposition", f'{disp}; filename="{os.path.basename(path)}"'
        )
        self.end_headers()
        with open(path, "rb") as fh:
            shutil.copyfileobj(fh, self.wfile)

    def _route(self):
        # works whether or not `tailscale serve` strips the /voz prefix
        path = self.path.split("?", 1)[0]
        return re.sub(r"^/voz", "", path).rstrip("/") or "/"

    def _query(self):
        q = self.path.split("?", 1)[1] if "?" in self.path else ""
        return {k: v[0] for k, v in parse_qs(q).items()}

    def _body(self, limit=1 << 20):
        length = int(self.headers.get("Content-Length", 0))
        if not 0 < length <= limit:
            return None
        return self.rfile.read(length)

    def do_GET(self):
        route, q = self._route(), self._query()
        if route == "/":
            return self._send(
                200,
                open(os.path.join(HERE, "booth.html"), "rb").read(),
                "text/html; charset=utf-8",
            )
        if route == "/entrenar":
            return self._send(
                200,
                open(os.path.join(HERE, "trainer.html"), "rb").read(),
                "text/html; charset=utf-8",
            )
        if route == "/entrenamiento/estado":
            st = ent_state(self.out)
            from train import ENCODERS, LANGS, MIN_MINUTES, folder_config, lang_of

            return self._json(
                {
                    "minutos": st["minutos"],
                    "meta_min": st["meta_min"],
                    "min_min": MIN_MINUTES,
                    "consentimiento": st["consentimiento"],
                    "corriendo": training_active(),
                    "codificadores": ENCODERS,
                    "examen_textos": LANGS[lang_of(ent_dir(self.out))]["examen"],
                    "modelo": folder_config(ent_dir(self.out)).get(
                        "modelo", "quetzaly-v1"
                    ),
                    "tomas": len(
                        [t for t in st["tomas"] if t["id"] != "00-consentimiento"]
                    ),
                    "modelos": train_models(self.out),
                }
            )
        m = re.fullmatch(
            r"/entrenamiento/audio/([\w-]{1,40})/(examen|muestras-contentvec|muestras-spin-v2)/([\w.-]+\.wav)",
            route,
        )
        if m:
            p = os.path.join(ent_dir(self.out), "modelos", m[1], m[2], m[3])
            if os.path.isfile(p):
                return self._file(p)
            return self._send(404, '{"error":"not found"}')
        if route == "/editar":
            return self._send(
                200,
                open(os.path.join(HERE, "editor.html"), "rb").read(),
                "text/html; charset=utf-8",
            )
        if route == "/guion":
            return self._json(guion.load(self.out))
        if route == "/takes":
            picks = guion.elegidas(self.out)
            items = []
            for path, stem, tono in reversed(guion.takes(self.out)):
                f = os.path.basename(path)
                items.append(
                    {
                        "file": f,
                        "id": stem,
                        "tono": tono,
                        "seconds": round((os.path.getsize(path) - 44) / (48000 * 3), 1),
                        "elegida": picks[tono].get(stem) == f,
                    }
                )
            return self._json(items)
        if route == "/estado":
            state = dict(FLOW)
            if not state["running"] and not state["result"] and not state["error"]:
                r = results(self.out)
                state["result"] = next(
                    (x for x in r if x["tono"] == q.get("tono", "casual")), None
                )
            return self._json(state)
        if route == "/entreno":
            return self._json(ent_state(self.out))
        if route.startswith("/entreno/toma/"):
            name = os.path.basename(unquote(route[len("/entreno/toma/") :]))
            p = os.path.join(ent_dir(self.out), name)
            if ENT_TAKE.match(name) and os.path.isfile(p):
                return self._file(p)
            return self._send(404, '{"error":"not found"}')
        if route == "/fuentes":
            return self._json(results(self.out))
        if route == "/modelos":
            from pipeline import voices  # stdlib-only import path

            return self._json({"voces": voices(), "pulidos": PULIDOS})
        if route == "/render/estado":
            return self._json(EDIT)
        if route == "/renders":
            return self._json(renders(self.out))
        m = re.fullmatch(
            rf"/final/({CARPETA})/(?:(ediciones/{RID}/(?:escenas/)?))?([\w.-]+)", route
        )
        if m:
            p = os.path.join(self.out, "final", m[1], m[2] or "", m[3])
            if os.path.isfile(p):
                return self._file(p, download="descargar" in q)
        m = re.fullmatch(r"/prueba/(\d{8}-\d{6}(?:-lectura|-pulida)?\.wav)", route)
        if m:
            p = os.path.join(self.out, "pruebas", m[1])
            if os.path.isfile(p):
                return self._file(p)
        if route.startswith("/take/"):
            name = os.path.basename(unquote(route[len("/take/") :]))
            p = os.path.join(self.out, name)
            if guion.parse_take(name) and os.path.isfile(p):
                return self._file(p)
        self._send(404, '{"error":"not found"}')

    def _cross_site(self):
        """A browser request started by another site (CSRF): every POST here
        records, spends money or uploads biometric data, so only this page may."""
        site = self.headers.get("Sec-Fetch-Site")
        if site and site not in ("same-origin", "none"):
            return True
        origin = self.headers.get("Origin")
        return bool(origin) and origin not in ALLOWED_ORIGINS

    def do_POST(self):
        if self._cross_site():
            return self._json({"error": "solicitud de otro sitio rechazada"}, 403)
        route, q = self._route(), self._query()
        if route == "/procesar":
            voice = re.sub(r"[^\w-]", "", q.get("voz", "US_Female_1")) or "none"
            tono = q.get("tono", "casual")
            if tono not in guion.TONOS:
                return self._json({"error": "tono desconocido"}, 400)
            if voice != "none" and not os.path.isdir(
                os.path.join(APPLIO, "logs", voice)
            ):
                return self._json({"error": "no existe ese modelo de voz"}, 400)
            with LOCK:
                if busy():
                    return self._json({"error": "ya hay un proceso corriendo"}, 409)
                FLOW["running"] = True
            cmd = [
                AUTOFLOW_PY,
                os.path.join(HERE, "pipeline.py"),
                self.out,
                "--voz",
                voice,
                "--tono",
                tono,
            ]
            threading.Thread(target=run_job, args=(FLOW, cmd), daemon=True).start()
            return self._json({"started": True}, 202)
        if route == "/render":
            body = self._body()
            try:
                params = json.loads(body or b"")
            except ValueError:
                return self._json({"error": "parámetros inválidos"}, 400)
            carpeta = str(params.pop("carpeta", ""))
            if not re.fullmatch(CARPETA, carpeta):
                return self._json({"error": "resultado inválido"}, 400)
            if params.get("pulido") not in PULIDOS:
                return self._json({"error": "pulido desconocido"}, 400)
            with LOCK:
                if busy():
                    return self._json({"error": "ya hay un proceso corriendo"}, 409)
                EDIT["running"] = True
            cmd = [
                AUTOFLOW_PY,
                os.path.join(HERE, "render.py"),
                self.out,
                carpeta,
                json.dumps(params),
            ]
            threading.Thread(target=run_job, args=(EDIT, cmd), daemon=True).start()
            return self._json({"started": True}, 202)
        if route == "/entrenamiento/iniciar":
            try:
                b = json.loads(self._body() or b"")
            except ValueError:
                return self._json({"error": "inválido"}, 400)
            from train import ENCODERS

            name = re.sub(r"[^\w-]", "", str(b.get("nombre", "")))[:40]
            if len(name) < 3:
                return self._json(
                    {"error": "ponle un nombre al modelo (letras, números, guiones)"},
                    400,
                )
            if b.get("codificador") not in (*ENCODERS, "comparar"):
                return self._json({"error": "codificador no permitido"}, 400)
            if not ent_state(self.out)["consentimiento"]:
                return self._json(
                    {
                        "error": "falta la autorización de la voz (cabina › Entrenar su voz)"
                    },
                    403,
                )
            params = {
                "nombre": name,
                "codificador": b["codificador"],
                "epocas": int(b.get("epocas", 300)),
                "cada": int(b.get("cada", 25)),
                "lote": int(b.get("lote", 6)),
                "donde": "nube" if b.get("donde") == "nube" else "local",
                "borrar_nube": bool(b.get("borrar_nube", True)),
            }
            with LOCK:
                if busy():
                    return self._json(
                        {
                            "error": "hay otro proceso corriendo (entrenamiento, render o auto-flow)"
                        },
                        409,
                    )
                subprocess.run(
                    ["systemctl", "--user", "reset-failed", TRAIN_UNIT],
                    capture_output=True,
                )
                r = subprocess.run(
                    [
                        "systemd-run",
                        "--user",
                        "--unit",
                        TRAIN_UNIT,
                        "--collect",
                        "-p",
                        f"WorkingDirectory={APPLIO}",
                        "-p",
                        "KillSignal=SIGTERM",
                        "-p",
                        "TimeoutStopSec=900",
                        AUTOFLOW_PY,
                        os.path.join(HERE, "train.py"),
                        self.out,
                        json.dumps(params),
                    ],
                    capture_output=True,
                    text=True,
                )
            if r.returncode != 0:
                return self._json(
                    {"error": "no se pudo iniciar: " + r.stderr[-200:]}, 500
                )
            print(f"training started: {params}", flush=True)
            return self._json({"started": True, **params}, 202)
        if route == "/entrenamiento/cancelar":
            subprocess.run(
                ["systemctl", "--user", "stop", TRAIN_UNIT], capture_output=True
            )
            return self._json({"ok": True})
        if route == "/entreno/acepto":
            try:
                b = json.loads(self._body() or b"")
            except ValueError:
                return self._json({"error": "inválido"}, 400)
            nombre = re.sub(r"\s+", " ", str(b.get("nombre", ""))).strip()[:80]
            st = ent_state(self.out)
            audio = next(
                (t["file"] for t in st["tomas"] if t["id"] == "00-consentimiento"), None
            )
            if not audio:
                return self._json(
                    {"error": "primero graba la autorización (texto 00) en voz alta"},
                    400,
                )
            if len(nombre) < 5 or not b.get("acepto"):
                return self._json(
                    {"error": "escribe tu nombre completo y marca que aceptas"}, 400
                )
            texto = next(
                (t["texto"] for t in st["textos"] if t["id"] == "00-consentimiento"), ""
            )
            rec = {
                "nombre": nombre,
                "aceptado_en": datetime.datetime.now()
                .astimezone()
                .isoformat(timespec="seconds"),
                "texto": texto,
                "grabacion": audio,
            }
            with open(
                os.path.join(ent_dir(self.out), "consentimiento.json"),
                "w",
                encoding="utf-8",
            ) as fh:
                json.dump(rec, fh, ensure_ascii=False, indent=1)
            print(f"training consent accepted by {nombre} (audio {audio})", flush=True)
            return self._json(rec)
        if route == "/entreno/save":
            scene = re.sub(r"[^\w-]+", "-", q.get("id", "")).strip("-")[:40]
            ronda = q.get("ronda", "1")
            st = ent_state(self.out)
            if scene not in {t["id"] for t in st["textos"]} or ronda not in (
                "1",
                "2",
                "3",
            ):
                return self._json({"error": "texto o ronda inválidos"}, 400)
            if scene != "00-consentimiento" and not st["consentimiento"]:
                return self._json(
                    {
                        "error": "falta la autorización: graba el texto 00 y acepta por escrito"
                    },
                    403,
                )
            data = self._body(limit=500 * 1024 * 1024)
            if not data or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
                return self._json({"error": "not a wav"}, 400)
            base = f"ent-{scene}-r{ronda}-{datetime.datetime.now().strftime('%H%M%S')}"
            name, n = f"{base}.wav", 1
            while os.path.exists(os.path.join(ent_dir(self.out), name)):
                n += 1
                name = f"{base}-{n}.wav"
            with open(os.path.join(ent_dir(self.out), name), "wb") as fh:
                fh.write(data)
            print(f"saved training take {name} ({len(data) // 1024} KB)", flush=True)
            return self._json({"file": name})
        if route == "/prueba":
            data = self._body(limit=40 * 1024 * 1024)
            if not data or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
                return self._json({"error": "la prueba no llegó como WAV"}, 400)
            try:
                silence = min(10.0, max(0.5, float(q.get("silencio", "3"))))
            except ValueError:
                return self._json({"error": "inválido"}, 400)
            label = re.sub(r"[^\w ()\-.áéíóúñÁÉÍÓÚÑ]", "", q.get("mic", ""))[:80]
            folder = os.path.join(self.out, "pruebas")
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(
                folder, datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".wav"
            )
            with open(path, "wb") as fh:
                fh.write(data)
            try:
                r = subprocess.run(
                    [
                        AUTOFLOW_PY,
                        os.path.join(HERE, "mictest.py"),
                        path,
                        str(silence),
                        label,
                    ],
                    cwd=APPLIO,
                    capture_output=True,
                    text=True,
                    timeout=180,
                )
                result = json.loads(r.stdout.strip().splitlines()[-1])
            except Exception as exc:
                return self._json(
                    {"error": "no se pudo analizar la prueba: " + str(exc)[-200:]}, 500
                )
            print(
                f"mic test {os.path.basename(path)}: {result.get('veredicto')} ({label})",
                flush=True,
            )
            return self._json(result)
        if route == "/elegir":
            try:
                b = json.loads(self._body() or b"")
            except ValueError:
                return self._json({"error": "inválido"}, 400)
            tono, scene, take = b.get("tono"), str(b.get("id", "")), b.get("file")
            if tono not in guion.TONOS or not re.fullmatch(r"[\w-]{1,40}", scene):
                return self._json({"error": "inválido"}, 400)
            if take is not None and (
                guion.parse_take(take) != (scene, tono)
                or not os.path.isfile(os.path.join(self.out, take))
            ):
                return self._json(
                    {"error": "esa toma no es de esta escena y tono"}, 400
                )
            guion.elegir(self.out, tono, scene, take)
            return self._json({"ok": True})
        if route != "/save":
            return self._send(404, '{"error":"not found"}')
        length = int(self.headers.get("Content-Length", 0))
        if not 44 < length < 500 * 1024 * 1024:
            return self._json({"error": "empty take"}, 400)
        data = self.rfile.read(length)
        if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
            return self._json({"error": "not a wav"}, 400)
        stem = re.sub(r"[^\w-]+", "-", q.get("name", "toma")).strip("-")[:40] or "toma"
        tono = q.get("tono", "casual")
        if tono not in guion.TONOS:
            tono = "casual"
        base = f"{stem}-{tono}-{datetime.datetime.now().strftime('%H%M%S')}"
        name, n = f"{base}.wav", 1
        while os.path.exists(
            os.path.join(self.out, name)
        ):  # same second, or the same time another day
            n += 1
            name = f"{base}-{n}.wav"
        with open(os.path.join(self.out, name), "wb") as fh:
            fh.write(data)
        print(f"saved {name} ({length // 1024} KB)", flush=True)
        self._json({"file": name})


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.join("assets", "recordings", "voz"))
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    Studio.out = os.path.abspath(args.out)
    print(f"studio on 127.0.0.1:{args.port}, takes -> {Studio.out}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Studio).serve_forever()


if __name__ == "__main__":
    main()

"""The script (guion.json) and the takes recorded against it. Stdlib only:
the booth server and the processing scripts run in different venvs.

guion.json is a list of scenes. Each scene carries its copy per tone:

    {"id": "01-gancho", "titulo": "La escena",
     "casual":      {"texto": "...", "nota": "..."},
     "profesional": {"texto": "...", "nota": "..."}}

The older form ({"id", "titulo", "texto", "nota"}) still loads: the same copy
serves both tones. A plain guion.txt is one scene.

A take is <scene id>-<tone>-<HHMMSS>.wav. Takes from before tones existed
(<scene id>-<HHMMSS>.wav) count as casual, which is how that script was written.
elegidas.json holds the take picked per scene and tone ("★ usar esta").
"""

import json
import os
import re

TONOS = ("casual", "profesional")
TAKE_RE = re.compile(r"^(?P<stem>[\w-]+?)(?:-(?P<tono>casual|profesional))?-(?P<hora>\d{6})(?:-\d+)?\.wav$")


def load(folder):
    """The scenes, normalized: [{id, titulo, casual: {texto, nota}, profesional: {...}}]."""
    for name in ("guion.json", "guion.txt"):
        path = os.path.join(folder, name)
        if os.path.exists(path):
            raw = open(path, encoding="utf-8").read()
            break
    else:
        return []
    try:
        scenes = json.loads(raw)
    except ValueError:
        scenes = [{"id": "toma", "titulo": "Guion", "texto": raw}] if raw.strip() else []
    out = []
    for s in scenes:
        same = {"texto": s.get("texto", ""), "nota": s.get("nota", "")}
        out.append({"id": s["id"], "titulo": s.get("titulo", s["id"]),
                    **{t: {**same, **(s.get(t) or {})} for t in TONOS}})
    return out


def parse_take(name):
    """(stem, tono) for a take file name, or None if it is not a take."""
    m = TAKE_RE.match(name)
    if not m:
        return None
    return m["stem"], m["tono"] or "casual"


def takes(folder):
    """Every take in the folder, oldest first: [(path, stem, tono)]."""
    out = []
    for f in os.listdir(folder):
        p = parse_take(f)
        if p:
            out.append((os.path.join(folder, f), *p))
    return sorted(out, key=lambda t: os.path.getmtime(t[0]))


def elegidas(folder):
    """{tono: {scene id: take file}} — the take picked for each scene."""
    path = os.path.join(folder, "elegidas.json")
    try:
        data = json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    return {t: dict(data.get(t) or {}) for t in TONOS}


def elegir(folder, tono, scene, take):
    """Pick (or, with take=None, un-pick) the take for one scene and tone."""
    data = elegidas(folder)
    if take:
        data[tono][scene] = take
    else:
        data[tono].pop(scene, None)
    path = os.path.join(folder, "elegidas.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, path)
    return data

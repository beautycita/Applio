"""Prueba de micrófono: is this mic, in this room, good enough for the ad?

    ~/.local/share/remifasol-autoflow/venv/bin/python tools/studio/mictest.py <prueba.wav> <silence seconds> [mic label]

The booth records a short clip: a few seconds of silence (the room), then her
reading a scene. This measures what made the 10-23 session unusable and says
APTO / APTO CON AJUSTES / NO APTO in plain Spanish:

  brillo    the 8-12 kHz level against the voice's 1-3 kHz body. Calibrated
            2026-10-26: the 10-23 Owl take -27 dB (its highs fade early: the
            "tinny" sound, with the echo); a full-band voice -10 dB; a 16 kHz
            USB mic about -55 dB (nothing past 8 kHz). Also reported: how high
            the voice still reaches over the room ("alcance").
  ruido     the room and the mic's own hiss, from the silent part.
  eco       how long a word rings after it ends, from clean word endings only
            (calibrated: the Owl take 0.42 s, a voice with no room 0.14 s).
  nivel     speech level, and whether it clipped.
  zumbido   mains hum (60/120/180 Hz) in the silent part.

It also bounces the reading through the SoundCraft "natural" chain so the
result can be heard raw and polished. Prints one JSON object.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as pl  # noqa: E402

RATE = pl.RATE
FR = 960  # 20 ms frames
FAR = ("owl", "array", "matriz", "webcam", "cámara", "camara", "camera", "realtek", "internal", "interno",
       "integrad", "built-in")


def load(path):
    raw = pl.run_bytes(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(RATE), "-f", "f32le", "-"])
    return np.frombuffer(raw, np.float32).astype(np.float64)


def frame_db(x):
    n = len(x) // FR
    if n == 0:
        return np.array([])
    return 10 * np.log10((x[:n * FR].reshape(n, FR) ** 2).mean(1) + 1e-12)


def ltas(x, win):
    """Long-term average power spectrum (linear) over x, Hann windows, 50 % overlap."""
    if len(x) < win:
        return None
    w = np.hanning(win)
    acc, n = np.zeros(win // 2 + 1), 0
    for i in range(0, len(x) - win, win // 2):
        acc += np.abs(np.fft.rfft(x[i:i + win] * w)) ** 2
        n += 1
    return acc / max(n, 1)


def band_db(spec, win, lo, hi):
    hz = RATE / win
    a, b = int(lo / hz), max(int(lo / hz) + 1, int(hi / hz))
    return 10 * np.log10(spec[a:b].mean() + 1e-20)


def voiced(x, fdb, floor):
    """The samples of frames that carry voice (15 dB over the room)."""
    keep = np.repeat(fdb > floor + 15, FR)
    return x[:len(keep)][keep]


def brillo_db(speech):
    """8-12 kHz against the 1-3 kHz body, in dB (power mean)."""
    win = 2048
    sp = ltas(speech, win)
    if sp is None:
        return None
    return float(band_db(sp, win, 8000, 12000) - band_db(sp, win, 1000, 3000))


def reach_khz(speech, silence):
    """Highest frequency (kHz) the voice keeps reaching: 500 Hz bands from 4 kHz up
    where speech is 6 dB over the room AND within 45 dB of its 1-3 kHz body."""
    win = 2048
    sp, si = ltas(speech, win), ltas(silence, win)
    if sp is None or si is None:
        return None
    body = band_db(sp, win, 1000, 3000)
    top = 4.0
    f = 4000
    while f + 500 <= 20000:
        s, n = band_db(sp, win, f, f + 500), band_db(si, win, f, f + 500)
        if s - n < 6 or s - body < -40:
            break
        f += 500
        top = f / 1000
    return top


def echo_s(x, floor):
    """Median ring-out (RT60, s) from clean word endings: a 5 ms-frame peak 30 dB
    over the room that falls 25 dB with no new syllable starting (no rise over
    3 dB), fitted between 5 and 25 dB down."""
    fr = 240
    n = len(x) // fr
    if n < 200:
        return None
    d = 10 * np.log10((x[:n * fr].reshape(n, fr) ** 2).mean(1) + 1e-12)
    d = np.convolve(d, np.ones(3) / 3, "same")
    slopes = []
    for i in range(4, len(d) - 100):
        if d[i] > floor + 30 and d[i] == d[i - 4:i + 5].max():
            seg = d[i:i + 100]
            j = int(np.argmax(seg < d[i] - 25))
            if j == 0:
                continue
            run = seg[:j + 1]
            if np.any(np.diff(run) > 3):
                continue
            k = np.where((run <= d[i] - 5) & (run >= d[i] - 25))[0]
            if len(k) >= 3:
                slopes.append(np.polyfit(k * 0.005, run[k], 1)[0])
    if len(slopes) < 8:
        return None
    return float(-60 / np.median(slopes))


def hum_db(silence):
    win = 8192
    sp = ltas(silence, win)
    if sp is None:
        return None
    hz = RATE / win
    worst = -99.0
    for h in (60, 120, 180):
        j = round(h / hz)
        near = np.concatenate([sp[j - 8:j - 3], sp[j + 4:j + 9]]).mean()
        worst = max(worst, 10 * np.log10(sp[j - 1:j + 2].max() / (near + 1e-20)))
    return float(worst)


def analyze(path, silence_s, label=""):
    x = load(path)
    cut = int(max(0.5, silence_s) * RATE)
    silence = x[int(0.3 * RATE):max(int(0.3 * RATE) + FR, cut - int(0.2 * RATE))]  # skip the click, and the breath before reading
    speech_all = x[cut:]
    sdb = frame_db(silence)
    fdb = frame_db(speech_all)
    # she may talk or breathe during the "silence": never let that raise the floor
    quiet = float(np.percentile(fdb, 10)) if len(fdb) else -90.0
    floor = min(float(np.median(sdb)), quiet) if len(sdb) else quiet
    speech = voiced(speech_all, fdb, floor)
    out = {"ruido_db": round(floor, 1), "microfono": label}
    checks = []  # (key, name, value text, colour, tip or None)

    if len(speech) < RATE * 2:
        out.update(veredicto="NO APTO", color="r", checks=[{
            "clave": "voz", "nombre": "Voz", "valor": "no se oyó", "color": "r",
            "tip": "No se escuchó voz en la prueba. Revisa que el micrófono elegido arriba sea el de solapa, que no esté en silencio en Windows, y lee en voz normal cuando lo indique la pantalla."}])
        return out

    # The room's own sound: the quietest moments of the whole clip (the silent
    # part if she kept quiet, the pauses between phrases if she did not).
    allx = x[int(0.3 * RATE):]
    adb = frame_db(allx)
    quiet_mask = np.repeat(adb < np.percentile(adb, 15) + 3, FR)
    room = allx[:len(quiet_mask)][quiet_mask]
    if len(room) < RATE // 2:
        room = silence
    level = float(np.median(fdb[fdb > floor + 15]))
    clips = int((np.abs(speech_all) > 0.98).sum())
    reach = reach_khz(speech, room)
    brillo = brillo_db(speech)
    echo = echo_s(speech_all, floor)
    hum = hum_db(room)
    out.update(nivel_db=round(level, 1), saturaciones=clips, alcance_khz=reach,
               brillo_db=None if brillo is None else round(brillo, 1),
               eco_s=None if echo is None else round(echo, 2), zumbido_db=None if hum is None else round(hum, 1))

    # brillo (the tinny test)
    reach_txt = f" · llega a {reach:g} kHz" if reach else ""
    if brillo is None:
        checks.append(("brillo", "Brillo", "sin medir", "y", "La prueba fue muy corta para medir el brillo. Repítela leyendo toda la ventana."))
    elif brillo >= -24:
        checks.append(("brillo", "Brillo", f"{brillo:.0f} dB{reach_txt}", "g", None))
    elif brillo >= -32:
        checks.append(("brillo", "Brillo", f"{brillo:.0f} dB{reach_txt}", "y",
                       "La voz llega con pocos agudos: se oirá algo opaca o «de teléfono». Revisa que el micrófono quede por fuera de la ropa, a un palmo de la boca, con la cabeza hacia arriba y sin nada que lo cubra."))
    else:
        checks.append(("brillo", "Brillo", f"{brillo:.0f} dB{reach_txt}", "r",
                       "Casi no hay agudos: esto es lo que sonó «metálico» la vez pasada. Si el micrófono elegido es el de solapa y está bien puesto, ese micrófono no sirve para el anuncio; si es otro (el Owl, la laptop), elige el de solapa."))
    # ruido
    if floor < -60:
        checks.append(("ruido", "Ruido de fondo", f"{floor:.0f} dB", "g", None))
    elif floor < -50:
        checks.append(("ruido", "Ruido de fondo", f"{floor:.0f} dB", "y",
                       "Hay algo de ruido: apaga ventilador y aire acondicionado, cierra la puerta, aleja la laptop."))
    else:
        checks.append(("ruido", "Ruido de fondo", f"{floor:.0f} dB", "r",
                       "Mucho ruido de fondo (cuarto o el propio micrófono). Apaga todo lo que zumbe y repite la prueba; si sigue igual en silencio total, el micrófono mete su propio ruido."))
    # eco
    if echo is None:
        checks.append(("eco", "Eco", "sin medir", "", None))
    elif echo < 0.25:
        checks.append(("eco", "Eco", f"{echo:.2f} s", "g", None))
    elif echo < 0.4:
        checks.append(("eco", "Eco", f"{echo:.2f} s", "y", "Algo de eco: cobijas o almohadas en las paredes cercanas, o graba en la recámara."))
    else:
        checks.append(("eco", "Eco", f"{echo:.2f} s", "r",
                       "Eco alto: la voz rebota. Revisa que el micrófono esté a un palmo de la boca y pon cobijas detrás de ti y en las paredes cercanas."))
    # nivel
    if clips:
        checks.append(("nivel", "Volumen", f"{level:.0f} dB · satura", "r", "Se satura: baja el volumen de entrada en Windows (Configuración › Sonido › Entrada)."))
    elif level < -38:
        checks.append(("nivel", "Volumen", f"{level:.0f} dB", "r", "Muy bajo: sube el volumen de entrada del micrófono de solapa en Windows hasta que la barra quede verde al hablar."))
    elif level < -30 or level > -12:
        checks.append(("nivel", "Volumen", f"{level:.0f} dB", "y", "Ajusta el volumen de entrada en Windows para que la barra quede verde al hablar normal."))
    else:
        checks.append(("nivel", "Volumen", f"{level:.0f} dB", "g", None))
    # zumbido
    if hum is not None and hum > 10:
        checks.append(("zumbido", "Zumbido", "sí", "y", "Zumbido eléctrico: desconecta el cargador de la laptop mientras grabas y aleja el cable del micrófono de cables de corriente."))
    # the mic itself
    if label and any(k in label.lower() for k in FAR):
        checks.append(("mic", "Micrófono", label, "r", "La prueba se hizo con un micrófono lejano, no con el de solapa. Elígelo arriba y repite."))

    colours = [c[3] for c in checks]
    if "r" in colours:
        out.update(veredicto="NO APTO", color="r")
    elif "y" in colours:
        out.update(veredicto="APTO, CON AJUSTES", color="y")
    else:
        out.update(veredicto="APTO", color="g")
    out["checks"] = [{"clave": k, "nombre": n, "valor": v, "color": c, "tip": t} for k, n, v, c, t in checks]
    return out


def main(path, silence_s, label=""):
    result = analyze(path, silence_s, label)
    # the reading alone, raw and through the SoundCraft natural chain
    base = os.path.splitext(path)[0]
    lectura = base + "-lectura.wav"
    pl.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{silence_s:.3f}", "-i", path, "-c:a", "pcm_s24le", lectura])
    try:
        pl.polish(lectura, base + "-pulida.wav", chain="natural")
        result["pulida"] = os.path.basename(base + "-pulida.wav")
    except Exception as exc:  # the verdict stands without the polished copy
        result["pulida_error"] = str(exc)[-200:]
    result["cruda"] = os.path.basename(lectura)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "")

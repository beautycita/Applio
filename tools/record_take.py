"""Sing over a backing track: play it in the headphones, record the mic, and
save a vocal take that is sample-aligned with the track.

The headphone feed is recorded alongside the mic (same ffmpeg process, same
clock); cross-correlating that feed against the original track tells exactly
where the track started in the recording, so the take is trimmed to match.

    .venv/bin/python tools/record_take.py assets/audios/senorita.wav
"""

import argparse
import os
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

DEFAULT_MIC = "alsa_input.usb-Owl_Labs__Inc._Meeting_Owl_Pro_1536c470-02.iec958-stereo"
DEFAULT_PHONES = "alsa_output.pci-0000_00_1f.3.analog-stereo"
RATE = 48000


def mono(path, rate=RATE):
    data, sr = sf.read(path, always_2d=True)
    data = data.mean(axis=1)
    if sr != rate:
        import soxr

        data = soxr.resample(data, sr, rate)
    return data.astype(np.float32)


def find_offset(recorded, reference, seconds=20):
    """Sample index in `recorded` where `reference` starts."""
    ref = reference[: seconds * RATE]
    corr = fftconvolve(recorded, ref[::-1], mode="valid")
    return int(np.argmax(np.abs(corr)))


def next_take_path(out_dir, stem):
    n = 1
    while os.path.exists(os.path.join(out_dir, f"{stem}_take{n}.wav")):
        n += 1
    return os.path.join(out_dir, f"{stem}_take{n}.wav")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("track", help="backing track to sing over")
    ap.add_argument("--mic", default=DEFAULT_MIC)
    ap.add_argument("--phones", default=DEFAULT_PHONES, help="PipeWire sink for playback")
    ap.add_argument("--out-dir", default=os.path.join("assets", "audios"))
    ap.add_argument("--start", type=float, default=0.0, help="start the track at this second")
    ap.add_argument("--seconds", type=float, default=None, help="only record this long")
    args = ap.parse_args()

    stem = os.path.splitext(os.path.basename(args.track))[0]
    out_path = next_take_path(args.out_dir, stem)

    with tempfile.TemporaryDirectory() as tmp:
        track = os.path.join(tmp, "track.wav")
        trim = ["-ss", str(args.start)] + (["-t", str(args.seconds)] if args.seconds else [])
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", *trim, "-i", args.track, "-ar", str(RATE), "-ac", "2", track],
            check=True,
        )
        mic_raw = os.path.join(tmp, "mic.wav")
        feed_raw = os.path.join(tmp, "feed.wav")
        recorder = subprocess.Popen(
            [
                "ffmpeg", "-v", "error", "-y",
                "-f", "pulse", "-i", args.mic,
                "-f", "pulse", "-i", f"{args.phones}.monitor",
                "-map", "0:a", "-ar", str(RATE), mic_raw,
                "-map", "1:a", "-ar", str(RATE), feed_raw,
            ],
            stdin=subprocess.PIPE,
        )
        time.sleep(1.0)
        print("Recording. The track starts now; sing along. Ctrl+C stops early.", flush=True)
        try:
            subprocess.run(["pw-play", "--target", args.phones, track], check=True)
            time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nStopped early.")
        recorder.send_signal(signal.SIGINT)
        recorder.wait(timeout=10)

        reference = mono(track)
        feed = mono(feed_raw)
        offset = find_offset(feed, reference)
        voice, _ = sf.read(mic_raw, always_2d=True)
        voice = voice.mean(axis=1)[offset : offset + len(reference)]
        if len(voice) < len(reference):
            voice = np.pad(voice, (0, len(reference) - len(voice)))
        peak = np.max(np.abs(voice)) or 1.0
        sf.write(out_path, (voice / peak * 0.9).astype(np.float32), RATE, subtype="PCM_16")

    print(f"Saved {out_path}  (track found {offset / RATE:.3f}s into the recording)")


if __name__ == "__main__":
    sys.exit(main())

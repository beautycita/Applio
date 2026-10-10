# ReMiFaSol

ReMiFaSol turns one voice into another: give it a recording of someone singing or
speaking, pick a voice model, and it re-sings the performance in that voice while
keeping the melody and timing. It also trains new voice models from your own recordings.

## What's inside

- **Inference**: convert a recording into another voice (pitch shift, index strength, pitch detector, effects).
- **TTS**: type text, synthesize it, and convert it to a voice model.
- **Training**: build your own voice model from 10–30 minutes of clean audio.
- **Download**: fetch voice models and pretrained models by link.
- **Voice Blender**, **Realtime**, **Extra** (audio analysis, F0 tools), **Plugins**, **TensorBoard**, **Settings**.

Voice models live in `logs/<voice-name>/` (a `.pth` plus an optional `.index`);
audio you want to convert goes in `assets/audios/`.

### Recording over a backing track

```bash
.venv/bin/python tools/record_take.py assets/audios/track.wav
```

Plays the track in your headphones while recording the mic, then saves
`assets/audios/track_takeN.wav` aligned sample-for-sample with the track.
Use `--start`/`--seconds` to rehearse a section and `--mic`/`--phones` to pick devices.

## Install (Linux / macOS)

```bash
./run-install.sh
```

The installer creates a Python 3.12 virtual environment with [uv](https://github.com/astral-sh/uv)
and picks the right PyTorch build for your GPU. Cards older than compute
capability 7.0 (Maxwell, Pascal) get the CUDA 12.6 wheels, newer cards get CUDA 12.8.

Model files download from the private Hugging Face repo `bc-deb/ReMiFaSol`, so log in
once with an account that can read it before the first launch:

```bash
.venv/bin/hf auth login
```

## Run

```bash
./run-remifasol.sh            # opens http://127.0.0.1:6969/
./run-remifasol.sh --share    # temporary public link
./run-remifasol.sh --port 7000
./run-tensorboard.sh          # training graphs
```


## License

MIT. See [LICENSE](./LICENSE).

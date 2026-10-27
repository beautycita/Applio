#!/bin/sh
# Runs INSIDE a Hugging Face Job (python:3.12 image, 1x L40S), started by
# train.py when the training page chooses "Nube". It trains one voice model:
#   in  : <REPO>/<RUN>/dataset/*.wav   (private dataset repo, already checked as Spanish on bc-dev)
#   out : <REPO>/<RUN>/<ENC>/*.pth + *.index   (every saved epoch; the Spanish exam runs on bc-dev)
# Env: REPO RUN MODEL ENC EPOCHS EVERY BATCH APPLIO_REF; secret HF_TOKEN.
# Progress lines "@@ <stage>" and Applio's own "| epoch=N |" are read from the job log.
set -eu
say() { echo "@@ $1"; }

say instalar
apt-get update -qq >/dev/null && apt-get install -y -qq ffmpeg >/dev/null
git clone -q https://github.com/beautycita/Applio /applio
cd /applio && git checkout -q "$APPLIO_REF"
# Applio writes each epoch's small model only when its runtime config exists (gitignored, made on first UI run)
cp assets/config_template.json assets/config.json
pip install -q --no-cache-dir torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu126
# the UI and audio-device packages are not needed to train
grep -vE '^(torch|torchaudio|gradio|sounddevice|edge-tts)==' requirements.txt > /tmp/req.txt
pip install -q --no-cache-dir -r /tmp/req.txt huggingface_hub
python core.py prerequisites --pretraineds-hifigan --models --no-exe

say datos
# the library, not the hf CLI: Applio's pinned click/typer break the CLI's exit (status 1 after a good download)
python -c "import os; from huggingface_hub import snapshot_download as d; d(os.environ['REPO'], repo_type='dataset', allow_patterns=[os.environ['RUN'] + '/dataset/*'], local_dir='/data')" >/dev/null

say preparar
python core.py preprocess --model-name "$MODEL" --dataset-path "/data/$RUN/dataset" --sample-rate 48000 \
  --cpu-cores 8 --cut-preprocess Automatic --chunk-len 3.0 --overlap-len 0.3 --normalization-mode post
say rasgos
python core.py extract --model-name "$MODEL" --f0-method rmvpe --cpu-cores 8 --gpu 0 --sample-rate 48000 \
  --embedder-model "$ENC" --include-mutes 2
say entrenar
python core.py train --model-name "$MODEL" --vocoder HiFi-GAN --save-every-epoch "$EVERY" --save-only-latest \
  --total-epoch "$EPOCHS" --sample-rate 48000 --batch-size "$BATCH" --gpu 0 --pretrained --index-algorithm Auto
say indice
python core.py index --model-name "$MODEL" --index-algorithm Auto

say subir
mkdir -p /out
cp logs/"$MODEL"/"$MODEL"_*e_*s.pth logs/"$MODEL"/*.index /out/
python -c "import os; from huggingface_hub import upload_folder as u; e = os.environ; u(folder_path='/out', path_in_repo=e['RUN'] + '/' + e['ENC'], repo_id=e['REPO'], repo_type='dataset', commit_message='trained ' + e['MODEL'])" >/dev/null
say listo

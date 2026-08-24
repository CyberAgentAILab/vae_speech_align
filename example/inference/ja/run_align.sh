#!/bin/bash
# Align the bundled Japanese samples with the pretrained CSJ model.
#
# Usage:
#   uv run bash run_align.sh
#
# The pretrained model (model.safetensors) is downloaded from the repository's
# GitHub Releases on first run.
#
# Outputs (under output/):
#   textgrid/*.textgrid  Praat TextGrids with token/state tiers
#   token/*.npy          per-phoneme durations in frames (10 ms shift)
#   state/*.npy          per-state durations in frames

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$SCRIPT_DIR"

# Fetch the pretrained model from GitHub Releases on first run.
MODEL_RELEASE_TAG="models-v3"
MODEL_ASSET="model_ja.safetensors"
MODEL_REPO="CyberAgentAILab/vae_speech_align"
MODEL_SHA256="2ab637d8d26ec47fa914c2692e1a16e7fed0db350979f84d8d45a6f32e3c0234"
if [ ! -f model.safetensors ]; then
    echo "Downloading the pretrained model ($MODEL_ASSET)..."
    if command -v gh > /dev/null 2>&1; then
        gh release download "$MODEL_RELEASE_TAG" --repo "$MODEL_REPO" \
            --pattern "$MODEL_ASSET" --output model.safetensors
    else
        curl -fL -o model.safetensors "https://github.com/$MODEL_REPO/releases/download/$MODEL_RELEASE_TAG/$MODEL_ASSET"
    fi
    if ! echo "$MODEL_SHA256  model.safetensors" \
            | shasum -a 256 --check --status -; then
        rm -f model.safetensors
        echo "ERROR: model.safetensors does not match the expected" \
            "SHA-256 checksum; the download may be corrupted or" \
            "tampered with" >&2
        exit 1
    fi
fi

python make_phoneme.py data/text data/phoneme_txt

python -m vae_speech_align.cli.align \
    --exp_config config.yaml \
    --model_ckpt model.safetensors \
    --out_dir output \
    --wav_dir data/wav \
    --phoneme_dir data/phoneme_txt \
    --device cuda \
    --batch_size 4 \
    --impl numba

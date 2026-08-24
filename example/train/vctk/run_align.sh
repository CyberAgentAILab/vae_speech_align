#!/bin/bash
# Run Viterbi alignment on the VCTK corpus.
#
# Usage:
#   uv run bash run_align.sh
#
# Prerequisites:
#   Run prepare_data.sh and run_train.sh first to obtain a trained model.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$SCRIPT_DIR"

python -m vae_speech_align.cli.align \
    --exp_config config.yaml \
    --model_ckpt output/train/model.safetensors \
    --out_dir output/align \
    --wav_dir data/wav \
    --phoneme_dir data/phoneme_txt \
    --device cuda \
    --batch_size 4 \
    --impl triton

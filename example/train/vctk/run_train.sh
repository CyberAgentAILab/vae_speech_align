#!/bin/bash
# Train an alignment model on the VCTK corpus.
#
# Usage:
#   uv run bash run_train.sh
#
# Prerequisites:
#   Run prepare_data.sh first to set up data/wav and data/phoneme_txt.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$SCRIPT_DIR"

python -m vae_speech_align.cli.train \
    --exp_config config.yaml \
    --out_dir output/train \
    --wav_dir data/wav \
    --phoneme_dir data/phoneme_txt \
    --device cuda \
    --impl triton

#!/bin/bash
# Prepare JVS data for alignment training and prediction.
#
# Usage:
#   uv run bash prepare_data.sh /path/to/jvs_ver1
#
# This script:
#   1. Extracts phoneme labels from the bundled phoneme.tar.gz
#   2. Copies and renames WAV files from the JVS corpus
#      jvs{id}/parallel100/wav24kHz16bit/{subset}_{num}.wav
#      -> jvs{id}_{subset}_{num}.wav

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 /path/to/jvs_ver1"
    exit 1
fi

JVS_DIR="$1"

if [ ! -d "$JVS_DIR" ]; then
    echo "Error: JVS directory not found: $JVS_DIR"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"

# --- Extract phoneme labels ---
echo "Extracting phoneme labels..."
tar xzf "$DATA_DIR/phoneme.tar.gz" -C "$DATA_DIR"
echo "  -> $DATA_DIR/phoneme_txt/ ($(ls "$DATA_DIR/phoneme_txt" | wc -l) files)"

# --- Copy and rename WAV files ---
echo "Copying WAV files from $JVS_DIR ..."

wav_dir="$DATA_DIR/wav"
mkdir -p "$wav_dir"

count=0
for phoneme_file in "$DATA_DIR/phoneme_txt"/*.txt; do
    base="$(basename "$phoneme_file" .txt)"
    # base = jvs001_VOICEACTRESS100_001
    speaker_id="${base%%_*}"
    wav_name="${base#*_}"  # VOICEACTRESS100_001

    src_wav="$JVS_DIR/$speaker_id/parallel100/wav24kHz16bit/$wav_name.wav"
    dst_wav="$wav_dir/$base.wav"

    if [ -f "$src_wav" ]; then
        cp "$src_wav" "$dst_wav"
        count=$((count + 1))
    else
        echo "  Warning: not found: $src_wav"
    fi
done
echo "  -> $wav_dir/ ($count files)"

echo "Done."

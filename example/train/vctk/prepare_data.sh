#!/bin/bash
# Prepare VCTK data for alignment training and prediction.
#
# Usage:
#   uv run bash prepare_data.sh /path/to/VCTK-Corpus
#
# Both official distributions are supported; the layout is detected
# automatically:
#   - VCTK 0.92 (wav48_silence_trimmed/*.flac): phoneme labels are
#     downloaded from kan-bayashi/VCTKCorpusFullContextLabel and
#     silence timing from nii-yamagishilab/vctk-silence-labels.
#   - VCTK 0.80 (wav48/*.wav): HTS mono labels in $VCTK_DIR/lab/mono
#     are used when present; otherwise the same labels are downloaded.
#
# This script:
#   1. Generates phoneme labels from HTS mono labels using
#      make_phoneme.py
#   2. Copies audio files with silence trimming using prepare_wav.py

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 /path/to/VCTK-Corpus"
    exit 1
fi

VCTK_DIR="$1"

if [ ! -d "$VCTK_DIR" ]; then
    echo "Error: VCTK directory not found: $VCTK_DIR"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"
DOWNLOAD_DIR="$DATA_DIR/downloads"

LABEL_REPO_URL="https://github.com/kan-bayashi/VCTKCorpusFullContextLabel.git"
SILENCE_REPO_URL="https://github.com/nii-yamagishilab/vctk-silence-labels.git"
DOWNLOADED_LAB_DIR="$DOWNLOAD_DIR/VCTKCorpusFullContextLabel/lab/mono"
SILENCE_FILE="$DOWNLOAD_DIR/vctk-silence-labels/vctk-silences.0.92.txt"

download_mono_labels() {
    if [ ! -d "$DOWNLOADED_LAB_DIR" ]; then
        echo "Downloading HTS mono labels..."
        git clone --depth 1 "$LABEL_REPO_URL" \
            "$DOWNLOAD_DIR/VCTKCorpusFullContextLabel"
    fi
}

download_silence_labels() {
    if [ ! -f "$SILENCE_FILE" ]; then
        echo "Downloading silence labels..."
        git clone --depth 1 "$SILENCE_REPO_URL" \
            "$DOWNLOAD_DIR/vctk-silence-labels"
    fi
}

if [ -d "$VCTK_DIR/wav48_silence_trimmed" ]; then
    echo "Detected VCTK 0.92 layout (wav48_silence_trimmed)."
    download_mono_labels
    download_silence_labels
    LAB_DIR="$DOWNLOADED_LAB_DIR"
    PREPARE_WAV_ARGS=(--silence_file "$SILENCE_FILE")
elif [ -d "$VCTK_DIR/wav48" ]; then
    echo "Detected VCTK 0.80 layout (wav48)."
    LAB_DIR="$VCTK_DIR/lab/mono"
    if [ ! -d "$LAB_DIR" ]; then
        download_mono_labels
        LAB_DIR="$DOWNLOADED_LAB_DIR"
    fi
    PREPARE_WAV_ARGS=(--lab_dir "$LAB_DIR")
else
    echo "Error: neither wav48_silence_trimmed nor wav48 found" \
        "under $VCTK_DIR"
    exit 1
fi

# --- Generate phoneme labels ---
echo "Generating phoneme labels..."
python "$SCRIPT_DIR/make_phoneme.py" "$LAB_DIR" "$DATA_DIR/phoneme_txt"

# --- Copy audio files with silence trimming ---
echo "Preparing WAV files (trimming silence)..."
python "$SCRIPT_DIR/prepare_wav.py" "$VCTK_DIR" "$DATA_DIR/phoneme_txt" \
    "$DATA_DIR/wav" "${PREPARE_WAV_ARGS[@]}"

echo "Done."

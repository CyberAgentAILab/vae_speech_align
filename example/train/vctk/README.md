# VCTK Alignment Example

Train a VAE-based speech alignment model on the
[CSTR VCTK Corpus](https://datashare.ed.ac.uk/handle/10283/3443).

## Setup

1. Download the VCTK corpus. Both official distributions are
   supported:
   - **Version 0.92** (recommended; available from the
     [official site](https://datashare.ed.ac.uk/handle/10283/3443)):
     `wav48_silence_trimmed/*.flac` with `_mic1`/`_mic2` tracks.
     `prepare_data.sh` uses the `mic1` track.
   - **Version 0.80**: `wav48/*.wav`. If the copy includes HTS mono
     labels under `lab/mono`, they are used as is.
2. Prepare the data (the corpus layout is detected automatically):
   ```bash
   uv run bash prepare_data.sh /path/to/VCTK-Corpus
   ```
   Phoneme sequences are generated from HTS mono label files, and
   leading/trailing silence is trimmed using per-utterance speech
   timing. `prepare_data.sh` downloads the following resources into
   `data/downloads/` as needed:
   - [kan-bayashi/VCTKCorpusFullContextLabel](https://github.com/kan-bayashi/VCTKCorpusFullContextLabel)
     — HTS mono labels (phoneme sequences; timing refers to the 0.80
     audio)
   - [nii-yamagishilab/vctk-silence-labels](https://github.com/nii-yamagishilab/vctk-silence-labels)
     — speech start/end times for the 0.92 audio (used for trimming
     with the 0.92 layout)
3. Train:
   ```bash
   uv run bash run_train.sh
   ```
   With the default configuration (`wav2vec2-large-xlsr-53` features,
   `num_workers: 4`), training uses more than 16 GB of host RAM and
   gets killed by the OOM killer on smaller machines; 32 GB or more
   is recommended.
4. Run alignment:
   ```bash
   uv run bash run_align.sh
   ```

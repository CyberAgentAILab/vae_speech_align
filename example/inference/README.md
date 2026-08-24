# Alignment Inference Example

Run phoneme alignment on your own audio with the pretrained models —
no training required. Two ready-to-run examples are provided:

| Directory | Language | Text front-end | Pretrained model |
|-----------|----------|----------------|------------------|
| [`ja/`](./ja) | Japanese | kana → phonemes (`vae_speech_align.g2p.kana`) | trained on [CSJ](https://clrd.ninjal.ac.jp/csj/en/) |
| [`en/`](./en) | English | G2P → VCTK ARPABET (`vae_speech_align.g2p.english`) | trained on [VCTK](../train/vctk) |

## Setup

The pretrained models (`ja/model.safetensors`, `en/model.safetensors`) are distributed
via [GitHub Releases](https://github.com/CyberAgentAILab/vae_speech_align/releases/tag/models-v3)
and are downloaded automatically the first time `run_align.sh` runs.
To fetch one manually, download the `model_ja.safetensors` / `model_en.safetensors`
asset and save it as `model.safetensors` in the corresponding directory.

Install the package (from the repository root):

```bash
uv sync          # or: pip install -e .
```

## Usage

```bash
cd example/inference/ja   # or example/inference/en
uv run bash run_align.sh
```

The script converts the sample texts to phoneme sequences and runs
Viterbi alignment on the bundled sample audio. Results are written to
`output/`:

- `textgrid/*.textgrid` — Praat TextGrids with token and state tiers
- `token/*.npy` — per-phoneme durations in frames (10 ms frame shift)
- `state/*.npy` — per-state durations in frames

To align your own recordings, put WAV files in `data/wav/` and matching
text files (same base name) in `data/text/` — katakana for `ja/`,
plain English for `en/` — then rerun `run_align.sh`.

## Directory layout

```
ja/ (en/ is analogous)
├── config.yaml          # model/feature configuration used at training
├── model.safetensors    # pretrained model (from GitHub Releases)
├── make_phoneme.py      # text → phoneme conversion
├── run_align.sh         # end-to-end alignment script
└── data/
    ├── all_phonemes.txt # phoneme inventory of the model
    ├── text/            # input sentences (one per file)
    ├── wav/             # input audio (resampled to 16 kHz on load)
    └── phoneme_txt/     # generated phoneme sequences (created by script)
```

## Sample audio

The WAV files under `data/wav/` were **generated with Gemini TTS**
(text-to-speech); they are synthetic speech, not human recordings.
The five sentences are parallel between Japanese and English:

| ID | Japanese (original) | Japanese input (kana) | English input |
|----|--------------------|----------------------|---------------|
| 01 | 明日の朝、雨が降る確率はどのくらい？ | アシタノアサ、アメガフルカクリツワドノクライ | What's the probability of rain tomorrow morning? |
| 02 | 明かりを消して。 | アカリオケシテ | Turn off the lights. |
| 03 | 遠くから微かな波の音が聞こえ、静かな夜がゆっくりと明けていく。 | トークカラカスカナナミノオトガキコエ、シズカナヨルガユックリトアケテイク | The faint sound of waves echoed from afar, as the quiet night slowly gave way to dawn. |
| 04 | まず、ボウルに卵と砂糖を入れて、白っぽくなるまでよく混ぜ合わせる。 | マズ、ボールニタマゴトサトーオイレテ、シロッポクナルマデヨクマゼアワセル | First, put the eggs and sugar in a bowl and mix well until the mixture turns pale. |
| 05 | まもなく、進行方向の右側に海が見えてまいります。 | マモナク、シンコーホーコーノミギガワニウミガミエテマイリマス | Soon, you will see the ocean on the right side in the direction of travel. |

Note that the kana input is pronunciation kana (yomigana): particles
are written as pronounced (は → ワ, を → オ) and long vowels with ー.

## Notes

- `run_align.sh` uses `--device cuda` but falls back to CPU
  automatically when CUDA is unavailable; the `numba` alignment
  implementation runs on both.
- The Viterbi DP backend is selected with `--impl`; both `numba` and
  `triton` are available and produce identical alignments:
  - `--impl numba` (used by the script) runs the DP on the CPU and
    needs no GPU kernel warm-up, which makes it the faster choice for
    a handful of short recordings like the bundled samples.
  - `--impl triton` runs the DP on the GPU (requires `--device cuda`).
    Each process pays a one-time kernel autotune, but the DP itself is
    then faster than `numba`, so it pays off for long recordings or
    large batches.
- The first run downloads the acoustic feature extractor from the
  Hugging Face Hub (`yky-h/japanese-hubert-base` for `ja/`,
  `facebook/wav2vec2-large-xlsr-53` for `en/`), and the English
  front-end downloads small NLTK resources on first use.
- The English model was trained with the recipe in
  [`example/train/vctk`](../train/vctk); the Japanese model was trained
  on the [CSJ (Corpus of Spontaneous Japanese)](https://clrd.ninjal.ac.jp/csj/en/)
  corpus.

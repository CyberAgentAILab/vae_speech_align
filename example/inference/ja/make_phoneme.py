"""Convert Japanese kana text files to phoneme text files.

Reads one katakana sentence per file from text_dir and writes the
corresponding space-separated phoneme sequence (wrapped in "sil") to
out_dir, using the same base filename.

Usage:
    python make_phoneme.py data/text data/phoneme_txt
"""

import argparse
from pathlib import Path

from vae_speech_align.g2p.kana import kana_text_to_phoneme


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "text_dir", type=str, help="Directory of kana text files"
    )
    parser.add_argument(
        "out_dir", type=str, help="Output directory for phoneme txt files"
    )
    args = parser.parse_args()

    text_dir = Path(args.text_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for text_file in sorted(text_dir.glob("*.txt")):
        kana_text = text_file.read_text().strip()
        phoneme_text = kana_text_to_phoneme(kana_text)
        (out_dir / text_file.name).write_text(phoneme_text + "\n")
        count += 1

    print(f"Created {count} phoneme files in {out_dir}")


if __name__ == "__main__":
    main()

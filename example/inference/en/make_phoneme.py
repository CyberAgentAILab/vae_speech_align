"""Convert English text files to VCTK-phoneme text files.

Reads one English sentence per file from text_dir and writes the
corresponding space-separated phoneme sequence (VCTK ARPABET set,
wrapped in "pau") to out_dir, using the same base filename.

Usage:
    python make_phoneme.py data/text data/phoneme_txt
"""

import argparse
from pathlib import Path

from vae_speech_align.g2p.english import EnglishG2p


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "text_dir", type=str, help="Directory of English text files"
    )
    parser.add_argument(
        "out_dir", type=str, help="Output directory for phoneme txt files"
    )
    args = parser.parse_args()

    text_dir = Path(args.text_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    g2p = EnglishG2p()

    count = 0
    for text_file in sorted(text_dir.glob("*.txt")):
        english_text = text_file.read_text().strip()
        phoneme_text = g2p.text_to_phoneme(english_text)
        (out_dir / text_file.name).write_text(phoneme_text + "\n")
        count += 1

    print(f"Created {count} phoneme files in {out_dir}")


if __name__ == "__main__":
    main()

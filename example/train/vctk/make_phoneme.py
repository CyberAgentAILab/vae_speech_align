"""Convert VCTK lab/mono files to phoneme text files.

Reads HTS-style monophone label files and writes space-separated
ARPABET phoneme sequences.  Consecutive pauses are merged.

Usage:
    python make_phoneme.py /path/to/VCTK-Corpus/lab/mono out_dir/
"""

import argparse
from pathlib import Path


def convert_lab_to_phoneme(in_label_file: Path) -> list[str]:
    """Read an HTS mono label file and return a phoneme list."""
    phonemes: list[str] = []
    with open(in_label_file) as f:
        for line in f:
            fields = line.strip().split()
            ph = fields[2]
            if phonemes and phonemes[-1] == "pau" and ph == "pau":
                continue
            phonemes.append(ph)

    return phonemes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "lab_dir", type=str, help="Path to VCTK-Corpus/lab/mono"
    )
    parser.add_argument(
        "out_dir", type=str, help="Output directory for phoneme txt files"
    )
    args = parser.parse_args()

    lab_dir = Path(args.lab_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for spk_dir in sorted(lab_dir.glob("*")):
        if not spk_dir.is_dir():
            continue
        # p315 has no corresponding wav files
        if spk_dir.name == "p315":
            continue

        for lab_file in sorted(spk_dir.glob("*.lab")):
            phonemes = convert_lab_to_phoneme(lab_file)
            out_file = out_dir / f"{lab_file.stem}.txt"
            out_file.write_text(" ".join(phonemes))
            count += 1

    print(f"Created {count} phoneme files in {out_dir}")


if __name__ == "__main__":
    main()

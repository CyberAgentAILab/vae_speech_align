"""Copy VCTK audio files with silence trimming.

Trims leading/trailing silence using per-utterance speech timing.  A
random margin (30-100 ms) is kept around the first and last non-pause
phoneme so that the waveform does not start or end abruptly.  The
timing source depends on the corpus version:

- VCTK 0.80 (``--lab_dir``): HTS mono label files, whose timestamps
  refer to the untrimmed ``wav48/<speaker>/<utterance>.wav`` files.
- VCTK 0.92 (``--silence_file``): the silence labels published by the
  corpus authors (https://github.com/nii-yamagishilab/vctk-silence-labels),
  whose timestamps refer to
  ``wav48_silence_trimmed/<speaker>/<utterance>_<mic>.flac``.

Usage:
    python prepare_wav.py /path/to/VCTK-Corpus phoneme_txt_dir out_dir \
        --lab_dir /path/to/VCTK-Corpus/lab/mono
    python prepare_wav.py /path/to/VCTK-Corpus-0.92 phoneme_txt_dir \
        out_dir --silence_file /path/to/vctk-silences.0.92.txt
"""

import argparse
from pathlib import Path
from typing import NamedTuple

import numpy as np
import soundfile as sf

HTK_TIME_UNITS_PER_SECOND = 10_000_000
MIN_MARGIN_SEC = 0.03
MAX_MARGIN_SEC = 0.1
PAUSE_PHONEME = "pau"
PROGRESS_INTERVAL = 1000


class TrimJob(NamedTuple):
    """One audio file with its speech span in seconds."""

    audio_file: Path
    speech_begin: float
    speech_end: float
    out_name: str


def read_speech_span_from_lab(lab_file: Path) -> tuple[float, float] | None:
    """Return (begin, end) seconds of speech in an HTS mono label file.

    Args:
        lab_file: HTS mono label file with lines of
            "<begin> <end> <phoneme>" in 100 ns units.

    Returns:
        Speech span in seconds, or None if the file only contains
        pauses.
    """
    speech_begin = None
    speech_end = None
    with open(lab_file) as label_stream:
        for line in label_stream:
            begin_field, end_field, phoneme = line.split()
            if phoneme == PAUSE_PHONEME:
                continue
            if speech_begin is None:
                speech_begin = int(begin_field) / HTK_TIME_UNITS_PER_SECOND
            speech_end = int(end_field) / HTK_TIME_UNITS_PER_SECOND
    if speech_begin is None or speech_end is None:
        return None
    return speech_begin, speech_end


def load_silence_spans(silence_file: Path) -> dict[str, tuple[float, float]]:
    """Load per-utterance speech spans from a VCTK silence label file.

    Args:
        silence_file: Text file with lines of
            "<utterance_id> <speech_begin> <speech_end>" in seconds
            (vctk-silences.0.92.txt).

    Returns:
        Mapping from utterance id to (begin, end) in seconds.
    """
    spans: dict[str, tuple[float, float]] = {}
    for line in silence_file.read_text().splitlines():
        fields = line.split()
        if len(fields) != 3:
            continue
        spans[fields[0]] = (float(fields[1]), float(fields[2]))
    return spans


def list_lab_jobs(
    vctk_dir: Path, phoneme_dir: Path, lab_dir: Path
) -> list[TrimJob]:
    """Build trim jobs for the VCTK 0.80 layout (wav48 + HTS labels)."""
    jobs = []
    for phoneme_file in sorted(phoneme_dir.glob("*.txt")):
        base = phoneme_file.stem
        speaker_id = base.split("_")[0]
        wav_file = vctk_dir / "wav48" / speaker_id / f"{base}.wav"
        if not wav_file.exists():
            print(f"  Warning: not found: {wav_file}")
            continue
        span = read_speech_span_from_lab(lab_dir / speaker_id / f"{base}.lab")
        if span is None:
            continue
        jobs.append(TrimJob(wav_file, span[0], span[1], base))
    return jobs


def list_silence_jobs(
    vctk_dir: Path, phoneme_dir: Path, silence_file: Path, mic: str
) -> list[TrimJob]:
    """Build trim jobs for the VCTK 0.92 layout (flac + silence labels)."""
    spans = load_silence_spans(silence_file)
    jobs = []
    for phoneme_file in sorted(phoneme_dir.glob("*.txt")):
        base = phoneme_file.stem
        speaker_id = base.split("_")[0]
        flac_file = (
            vctk_dir
            / "wav48_silence_trimmed"
            / speaker_id
            / f"{base}_{mic}.flac"
        )
        if not flac_file.exists():
            print(f"  Warning: not found: {flac_file}")
            continue
        if base not in spans:
            print(f"  Warning: no silence label for {base}; skipped")
            continue
        speech_begin, speech_end = spans[base]
        jobs.append(TrimJob(flac_file, speech_begin, speech_end, base))
    return jobs


def random_margin_sec(rng: np.random.Generator) -> float:
    return (MAX_MARGIN_SEC - MIN_MARGIN_SEC) * rng.random() + MIN_MARGIN_SEC


def trim_and_save(
    job: TrimJob, out_file: Path, rng: np.random.Generator
) -> None:
    """Read audio, keep the speech span plus margins, and write a WAV."""
    trim_begin = max(job.speech_begin - random_margin_sec(rng), 0.0)
    trim_end = job.speech_end + random_margin_sec(rng)
    data, sample_rate = sf.read(job.audio_file)
    begin_sample = int(trim_begin * sample_rate)
    end_sample = min(int(trim_end * sample_rate), len(data))
    sf.write(str(out_file), data[begin_sample:end_sample], sample_rate)


def trim_all(jobs: list[TrimJob], out_dir: Path, seed: int) -> int:
    """Trim every job and return the number of written files."""
    rng = np.random.default_rng(seed)
    for job_index, job in enumerate(jobs, start=1):
        trim_and_save(job, out_dir / f"{job.out_name}.wav", rng)
        if job_index % PROGRESS_INTERVAL == 0:
            print(f"  {job_index}/{len(jobs)}")
    return len(jobs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("vctk_dir", type=str, help="Path to the VCTK corpus")
    parser.add_argument(
        "phoneme_dir", type=str, help="Directory with phoneme txt files"
    )
    parser.add_argument(
        "out_dir", type=str, help="Output directory for trimmed WAV files"
    )
    timing_source = parser.add_mutually_exclusive_group(required=True)
    timing_source.add_argument(
        "--lab_dir",
        type=str,
        help="HTS mono label directory (VCTK 0.80 layout)",
    )
    timing_source.add_argument(
        "--silence_file",
        type=str,
        help="Path to vctk-silences.0.92.txt (VCTK 0.92 layout)",
    )
    parser.add_argument(
        "--mic",
        type=str,
        default="mic1",
        choices=["mic1", "mic2"],
        help="Microphone track to use with the VCTK 0.92 layout",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for margin"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vctk_dir = Path(args.vctk_dir)
    phoneme_dir = Path(args.phoneme_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.silence_file:
        jobs = list_silence_jobs(
            vctk_dir, phoneme_dir, Path(args.silence_file), args.mic
        )
    else:
        jobs = list_lab_jobs(vctk_dir, phoneme_dir, Path(args.lab_dir))

    count = trim_all(jobs, out_dir, args.seed)
    print(f"Trimmed and saved {count} WAV files to {out_dir}")


if __name__ == "__main__":
    main()

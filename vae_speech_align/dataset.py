from pathlib import Path

import torch
import torchaudio

from vae_speech_align import config

WAV_SAMPLE_RATE = 16000


def get_token_indices(
    tokens: list[list[str]], token_index_dict: dict[str, int]
) -> list[torch.Tensor]:
    """Convert token lists to 1-indexed LongTensors (0 is padding)."""
    xs: list[torch.Tensor] = []
    for token_seq in tokens:
        unknown = [p for p in token_seq if p not in token_index_dict]
        if unknown:
            raise KeyError(
                f"token(s) {unknown} are not in the token list; "
                "check the label files and token_list_file"
            )
        xs.append(
            torch.LongTensor([1 + token_index_dict[p] for p in token_seq])
        )
    return xs


class Dataset(torch.utils.data.Dataset):
    """Dataset that pairs WAV files with their token label files."""

    def __init__(
        self,
        *,
        token_dir: Path,
        wav_dir: Path,
    ) -> None:
        base_list = []
        for fn in wav_dir.glob("*.wav"):
            base_list.append(fn.stem)

        self.base_list = sorted(base_list)
        self.token_dir = token_dir
        self.wav_dir = wav_dir

    def __len__(self) -> int:
        return len(self.base_list)

    def __getitem__(self, idx: int) -> tuple[list[str], torch.Tensor]:
        base = self.base_list[idx]

        with open(self.token_dir / f"{base}.txt") as f:
            tokens = f.readline().rstrip().split(" ")

        wav, sample_rate = torchaudio.load(self.wav_dir / f"{base}.wav")

        if sample_rate != WAV_SAMPLE_RATE:
            transform = torchaudio.transforms.Resample(
                sample_rate, WAV_SAMPLE_RATE
            )
            wav = transform(wav)

        wav = wav.squeeze(0)

        return tokens, wav


class Collate:
    """Collate (tokens, wav) samples into padded index/wave batches."""

    def __init__(self, *, token_list: list[str]) -> None:
        self.token_index_dict = dict(zip(token_list, range(len(token_list))))

    def __call__(
        self,
        batch: list[tuple[list[str], torch.Tensor]],
    ) -> tuple[
        torch.Tensor, torch.Tensor, list[list[str]], torch.Tensor, torch.Tensor
    ]:
        tokens, ys = [list(sample) for sample in zip(*batch)]

        y_lengths = torch.LongTensor([yi.size(0) for yi in ys])

        xs = get_token_indices(tokens, self.token_index_dict)
        x_lengths = torch.LongTensor([len(c) for c in tokens])
        x_concat = torch.nn.utils.rnn.pad_sequence(
            xs, batch_first=True
        )  # index 0 for None padding

        y_concat = torch.nn.utils.rnn.pad_sequence(ys, batch_first=True)

        return x_concat, y_concat, tokens, x_lengths, y_lengths


def create_dataloader(
    conf: config.AlignmentExpConfig,
    wav_dir: Path,
    token_dir: Path,
    shuffle: bool,
    batch_size: int | None = None,
) -> torch.utils.data.DataLoader:
    """Build a DataLoader from config, WAV dir, and token label dir."""
    token_list_file = Path(conf.features.linguistic.token_list_file)
    token_list = [
        line.strip() for line in token_list_file.read_text().splitlines()
    ]

    num_tokens = conf.features.linguistic.num_tokens
    if num_tokens != len(token_list):
        raise ValueError(
            f"num_tokens ({num_tokens}) does not match the number of tokens "
            f"({len(token_list)}) in {token_list_file}"
        )

    dataset = Dataset(
        token_dir=token_dir,
        wav_dir=wav_dir,
    )

    collate_fn = Collate(
        token_list=token_list,
    )

    if batch_size is None:
        batch_size = conf.training.batch_size

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=conf.training.num_workers,
        pin_memory=conf.training.pin_memory,
    )

    return dataloader

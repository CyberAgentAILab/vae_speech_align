"""Tests for vae_speech_align.dataset."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torchaudio

from vae_speech_align.dataset import (
    WAV_SAMPLE_RATE,
    Collate,
    Dataset,
    create_dataloader,
    get_token_indices,
)

PHONEMES = ["a", "i", "u", "e", "o"]
SR = 16000


def _make_data(tmp_path, num_utts=3, sr=SR):
    """Create synthetic WAV/phoneme files; return (wav_dir, phoneme_dir)."""
    wav_dir = tmp_path / "wav"
    wav_dir.mkdir()
    phoneme_dir = tmp_path / "phoneme"
    phoneme_dir.mkdir()

    rng = np.random.default_rng(0)
    for i in range(num_utts):
        n_samples = int(sr * (0.5 + 0.5 * rng.random()))
        wav = (
            torch.from_numpy(rng.standard_normal(n_samples).astype(np.float32))
            * 0.01
        )
        torchaudio.save(str(wav_dir / f"utt{i:03d}.wav"), wav.unsqueeze(0), sr)

        n_phones = rng.integers(2, 5)
        phones = [PHONEMES[j % len(PHONEMES)] for j in range(n_phones)]
        (phoneme_dir / f"utt{i:03d}.txt").write_text(" ".join(phones))

    return wav_dir, phoneme_dir


# ---------------------------------------------------------------------------
# get_token_indices
# ---------------------------------------------------------------------------


class TestGetTokenIndices:
    def test_single_utterance(self):
        """Test that one utterance yields one tensor of 1-based indices."""
        index_dict = {"a": 0, "i": 1, "u": 2}
        result = get_token_indices([["a", "i", "u"]], index_dict)
        assert len(result) == 1
        assert torch.equal(result[0], torch.LongTensor([1, 2, 3]))

    def test_multiple_utterances(self):
        """Test that each utterance yields its own tensor of indices."""
        index_dict = {"a": 0, "i": 1}
        result = get_token_indices([["a"], ["i", "a"]], index_dict)
        assert len(result) == 2
        assert torch.equal(result[0], torch.LongTensor([1]))
        assert torch.equal(result[1], torch.LongTensor([2, 1]))

    def test_indices_are_one_based(self):
        """Test that indices are shifted by one, reserving 0 for padding."""
        index_dict = {"a": 0}
        result = get_token_indices([["a"]], index_dict)
        # 0 is reserved for padding, so "a" (index 0) -> 1
        assert result[0].item() == 1

    def test_unknown_token_raises_keyerror(self):
        """Test that a token missing from the dict raises KeyError."""
        index_dict = {"a": 0}
        with pytest.raises(KeyError):
            get_token_indices([["a", "zzz"]], index_dict)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class TestDataset:
    def test_length(self, tmp_path):
        """Test that len(dataset) equals the number of WAV files."""
        wav_dir, phoneme_dir = _make_data(tmp_path, num_utts=5)
        ds = Dataset(wav_dir=wav_dir, token_dir=phoneme_dir)
        assert len(ds) == 5

    def test_getitem_returns_tokens_and_wav(self, tmp_path):
        """Test that __getitem__ returns a list of token strings and a
        1-D waveform tensor."""
        wav_dir, phoneme_dir = _make_data(tmp_path, num_utts=2)
        ds = Dataset(wav_dir=wav_dir, token_dir=phoneme_dir)
        phonemes, wav = ds[0]
        assert isinstance(phonemes, list)
        assert all(isinstance(p, str) for p in phonemes)
        assert isinstance(wav, torch.Tensor)
        assert wav.ndim == 1

    def test_sorted_order(self, tmp_path):
        """Test that utterance basenames are stored in sorted order."""
        wav_dir, phoneme_dir = _make_data(tmp_path, num_utts=3)
        ds = Dataset(wav_dir=wav_dir, token_dir=phoneme_dir)
        assert ds.base_list == sorted(ds.base_list)

    def test_resamples_to_16khz(self, tmp_path):
        """Test that a wav at another sample rate is resampled to 16 kHz."""
        wav_dir = tmp_path / "wav"
        wav_dir.mkdir()
        phoneme_dir = tmp_path / "phoneme"
        phoneme_dir.mkdir()

        src_sr = 8000
        duration_sec = 0.5
        wav = torch.zeros(1, int(src_sr * duration_sec))
        torchaudio.save(str(wav_dir / "utt000.wav"), wav, src_sr)
        (phoneme_dir / "utt000.txt").write_text("a i")

        ds = Dataset(wav_dir=wav_dir, token_dir=phoneme_dir)
        _, loaded = ds[0]
        assert loaded.shape[0] == int(WAV_SAMPLE_RATE * duration_sec)


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------


class TestCollate:
    def test_batch_shapes(self, tmp_path):
        """Test that all collate outputs share the batch-size leading dim."""
        wav_dir, phoneme_dir = _make_data(tmp_path, num_utts=3)
        ds = Dataset(wav_dir=wav_dir, token_dir=phoneme_dir)

        collate_fn = Collate(token_list=PHONEMES)
        batch = [ds[i] for i in range(len(ds))]
        x, wav, phonemes, x_lengths, wav_lengths = collate_fn(batch)

        B = len(batch)
        assert x.shape[0] == B
        assert wav.shape[0] == B
        assert len(phonemes) == B
        assert x_lengths.shape == (B,)
        assert wav_lengths.shape == (B,)

    def test_padding(self, tmp_path):
        """Test that phoneme index and wav tensors are padded to the
        maximum lengths in the batch."""
        wav_dir, phoneme_dir = _make_data(tmp_path, num_utts=3)
        ds = Dataset(wav_dir=wav_dir, token_dir=phoneme_dir)

        collate_fn = Collate(token_list=PHONEMES)
        batch = [ds[i] for i in range(len(ds))]
        x, wav, phonemes, x_lengths, wav_lengths = collate_fn(batch)

        # x should be padded to max phoneme length; padding index is 0
        assert x.shape[1] == x_lengths.max().item()
        # wav should be padded to max wav length
        assert wav.shape[1] == wav_lengths.max().item()

    def test_x_lengths_match_token_counts(self, tmp_path):
        """Test that x_lengths equal the per-utterance token counts."""
        wav_dir, phoneme_dir = _make_data(tmp_path, num_utts=3)
        ds = Dataset(wav_dir=wav_dir, token_dir=phoneme_dir)

        collate_fn = Collate(token_list=PHONEMES)
        batch = [ds[i] for i in range(len(ds))]
        x, wav, phonemes, x_lengths, wav_lengths = collate_fn(batch)

        for i, ph in enumerate(phonemes):
            assert x_lengths[i].item() == len(ph)


# ---------------------------------------------------------------------------
# create_dataloader
# ---------------------------------------------------------------------------


def _make_conf(token_list_file, batch_size=2, num_workers=0, pin_memory=False):
    """Minimal config namespace with the fields create_dataloader accesses."""
    return SimpleNamespace(
        features=SimpleNamespace(
            linguistic=SimpleNamespace(
                token_list_file=str(token_list_file),
                num_tokens=len(PHONEMES),
            ),
        ),
        training=SimpleNamespace(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    )


class TestCreateDataloader:
    def test_yields_batches(self, tmp_path):
        """Test that the dataloader yields batches sized by the config
        batch_size."""
        wav_dir, phoneme_dir = _make_data(tmp_path, num_utts=4)
        token_list_file = tmp_path / "phonemes.txt"
        token_list_file.write_text("\n".join(PHONEMES))

        conf = _make_conf(token_list_file, batch_size=2)
        dataloader = create_dataloader(
            conf, wav_dir=wav_dir, token_dir=phoneme_dir, shuffle=False
        )

        batches = list(dataloader)
        assert len(batches) == 2
        x, wav, phonemes, x_lengths, wav_lengths = batches[0]
        assert x.shape[0] == 2
        assert wav.shape[0] == 2

    def test_batch_size_override(self, tmp_path):
        """Test that an explicit batch_size overrides the config value."""
        wav_dir, phoneme_dir = _make_data(tmp_path, num_utts=4)
        token_list_file = tmp_path / "phonemes.txt"
        token_list_file.write_text("\n".join(PHONEMES))

        conf = _make_conf(token_list_file, batch_size=2)
        dataloader = create_dataloader(
            conf,
            wav_dir=wav_dir,
            token_dir=phoneme_dir,
            shuffle=False,
            batch_size=1,
        )
        assert len(list(dataloader)) == 4

    def test_num_tokens_mismatch_raises(self, tmp_path):
        """Test that ValueError is raised when num_tokens disagrees with
        the token list file."""
        wav_dir, phoneme_dir = _make_data(tmp_path, num_utts=2)
        token_list_file = tmp_path / "phonemes.txt"
        token_list_file.write_text("\n".join(PHONEMES + ["extra"]))

        conf = _make_conf(
            token_list_file
        )  # num_tokens = len(PHONEMES), file has one more
        with pytest.raises(ValueError):
            create_dataloader(
                conf, wav_dir=wav_dir, token_dir=phoneme_dir, shuffle=False
            )

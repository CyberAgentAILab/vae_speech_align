"""Tests for vae_speech_align.utils."""

import numpy as np
import pytest
import torch
from praatio import textgrid as praatio_textgrid
from praatio.data_classes.textgrid import Textgrid

from vae_speech_align.utils import (
    FRAME_SHIFT,
    RunningAverage,
    _to_numpy,
    length_to_input_mask,
    write_textgrid,
)


def _read_textgrid(path) -> Textgrid:
    return praatio_textgrid.openTextgrid(str(path), includeEmptyIntervals=True)


# ---------------------------------------------------------------------------
# _to_numpy
# ---------------------------------------------------------------------------


class TestToNumpy:
    def test_tensor_to_numpy(self):
        """Test that a torch.Tensor is converted to an equal numpy array."""
        t = torch.tensor([1.0, 2.0, 3.0])
        result = _to_numpy(t)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_numpy_passthrough(self):
        """Test that a numpy array is returned unchanged (same object)."""
        a = np.array([1.0, 2.0])
        result = _to_numpy(a)
        assert result is a

    def test_list_passthrough(self):
        """Test that a plain list is returned unchanged (same object)."""
        lst = [1, 2, 3]
        result = _to_numpy(lst)
        assert result is lst

    def test_tensor_detaches_grad(self):
        """Test that a tensor in the autograd graph is detached and
        converted to a numpy array without error."""
        t = torch.tensor([1.0, 2.0], requires_grad=True) * 2
        result = _to_numpy(t)
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# write_textgrid
# ---------------------------------------------------------------------------


class TestWriteTextgrid:
    def test_phoneme_only(self, tmp_path):
        """Test that phoneme durations are written as a "tokens" tier
        with boundaries at cumulative durations times FRAME_SHIFT."""
        out_file = tmp_path / "test.textgrid"
        phonemes = ["a", "i", "u"]
        durations = np.array([10, 20, 30])

        write_textgrid(out_file, phonemes, durations)

        grid = _read_textgrid(out_file)
        assert "tokens" in grid.tierNames
        tokens = grid.getTier("tokens").entries
        assert len(tokens) == 3
        assert tokens[0].label == "a"
        assert tokens[0].start == pytest.approx(0.0)
        assert tokens[0].end == pytest.approx(10 * FRAME_SHIFT)
        assert tokens[2].end == pytest.approx(60 * FRAME_SHIFT)

    def test_state_labels_follow_tokens(self, tmp_path):
        """Test that each state interval is labeled with its parent
        token and the state index within that token."""
        out_file = tmp_path / "out.textgrid"
        write_textgrid(
            out_file,
            ["a", "b"],
            [30, 30],
            state_durations=[10, 10, 10, 10, 10, 10],
        )
        grid = _read_textgrid(out_file)
        assert [iv.label for iv in grid.getTier("states").entries] == [
            "a1",
            "a2",
            "a3",
            "b1",
            "b2",
            "b3",
        ]

    def test_mismatched_state_count_raises(self, tmp_path):
        """Test that a state count that is not a multiple of the token
        count raises ValueError."""
        with pytest.raises(ValueError, match="multiple"):
            write_textgrid(
                tmp_path / "out.textgrid",
                ["a", "b"],
                [30, 30],
                state_durations=[10, 10, 10],
            )

    def test_custom_frame_shift_scales_boundaries(self, tmp_path):
        """Test that passing frame_shift scales the interval times
        instead of the 10 ms default."""
        out_file = tmp_path / "out.textgrid"
        write_textgrid(out_file, ["a", "b"], [10, 20], frame_shift=0.02)
        grid = _read_textgrid(out_file)
        tokens = grid.getTier("tokens").entries
        assert tokens[0].end == pytest.approx(0.2)
        assert tokens[1].end == pytest.approx(0.6)

    def test_with_state_durations(self, tmp_path):
        """Test that passing state_durations adds a "states" tier
        alongside the "tokens" tier."""
        out_file = tmp_path / "test.textgrid"
        phonemes = ["a", "i"]
        token_durations = np.array([10, 20])
        state_durations = np.array([3, 7, 8, 12])

        write_textgrid(
            out_file,
            phonemes,
            token_durations,
            state_durations=state_durations,
        )

        grid = _read_textgrid(out_file)
        assert "tokens" in grid.tierNames
        assert "states" in grid.tierNames
        assert len(grid.getTier("tokens").entries) == 2
        assert len(grid.getTier("states").entries) == 4

    def test_torch_tensor_input(self, tmp_path):
        """Test that torch.Tensor durations are accepted and written
        the same way as numpy inputs."""
        out_file = tmp_path / "test.textgrid"
        phonemes = ["a", "i"]
        token_durations = torch.tensor([10, 20])
        state_durations = torch.tensor([3, 7, 8, 12])

        write_textgrid(
            out_file,
            phonemes,
            token_durations,
            state_durations=state_durations,
        )

        grid = _read_textgrid(out_file)
        assert len(grid.getTier("tokens").entries) == 2
        assert len(grid.getTier("states").entries) == 4


# ---------------------------------------------------------------------------
# RunningAverage
# ---------------------------------------------------------------------------


class TestRunningAverage:
    def test_initial_value(self):
        """Test that the first added value becomes the average as-is."""
        avg = RunningAverage(alpha=0.9)
        avg.add(10.0)
        assert avg.value == pytest.approx(10.0)

    def test_ema_update(self):
        """Test that a second value is blended as
        alpha * old + (1 - alpha) * new."""
        avg = RunningAverage(alpha=0.5)
        avg.add(10.0)
        avg.add(20.0)
        # 0.5 * 10 + 0.5 * 20 = 15
        assert avg.value == pytest.approx(15.0)

    def test_multiple_updates(self):
        """Test that successive values are folded into the exponential
        moving average step by step."""
        avg = RunningAverage(alpha=0.5)
        avg.add(0.0)
        avg.add(10.0)
        avg.add(10.0)
        # step1: 0.0, step2: 5.0, step3: 7.5
        assert avg.value == pytest.approx(7.5)

    def test_value_before_add_is_none(self):
        """Test that value is None before any value has been added."""
        avg = RunningAverage()
        assert avg.value is None


# ---------------------------------------------------------------------------
# length_to_input_mask
# ---------------------------------------------------------------------------


class TestLengthToInputMask:
    def test_basic(self):
        """Test that True marks positions before each sequence length,
        with mask width equal to the maximum length."""
        lengths = torch.tensor([3, 2])
        mask = length_to_input_mask(lengths)
        expected = torch.tensor(
            [
                [True, True, True],
                [True, True, False],
            ]
        )
        assert torch.equal(mask, expected)

    def test_with_max_length(self):
        """Test that an explicit max_length sets the mask width and the
        extra positions are False."""
        lengths = torch.tensor([2, 1])
        mask = length_to_input_mask(lengths, max_length=4)
        expected = torch.tensor(
            [
                [True, True, False, False],
                [True, False, False, False],
            ]
        )
        assert torch.equal(mask, expected)

    def test_single_element(self):
        """Test that a single full-length sequence yields an all-True
        (1, length) mask."""
        lengths = torch.tensor([5])
        mask = length_to_input_mask(lengths)
        assert mask.shape == (1, 5)
        assert mask.all()

    def test_zero_length(self):
        """Test that a zero-length sequence produces an all-False row."""
        lengths = torch.tensor([3, 0])
        mask = length_to_input_mask(lengths)
        expected = torch.tensor(
            [
                [True, True, True],
                [False, False, False],
            ]
        )
        assert torch.equal(mask, expected)

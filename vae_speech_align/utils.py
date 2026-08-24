from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from praatio import textgrid as praatio_textgrid
from praatio.utilities.constants import Interval

FRAME_SHIFT = 0.01

ArrayLike = torch.Tensor | np.ndarray | Sequence[int]


def _to_numpy(x: ArrayLike) -> np.ndarray | Sequence[int]:
    """Convert a torch.Tensor to a numpy array if needed."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def write_textgrid(
    out_textgrid_file: str | Path,
    tokens: Sequence[str],
    token_durations: ArrayLike,
    state_durations: ArrayLike | None = None,
    frame_shift: float = FRAME_SHIFT,
) -> None:
    """Write token (and optional state) durations to a TextGrid file.

    frame_shift is the duration of one frame in seconds; it defaults
    to 10 ms but depends on the acoustic feature rate and the model's
    acoustic upsampling.
    """
    token_durations = _to_numpy(token_durations)
    state_durations = (
        _to_numpy(state_durations) if state_durations is not None else None
    )

    grid = praatio_textgrid.Textgrid()

    token_intervals = []
    token_boundaries = np.cumsum(token_durations) * frame_shift
    for i, entity in enumerate(tokens):
        if i == 0:
            start_time = 0.0
        else:
            start_time = float(token_boundaries[i - 1])
        end_time = float(token_boundaries[i])

        token_intervals.append(Interval(start_time, end_time, entity))

    grid.addTier(praatio_textgrid.IntervalTier("tokens", token_intervals))

    if state_durations is not None:
        if len(state_durations) % len(tokens) != 0:
            raise ValueError(
                "state_durations length must be a multiple of the number "
                "of tokens"
            )
        states_per_token = len(state_durations) // len(tokens)

        state_intervals = []
        state_boundaries = np.cumsum(state_durations) * frame_shift
        for i in range(len(state_durations)):
            if i == 0:
                start_time = 0.0
            else:
                start_time = float(state_boundaries[i - 1])
            end_time = float(state_boundaries[i])

            # label each state with its parent token and the state's
            # index within that token, e.g. "a1", "a2", "a3"
            label = (
                f"{tokens[i // states_per_token]}"
                f"{i % states_per_token + 1}"
            )
            state_intervals.append(Interval(start_time, end_time, label))

        grid.addTier(praatio_textgrid.IntervalTier("states", state_intervals))

    grid.save(
        str(out_textgrid_file),
        format="long_textgrid",
        includeBlankSpaces=True,
    )


class RunningAverage:
    """Exponential moving average tracker."""

    def __init__(self, alpha: float = 0.98) -> None:
        self._value: float | None = None
        self.alpha = alpha

    def add(self, new_value: float) -> None:
        if self._value is None:
            self._value = new_value
        else:
            self._value = (
                self.alpha * self._value + (1.0 - self.alpha) * new_value
            )

    @property
    def value(self) -> float | None:
        return self._value


def length_to_input_mask(
    lengths: torch.Tensor, max_length: int | None = None
) -> torch.Tensor:
    """Create a bool mask [B, max_length]; True marks valid positions."""
    B = lengths.shape[0]
    device = lengths.device
    dtype = lengths.dtype

    if max_length is None:
        max_length = int(torch.max(lengths).item())

    indices = (
        torch.arange(max_length, device=device, dtype=dtype)
        .unsqueeze(0)
        .tile(B, 1)
    )

    mask = indices < lengths.unsqueeze(1)
    return mask

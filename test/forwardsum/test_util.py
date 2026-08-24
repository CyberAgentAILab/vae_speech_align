import pytest
import torch
from data import make_data_simple_2d

from vae_speech_align.forwardsum.matrix_util import (
    NEGATIVE_INF_THRESHOLD,
    make_padded_log_prob_output_matrix,
)


class TestNextPowerOfTwo:
    """Tests for the triton block-sizing helper."""

    def test_next_power_of_two(self) -> None:
        """Test the boundary behavior of triton.next_power_of_2, which the
        kernels rely on for block sizing."""
        triton = pytest.importorskip("triton")
        assert triton.next_power_of_2(31) == 32
        assert triton.next_power_of_2(32) == 32
        assert triton.next_power_of_2(33) == 64


class TestTimeWiseNormalization:
    """Tests for time-wise normalization of the output matrix."""

    def test_normalized_output_matrix(self) -> None:
        """Test that time_wise_normalize makes the state probabilities of
        each valid frame sum to one, unlike the unnormalized matrix."""
        y, state_output_mean, y_lengths, state_lengths = make_data_simple_2d()

        B, T, D = y.size()
        _, K, _ = state_output_mean.size()

        state_output_var = torch.ones_like(state_output_mean)

        log_prob_output_matrix = make_padded_log_prob_output_matrix(
            y,
            y_lengths,
            state_lengths,
            state_output_mean,
            state_output_var,
            time_wise_normalize=False,
        )

        log_prob_output_matrix_normalized = make_padded_log_prob_output_matrix(
            y,
            y_lengths,
            state_lengths,
            state_output_mean,
            state_output_var,
            time_wise_normalize=True,
        )

        prob_sum = torch.exp(log_prob_output_matrix).sum(dim=-1)
        prob_sum_normalized = torch.exp(log_prob_output_matrix_normalized).sum(
            dim=-1
        )

        assert prob_sum_normalized.size() == (B, T + 2)
        assert torch.allclose(
            prob_sum_normalized[0, 1 : y_lengths[0] + 1],
            torch.ones_like(prob_sum_normalized[0, 1 : y_lengths[0] + 1]),
        )
        assert torch.allclose(
            prob_sum_normalized[1, 1 : y_lengths[1] + 1],
            torch.ones_like(prob_sum_normalized[1, 1 : y_lengths[1] + 1]),
        )

        assert not torch.allclose(
            prob_sum[0, 1 : y_lengths[0] + 1],
            torch.ones_like(prob_sum[0, 1 : y_lengths[0] + 1]),
        )
        assert not torch.allclose(
            prob_sum[1, 1 : y_lengths[1] + 1],
            torch.ones_like(prob_sum[1, 1 : y_lengths[1] + 1]),
        )

    def test_normalization_keeps_invalid_positions_masked(self) -> None:
        """Test that frames and states beyond each element's lengths
        stay at NEGATIVE_INF under time_wise_normalize (a softmax over
        an all-masked frame must not become a uniform distribution)."""
        y, state_output_mean, y_lengths, state_lengths = make_data_simple_2d()

        state_output_var = torch.ones_like(state_output_mean)

        mat = make_padded_log_prob_output_matrix(
            y,
            y_lengths,
            state_lengths,
            state_output_mean,
            state_output_var,
            time_wise_normalize=True,
        )

        # the second batch element is shorter than the padded size
        assert (mat[1, y_lengths[1] + 1 :, :] < NEGATIVE_INF_THRESHOLD).all()
        assert (
            mat[1, :, state_lengths[1] + 1 :] < NEGATIVE_INF_THRESHOLD
        ).all()

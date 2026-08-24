"""Tests for vae_speech_align.forwardsum.matrix_util."""

import pytest
import torch
from data import make_data_simple_2d

from vae_speech_align.forwardsum.matrix_util import (
    NEGATIVE_INF,
    MatchingFuncType,
    calc_gaussian_log_likelihood_matrix,
    calc_inner_product_matrix,
    gather_terminal_log_likelihoods,
    make_backward_begin_mask,
    make_forward_begin_mask,
    make_padded_BTK_matrix,
    make_padded_log_prob_output_matrix,
    mask_invalid_log_probs,
)

# ---------------------------------------------------------------------------
# calc_gaussian_log_likelihood_matrix
# ---------------------------------------------------------------------------


class TestGaussianLogLikelihoodMatrix:
    def test_matches_torch_distributions(self):
        """Test that the vectorized matrix agrees with per-(b, k)
        torch.distributions.Normal log-probabilities."""
        torch.manual_seed(0)
        B, T, K, D = 2, 6, 4, 3
        x = torch.randn(B, T, D)
        mean = torch.randn(B, K, D)
        var = torch.rand(B, K, D) + 0.5

        ll = calc_gaussian_log_likelihood_matrix(x, mean, var)
        assert ll.shape == (B, T, K)

        for b in range(B):
            for k in range(K):
                dist = torch.distributions.Normal(mean[b, k], var[b, k].sqrt())
                ref = dist.log_prob(x[b]).sum(dim=-1)  # [T]
                assert torch.allclose(
                    ll[b, :, k], ref, atol=1e-4
                ), f"mismatch at b={b}, k={k}"

    def test_peak_at_mean(self):
        """Test that the likelihood is maximized when the observation
        equals the state mean."""
        mean = torch.tensor([[[0.0, 0.0], [5.0, 5.0]]])  # [1, 2, 2]
        var = torch.ones_like(mean)
        x = torch.tensor([[[0.0, 0.0], [5.0, 5.0]]])  # [1, 2, 2]
        ll = calc_gaussian_log_likelihood_matrix(x, mean, var)
        assert ll[0, 0, 0] > ll[0, 0, 1]
        assert ll[0, 1, 1] > ll[0, 1, 0]


# ---------------------------------------------------------------------------
# calc_inner_product_matrix
# ---------------------------------------------------------------------------


class TestInnerProductMatrix:
    def test_matches_einsum(self):
        """Test that the inner-product matrix equals the einsum
        reference."""
        torch.manual_seed(1)
        x = torch.randn(2, 5, 3)
        mean = torch.randn(2, 4, 3)
        result = calc_inner_product_matrix(x, mean)
        ref = torch.einsum("btd,bkd->btk", x, mean)
        assert torch.allclose(result, ref)


# ---------------------------------------------------------------------------
# make_padded_log_prob_output_matrix
# ---------------------------------------------------------------------------


def _setup():
    y, mean, y_lengths, state_lengths = make_data_simple_2d()
    var = torch.ones_like(mean)
    return y, mean, var, y_lengths, state_lengths


class TestMakePaddedLogProbOutputMatrix:
    def test_inner_product_type(self):
        """Test that INNER_PRODUCT matching pads the shape by one on
        each border and keeps raw inner products in the interior."""
        y, mean, var, y_lengths, state_lengths = _setup()
        mat = make_padded_log_prob_output_matrix(
            y,
            y_lengths,
            state_lengths,
            mean,
            var,
            matching_func_type=MatchingFuncType.INNER_PRODUCT,
        )
        B, T, _ = y.size()
        K = mean.size(1)
        assert mat.shape == (B, T + 2, K + 2)
        # interior valid entries equal the raw inner product
        ip = calc_inner_product_matrix(y, mean)
        t, k = 0, 0
        assert mat[0, t + 1, k + 1].item() == pytest.approx(ip[0, t, k].item())

    def test_unknown_matching_type_raises(self):
        """Test that an unknown matching function type raises
        NotImplementedError."""
        y, mean, var, y_lengths, state_lengths = _setup()
        with pytest.raises(NotImplementedError):
            make_padded_log_prob_output_matrix(
                y,
                y_lengths,
                state_lengths,
                mean,
                var,
                matching_func_type="bogus",
            )

    def test_bias_is_added(self):
        """Test that a bias matrix shifts the valid interior entries by
        its value."""
        y, mean, var, y_lengths, state_lengths = _setup()
        base = make_padded_log_prob_output_matrix(
            y,
            y_lengths,
            state_lengths,
            mean,
            var,
            matching_func_type=MatchingFuncType.INNER_PRODUCT,
        )
        bias_value = 3.0
        bias = torch.full((y.size(0), y.size(1), mean.size(1)), bias_value)
        biased = make_padded_log_prob_output_matrix(
            y,
            y_lengths,
            state_lengths,
            mean,
            var,
            matching_func_type=MatchingFuncType.INNER_PRODUCT,
            bias=bias,
        )
        # valid interior entries are shifted by the bias
        diff = biased[0, 1, 1] - base[0, 1, 1]
        assert diff.item() == pytest.approx(bias_value)


# ---------------------------------------------------------------------------
# mask_invalid_log_probs
# ---------------------------------------------------------------------------


class TestMaskInvalidLogProbs:
    def test_masks_beyond_lengths_without_mutating_input(self):
        """Test that out-of-length entries become NEGATIVE_INF while
        valid entries and the input matrix itself are untouched."""
        B, T, K = 2, 4, 3
        output_lengths = torch.tensor([4, 2])
        state_lengths = torch.tensor([3, 1])
        base = torch.zeros(B, T, K)

        masked = mask_invalid_log_probs(base, output_lengths, state_lengths)

        assert (base == 0).all()  # input is not modified in place
        assert (masked[0] == 0).all()  # full-length element is untouched
        assert (masked[1, :2, :1] == 0).all()
        assert (masked[1, 2:, :] == NEGATIVE_INF).all()
        assert (masked[1, :, 1:] == NEGATIVE_INF).all()


# ---------------------------------------------------------------------------
# make_padded_BTK_matrix
# ---------------------------------------------------------------------------


class TestMakePaddedBTKMatrix:
    def test_padding_and_masking(self):
        """Test that borders and positions beyond each element's
        lengths become NEGATIVE_INF while valid entries are kept."""
        B, T, K = 2, 5, 3
        output_lengths = torch.tensor([5, 4])
        state_lengths = torch.tensor([3, 2])
        base = torch.zeros(B, T, K)

        padded = make_padded_BTK_matrix(
            base.clone(), output_lengths, state_lengths
        )

        assert padded.shape == (B, T + 2, K + 2)
        # border padding is NEGATIVE_INF
        assert (padded[:, 0, :] == NEGATIVE_INF).all()
        assert (padded[:, -1, :] == NEGATIVE_INF).all()
        assert (padded[:, :, 0] == NEGATIVE_INF).all()
        assert (padded[:, :, -1] == NEGATIVE_INF).all()
        # positions beyond each element's lengths are masked
        assert (
            padded[1, 5, 1:3] == NEGATIVE_INF
        ).all()  # t = output_lengths[1]
        assert (
            padded[1, 1:5, 3] == NEGATIVE_INF
        ).all()  # k = state_lengths[1]
        # valid entries keep their original value
        assert (padded[0, 1:6, 1:4] == 0).all()
        assert (padded[1, 1:5, 1:3] == 0).all()


# ---------------------------------------------------------------------------
# gather_terminal_log_likelihoods
# ---------------------------------------------------------------------------


class TestGatherTerminalLogLikelihoods:
    def test_matches_per_batch_loop(self):
        """The vectorized gather equals a per-batch loop.

        The oracle collects alpha[b, output_lengths[b], state_lengths[b]]
        one batch element at a time and stacks the results.
        """
        torch.manual_seed(2)
        B, T, K = 4, 7, 5
        alpha = torch.randn(B, T, K)
        # ragged lengths so each batch element indexes a different (t, k)
        output_lengths = torch.tensor([6, 4, 5, 3])
        state_lengths = torch.tensor([4, 2, 3, 1])

        result = gather_terminal_log_likelihoods(
            alpha, output_lengths, state_lengths
        )

        ref = torch.stack(
            [alpha[b, output_lengths[b], state_lengths[b]] for b in range(B)],
            dim=0,
        )

        assert result.shape == (B,)
        assert torch.equal(result, ref)


# ---------------------------------------------------------------------------
# begin masks
# ---------------------------------------------------------------------------


class TestBeginMasks:
    def test_forward_begin_mask(self):
        """Test that the forward begin mask is True only at (t=0, k=0)
        for every batch element."""
        mask = make_forward_begin_mask(2, 4, 3, torch.device("cpu"))
        assert mask.shape == (2, 4, 3)
        assert mask[:, 0, 0].all()
        assert mask.sum() == 2

    def test_backward_begin_mask(self):
        """Test that the backward begin mask is True only at each
        element's (begin_time, begin_state)."""
        begin_times = torch.tensor([3, 2])
        begin_states = torch.tensor([2, 1])
        mask = make_backward_begin_mask(
            2, 4, 3, begin_times, begin_states, torch.device("cpu")
        )
        assert mask.shape == (2, 4, 3)
        assert mask[0, 3, 2]
        assert mask[1, 2, 1]
        assert mask.sum() == 2

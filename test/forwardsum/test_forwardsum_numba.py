import torch
from data import make_data_simple_2d
from model_dp import ForwardSumDP, ForwardSumNumba

from vae_speech_align.forwardsum import (
    forwardsum_gamma_numba,
    forwardsum_gamma_torch,
    forwardsum_likelihood_numba,
    forwardsum_likelihood_torch,
    forwardsum_viterbi_numba,
    forwardsum_viterbi_torch,
)
from vae_speech_align.forwardsum.matrix_util import (
    make_padded_BTK_matrix,
    make_padded_log_prob_output_matrix,
)
from vae_speech_align.forwardsum.numba_impl import (
    forwardsum_backward_numba,
    forwardsum_forward_numba,
)
from vae_speech_align.forwardsum.torch_impl import (
    forwardsum_backward_torch,
    forwardsum_forward_torch,
)

NUMERICAL_ATOL = 1e-5


def setup_input() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    )

    return (log_prob_output_matrix, y_lengths, state_lengths)


class TestForwardSumNumba:
    """Tests comparing the numba backend against the torch DP."""

    def test_likelihood(self) -> None:
        """Test that the numba likelihood matches the torch DP per batch
        element."""
        log_prob_output_matrix, y_lengths, state_lengths = setup_input()

        log_likelihoods_torch = forwardsum_likelihood_torch(
            log_prob_output_matrix,
            y_lengths,
            state_lengths,
        )

        device = torch.device("cpu")

        log_likelihoods_numba = forwardsum_likelihood_numba(
            log_prob_output_matrix.to(device),
            y_lengths.to(device),
            state_lengths.to(device),
        )

        assert len(log_likelihoods_torch) == 2

        assert (
            log_likelihoods_torch[0].cpu().item()
            == log_likelihoods_numba[0].cpu().item()
        )
        assert (
            log_likelihoods_torch[1].cpu().item()
            == log_likelihoods_numba[1].cpu().item()
        )

    def test_alpha_beta(self) -> None:
        """Test that numba alpha/beta tables match torch and that the
        terminal alpha equals the initial beta (same total likelihood)."""
        log_prob_output_matrix, y_lengths, state_lengths = setup_input()

        device = torch.device("cpu")

        alpha_torch = forwardsum_forward_torch(log_prob_output_matrix)
        alpha_triton = forwardsum_forward_numba(
            log_prob_output_matrix.to(device)
        )

        beta_torch = forwardsum_backward_torch(
            log_prob_output_matrix,
            y_lengths,
            state_lengths,
        )

        beta_triton = forwardsum_backward_numba(
            log_prob_output_matrix.to(device),
            y_lengths.to(device),
            state_lengths.to(device),
        )

        assert (
            alpha_torch[0, y_lengths[0], state_lengths[0]].cpu().item()
            == alpha_triton[0, y_lengths[0], state_lengths[0]].cpu().item()
        )
        assert (
            alpha_torch[1, y_lengths[1], state_lengths[1]].cpu().item()
            == alpha_triton[1, y_lengths[1], state_lengths[1]].cpu().item()
        )
        assert torch.allclose(
            alpha_torch[0, y_lengths[0], state_lengths[0]].cpu(),
            beta_torch[0, 1, 1].cpu(),
        )
        assert torch.allclose(
            alpha_torch[1, y_lengths[1], state_lengths[1]].cpu(),
            beta_torch[1, 1, 1].cpu(),
        )
        assert (
            beta_torch[0, 1, 1].cpu().item()
            == beta_triton[0, 1, 1].cpu().item()
        )
        assert (
            beta_torch[1, 1, 1].cpu().item()
            == beta_triton[1, 1, 1].cpu().item()
        )

    def test_likelihood_gradient(self) -> None:
        """Test that backprop through the numba backend yields the same
        loss and matrix gradient as torch autograd."""
        log_prob_output_matrix, y_lengths, state_lengths = setup_input()

        device = torch.device("cpu")

        model_torch = ForwardSumDP(
            log_prob_output_matrix,
            y_lengths,
            state_lengths,
        )
        model_torch.to(device)

        log_likelihoods_torch = model_torch.forward()
        loss_torch = -log_likelihoods_torch.sum()
        loss_torch.backward()

        model_numba = ForwardSumNumba(
            log_prob_output_matrix,
            y_lengths,
            state_lengths,
        )
        model_numba.to(device)

        log_likelihoods_numba = model_numba.forward()
        loss_numba = -log_likelihoods_numba.sum()
        loss_numba.backward()

        assert loss_numba == loss_torch

        assert model_torch.log_prob_output_matrix.grad is not None
        assert model_numba.log_prob_output_matrix.grad is not None
        assert torch.allclose(
            model_torch.log_prob_output_matrix.grad[:, 1:-1, :],
            model_numba.log_prob_output_matrix.grad[:, 1:-1, :],
            atol=NUMERICAL_ATOL,
        )

    def test_gamma(self) -> None:
        """Test that numba gamma occupancies match the torch DP."""
        log_prob_output_matrix, y_lengths, state_lengths = setup_input()

        gamma_torch = forwardsum_gamma_torch(
            log_prob_output_matrix,
            y_lengths,
            state_lengths,
        )

        gamma_numba = forwardsum_gamma_numba(
            log_prob_output_matrix,
            y_lengths,
            state_lengths,
        )

        assert torch.allclose(
            gamma_torch[:, 1:-1, :],
            gamma_numba[:, 1:-1, :],
            atol=NUMERICAL_ATOL,
        )

    def test_viterbi(self) -> None:
        """Test that numba Viterbi returns the same best-path likelihood
        and (float-typed) path matrix as the torch DP."""
        log_prob_output_matrix, y_lengths, state_lengths = setup_input()

        log_likelihoods_torch, path_torch = forwardsum_viterbi_torch(
            log_prob_output_matrix,
            y_lengths,
            state_lengths,
        )

        assert path_torch.sum() == y_lengths.sum()

        device = torch.device("cpu")

        log_likelihoods_numba, path_numba = forwardsum_viterbi_numba(
            log_prob_output_matrix.to(device),
            y_lengths.to(device),
            state_lengths.to(device),
        )

        assert (
            log_likelihoods_torch[0].cpu().item()
            == log_likelihoods_numba[0].cpu().item()
        )
        assert (
            log_likelihoods_torch[1].cpu().item()
            == log_likelihoods_numba[1].cpu().item()
        )

        assert path_torch.dtype == torch.float
        assert path_numba.dtype == torch.float

        assert torch.allclose(
            torch.as_tensor(path_torch.cpu(), dtype=torch.float),
            path_numba.cpu(),
        )

    def test_viterbi_tie_break_matches_numba(self) -> None:
        """Test that torch and numba resolve Viterbi ties identically.

        A uniform log-prob matrix ties transition and stay at every DP
        step, so the backends must break the tie the same way or the
        traced paths diverge.
        """
        B, T, K = 1, 5, 3
        output_lengths = torch.tensor([T])
        state_lengths = torch.tensor([K])
        base = torch.zeros(B, T, K)
        padded = make_padded_BTK_matrix(base, output_lengths, state_lengths)

        _, path_torch = forwardsum_viterbi_torch(
            padded, output_lengths, state_lengths
        )
        _, path_numba = forwardsum_viterbi_numba(
            padded, output_lengths, state_lengths
        )

        assert path_torch.sum() == T
        assert torch.allclose(
            path_torch.cpu().float(), path_numba.cpu().float()
        )

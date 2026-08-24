import pytest
import torch
from data import make_data_simple_2d
from model_dp import ForwardSumDP, ForwardSumTriton

pytest.importorskip("triton")

from vae_speech_align.forwardsum import (  # noqa: E402
    forwardsum_gamma_torch,
    forwardsum_gamma_triton,
    forwardsum_likelihood_torch,
    forwardsum_likelihood_triton,
    forwardsum_viterbi_torch,
    forwardsum_viterbi_triton,
)
from vae_speech_align.forwardsum.matrix_util import (  # noqa: E402
    NEGATIVE_INF,
    make_padded_log_prob_output_matrix,
)
from vae_speech_align.forwardsum.torch_impl import (  # noqa: E402
    forwardsum_backward_torch,
    forwardsum_forward_torch,
)
from vae_speech_align.forwardsum.triton_likelihood_impl import (  # noqa: E402
    forwardsum_backward_triton,
    forwardsum_forward_triton,
)
from vae_speech_align.forwardsum.triton_util import (  # noqa: E402
    TritonConfigK,
)


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


class TestForwardSumTriton:
    """Tests comparing the triton backend against the torch DP."""

    def test_likelihood(self) -> None:
        """Test that the triton likelihood on CUDA matches the torch DP
        per batch element."""
        log_prob_output_matrix, y_lengths, state_lengths = setup_input()

        log_likelihoods_torch = forwardsum_likelihood_torch(
            log_prob_output_matrix,
            y_lengths,
            state_lengths,
        )

        device = torch.device("cuda")

        log_likelihoods_triton = forwardsum_likelihood_triton(
            log_prob_output_matrix.to(device),
            y_lengths.to(device),
            state_lengths.to(device),
        )

        assert len(log_likelihoods_torch) == 2

        assert (
            log_likelihoods_torch[0].cpu().item()
            == log_likelihoods_triton[0].cpu().item()
        )
        assert (
            log_likelihoods_torch[1].cpu().item()
            == log_likelihoods_triton[1].cpu().item()
        )

    def test_alpha_beta(self) -> None:
        """Test that triton alpha/beta tables match torch and that the
        terminal alpha equals the initial beta (same total likelihood)."""
        log_prob_output_matrix, y_lengths, state_lengths = setup_input()

        device = torch.device("cuda")

        alpha_torch = forwardsum_forward_torch(log_prob_output_matrix)
        alpha_triton = forwardsum_forward_triton(
            log_prob_output_matrix.to(device)
        )

        beta_torch = forwardsum_backward_torch(
            log_prob_output_matrix,
            y_lengths,
            state_lengths,
        )

        beta_triton = forwardsum_backward_triton(
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
        """Test that backprop through the triton backend yields the same
        loss and matrix gradient as torch autograd."""
        log_prob_output_matrix, y_lengths, state_lengths = setup_input()

        device = torch.device("cuda")

        model_torch = ForwardSumDP(
            log_prob_output_matrix,
            y_lengths,
            state_lengths,
        )
        model_torch.to(device)

        log_likelihoods_torch = model_torch.forward()
        loss_torch = -log_likelihoods_torch.sum()
        loss_torch.backward()

        model_triton = ForwardSumTriton(
            log_prob_output_matrix,
            y_lengths,
            state_lengths,
        )
        model_triton.to(device)

        log_likelihoods_triton = model_triton.forward()
        loss_triton = -log_likelihoods_triton.sum()
        loss_triton.backward()

        assert loss_triton == loss_torch

        assert model_torch.log_prob_output_matrix.grad is not None
        assert model_triton.log_prob_output_matrix.grad is not None
        assert torch.allclose(
            model_torch.log_prob_output_matrix.grad[:, 1:-1, :],
            model_triton.log_prob_output_matrix.grad[:, 1:-1, :],
        )

    def test_gamma(self) -> None:
        """Test that triton gamma occupancies match the torch DP."""
        log_prob_output_matrix, y_lengths, state_lengths = setup_input()

        device = torch.device("cuda")

        gamma_torch = forwardsum_gamma_torch(
            log_prob_output_matrix,
            y_lengths,
            state_lengths,
        )

        gamma_triton = forwardsum_gamma_triton(
            log_prob_output_matrix.to(device),
            y_lengths.to(device),
            state_lengths.to(device),
        )

        assert torch.allclose(
            gamma_torch[:, 1:-1, :], gamma_triton[:, 1:-1, :].cpu()
        )

    def test_viterbi(self) -> None:
        """Test that triton Viterbi returns the same best-path likelihood
        and (float-typed) path matrix as the torch DP."""
        log_prob_output_matrix, y_lengths, state_lengths = setup_input()

        log_likelihoods_torch, path_torch = forwardsum_viterbi_torch(
            log_prob_output_matrix,
            y_lengths,
            state_lengths,
        )

        assert path_torch.sum() == y_lengths.sum()

        device = torch.device("cuda")

        log_likelihoods_triton, path_triton = forwardsum_viterbi_triton(
            log_prob_output_matrix.to(device),
            y_lengths.to(device),
            state_lengths.to(device),
        )

        assert (
            log_likelihoods_torch[0].cpu().item()
            == log_likelihoods_triton[0].cpu().item()
        )
        assert (
            log_likelihoods_torch[1].cpu().item()
            == log_likelihoods_triton[1].cpu().item()
        )

        assert path_torch.dtype == torch.float
        assert path_triton.dtype == torch.float

        assert torch.allclose(
            torch.as_tensor(path_torch.cpu(), dtype=torch.float),
            path_triton.cpu(),
        )

    @pytest.mark.parametrize(
        "BLOCK_SIZE_K, num_warps, num_stages",
        [
            # BLOCK SIZE
            (1024, 1, 2),
            (256, 1, 2),
            (16, 1, 2),
            (8, 1, 2),
            (4, 1, 2),
            # num_warps
            (1024, 32, 2),
            (1024, 16, 2),
            (1024, 8, 2),
            (1024, 4, 2),
            (1024, 2, 2),
            # num_stages,
            (1024, 4, 2),
            (1024, 4, 3),
            (1024, 4, 4),
        ],
    )
    def test_autotune_config(
        self, BLOCK_SIZE_K: int, num_warps: int, num_stages: int
    ) -> None:
        """Test that alpha, beta, and Viterbi are invariant to the kernel
        launch configuration (block size, warps, stages) vs autotuning."""
        log_prob_output_matrix, y_lengths, state_lengths = setup_input()

        device = torch.device("cuda")

        log_prob_output_matrix = log_prob_output_matrix.to(device)
        y_lengths = y_lengths.to(device)
        state_lengths = state_lengths.to(device)

        triton_custom_config = TritonConfigK(
            BLOCK_SIZE_K, num_warps, num_stages
        )

        alpha_autotune = forwardsum_forward_triton(log_prob_output_matrix)
        alpha_custom = forwardsum_forward_triton(
            log_prob_output_matrix, triton_custom_config=triton_custom_config
        )

        assert torch.allclose(alpha_autotune.cpu(), alpha_custom.cpu())

        beta_autotune = forwardsum_backward_triton(
            log_prob_output_matrix,
            output_lengths=y_lengths,
            state_lengths=state_lengths,
        )
        beta_custom = forwardsum_backward_triton(
            log_prob_output_matrix,
            output_lengths=y_lengths,
            state_lengths=state_lengths,
            triton_custom_config=triton_custom_config,
        )

        assert torch.allclose(beta_autotune.cpu(), beta_custom.cpu())

        log_likelihoods_autotune, path_autotune = forwardsum_viterbi_triton(
            log_prob_output_matrix,
            y_lengths,
            state_lengths,
        )
        log_likelihoods_custom, path_custom = forwardsum_viterbi_triton(
            log_prob_output_matrix,
            y_lengths,
            state_lengths,
            triton_custom_config=triton_custom_config,
        )

        assert torch.allclose(
            log_likelihoods_autotune.cpu(), log_likelihoods_custom.cpu()
        )
        assert torch.allclose(path_autotune.cpu(), path_custom.cpu())

    @pytest.mark.parametrize(
        "max_y_length, max_state_length, batch_size",
        [
            (456, 80, 2),
            (522, 91, 4),
            (248, 47, 8),
            (908, 163, 10),
            (385, 69, 8),
            (789, 140, 8),
            (686, 132, 8),
        ],
    )
    def test_multiple_size(
        self, max_y_length: int, max_state_length: int, batch_size: int
    ) -> None:
        """Test that triton matches torch on random matrices across a
        range of realistic sequence lengths and batch sizes."""
        device = torch.device("cuda")

        torch.manual_seed(211)

        ll = torch.randn((batch_size, max_y_length, max_state_length))
        log_prob_output_matrix = torch.nn.functional.pad(
            ll, (1, 1, 1, 1), "constant", NEGATIVE_INF
        ).to(device)
        y_lengths = (
            torch.zeros((batch_size,), dtype=torch.int64, device=device)
            + max_y_length
        )
        state_lengths = (
            torch.zeros((batch_size,), dtype=torch.int64, device=device)
            + max_state_length
        )

        alpha_triton = forwardsum_forward_triton(log_prob_output_matrix)
        alpha_torch = forwardsum_forward_torch(log_prob_output_matrix)

        assert torch.allclose(
            alpha_triton[:, 1:-1, 1:-1].cpu(),
            alpha_torch[:, 1:-1, 1:-1].cpu(),
            rtol=1e-2,
        )

        beta_triton = forwardsum_backward_triton(
            log_prob_output_matrix,
            output_lengths=y_lengths,
            state_lengths=state_lengths,
        )
        beta_torch = forwardsum_backward_torch(
            log_prob_output_matrix,
            output_lengths=y_lengths,
            state_lengths=state_lengths,
        )

        assert torch.allclose(
            beta_triton[:, 1:-1, 1:-1].cpu(),
            beta_torch[:, 1:-1, 1:-1].cpu(),
            rtol=1e-2,
        )

        log_likelihoods_triton, path_triton = forwardsum_viterbi_triton(
            log_prob_output_matrix,
            y_lengths,
            state_lengths,
        )
        log_likelihoods_torch, path_torch = forwardsum_viterbi_torch(
            log_prob_output_matrix,
            y_lengths,
            state_lengths,
        )

        assert torch.allclose(
            log_likelihoods_triton.cpu(),
            log_likelihoods_torch.cpu(),
            rtol=1e-2,
        )
        assert torch.allclose(path_triton.cpu(), path_torch.cpu())

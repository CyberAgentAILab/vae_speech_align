import pytest
import torch
from data import make_forwardsum_data_numberable_pathes
from model_dp import ForwardSumAllPass, ForwardSumDP
from util import calc_one_pass_forwardsum_likelihood, make_mask

from vae_speech_align.forwardsum.matrix_util import (
    forwardsum_gamma_from_alphabeta,
    make_padded_log_prob_output_matrix,
)
from vae_speech_align.forwardsum.torch_impl import (
    forwardsum_backward_torch,
    forwardsum_forward_torch,
    forwardsum_viterbi_delta_torch,
    forwardsum_viterbi_path_torch,
    forwardsum_viterbi_torch,
)


class TestForwardSumDP:
    """Tests comparing the torch DP against brute-force enumeration."""

    def test_pass_likelihood(self) -> None:
        """Test that the mask-based one-path likelihood helper of util.py
        matches directly indexing and summing the matrix entries."""
        y, output_mean, output_var = make_forwardsum_data_numberable_pathes()

        log_prob_output_matrix = make_padded_log_prob_output_matrix(
            y,
            torch.LongTensor([y.size(1)]),
            torch.LongTensor([output_mean.size(1)]),
            output_mean,
            output_var,
        )

        B = log_prob_output_matrix.size(0)
        T = log_prob_output_matrix.size(1)
        K = log_prob_output_matrix.size(2)

        state_pass = [1, 2, 2, 2, 3]

        output_mask = make_mask(B, T, K, state_pass)

        assert output_mask.sum() == 5

        output_prob_by_mask = log_prob_output_matrix[output_mask].sum()
        output_prob_direct = (
            log_prob_output_matrix[0, 1, state_pass[0]]
            + log_prob_output_matrix[0, 2, state_pass[1]]
            + log_prob_output_matrix[0, 3, state_pass[2]]
            + log_prob_output_matrix[0, 4, state_pass[3]]
            + log_prob_output_matrix[0, 5, state_pass[4]]
        )

        assert output_prob_by_mask == output_prob_direct

        assert output_prob_by_mask == calc_one_pass_forwardsum_likelihood(
            state_pass, log_prob_output_matrix
        )

    def test_likelihood(self) -> None:
        """Test that the forward (alpha) and backward (beta) torch DPs both
        reproduce the logsumexp over all six enumerable paths, and that the
        resulting gamma occupancies sum to the number of frames."""
        y, output_mean, output_var = make_forwardsum_data_numberable_pathes()

        log_prob_output_matrix = make_padded_log_prob_output_matrix(
            y,
            torch.LongTensor([y.size(1)]),
            torch.LongTensor([output_mean.size(1)]),
            output_mean,
            output_var,
        )

        state_pass_list = [
            [1, 1, 1, 2, 3],
            [1, 1, 2, 2, 3],
            [1, 1, 2, 3, 3],
            [1, 2, 2, 2, 3],
            [1, 2, 2, 3, 3],
            [1, 2, 3, 3, 3],
        ]

        log_prob_list = []
        for state_pass in state_pass_list:
            pass_likelohood = calc_one_pass_forwardsum_likelihood(
                state_pass, log_prob_output_matrix
            )

            log_prob_list.append(pass_likelohood)

        likelihood_direct = torch.logsumexp(torch.stack(log_prob_list), dim=0)

        # torch DP
        alpha = forwardsum_forward_torch(
            log_prob_output_matrix,
        )

        assert pytest.approx(alpha[0, -2, -2]) == likelihood_direct.item()

        # torch DP (backward algorithm)
        beta = forwardsum_backward_torch(
            log_prob_output_matrix,
            torch.LongTensor([y.size(1)]),
            torch.LongTensor([output_mean.size(1)]),
        )

        assert pytest.approx(beta[0, 1, 1]) == likelihood_direct.item()

        gamma = forwardsum_gamma_from_alphabeta(
            alpha,
            beta,
            log_prob_output_matrix,
        )

        assert gamma.sum() == pytest.approx(5.0)

    def test_likelihood_gradient(self) -> None:
        """Test that the torch DP model and the brute-force AllPass model
        agree on both the likelihood and its gradient w.r.t. the matrix."""
        y, output_mean, output_var = make_forwardsum_data_numberable_pathes()

        output_lengths = torch.LongTensor([y.size(1)])
        state_lengths = torch.LongTensor([output_mean.size(1)])

        log_prob_output_matrix = make_padded_log_prob_output_matrix(
            y,
            output_lengths,
            state_lengths,
            output_mean,
            output_var,
        )

        model_direct = ForwardSumAllPass(
            log_prob_output_matrix,
        )

        state_pass_list = [
            [1, 1, 1, 2, 3],
            [1, 1, 2, 2, 3],
            [1, 1, 2, 3, 3],
            [1, 2, 2, 2, 3],
            [1, 2, 2, 3, 3],
            [1, 2, 3, 3, 3],
        ]

        likelihood_direct = model_direct(state_pass_list)
        likelihood_direct.backward()

        model_dp_torch = ForwardSumDP(
            log_prob_output_matrix,
            output_lengths,
            state_lengths,
        )

        likelihood_dp_torch = model_dp_torch()
        likelihood_dp_torch.backward()

        assert likelihood_direct.detach().item() == pytest.approx(
            likelihood_dp_torch[0].detach().item()
        )

        assert model_direct.log_prob_output_matrix.grad is not None
        assert model_dp_torch.log_prob_output_matrix.grad is not None
        assert torch.allclose(
            model_direct.log_prob_output_matrix.grad[:, 1:-1, :],
            model_dp_torch.log_prob_output_matrix.grad[:, 1:-1, :],
        )

    def test_viterbi(self) -> None:
        """Test that Viterbi (delta/psi and the unified wrapper) finds the
        same best path and score as brute-force max over all paths."""
        y, output_mean, output_var = make_forwardsum_data_numberable_pathes()

        output_lengths = torch.LongTensor([y.size(1)])
        state_lengths = torch.LongTensor([output_mean.size(1)])

        log_prob_output_matrix = make_padded_log_prob_output_matrix(
            y,
            output_lengths,
            state_lengths,
            output_mean,
            output_var,
        )

        state_pass_list = [
            [1, 1, 1, 2, 3],
            [1, 1, 2, 2, 3],
            [1, 1, 2, 3, 3],
            [1, 2, 2, 2, 3],
            [1, 2, 2, 3, 3],
            [1, 2, 3, 3, 3],
        ]

        log_prob_list = []
        for state_pass in state_pass_list:
            pass_likelohood = calc_one_pass_forwardsum_likelihood(
                state_pass, log_prob_output_matrix
            )

            log_prob_list.append(pass_likelohood)

        max_direct = torch.max(torch.stack(log_prob_list), dim=0)

        assert max_direct.indices.item() == 3

        # torch DP
        delta, psi = forwardsum_viterbi_delta_torch(
            log_prob_output_matrix,
        )

        # trace path
        path = forwardsum_viterbi_path_torch(
            delta,
            psi,
            output_lengths,
            state_lengths,
        )

        # assert path
        assert state_pass_list[max_direct.indices] == [1, 2, 2, 2, 3]
        assert path.sum() == 5.0
        for t, k in enumerate(state_pass_list[max_direct.indices], start=1):
            assert path[0, t, k] == 1.0

        # assert path likelihood
        assert delta[0, -2, -2] == max_direct.values

        # assert unified function
        delta_unified_func, path_unified_func = forwardsum_viterbi_torch(
            log_prob_output_matrix,
            output_lengths,
            state_lengths,
        )

        assert torch.allclose(path, path_unified_func)

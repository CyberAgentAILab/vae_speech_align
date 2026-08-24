from typing import Iterable

import torch
from util import calc_one_pass_forwardsum_likelihood

from vae_speech_align.forwardsum import (
    forwardsum_likelihood_numba,
    forwardsum_likelihood_torch,
    forwardsum_viterbi_numba,
    forwardsum_viterbi_torch,
)


class ForwardSumAllPass(torch.nn.Module):
    """Compute the forward-sum likelihood without dynamic programming.

    Applies logsumexp over explicitly enumerated state paths, which is
    tractable only when the paths are few enough to list by hand. Serves
    as the ground truth that the DP implementations (torch, numba,
    triton) are compared against, for gradients as well as values.
    """

    def __init__(self, initial_log_prob_output_matrix: torch.Tensor):
        super().__init__()

        self.log_prob_output_matrix = torch.nn.Parameter(
            initial_log_prob_output_matrix.clone()
        )

    def forward(self, state_pass_list: Iterable[list[int]]) -> torch.Tensor:
        log_prob_list = []
        for state_pass in state_pass_list:
            pass_likelohood = calc_one_pass_forwardsum_likelihood(
                state_pass, self.log_prob_output_matrix
            )

            log_prob_list.append(pass_likelohood)

        return torch.logsumexp(torch.stack(log_prob_list), dim=0)


class ForwardSumDP(torch.nn.Module):
    """Compute the forward-sum likelihood with the pure-torch DP.

    Wraps forwardsum_likelihood_torch / forwardsum_viterbi_torch so that
    the dynamic-programming result can be checked against
    ForwardSumAllPass and differentiated through.
    """

    def __init__(
        self,
        initial_log_prob_output_matrix: torch.Tensor,
        output_lengths: torch.Tensor,
        state_lengths: torch.Tensor,
    ):
        super().__init__()

        self.log_prob_output_matrix = torch.nn.Parameter(
            initial_log_prob_output_matrix.clone()
        )
        self.output_lengths = output_lengths
        self.state_lengths = state_lengths

    def forward(self) -> torch.Tensor:
        likelihood = forwardsum_likelihood_torch(
            self.log_prob_output_matrix,
            self.output_lengths,
            self.state_lengths,
        )

        return likelihood

    def viterbi(self) -> tuple[torch.Tensor, torch.Tensor]:
        likelihood, path = forwardsum_viterbi_torch(
            self.log_prob_output_matrix,
            self.output_lengths,
            self.state_lengths,
        )

        return likelihood, path


class ForwardSumNumba(torch.nn.Module):
    """Compute the forward-sum likelihood with the numba DP.

    Same interface as ForwardSumDP, backed by
    forwardsum_likelihood_numba / forwardsum_viterbi_numba.
    """

    def __init__(
        self,
        initial_log_prob_output_matrix: torch.Tensor,
        output_lengths: torch.Tensor,
        state_lengths: torch.Tensor,
    ):
        super().__init__()

        self.log_prob_output_matrix = torch.nn.Parameter(
            initial_log_prob_output_matrix.clone()
        )
        self.output_lengths = output_lengths
        self.state_lengths = state_lengths

    def forward(self) -> torch.Tensor:
        likelihood = forwardsum_likelihood_numba(
            self.log_prob_output_matrix,
            self.output_lengths,
            self.state_lengths,
        )

        return likelihood

    def viterbi(self) -> tuple[torch.Tensor, torch.Tensor]:
        likelihood, path = forwardsum_viterbi_numba(
            self.log_prob_output_matrix,
            self.output_lengths,
            self.state_lengths,
        )

        return likelihood, path


class ForwardSumTriton(torch.nn.Module):
    """Compute the forward-sum likelihood with the triton DP.

    Same interface as ForwardSumDP, backed by
    forwardsum_likelihood_triton / forwardsum_viterbi_triton. The triton
    functions are imported inside the methods so this module stays
    importable when triton is unavailable.
    """

    def __init__(
        self,
        initial_log_prob_output_matrix: torch.Tensor,
        output_lengths: torch.Tensor,
        state_lengths: torch.Tensor,
    ):
        super().__init__()

        self.log_prob_output_matrix = torch.nn.Parameter(
            initial_log_prob_output_matrix.clone()
        )
        self.output_lengths = output_lengths
        self.state_lengths = state_lengths

    def forward(self) -> torch.Tensor:
        from vae_speech_align.forwardsum import forwardsum_likelihood_triton

        # annotated because the lazy triton export is Any to mypy
        likelihood: torch.Tensor = forwardsum_likelihood_triton(
            self.log_prob_output_matrix,
            self.output_lengths,
            self.state_lengths,
        )

        return likelihood

    def viterbi(self) -> tuple[torch.Tensor, torch.Tensor]:
        from vae_speech_align.forwardsum import forwardsum_viterbi_triton

        likelihood, path = forwardsum_viterbi_triton(
            self.log_prob_output_matrix,
            self.output_lengths,
            self.state_lengths,
        )

        return likelihood, path

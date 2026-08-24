from typing import Callable

import torch

from vae_speech_align.forwardsum.matrix_util import (
    NEGATIVE_INF,
    NEGATIVE_INF_THRESHOLD,
    forwardsum_gamma_from_alphabeta,
    gather_terminal_log_likelihoods,
    make_backward_begin_mask,
    make_forward_begin_mask,
)


def forwardsum_forward_torch(
    log_prob_output_matrix: torch.Tensor,
    begin_time_list: torch.Tensor | None = None,
    begin_state_list: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute log alpha [B, T, K] with the forward algorithm.

    alpha[b, t, k] is the log-probability of starting in state 1 at
    time 1 and being in state k at time t, summed over all monotonic
    state sequences in between; it includes the emission term
    b_k(o_t) of its own time step. When begin_time_list and
    begin_state_list are given, the recursion starts from those
    positions instead of (1, 1) — forwardsum_backward_torch uses this
    to run the same recursion on flipped inputs.
    """
    B = log_prob_output_matrix.size(0)
    K = log_prob_output_matrix.size(-1)
    T = log_prob_output_matrix.size(-2)

    device = log_prob_output_matrix.device

    alpha_list = []

    if begin_time_list is not None and begin_state_list is not None:
        # implement for backward algorithm
        begin_mask = make_backward_begin_mask(
            B, T, K, begin_time_list, begin_state_list, device
        )
    else:
        begin_mask = make_forward_begin_mask(B, T, K, device)

    # t = 0
    alpha0 = torch.zeros((B, K), device=device)
    alpha0[~begin_mask[:, 0, :]] = NEGATIVE_INF
    alpha_list.append(alpha0)

    for t in range(1, T - 1):
        alpha_t = torch.zeros((B, K), device=device)

        alpha_t[:] = NEGATIVE_INF

        tmp = torch.stack(
            [alpha_list[t - 1][:, 0 : K - 2], alpha_list[t - 1][:, 1 : K - 1]]
        )

        alpha_t[:, 1 : K - 1] = (
            torch.logsumexp(tmp, dim=0)
            + log_prob_output_matrix[:, t, 1 : K - 1]
        )

        # for the beginning of backward
        alpha_t[begin_mask[:, t, :]] = 0.0

        alpha_list.append(alpha_t)

    # t = T + 1 (Tp1)
    alpha_Tp1 = torch.zeros((B, K), device=device)
    alpha_Tp1[:] = NEGATIVE_INF
    alpha_list.append(alpha_Tp1)

    alpha = torch.stack(alpha_list, dim=0).transpose(
        0, 1
    )  # [T][B, K] -> [B, T, K]

    return alpha


def forwardsum_backward_torch(
    log_prob_output_matrix: torch.Tensor,
    output_lengths: torch.Tensor,
    state_lengths: torch.Tensor,
) -> torch.Tensor:
    """Compute log beta [B, T, K] with the backward algorithm.

    beta[b, t, k] is the log-probability of the whole remaining state
    sequence: being in state k at time t and ending in the last state
    K at the last time T (per-element lengths), summed over all
    monotonic continuations; like alpha it includes the emission term
    b_k(o_t) of its own time step. Implemented as the forward
    recursion on the time/state-flipped matrix, so beta[:, 1, 1]
    equals the total log-likelihood.
    """
    K = log_prob_output_matrix.size(-1)
    T = log_prob_output_matrix.size(-2)

    begin_time_list = T - 2 - output_lengths
    begin_state_list = K - 2 - state_lengths

    beta = forwardsum_forward_torch(
        log_prob_output_matrix.flip(dims=(-2, -1)),
        begin_time_list=begin_time_list,
        begin_state_list=begin_state_list,
    )
    beta = beta.flip(dims=(-2, -1))
    return beta


class ForwardSumLogLikelihoodFunctionTorch(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore
        ctx,
        log_prob_output_matrix,
        output_lengths,
        state_lengths,
    ):
        alpha = forwardsum_forward_torch(log_prob_output_matrix)

        log_likelihoods = gather_terminal_log_likelihoods(
            alpha, output_lengths, state_lengths
        )

        variables = [
            log_prob_output_matrix,
            output_lengths,
            state_lengths,
            alpha,
            log_likelihoods,
        ]
        ctx.save_for_backward(*variables)

        return log_likelihoods

    @staticmethod
    def backward(ctx, dL_dll):  # type: ignore
        (
            log_prob_output_matrix,
            output_lengths,
            state_lengths,
            alpha,
            log_likelihoods,
        ) = ctx.saved_tensors

        beta = forwardsum_backward_torch(
            log_prob_output_matrix,
            output_lengths,
            state_lengths,
        )

        gamma = forwardsum_gamma_from_alphabeta(
            alpha,
            beta,
            log_prob_output_matrix,
        )

        # minibatch-wise
        dL_doutput = dL_dll[:, None, None, None] * gamma

        return dL_doutput, None, None


forwardsum_likelihood_torch: Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
] = ForwardSumLogLikelihoodFunctionTorch.apply


def forwardsum_gamma_torch(
    log_prob_output_matrix: torch.Tensor,
    output_lengths: torch.Tensor,
    state_lengths: torch.Tensor,
) -> torch.Tensor:
    """Compute the posterior gamma [B, T, K].

    gamma[b, t, k] is the state occupancy posterior: the probability
    of being in state k at time t given the whole observation
    sequence. See forwardsum_gamma_from_alphabeta for the derivation
    from alpha and beta.
    """
    alpha = forwardsum_forward_torch(log_prob_output_matrix)

    beta = forwardsum_backward_torch(
        log_prob_output_matrix,
        output_lengths,
        state_lengths,
    )

    gamma = forwardsum_gamma_from_alphabeta(
        alpha,
        beta,
        log_prob_output_matrix,
    )

    return gamma


def forwardsum_viterbi_delta_torch(
    log_prob_output_matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the Viterbi score delta and backpointers psi.

    delta[b, t, k] is the log-probability of the most probable
    monotonic path that starts in state 1 at time 1 and reaches state
    k at time t (the forward algorithm with max instead of sum).
    psi[b, t, k] records the route of that best path: 1 if (t, k) was
    reached by a transition from state k - 1, 0 if by staying in
    state k.
    """
    B = log_prob_output_matrix.size(0)
    K = log_prob_output_matrix.size(-1)
    T = log_prob_output_matrix.size(-2)

    device = log_prob_output_matrix.device

    delta_list = []
    psi_list = []

    begin_mask = make_forward_begin_mask(B, T, K, device=device)

    # t = 0
    delta0 = torch.zeros((B, K), device=device)
    delta0[~begin_mask[:, 0, :]] = NEGATIVE_INF
    delta_list.append(delta0)

    psi0 = torch.zeros((B, K), dtype=torch.int32, device=device)
    psi0[begin_mask[:, 0, :]] = 1
    psi_list.append(psi0)

    for t in range(1, T - 1):
        delta_t = torch.zeros((B, K), device=device)
        psi_t = torch.zeros((B, K), dtype=torch.int32, device=device)

        delta_t[:] = NEGATIVE_INF

        delta_prev_transition = delta_list[t - 1][:, 0 : K - 2]
        delta_prev_stay = delta_list[t - 1][:, 1 : K - 1]

        # Break ties toward "stay" so this backend agrees with the numba
        # and triton kernels (which use strict > / >=); torch.max would
        # otherwise pick the transition on a tie and is non-deterministic
        # across CPU/CUDA.
        is_transition = delta_prev_transition > delta_prev_stay
        best_prev = torch.where(
            is_transition, delta_prev_transition, delta_prev_stay
        )
        delta_t[:, 1 : K - 1] = (
            best_prev + log_prob_output_matrix[:, t, 1 : K - 1]
        )

        psi_t[:, 1 : K - 1] = is_transition.to(torch.int32)

        # for the beginning of backward
        delta_t[begin_mask[:, t, :]] = 0.0
        psi_t[begin_mask[:, t, :]] = 1

        delta_list.append(delta_t)
        psi_list.append(psi_t)

    delta_Tp1 = torch.zeros((B, K), device=device)
    delta_Tp1[:] = NEGATIVE_INF
    delta_list.append(delta_Tp1)

    psi_Tp1 = torch.zeros((B, K), dtype=torch.int32, device=device)
    psi_list.append(psi_Tp1)

    delta = torch.stack(delta_list, dim=0).transpose(
        0, 1
    )  # [T][B, K] -> [B, T, K]
    psi = torch.stack(psi_list, dim=0).transpose(
        0, 1
    )  # [T][B, K] -> [B, T, K]

    return delta, psi


def forwardsum_viterbi_path_torch(
    delta: torch.Tensor,
    psi: torch.Tensor,
    output_lengths: torch.Tensor,
    state_lengths: torch.Tensor,
) -> torch.Tensor:
    """Trace the psi backpointers into a 0/1 path matrix.

    Starting from each element's terminal position (output_lengths[b],
    state_lengths[b]), follows psi backwards and marks the visited
    (t, k) cells of the most probable path with 1.
    """
    T = psi.size(-2)
    B = psi.size(0)

    device = delta.device

    trace_states = torch.zeros_like(state_lengths)
    path = torch.zeros_like(delta)

    for t in range(T - 2, 0, -1):
        trace_states = torch.where(
            output_lengths == t, state_lengths.detach(), trace_states
        )

        path[torch.arange(B, device=device), t, trace_states] = 1
        path[:, t, :][delta[:, t, :] < NEGATIVE_INF_THRESHOLD] = 0.0

        trace_states = torch.where(
            psi[torch.arange(B, device=device), t, trace_states] == 1,
            trace_states - 1,
            trace_states,
        )

    return path


def forwardsum_viterbi_torch(
    log_prob_output_matrix: torch.Tensor,
    output_lengths: torch.Tensor,
    state_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the best-path log-likelihoods and the Viterbi path.

    Combines the delta/psi recursion with the traceback: the
    log-likelihood is delta at each element's terminal position, and
    the path is the traced 0/1 alignment matrix.
    """
    delta, psi = forwardsum_viterbi_delta_torch(
        log_prob_output_matrix,
    )
    path = forwardsum_viterbi_path_torch(
        delta, psi, output_lengths, state_lengths
    )

    log_likelihoods = gather_terminal_log_likelihoods(
        delta, output_lengths, state_lengths
    )

    return log_likelihoods, path

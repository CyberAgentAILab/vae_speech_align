from typing import Callable

import numba
import numpy as np
import torch

from vae_speech_align.forwardsum.matrix_util import (
    NEGATIVE_INF,
    forwardsum_gamma_from_alphabeta,
    gather_terminal_log_likelihoods,
)


@numba.jit(fastmath=True, cache=True)
def logsumexp_vals(val1: numba.float32, val2: numba.float32) -> numba.float32:
    if val1 > val2:
        val_max = val1
        val_min = val2
    else:
        val_max = val2
        val_min = val1

    expdiff = np.exp(val_min - val_max)
    return val_max + np.log1p(expdiff)


@numba.njit(fastmath=True, cache=True)
def forwardsum_alpha_each(
    log_prob_output_matrix: np.ndarray,
    begin_time: np.ndarray,
    begin_state: np.ndarray,
    alpha: np.ndarray,
    negative_inf: float,
) -> None:

    T = log_prob_output_matrix.shape[0]
    K = log_prob_output_matrix.shape[1]

    if begin_time == 0:
        alpha[begin_time, begin_state] = 0

    for t in range(1, T - 1):
        for k in range(1, K - 1):
            alpha_prev_transition = alpha[t - 1, k - 1]
            alpha_prev_stay = alpha[t - 1, k]

            alpha_prev_sum = logsumexp_vals(
                alpha_prev_transition, alpha_prev_stay
            )
            alpha[t, k] = alpha_prev_sum + log_prob_output_matrix[t, k]

        if t == begin_time:
            alpha[begin_time, begin_state] = 0


@numba.njit(fastmath=True, cache=True, parallel=True)
def forwardsum_alpha_base(
    log_prob_output_matrix: np.ndarray,
    begin_time: np.ndarray,
    begin_state: np.ndarray,
    alpha: np.ndarray,
    negative_inf: float,
) -> None:
    B = log_prob_output_matrix.shape[0]

    for b in numba.prange(B):
        forwardsum_alpha_each(
            log_prob_output_matrix[b],
            begin_time[b],
            begin_state[b],
            alpha[b],
            negative_inf,
        )


def forwardsum_forward_numba(
    log_prob_output_matrix: torch.Tensor,
    begin_time_list: torch.Tensor | None = None,
    begin_state_list: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute log alpha with the forward algorithm (numba backend).

    See forwardsum_forward_torch for the definition of alpha.
    """
    B = log_prob_output_matrix.size(0)
    K = log_prob_output_matrix.size(-1)
    T = log_prob_output_matrix.size(-2)

    device = log_prob_output_matrix.device
    dtype = log_prob_output_matrix.dtype

    alpha = np.empty((B, T, K), dtype=np.float32)
    alpha[:] = NEGATIVE_INF

    log_prob_output_matrix_npy = log_prob_output_matrix.cpu().numpy()

    if begin_time_list is not None and begin_state_list is not None:
        begin_time = begin_time_list.cpu().numpy().astype(np.int32)
        begin_state = begin_state_list.cpu().numpy().astype(np.int32)
    else:
        begin_time = np.zeros(B, dtype=np.int32)
        begin_state = np.zeros(B, dtype=np.int32)

    forwardsum_alpha_base(
        log_prob_output_matrix_npy,
        begin_time,
        begin_state,
        alpha,
        NEGATIVE_INF,
    )

    return torch.from_numpy(alpha).to(device=device, dtype=dtype)


def forwardsum_backward_numba(
    log_prob_output_matrix: torch.Tensor,
    output_lengths: torch.Tensor,
    state_lengths: torch.Tensor,
) -> torch.Tensor:
    """Compute log beta with the backward algorithm (numba backend).

    See forwardsum_backward_torch for the definition of beta.
    """
    K = log_prob_output_matrix.size(-1)
    T = log_prob_output_matrix.size(-2)

    begin_time_list = T - 2 - output_lengths
    begin_state_list = K - 2 - state_lengths

    beta = forwardsum_forward_numba(
        log_prob_output_matrix.flip(dims=(-2, -1)),
        begin_time_list=begin_time_list,
        begin_state_list=begin_state_list,
    )
    beta = beta.flip(dims=(-2, -1))
    return beta


class ForwardSumLogLikelihoodFunctionNumba(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore
        ctx,
        log_prob_output_matrix,
        output_lengths,
        state_lengths,
    ):
        alpha = forwardsum_forward_numba(log_prob_output_matrix)

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

        beta = forwardsum_backward_numba(
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


forwardsum_likelihood_numba: Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
] = ForwardSumLogLikelihoodFunctionNumba.apply


def forwardsum_gamma_numba(
    log_prob_output_matrix: torch.Tensor,
    output_lengths: torch.Tensor,
    state_lengths: torch.Tensor,
    return_likelihood: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Compute the posterior gamma (numba backend).

    See forwardsum_gamma_torch for the definition of gamma.
    """
    alpha = forwardsum_forward_numba(log_prob_output_matrix)

    beta = forwardsum_backward_numba(
        log_prob_output_matrix,
        output_lengths,
        state_lengths,
    )

    gamma = forwardsum_gamma_from_alphabeta(
        alpha,
        beta,
        log_prob_output_matrix,
    )

    if return_likelihood:
        log_likelihoods = gather_terminal_log_likelihoods(
            alpha, output_lengths, state_lengths
        )

        return gamma, log_likelihoods
    else:
        return gamma


@numba.njit(fastmath=True, cache=True)
def forwardsum_viterbi_delta_each(
    log_prob_output_matrix: np.ndarray,
    delta: np.ndarray,
    psi: np.ndarray,
    negative_inf: float,
) -> None:
    T = log_prob_output_matrix.shape[0]
    K = log_prob_output_matrix.shape[1]

    for t in range(1, T - 1):
        for k in range(1, K - 1):
            delta_prev_transition = delta[t - 1, k - 1]
            delta_prev_stay = delta[t - 1, k]

            is_transition = delta_prev_transition > delta_prev_stay
            delta_prev_max = (
                delta_prev_transition if is_transition else delta_prev_stay
            )

            delta[t, k] = delta_prev_max + log_prob_output_matrix[t, k]
            psi[t, k] = np.int32(is_transition)


@numba.njit(fastmath=True, cache=True, parallel=True)
def forwardsum_viterbi_delta_base(
    log_prob_output_matrix: np.ndarray,
    delta: np.ndarray,
    psi: np.ndarray,
    negative_inf: float,
) -> None:
    B = log_prob_output_matrix.shape[0]

    for b in numba.prange(B):
        forwardsum_viterbi_delta_each(
            log_prob_output_matrix[b], delta[b], psi[b], negative_inf
        )


def forwardsum_viterbi_delta_numba(
    log_prob_output_matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    B = log_prob_output_matrix.size(0)
    K = log_prob_output_matrix.size(-1)
    T = log_prob_output_matrix.size(-2)

    device = log_prob_output_matrix.device
    dtype = log_prob_output_matrix.dtype

    delta_npy = np.empty((B, T, K), dtype=np.float32)
    delta_npy[:] = NEGATIVE_INF
    delta_npy[:, 0, 0] = 0.0

    psi_npy = np.zeros((B, T, K), dtype=np.int32)

    log_prob_output_matrix_npy = (
        log_prob_output_matrix.cpu().numpy().astype(np.float32)
    )

    forwardsum_viterbi_delta_base(
        log_prob_output_matrix_npy, delta_npy, psi_npy, NEGATIVE_INF
    )

    delta = torch.from_numpy(delta_npy).to(device=device, dtype=dtype)
    psi = torch.from_numpy(psi_npy).to(device=device)

    return delta, psi


@numba.njit(fastmath=True, cache=True)
def forwardsum_viterbi_path_each(
    psi: np.ndarray, output_length: int, state_length: int, path: np.ndarray
) -> None:
    k = state_length

    for t in range(output_length, 0, -1):
        path[t, k] = 1.0

        if psi[t, k] > 0:
            k -= 1


@numba.njit(fastmath=True, cache=True, parallel=True)
def forwardsum_viterbi_path_base(
    psi: np.ndarray,
    output_lengths: np.ndarray,
    state_lengths: np.ndarray,
    path: np.ndarray,
) -> None:
    B = psi.shape[0]

    for b in numba.prange(B):
        forwardsum_viterbi_path_each(
            psi[b], output_lengths[b], state_lengths[b], path[b]
        )


def forwardsum_viterbi_path_numba(
    delta: torch.Tensor,
    psi: torch.Tensor,
    output_lengths: torch.Tensor,
    state_lengths: torch.Tensor,
) -> torch.Tensor:

    device = psi.device
    dtype = delta.dtype

    path = np.zeros_like(delta.cpu().numpy())
    output_lengths_npy = output_lengths.cpu().numpy().astype(np.int32)
    state_lengths_npy = state_lengths.cpu().numpy().astype(np.int32)
    psi_npy = psi.cpu().numpy().astype(np.int32)

    forwardsum_viterbi_path_base(
        psi_npy, output_lengths_npy, state_lengths_npy, path
    )

    return torch.from_numpy(path).to(device=device, dtype=dtype)


def forwardsum_viterbi_numba(
    log_prob_output_matrix: torch.Tensor,
    output_lengths: torch.Tensor,
    state_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return best-path log-likelihoods and path (numba backend).

    See forwardsum_viterbi_torch for the definitions of delta, psi,
    and the traced path.
    """
    delta, psi = forwardsum_viterbi_delta_numba(
        log_prob_output_matrix,
    )
    path = forwardsum_viterbi_path_numba(
        delta,
        psi,
        output_lengths,
        state_lengths,
    )

    log_likelihoods = gather_terminal_log_likelihoods(
        delta, output_lengths, state_lengths
    )

    return log_likelihoods, path

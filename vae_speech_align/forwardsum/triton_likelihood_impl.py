from typing import Callable

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from vae_speech_align.forwardsum.matrix_util import (
    NEGATIVE_INF,
    forwardsum_gamma_from_alphabeta,
    gather_terminal_log_likelihoods,
    make_backward_begin_mask,
    make_forward_begin_mask,
)
from vae_speech_align.forwardsum.triton_util import TritonConfigK, configs_K

negative_inf = triton.language.constexpr(NEGATIVE_INF)


@triton.jit  # type: ignore
def logsumexp_vecs(vec1, vec2):
    val_max = tl.maximum(vec1, vec2)
    val_min = tl.minimum(vec1, vec2)
    expdiff = tl.exp(val_min - val_max)
    return val_max + libdevice.log1p(expdiff)


@triton.jit  # type: ignore
def forwardsum_alpha_set_t0(
    begin_mask_ptr,
    alpha_ptr,
    K: int,
    batch_index: int,
    BTK_stride_B: int,
    BTK_stride_T: int,
    BTK_stride_K: int,
    BLOCK_SIZE_K: tl.constexpr,
):
    t = 0

    for K_chunk_i in range(tl.cdiv(K, BLOCK_SIZE_K)):
        K_offsets = K_chunk_i * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        K_mask = K_offsets < K  # 0 to K - 1

        # load
        alpha_0_offsets = (
            batch_index * BTK_stride_B
            + t * BTK_stride_T
            + K_offsets * BTK_stride_K
        )
        alpha_0 = tl.load(
            alpha_ptr + alpha_0_offsets, mask=K_mask, other=negative_inf
        )  # 0 to K - 1
        begin_mask_0 = tl.load(
            begin_mask_ptr + alpha_0_offsets, mask=K_mask, other=0
        )  # 0 to K - 1

        # calc_value
        alpha_0 = tl.where(begin_mask_0, 0.0, alpha_0)

        # store
        tl.store(
            alpha_ptr + alpha_0_offsets, alpha_0, mask=K_mask
        )  # 0 to K - 1


@triton.jit  # type: ignore
def forwardsum_alpha_step(
    log_prob_output_matrix_ptr,
    alpha_ptr,
    K: int,
    batch_index: int,
    t: int,
    BTK_stride_B: int,
    BTK_stride_T: int,
    BTK_stride_K: int,
    BLOCK_SIZE_K: tl.constexpr,
):
    for K_chunk_i in range(tl.cdiv(K, BLOCK_SIZE_K)):
        K_offsets = K_chunk_i * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        K_minus1_offsets = K_offsets - 1
        K_mask = K_offsets < K - 1  # (0 to K - 2) or (1 to K - 1)
        K_minus1_mask = (K_minus1_offsets >= 0) & (K_minus1_offsets < K - 1)

        # # load
        alpha_prev_transition_offsets = (
            batch_index * BTK_stride_B
            + (t - 1) * BTK_stride_T
            + K_minus1_offsets * BTK_stride_K
        )
        alpha_prev_transition = tl.load(
            alpha_ptr + alpha_prev_transition_offsets,
            mask=K_minus1_mask,
            other=negative_inf,
        )  # 0 to K-2

        alpha_prev_stay_offsets = (
            batch_index * BTK_stride_B
            + (t - 1) * BTK_stride_T
            + K_offsets * BTK_stride_K
        )
        alpha_prev_stay = tl.load(
            alpha_ptr + alpha_prev_stay_offsets,
            mask=K_mask,
            other=negative_inf,
        )  # 1 to K-1

        log_prob_output_offsets = (
            batch_index * BTK_stride_B
            + t * BTK_stride_T
            + K_offsets * BTK_stride_K
        )
        log_prob_output_vec = tl.load(
            log_prob_output_matrix_ptr + log_prob_output_offsets,
            mask=K_mask,
            other=negative_inf,
        )  # 1 to K - 1

        # calc value
        alpha_prev_sum = logsumexp_vecs(alpha_prev_transition, alpha_prev_stay)
        alpha = alpha_prev_sum + log_prob_output_vec

        # store
        alpha_offsets = (
            batch_index * BTK_stride_B
            + t * BTK_stride_T
            + K_offsets * BTK_stride_K
        )
        tl.store(alpha_ptr + alpha_offsets, alpha, mask=K_mask)
        # the next DP step loads the row stored above; the barrier
        # orders the global store before those loads (without it the
        # compiler/warps may overlap iterations and read stale rows)
        tl.debug_barrier()


@triton.jit  # type: ignore
def forwardsum_alpha_set_begin_state_for_backward(
    begin_mask_ptr,
    alpha_ptr,
    K: int,
    batch_index: int,
    t: int,
    BTK_stride_B: int,
    BTK_stride_T: int,
    BTK_stride_K: int,
    BLOCK_SIZE_K: tl.constexpr,
):
    for K_chunk_i in range(tl.cdiv(K, BLOCK_SIZE_K)):
        K_offsets = K_chunk_i * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        K_mask = K_offsets < K  # 0 to K - 1

        alpha_t_offset = (
            batch_index * BTK_stride_B
            + t * BTK_stride_T
            + K_offsets * BTK_stride_K
        )
        alpha_t = tl.load(
            alpha_ptr + alpha_t_offset, mask=K_mask, other=negative_inf
        )
        begin_mask_t = tl.load(
            begin_mask_ptr + alpha_t_offset, mask=K_mask, other=0
        )
        alpha_t = tl.where(begin_mask_t, 0.0, alpha_t)

        tl.store(alpha_ptr + alpha_t_offset, alpha_t, mask=K_mask)
        # the next DP step loads the row stored above; the barrier
        # orders the global store before those loads (without it the
        # compiler/warps may overlap iterations and read stale rows)
        tl.debug_barrier()


@triton.jit  # type: ignore
def forwardsum_alpha_loop_base(
    log_prob_output_matrix_ptr,
    begin_mask_ptr,
    alpha_ptr,
    B: int,
    T: int,
    K: int,
    BTK_stride_B: int,
    BTK_stride_T: int,
    BTK_stride_K: int,
    K_next_power_of_2: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    batch_index = tl.program_id(axis=0)

    # set initial state
    forwardsum_alpha_set_t0(
        begin_mask_ptr,
        alpha_ptr,
        K,
        batch_index,
        BTK_stride_B,
        BTK_stride_T,
        BTK_stride_K,
        BLOCK_SIZE_K,
    )

    for t in range(1, T - 1):
        forwardsum_alpha_step(
            log_prob_output_matrix_ptr,
            alpha_ptr,
            K,
            batch_index,
            t,
            BTK_stride_B,
            BTK_stride_T,
            BTK_stride_K,
            BLOCK_SIZE_K,
        )

        forwardsum_alpha_set_begin_state_for_backward(
            begin_mask_ptr,
            alpha_ptr,
            K,
            batch_index,
            t,
            BTK_stride_B,
            BTK_stride_T,
            BTK_stride_K,
            BLOCK_SIZE_K,
        )


forwardsum_alpha_loop = triton.autotune(
    configs=configs_K,
    key=["K_next_power_of_2"],
)(forwardsum_alpha_loop_base)


def forwardsum_forward_triton(
    log_prob_output_matrix: torch.Tensor,
    begin_time_list: torch.Tensor | None = None,
    begin_state_list: torch.Tensor | None = None,
    triton_custom_config: TritonConfigK | None = None,
) -> torch.Tensor:
    """Compute log alpha with the forward algorithm (triton backend).

    See forwardsum_forward_torch for the definition of alpha.
    """
    B = log_prob_output_matrix.size(0)
    K = log_prob_output_matrix.size(-1)
    T = log_prob_output_matrix.size(-2)

    device = log_prob_output_matrix.device

    if begin_time_list is not None and begin_state_list is not None:
        begin_mask = make_backward_begin_mask(
            B, T, K, begin_time_list, begin_state_list, device
        )
    else:
        begin_mask = make_forward_begin_mask(B, T, K, device)

    alpha = torch.full((B, T, K), NEGATIVE_INF, device=device)
    grid = (B,)

    BTK_stride_B, BTK_stride_T, BTK_stride_K = alpha.stride()

    if triton_custom_config is None:
        forwardsum_alpha_loop[grid](
            log_prob_output_matrix,
            begin_mask,
            alpha,
            B,
            T,
            K,
            BTK_stride_B,
            BTK_stride_T,
            BTK_stride_K,
            triton.next_power_of_2(K),
        )
    else:
        forwardsum_alpha_loop_base[grid](
            log_prob_output_matrix,
            begin_mask,
            alpha,
            B,
            T,
            K,
            BTK_stride_B,
            BTK_stride_T,
            BTK_stride_K,
            triton.next_power_of_2(K),
            BLOCK_SIZE_K=triton_custom_config.BLOCK_SIZE_K,
            num_warps=triton_custom_config.num_warps,
            num_stages=triton_custom_config.num_stages,
        )

    return alpha


def forwardsum_backward_triton(
    log_prob_output_matrix: torch.Tensor,
    output_lengths: torch.Tensor,
    state_lengths: torch.Tensor,
    triton_custom_config: TritonConfigK | None = None,
) -> torch.Tensor:
    """Compute log beta with the backward algorithm (triton backend).

    See forwardsum_backward_torch for the definition of beta.
    """
    K = log_prob_output_matrix.size(-1)
    T = log_prob_output_matrix.size(-2)

    begin_time_list = T - 2 - output_lengths
    begin_state_list = K - 2 - state_lengths

    beta = forwardsum_forward_triton(
        log_prob_output_matrix.flip(dims=(-2, -1)),
        begin_time_list=begin_time_list,
        begin_state_list=begin_state_list,
        triton_custom_config=triton_custom_config,
    )
    beta = beta.flip(dims=(-2, -1))
    return beta


class ForwardSumLogLikelihoodFunctionTriton(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore
        ctx,
        log_prob_output_matrix,
        output_lengths,
        state_lengths,
    ):
        alpha = forwardsum_forward_triton(log_prob_output_matrix)

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

        beta = forwardsum_backward_triton(
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


forwardsum_likelihood_triton: Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
] = ForwardSumLogLikelihoodFunctionTriton.apply


def forwardsum_gamma_triton(
    log_prob_output_matrix: torch.Tensor,
    output_lengths: torch.Tensor,
    state_lengths: torch.Tensor,
    return_likelihood: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Compute the posterior gamma (triton backend).

    See forwardsum_gamma_torch for the definition of gamma.
    """
    alpha = forwardsum_forward_triton(log_prob_output_matrix)

    beta = forwardsum_backward_triton(
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

import torch
import triton
import triton.language as tl

from vae_speech_align.forwardsum.matrix_util import (
    NEGATIVE_INF,
    gather_terminal_log_likelihoods,
)
from vae_speech_align.forwardsum.triton_util import TritonConfigK, configs_K

negative_inf = triton.language.constexpr(NEGATIVE_INF)


@triton.jit  # type: ignore
def forwardsum_delta_step(
    log_prob_output_matrix_ptr,
    delta_ptr,
    psi_ptr,
    K: int,
    batch_index: int,
    t: int,
    BTK_stride_B: int,
    BTK_stride_T: int,
    BTK_stride_K: int,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Set delta[b, t, k]
    """
    for K_chunk_i in range(tl.cdiv(K, BLOCK_SIZE_K)):
        K_offsets = K_chunk_i * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        K_mask = K_offsets < K - 1  # (0 to K - 2) or (1 to K - 1)

        psi = tl.zeros((BLOCK_SIZE_K,), dtype=tl.int32)

        # load
        delta_prev_transition_offsets = (
            batch_index * BTK_stride_B
            + (t - 1) * BTK_stride_T
            + K_offsets * BTK_stride_K
        )
        delta_prev_transition = tl.load(
            delta_ptr + delta_prev_transition_offsets,
            mask=K_mask,
            other=negative_inf,
        )  # 0 to K-2

        delta_prev_stay_offsets = (
            batch_index * BTK_stride_B
            + (t - 1) * BTK_stride_T
            + (K_offsets + 1) * BTK_stride_K
        )
        delta_prev_stay = tl.load(
            delta_ptr + delta_prev_stay_offsets,
            mask=K_mask,
            other=negative_inf,
        )  # 1 to K-1

        log_prob_output_offsets = (
            batch_index * BTK_stride_B
            + t * BTK_stride_T
            + (K_offsets + 1) * BTK_stride_K
        )
        log_prob_output_vec = tl.load(
            log_prob_output_matrix_ptr + log_prob_output_offsets,
            mask=K_mask,
            other=negative_inf,
        )  # 1 to K - 1

        # calc value
        is_stay = delta_prev_stay >= delta_prev_transition
        psi = tl.where(is_stay, psi, psi + 1)
        delta_prev_max = tl.where(
            is_stay, delta_prev_stay, delta_prev_transition
        )
        delta = delta_prev_max + log_prob_output_vec

        # store
        delta_offsets = (
            batch_index * BTK_stride_B
            + t * BTK_stride_T
            + (K_offsets + 1) * BTK_stride_K
        )
        tl.store(delta_ptr + delta_offsets, delta, mask=K_mask)
        tl.store(psi_ptr + delta_offsets, psi, mask=K_mask)
        # the next DP step loads the row stored above; the barrier
        # orders the global store before those loads (without it the
        # compiler/warps may overlap iterations and read stale rows)
        tl.debug_barrier()


@triton.jit  # type: ignore
def forwardsum_delta_loop_base(
    log_prob_output_matrix_ptr,
    delta_ptr,
    psi_ptr,
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

    for t in range(1, T - 1):
        forwardsum_delta_step(
            log_prob_output_matrix_ptr,
            delta_ptr,
            psi_ptr,
            K,
            batch_index,
            t,
            BTK_stride_B,
            BTK_stride_T,
            BTK_stride_K,
            BLOCK_SIZE_K,
        )


forwardsum_delta_loop = triton.autotune(
    configs=configs_K,
    key=["K_next_power_of_2"],
)(forwardsum_delta_loop_base)


def forwardsum_viterbi_delta_triton(
    log_prob_output_matrix: torch.Tensor,
    triton_custom_config: TritonConfigK | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B = log_prob_output_matrix.size(0)
    K = log_prob_output_matrix.size(-1)
    T = log_prob_output_matrix.size(-2)

    device = log_prob_output_matrix.device

    delta = torch.full((B, T, K), NEGATIVE_INF, device=device)
    # DP base case: the path starts at (t=0, k=0) with probability one
    delta[:, 0, 0] = 0.0

    psi = torch.zeros((B, T, K), device=device, dtype=torch.int32)
    grid = (B,)

    BTK_stride_B, BTK_stride_T, BTK_stride_K = delta.stride()

    if triton_custom_config is None:
        forwardsum_delta_loop[grid](
            log_prob_output_matrix,
            delta,
            psi,
            B,
            T,
            K,
            BTK_stride_B,
            BTK_stride_T,
            BTK_stride_K,
            triton.next_power_of_2(K),
        )
    else:
        forwardsum_delta_loop_base[grid](
            log_prob_output_matrix,
            delta,
            psi,
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
    return delta, psi


@triton.jit  # type: ignore
def forwardsum_viterbi_path_loop_base(
    psi_ptr,
    output_lengths_ptr,
    state_lengths_ptr,
    path_ptr,
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

    # load
    state_length = tl.load(state_lengths_ptr + batch_index)
    output_length = tl.load(output_lengths_ptr + batch_index)

    k = -1

    for t in range(T - 2, 0, -1):
        # set initial point
        k = tl.where(t == output_length, state_length, k)

        # load
        psi_offset = (
            batch_index * BTK_stride_B + t * BTK_stride_T + k * BTK_stride_K
        )
        psi_tk = tl.load(psi_ptr + psi_offset)

        # store
        value = tl.where(k > 0, 1.0, 0.0)
        tl.store(path_ptr + psi_offset, value)

        # update
        k = tl.where(psi_tk == 1, k - 1, k)


forwardsum_viterbi_path_loop = triton.autotune(
    configs=configs_K,
    key=["K_next_power_of_2"],
)(forwardsum_viterbi_path_loop_base)


def forwardsum_viterbi_path_triton(
    delta: torch.Tensor,
    psi: torch.Tensor,
    output_lengths: torch.Tensor,
    state_lengths: torch.Tensor,
    triton_custom_config: TritonConfigK | None = None,
) -> torch.Tensor:
    T = psi.size(-2)
    K = psi.size(-1)
    B = psi.size(0)

    device = psi.device

    path = torch.zeros_like(delta)

    grid = (B,)

    BTK_stride_B, BTK_stride_T, BTK_stride_K = path.stride()

    if triton_custom_config is None:
        forwardsum_viterbi_path_loop[grid](
            psi,
            torch.as_tensor(output_lengths, dtype=torch.int32, device=device),
            torch.as_tensor(state_lengths, dtype=torch.int32, device=device),
            path,
            B,
            T,
            K,
            BTK_stride_B,
            BTK_stride_T,
            BTK_stride_K,
            triton.next_power_of_2(K),
        )
    else:
        forwardsum_viterbi_path_loop_base[grid](
            psi,
            torch.as_tensor(output_lengths, dtype=torch.int32, device=device),
            torch.as_tensor(state_lengths, dtype=torch.int32, device=device),
            path,
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

    return path


def forwardsum_viterbi_triton(
    log_prob_output_matrix: torch.Tensor,
    output_lengths: torch.Tensor,
    state_lengths: torch.Tensor,
    triton_custom_config: TritonConfigK | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return best-path log-likelihoods and path (triton backend).

    See forwardsum_viterbi_torch for the definitions of delta, psi,
    and the traced path.
    """
    delta, psi = forwardsum_viterbi_delta_triton(
        log_prob_output_matrix,
        triton_custom_config=triton_custom_config,
    )
    path = forwardsum_viterbi_path_triton(
        delta,
        psi,
        output_lengths,
        state_lengths,
        triton_custom_config=triton_custom_config,
    )

    log_likelihoods = gather_terminal_log_likelihoods(
        delta, output_lengths, state_lengths
    )

    return log_likelihoods, path

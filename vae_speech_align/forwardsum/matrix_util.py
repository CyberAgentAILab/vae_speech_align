import math
from enum import Enum, auto

import torch

NEGATIVE_INF = -1.0e10
NEGATIVE_INF_THRESHOLD = -1.0e5


class MatchingFuncType(Enum):
    GAUSSIAN = auto()
    INNER_PRODUCT = auto()


def calc_gaussian_log_likelihood_matrix(
    x: torch.Tensor, mean: torch.Tensor, var: torch.Tensor
) -> torch.Tensor:
    """
    Input:
        x: (B, T, D)
        mean: (B, K, D)
        var: (B, K, D)
    Output:
        log likelihood matrix: (B, T, K)
    """

    D = x.size(-1)

    xSx = (x * x) @ (1 / var).transpose(-1, -2)
    mSm = (mean * mean / var).sum(dim=-1, keepdim=True).transpose(-1, -2)
    xSm = x @ (mean / var).transpose(-1, -2)

    ll1: float = -0.5 * D * math.log(2 * math.pi)
    ll2: torch.Tensor = -0.5 * torch.log(var).sum(
        dim=-1, keepdim=True
    ).transpose(-1, -2)
    ll3: torch.Tensor = -0.5 * xSx - 0.5 * mSm + xSm

    log_likelihood = ll1 + ll2 + ll3

    return log_likelihood


def calc_inner_product_matrix(
    x: torch.Tensor, mean: torch.Tensor
) -> torch.Tensor:
    """
    Input:
        x: (B, T, D)
        mean: (B, K, D)
    Output:
        inner product matrix: (B, T, K)
    """

    xm = x @ mean.transpose(-1, -2)

    return xm


# The observations, per-batch lengths, state Gaussian parameters, and
# matching/normalization options are distinct tensors/flags that cannot be
# meaningfully grouped, so the argument-count rule (CFQ002) is waived here.
def make_padded_log_prob_output_matrix(  # noqa: CFQ002
    y: torch.Tensor,
    output_lengths: torch.Tensor,
    state_lengths: torch.Tensor,
    output_mean: torch.Tensor,
    output_var: torch.Tensor,
    time_wise_normalize: bool = False,
    matching_func_type: MatchingFuncType = MatchingFuncType.GAUSSIAN,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute padded [B, T+2, K+2] log-prob matrix for forward-sum."""

    device = y.device
    state_lengths = state_lengths.to(device)
    output_lengths = output_lengths.to(device)

    if matching_func_type == MatchingFuncType.GAUSSIAN:
        ll = calc_gaussian_log_likelihood_matrix(
            y, output_mean, output_var
        )  # [B, T, K]
    elif matching_func_type == MatchingFuncType.INNER_PRODUCT:
        ll = calc_inner_product_matrix(y, output_mean)  # [B, T, K]
    else:
        raise NotImplementedError()

    if bias is not None:
        ll = ll + bias

    # Normalization is a log-probability concern, so it lives here rather
    # than in the shape-only helpers below. The order is load-bearing:
    # masking first restricts the softmax to valid states, and padding
    # afterwards keeps out-of-length frames at NEGATIVE_INF (a softmax over
    # an all-masked frame would turn it into a uniform distribution).
    ll = mask_invalid_log_probs(ll, output_lengths, state_lengths)

    if time_wise_normalize:
        ll = torch.nn.functional.log_softmax(ll, dim=-1)

    return make_padded_BTK_matrix(ll, output_lengths, state_lengths)


def lengths_to_matrix_mask(
    output_lengths: torch.Tensor,
    state_lengths: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Build a [B, T, K] bool mask of each element's valid region.

    True at (b, t, k) iff t < output_lengths[b] and
    k < state_lengths[b], with T and K taken as the batch maxima.
    """
    B = len(output_lengths)
    K = max(state_lengths)
    T = max(output_lengths)

    state_lengths = state_lengths.to(device)
    output_lengths = output_lengths.to(device)

    state_indices = torch.tile(
        torch.arange(K, device=device), (B, 1)
    )  # [B, K]
    mask_state = state_indices < state_lengths.unsqueeze(
        -1
    )  # ([B, K], [B, 1]) -> [B, K]

    output_indices = torch.tile(
        torch.arange(T, device=device), (B, 1)
    )  # [B, T]
    mask_output = output_indices < output_lengths.unsqueeze(
        -1
    )  # ([B, T], [B, 1]) -> [B, T]

    # [B, T, 1], [B, 1, K] -> [B, T, K]
    return mask_output.unsqueeze(-1) & mask_state.unsqueeze(-2)


def mask_invalid_log_probs(
    base_matrix: torch.Tensor,
    output_lengths: torch.Tensor,
    state_lengths: torch.Tensor,
) -> torch.Tensor:
    """Set positions beyond each element's lengths to NEGATIVE_INF.

    Returns a new [B, T, K] matrix where entries at t >=
    output_lengths[b] or k >= state_lengths[b] are replaced with
    NEGATIVE_INF; the input matrix is left unmodified.
    """
    mask = lengths_to_matrix_mask(
        output_lengths, state_lengths, base_matrix.device
    )

    assert base_matrix.size() == mask.size()

    return base_matrix.masked_fill(~mask, NEGATIVE_INF)


def make_padded_BTK_matrix(
    base_matrix: torch.Tensor,
    output_lengths: torch.Tensor,
    state_lengths: torch.Tensor,
) -> torch.Tensor:
    """Mask invalid positions and pad [B,T,K] to [B,T+2,K+2].

    Pure shape preparation for the forward-sum DP: out-of-length
    entries become NEGATIVE_INF and a one-cell NEGATIVE_INF border is
    added on every side.
    """
    base_matrix = mask_invalid_log_probs(
        base_matrix, output_lengths, state_lengths
    )

    return torch.nn.functional.pad(
        base_matrix, (1, 1, 1, 1), "constant", NEGATIVE_INF
    )


def make_forward_begin_mask(
    B: int, T: int, K: int, device: torch.device
) -> torch.Tensor:
    """Create a mask marking position (0,0) for the forward pass."""
    mask = torch.zeros((B, T, K), dtype=torch.bool, device=device)
    mask[:, 0, 0] = True
    return mask


def make_backward_begin_mask(
    B: int,
    T: int,
    K: int,
    begin_time_list: torch.Tensor,
    begin_state_list: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Mark the final position per batch element for the backward pass."""
    mask = torch.zeros((B, T, K), dtype=torch.bool, device=device)
    batch_indices = torch.arange(B, device=device)
    mask[
        batch_indices,
        begin_time_list.to(device),
        begin_state_list.to(device),
    ] = True
    return mask


def gather_terminal_log_likelihoods(
    alpha: torch.Tensor,
    output_lengths: torch.Tensor,
    state_lengths: torch.Tensor,
) -> torch.Tensor:
    """Gather alpha[b, output_lengths[b], state_lengths[b]] over the batch.

    Input:
        alpha: (B, T, K)
        output_lengths: (B,)
        state_lengths: (B,)
    Output:
        log likelihoods: (B,)
    """
    device = alpha.device
    batch_indices = torch.arange(alpha.size(0), device=device)
    return alpha[
        batch_indices, output_lengths.to(device), state_lengths.to(device)
    ]


def forwardsum_gamma_from_alphabeta(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    log_prob_output_matrix: torch.Tensor,
) -> torch.Tensor:
    """Compute posterior gamma from forward (alpha) and backward (beta).

    See forwardsum_gamma_torch for the definition of gamma; this
    helper performs the derivation.
    rho[t, k] = exp(alpha[t, k]) * exp(beta[t, k]) / exp(b_k(o_t)) is
    the probability of the paths that start at state 1 at time 1, pass
    through state k at time t, and reach state K at time T. Both alpha
    and beta contain the emission term b_k(o_t)
    (= log_prob_output_matrix[t, k]), so it is divided out once.
    Normalizing by the total likelihood sum_k rho[t, k]
    (= exp(beta[1, 1])) gives gamma[t, k] = rho[t, k] / likelihood,
    evaluated here in the log domain in a single exp. Entries whose
    emission score is masked (below NEGATIVE_INF_THRESHOLD) are forced
    to zero.
    """
    log_likelihood = beta[:, 1, 1]
    gamma = torch.exp(
        alpha + beta - log_prob_output_matrix - log_likelihood[:, None, None]
    )
    gamma = torch.where(
        log_prob_output_matrix < NEGATIVE_INF_THRESHOLD,
        torch.zeros_like(gamma),
        gamma,
    )
    return gamma

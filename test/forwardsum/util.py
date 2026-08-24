import torch


def make_mask(B: int, T: int, K: int, state_list: list[int]) -> torch.Tensor:
    """Return a [B, T, K] boolean mask that is True along one path.

    Marks (t, k) for the first batch element only, with t starting at
    1 to account for the one-cell border padding of the log-prob
    output matrix.

    Args:
        B: Batch size of the mask.
        T: Padded number of frames.
        K: Padded number of states.
        state_list: State index visited at each frame, in frame order.

    Returns:
        The boolean mask selecting the path entries.
    """
    mask = torch.zeros((B, T, K), dtype=torch.bool)
    for t, k in enumerate(state_list, start=1):
        mask[0, t, k] = True
    return mask


def calc_one_pass_forwardsum_likelihood(
    state_pass: list[int], log_prob_output_matrix: torch.Tensor
) -> torch.Tensor:
    """Return the log-likelihood of a single alignment path.

    Sums the log-prob matrix entries the path visits. Serves as the
    brute-force ground truth in the DP tests: logsumexp over all paths
    gives the forward-sum likelihood, max gives the Viterbi score.

    Args:
        state_pass: State index visited at each frame, in frame order.
        log_prob_output_matrix: Padded [B, T, K] log-prob matrix; only
            the first batch element is used.

    Returns:
        The scalar log-likelihood of the path.
    """
    B = log_prob_output_matrix.size(0)
    T = log_prob_output_matrix.size(1)
    K = log_prob_output_matrix.size(2)

    output_mask = make_mask(B, T, K, state_pass)
    log_output_prob = log_prob_output_matrix[output_mask].sum()

    return log_output_prob

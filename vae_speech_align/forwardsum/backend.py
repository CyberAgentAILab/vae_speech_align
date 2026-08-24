from dataclasses import dataclass
from typing import Callable

import torch

from vae_speech_align.config import AlignmentImplementation
from vae_speech_align.forwardsum.numba_impl import (
    forwardsum_gamma_numba,
    forwardsum_likelihood_numba,
    forwardsum_viterbi_numba,
)


@dataclass(frozen=True)
class ForwardSumBackend:
    """Forward-sum DP entry points of one implementation.

    All three callables share the positional signature
    (log_prob_output_matrix, output_lengths, state_lengths); gamma
    additionally accepts return_likelihood=True to also return the
    per-element log-likelihoods.
    """

    likelihood: Callable[..., torch.Tensor]
    gamma: Callable[..., torch.Tensor | tuple[torch.Tensor, torch.Tensor]]
    viterbi: Callable[..., tuple[torch.Tensor, torch.Tensor]]


def create_forwardsum_backend(
    impl: AlignmentImplementation,
) -> ForwardSumBackend:
    """Return the DP entry points for the given implementation.

    The triton modules are imported lazily so that triton stays an
    optional dependency.
    """
    if impl is AlignmentImplementation.TRITON:
        from vae_speech_align.forwardsum import (
            triton_likelihood_impl,
            triton_viterbi_impl,
        )

        return ForwardSumBackend(
            likelihood=triton_likelihood_impl.forwardsum_likelihood_triton,
            gamma=triton_likelihood_impl.forwardsum_gamma_triton,
            viterbi=triton_viterbi_impl.forwardsum_viterbi_triton,
        )

    return ForwardSumBackend(
        likelihood=forwardsum_likelihood_numba,
        gamma=forwardsum_gamma_numba,
        viterbi=forwardsum_viterbi_numba,
    )

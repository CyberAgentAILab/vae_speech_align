from importlib import import_module
from typing import Any

from vae_speech_align.forwardsum.numba_impl import (
    forwardsum_gamma_numba,
    forwardsum_likelihood_numba,
    forwardsum_viterbi_numba,
)
from vae_speech_align.forwardsum.torch_impl import (
    forwardsum_gamma_torch,
    forwardsum_likelihood_torch,
    forwardsum_viterbi_torch,
)

_TRITON_EXPORTS = {
    "forwardsum_gamma_triton": (
        "vae_speech_align.forwardsum.triton_likelihood_impl"
    ),
    "forwardsum_likelihood_triton": (
        "vae_speech_align.forwardsum.triton_likelihood_impl"
    ),
    "forwardsum_viterbi_triton": (
        "vae_speech_align.forwardsum.triton_viterbi_impl"
    ),
}


def __getattr__(name: str) -> Any:
    """Import Triton exports only when callers request them."""
    if name not in _TRITON_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    triton_module = import_module(_TRITON_EXPORTS[name])
    triton_export = getattr(triton_module, name)
    globals()[name] = triton_export
    return triton_export


__all__ = (
    "forwardsum_likelihood_torch",
    "forwardsum_viterbi_torch",
    "forwardsum_gamma_torch",
    "forwardsum_likelihood_numba",
    "forwardsum_gamma_numba",
    "forwardsum_viterbi_numba",
    "forwardsum_likelihood_triton",
    "forwardsum_viterbi_triton",
    "forwardsum_gamma_triton",
)

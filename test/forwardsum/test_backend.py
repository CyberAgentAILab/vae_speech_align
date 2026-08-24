"""Tests for vae_speech_align.forwardsum.backend."""

import pytest

from vae_speech_align.config import AlignmentImplementation
from vae_speech_align.forwardsum.backend import create_forwardsum_backend
from vae_speech_align.forwardsum.numba_impl import (
    forwardsum_gamma_numba,
    forwardsum_likelihood_numba,
    forwardsum_viterbi_numba,
)


class TestCreateForwardSumBackend:
    def test_numba_backend_wiring(self):
        """Test that the NUMBA backend bundles the numba entry points."""
        backend = create_forwardsum_backend(AlignmentImplementation.NUMBA)
        assert backend.likelihood is forwardsum_likelihood_numba
        assert backend.gamma is forwardsum_gamma_numba
        assert backend.viterbi is forwardsum_viterbi_numba

    def test_triton_backend_wiring(self):
        """Test that the TRITON backend bundles the triton entry points."""
        pytest.importorskip("triton")
        from vae_speech_align.forwardsum.triton_likelihood_impl import (
            forwardsum_gamma_triton,
            forwardsum_likelihood_triton,
        )
        from vae_speech_align.forwardsum.triton_viterbi_impl import (
            forwardsum_viterbi_triton,
        )

        backend = create_forwardsum_backend(AlignmentImplementation.TRITON)
        assert backend.likelihood is forwardsum_likelihood_triton
        assert backend.gamma is forwardsum_gamma_triton
        assert backend.viterbi is forwardsum_viterbi_triton

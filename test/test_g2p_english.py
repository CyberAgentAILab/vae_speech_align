"""Tests for vae_speech_align.g2p.english."""

import pytest

from vae_speech_align.g2p.english import (
    VCTK_PHONEMES,
    EnglishG2p,
    arpabet_to_vctk_phonemes,
)


class TestArpabetToVctkPhonemes:
    def test_strips_stress_and_lowercases(self):
        """Test that stress digits are stripped and phones lowercased."""
        result = arpabet_to_vctk_phonemes(["R", "EY1", "N"])
        assert result == ["pau", "r", "ey", "n", "pau"]

    def test_unstressed_schwa_becomes_ax(self):
        """Test that the unstressed schwa AH0 maps to "ax"."""
        result = arpabet_to_vctk_phonemes(["AH0"])
        assert result == ["pau", "ax", "pau"]

    def test_stressed_ah_stays_ah(self):
        """Test that stressed AH1 and AH2 map to "ah", not "ax"."""
        assert arpabet_to_vctk_phonemes(["AH1"])[1] == "ah"
        assert arpabet_to_vctk_phonemes(["AH2"])[1] == "ah"

    def test_word_separators_are_dropped(self):
        """Test that space word-separator tokens are dropped."""
        result = arpabet_to_vctk_phonemes(["DH", "AH0", " ", "K", "AE1", "T"])
        assert result == ["pau", "dh", "ax", "k", "ae", "t", "pau"]

    def test_punctuation_becomes_pause(self):
        """Test that a punctuation token becomes an internal "pau"."""
        result = arpabet_to_vctk_phonemes(["K", ",", "T"])
        assert result == ["pau", "k", "pau", "t", "pau"]

    def test_consecutive_pauses_are_merged(self):
        """Test that consecutive pause tokens collapse into a single
        "pau" instead of duplicating the final pause."""
        result = arpabet_to_vctk_phonemes(["T", " ", ".", "?"])
        assert result == ["pau", "t", "pau"]

    def test_empty_input_gives_single_pause(self):
        """Test that an empty token list yields a single "pau"."""
        assert arpabet_to_vctk_phonemes([]) == ["pau"]

    def test_output_is_within_vctk_set(self):
        """Test that every output phoneme belongs to VCTK_PHONEMES."""
        tokens = ["W", "AH1", "T", "S", " ", "DH", "AH0", " ", "?"]
        for phoneme in arpabet_to_vctk_phonemes(tokens):
            assert phoneme in VCTK_PHONEMES

    def test_unknown_token_raises(self):
        """Test that an unknown ARPABET token raises ValueError."""
        with pytest.raises(ValueError):
            arpabet_to_vctk_phonemes(["QQ1"])

    def test_stress_digits_all_map_identically(self):
        """Test that stress digits 0, 1, and 2 all yield the same
        phoneme for a non-schwa vowel."""
        for stress in ("0", "1", "2"):
            assert arpabet_to_vctk_phonemes([f"IY{stress}"])[1] == "iy"


@pytest.fixture(scope="module")
def g2p():
    """Return a shared EnglishG2p instance (slow to construct)."""
    return EnglishG2p()


class TestEnglishG2p:
    def test_simple_sentence(self, g2p):
        """Test that a plain sentence yields in-set phonemes wrapped
        in leading and trailing pauses."""
        result = g2p.text_to_phoneme("Turn off the lights.")
        phonemes = result.split()
        assert phonemes[0] == "pau"
        assert phonemes[-1] == "pau"
        assert all(p in VCTK_PHONEMES for p in phonemes)

    def test_question_sentence(self, g2p):
        """Test that a question yields in-set phonemes and that the
        content words produce phones beyond pauses."""
        result = g2p.text_to_phoneme("What's the probability of rain?")
        phonemes = result.split()
        assert all(p in VCTK_PHONEMES for p in phonemes)
        assert len([p for p in phonemes if p != "pau"]) > 10

    def test_comma_produces_internal_pause(self, g2p):
        """Test that a mid-sentence comma adds an internal pause."""
        result = g2p.text_to_phoneme("First, mix well.")
        phonemes = result.split()
        assert phonemes.count("pau") >= 3  # leading, comma, trailing

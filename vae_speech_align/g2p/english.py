"""English text to phonemes in the VCTK ARPABET phoneme set.

Converts English text to the lowercase ARPABET phoneme set used by the
VCTK example model (see example/train/vctk/data/all_phonemes.txt),
using g2p_en for grapheme-to-phoneme conversion.
"""

import nltk
from g2p_en import G2p

# Phoneme inventory of the VCTK example model
# (example/train/vctk/data/all_phonemes.txt).
VCTK_PHONEMES = frozenset(
    (
        "aa ae ah ao aw ax ay b ch d dh eh er ey f g hh ih iy jh k l m n "
        "ng ow oy p pau r s sh t th uh uw v w y z zh"
    ).split()
)

PAUSE_PHONEME = "pau"

# g2p_en emits raw punctuation tokens; these become pauses.
_PUNCTUATION_TO_PAUSE = frozenset({",", ";", ":", ".", "!", "?", "-", "--"})

_WORD_SEPARATOR = " "

# The VCTK label set distinguishes the reduced vowel (schwa) "ax" from
# the full vowel "ah"; g2p_en encodes that distinction only in the
# stress digit (AH0 = unstressed schwa).
_SCHWA_ARPABET = "AH0"
_SCHWA_VCTK = "ax"


def _strip_stress(arpabet_token: str) -> str:
    if arpabet_token and arpabet_token[-1].isdigit():
        return arpabet_token[:-1]
    return arpabet_token


def _append_pause(phonemes: list[str]) -> None:
    # collapse runs of pauses so silences appear once
    if phonemes and phonemes[-1] == PAUSE_PHONEME:
        return
    phonemes.append(PAUSE_PHONEME)


def arpabet_to_vctk_phonemes(arpabet_tokens: list[str]) -> list[str]:
    """Map g2p_en ARPABET output onto the VCTK phoneme set.

    Stress digits are stripped, the unstressed schwa (AH0) becomes
    "ax", punctuation becomes "pau", and the sequence is wrapped in
    leading/trailing pauses following the VCTK label convention.

    Args:
        arpabet_tokens: Tokens as returned by g2p_en.G2p() — uppercase
            ARPABET phones with stress digits, plus spaces and
            punctuation.

    Returns:
        Phonemes restricted to VCTK_PHONEMES, starting and ending with
        "pau".

    Raises:
        ValueError: If a token maps outside VCTK_PHONEMES.

    Example:
        >>> arpabet_to_vctk_phonemes(
        ...     ["L", "AY1", "T", "S", " ", ".", "AH0"]
        ... )
        ['pau', 'l', 'ay', 't', 's', 'pau', 'ax', 'pau']
    """
    phonemes = [PAUSE_PHONEME]

    for token in arpabet_tokens:
        if token == _WORD_SEPARATOR:
            continue
        if token in _PUNCTUATION_TO_PAUSE:
            _append_pause(phonemes)
            continue

        if token == _SCHWA_ARPABET:
            phoneme = _SCHWA_VCTK
        else:
            phoneme = _strip_stress(token).lower()

        if phoneme not in VCTK_PHONEMES:
            raise ValueError(
                f"Token {token!r} maps to {phoneme!r}, which is not in "
                f"the VCTK phoneme set"
            )
        phonemes.append(phoneme)

    _append_pause(phonemes)
    return phonemes


def _ensure_nltk_tagger() -> None:
    # g2p_en only fetches the legacy "averaged_perceptron_tagger"
    # resource, but nltk>=3.9 looks up the "_eng"-suffixed name, so
    # fetch it here to keep first use from failing.
    try:
        nltk.data.find("taggers/averaged_perceptron_tagger_eng")
    except LookupError:
        nltk.download("averaged_perceptron_tagger_eng", quiet=True)


class EnglishG2p:
    """Convert English text to VCTK phoneme strings via g2p_en."""

    def __init__(self) -> None:
        _ensure_nltk_tagger()
        self._g2p = G2p()

    def text_to_phoneme(self, text: str) -> str:
        """Convert an English sentence to a VCTK phoneme string.

        Args:
            text: English text (one sentence or utterance).

        Returns:
            Space-separated phonemes wrapped in "pau", e.g.
            "pau t er n ao f dh ax l ay t s pau".
        """
        arpabet_tokens = self._g2p(text)
        phonemes = arpabet_to_vctk_phonemes(arpabet_tokens)
        return " ".join(phonemes)

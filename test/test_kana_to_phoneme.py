"""Tests for vae_speech_align.g2p.kana."""

from vae_speech_align.g2p.kana import (
    alphabet_to_kana_table,
    kana_text_to_phoneme,
    kana_to_phoneme_table,
)


class TestKanaTextToPhoneme:
    def test_simple_vowels(self):
        """Test that vowel kana map one-to-one to vowel phonemes."""
        assert kana_text_to_phoneme("アイウエオ") == "sil a i u e o sil"

    def test_consonant_vowel_pairs(self):
        """Test that CV kana expand to consonant-vowel phoneme pairs."""
        assert kana_text_to_phoneme("カサタ") == "sil k a s a t a sil"

    def test_digraph_takes_precedence_over_single_kana(self):
        """Test that "キャ" is looked up as one unit (ky a), not as
        "キ" (kj i) followed by "ャ" (y a)."""
        assert kana_text_to_phoneme("キャ") == "sil ky a sil"

    def test_digraph_followed_by_single_kana(self):
        """Test that a digraph match does not consume the next kana."""
        assert kana_text_to_phoneme("シャシ") == "sil sy a sj i sil"

    def test_special_morae(self):
        """Test that the moraic nasal, geminate, and long-vowel marks
        map to N, Q, and H."""
        assert kana_text_to_phoneme("ンッー") == "sil N Q H sil"

    def test_long_vowel_word(self):
        """Test that long-vowel marks yield "H" after each vowel."""
        assert kana_text_to_phoneme("コーヒー") == "sil k o H hj i H sil"

    def test_pause_symbol(self):
        """Test that the ideographic comma maps to a "pau" phoneme."""
        assert kana_text_to_phoneme("ア、イ") == "sil a pau i sil"

    def test_fullwidth_alphabet_converted_to_kana(self):
        """Test that a fullwidth letter is read via its kana form
        ("Ａ" -> "エー" -> "e H")."""
        assert kana_text_to_phoneme("Ａ") == "sil e H sil"

    def test_fullwidth_lowercase_alphabet(self):
        """Test that a fullwidth lowercase letter is converted the
        same way as its uppercase form."""
        assert kana_text_to_phoneme("ａ") == kana_text_to_phoneme("Ａ")

    def test_unknown_characters_are_skipped(self):
        """Test that characters not in the table (kanji, ASCII) are
        silently ignored."""
        assert kana_text_to_phoneme("ア亜xイ") == "sil a i sil"

    def test_empty_string(self):
        """Test that an empty string yields only the "sil sil" pair."""
        assert kana_text_to_phoneme("") == "sil sil"

    def test_none_returns_empty_string(self):
        """Test that None input returns an empty string."""
        assert kana_text_to_phoneme(None) == ""

    def test_small_vowels_standalone(self):
        """Test that a small vowel joins a digraph when possible and
        otherwise maps to its plain vowel."""
        assert kana_text_to_phoneme("ファ") == "sil F a sil"
        assert kana_text_to_phoneme("ァ") == "sil a sil"

    def test_voiced_and_semivoiced(self):
        """Test that voiced, semivoiced, and "ヴ" kana map to b, p, v."""
        assert kana_text_to_phoneme("バパヴ") == "sil b a p a v u sil"

    def test_palatalized_digraphs_keep_their_vowel(self):
        """Test that each palatalized digraph retains its own vowel
        instead of collapsing onto "u"."""
        assert kana_text_to_phoneme("テャテュテョ") == "sil ty a ty u ty o sil"
        assert kana_text_to_phoneme("デャデュデョ") == "sil dy a dy u dy o sil"


class TestTables:
    def test_all_table_entries_produce_phonemes(self):
        """Test that every kana table entry round-trips through the
        converter to its own phonemes."""
        for kana, phoneme in kana_to_phoneme_table.items():
            result = kana_text_to_phoneme(kana)
            assert result == f"sil {phoneme} sil", f"mismatch for {kana}"

    def test_alphabet_table_values_are_convertible(self):
        """Test that every alphabet reading converts to a non-empty
        phoneme sequence."""
        for alpha, kana in alphabet_to_kana_table.items():
            result = kana_text_to_phoneme(kana)
            phonemes = result.split()
            assert phonemes[0] == "sil" and phonemes[-1] == "sil"
            assert len(phonemes) > 2, f"no phonemes for {alpha} ({kana})"

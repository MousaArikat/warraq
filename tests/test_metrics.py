"""
Unit tests for arabic_normalize and metrics.

Golden values are computed by hand — per §6.6 rule 8, "test the tests."
Each golden value has a comment showing the calculation.
"""

import pytest

from evals.arabic_normalize import normalize_arabic
from evals.metrics import compute_cer, compute_wer


# ── normalize_arabic: strip_tashkeel ─────────────────────────────────────────

class TestStripTashkeel:
    def test_removes_fatha(self):
        # كَتَبَ  →  كتب  (three fatha marks removed)
        assert normalize_arabic("كَتَبَ") == "كتب"

    def test_removes_shadda(self):
        # مُحَمَّد  →  محمد
        assert normalize_arabic("مُحَمَّد") == "محمد"

    def test_disabled_preserves_diacritics(self):
        assert normalize_arabic("كَتَبَ", strip_tashkeel=False) == "كَتَبَ"


# ── normalize_arabic: normalize_alef ─────────────────────────────────────────

class TestNormalizeAlef:
    def test_alef_hamza_above(self):
        # أ (U+0623) → ا (U+0627)
        assert normalize_arabic("أحمد", strip_tashkeel=False) == "احمد"

    def test_alef_hamza_below(self):
        # إ (U+0625) → ا
        assert normalize_arabic("إبراهيم", strip_tashkeel=False) == "ابراهيم"

    def test_alef_madda(self):
        # آ (U+0622) → ا
        assert normalize_arabic("آمن", strip_tashkeel=False) == "امن"

    def test_alef_wasla(self):
        # ٱ (U+0671) → ا
        assert normalize_arabic("ٱلله", strip_tashkeel=False) == "الله"

    def test_disabled(self):
        assert normalize_arabic("أإآ", normalize_alef=False, strip_tashkeel=False) == "أإآ"


# ── normalize_arabic: ta marbuta ─────────────────────────────────────────────

class TestTaMarbuta:
    def test_off_by_default(self):
        assert normalize_arabic("مدرسة", strip_tashkeel=False) == "مدرسة"

    def test_maps_when_enabled(self):
        # ة (U+0629) → ه (U+0647)
        assert normalize_arabic(
            "مدرسة", normalize_ta_marbuta=True, strip_tashkeel=False
        ) == "مدرسه"


# ── normalize_arabic: alef maqsura ───────────────────────────────────────────

class TestAlefMaqsura:
    def test_off_by_default(self):
        assert normalize_arabic("على", strip_tashkeel=False) == "على"

    def test_maps_when_enabled(self):
        # ى (U+0649) → ي (U+064A)
        assert normalize_arabic(
            "على", normalize_alef_maqsura=True, strip_tashkeel=False
        ) == "علي"


# ── normalize_arabic: tatweel ────────────────────────────────────────────────

class TestTatweel:
    def test_removes_tatweel(self):
        # ـ (U+0640) is a purely visual stretching character
        assert normalize_arabic("مـرحـبـا") == "مرحبا"

    def test_disabled(self):
        assert normalize_arabic("مـرحـبـا", strip_tatweel=False) == "مـرحـبـا"


# ── normalize_arabic: whitespace ─────────────────────────────────────────────

class TestNormalizeWhitespace:
    def test_off_by_default(self):
        # Default must NOT touch whitespace — markdown ground truth has meaningful newlines
        assert normalize_arabic("a  b\tc\nd", strip_tashkeel=False) == "a  b\tc\nd"

    def test_collapses_multiple_spaces(self):
        # "a  b" (2 spaces) → "a b" (1 space)
        assert normalize_arabic("a  b", strip_tashkeel=False, normalize_whitespace=True) == "a b"

    def test_collapses_tabs(self):
        # tab → single space
        assert normalize_arabic("a\tb", strip_tashkeel=False, normalize_whitespace=True) == "a b"

    def test_collapses_newlines(self):
        # newline → single space
        assert normalize_arabic("a\nb", strip_tashkeel=False, normalize_whitespace=True) == "a b"

    def test_collapses_mixed_whitespace(self):
        # " \t\n " (space + tab + newline + space) all collapse to one space
        assert normalize_arabic("a \t\n b", strip_tashkeel=False, normalize_whitespace=True) == "a b"

    def test_strips_leading_trailing(self):
        # Leading/trailing whitespace removed
        assert normalize_arabic("  مرحبا  ", strip_tashkeel=False, normalize_whitespace=True) == "مرحبا"

    def test_arabic_text_with_tabs(self):
        # Real KITAB ground truth often has tab-separated columns — collapses cleanly
        # "كتب\t\tمقالة" → "كتب مقالة"  (2 tabs → 1 space)
        assert normalize_arabic("كتب\t\tمقالة", strip_tashkeel=False, normalize_whitespace=True) == "كتب مقالة"


# ── normalize_arabic: digits ─────────────────────────────────────────────────

class TestDigits:
    def test_arabic_indic_to_western_default(self):
        # ٢٠٢٤ → 2024
        assert normalize_arabic("٢٠٢٤", strip_tashkeel=False) == "2024"

    def test_western_to_arabic_indic(self):
        assert normalize_arabic(
            "2024", strip_tashkeel=False, digits="arabic_indic"
        ) == "٢٠٢٤"

    def test_none_leaves_unchanged(self):
        assert normalize_arabic("٢٠٢٤", strip_tashkeel=False, digits=None) == "٢٠٢٤"

    def test_mixed_digits(self):
        # ١2٣ → 123  (mixed Arabic-Indic and Western)
        assert normalize_arabic("١2٣", strip_tashkeel=False) == "123"


# ── compute_cer ───────────────────────────────────────────────────────────────

class TestCER:
    def test_perfect_match(self):
        assert compute_cer("كتب", "كتب") == pytest.approx(0.0)

    def test_one_substitution(self):
        # ref "كتب" (3 chars), hyp "كتم" → 1 substitution → CER = 1/3 ≈ 0.333
        assert compute_cer("كتب", "كتم") == pytest.approx(1 / 3)

    def test_complete_miss(self):
        # ref "اب" (2 chars), hyp "جد" → 2 substitutions → CER = 1.0
        assert compute_cer("اب", "جد") == pytest.approx(1.0)

    def test_normalization_erases_alef_difference(self):
        # أحمد vs احمد differ only in alef hamza → after normalize, identical → CER = 0
        assert compute_cer("أحمد", "احمد") == pytest.approx(0.0)

    def test_no_normalization_sees_alef_difference(self):
        # Without normalization أ ≠ ا → 1 substitution in 4 chars → CER = 0.25
        assert compute_cer("أحمد", "احمد", normalize=False) == pytest.approx(0.25)

    def test_tashkeel_irrelevant_after_normalization(self):
        # كَتَبَ vs كتب → tashkeel stripped → identical
        assert compute_cer("كَتَبَ", "كتب") == pytest.approx(0.0)

    def test_digit_normalization_in_cer(self):
        # ٢٠٢٤ vs 2024 → after normalization both become "2024"
        assert compute_cer("عام ٢٠٢٤", "عام 2024") == pytest.approx(0.0)


# ── compute_wer ───────────────────────────────────────────────────────────────

class TestWER:
    def test_perfect_match(self):
        assert compute_wer("مرحبا بالعالم", "مرحبا بالعالم") == pytest.approx(0.0)

    def test_one_word_substitution(self):
        # 2 words, 1 substitution → WER = 1/2 = 0.5
        assert compute_wer("مرحبا بالعالم", "مرحبا بالعرب") == pytest.approx(0.5)

    def test_all_wrong(self):
        # 2 words, 2 substitutions → WER = 1.0
        assert compute_wer("مرحبا بالعالم", "صباح الخير") == pytest.approx(1.0)

    def test_normalization_erases_tashkeel(self):
        # مَرْحَبًا vs مرحبا → after normalization both "مرحبا" → WER = 0
        assert compute_wer("مَرْحَبًا", "مرحبا") == pytest.approx(0.0)

    def test_digit_normalization_in_wer(self):
        # عام ٢٠٢٤ vs عام 2024 → digits normalized → WER = 0
        assert compute_wer("عام ٢٠٢٤", "عام 2024") == pytest.approx(0.0)

    def test_no_normalization_digit_mismatch(self):
        # ٢٠٢٤ ≠ 2024 as tokens when not normalized → WER = 1/2 on "عام ٢٠٢٤"
        assert compute_wer("عام ٢٠٢٤", "عام 2024", normalize=False) == pytest.approx(0.5)

"""Arabic text normalization utilities for the eval harness.

All flags default to the settings that produce the fairest OCR comparison:
strip diacritics and tatweel (pure noise for OCR evaluation), normalize alef
variants (different keyboards/fonts produce different codepoints for the same
letter), and convert Arabic-Indic digits to Western (models mix them freely).

ta_marbuta and alef_maqsura normalization are OFF by default because they
conflate genuinely distinct letters and can inflate CER/WER; turn them on only
when comparing systems known to disagree only on those characters.
"""

import re

# ── Unicode constants ──────────────────────────────────────────────────────────

# Tashkeel: harakat + shadda + sukun + quranic annotation marks
_TASHKEEL_RE = re.compile(
    r"[ؐ-ؚ"   # Arabic extended A (sign above/below)
    r"ً-ٟ"    # Fathatan … Hamza below (the main harakat block)
    r"ٰ"           # Superscript alef (alef khanjariyya)
    r"ۖ-ۜ"    # Quranic annotation signs (small high letters)
    r"۟-ۤ"    # More quranic signs
    r"ۧۨ"     # Small high yeh / noon
    r"۪-ۭ]"   # Quranic marks at end of block
)

_TATWEEL = "ـ"  # ـ  kashida / tatweel

# Alef variants → bare alef (U+0627)
_ALEF_VARIANTS = str.maketrans(
    {
        "أ": "ا",  # أ alef with hamza above
        "إ": "ا",  # إ alef with hamza below
        "آ": "ا",  # آ alef with madda above
        "ٱ": "ا",  # ٱ alef wasla
    }
)

_TA_MARBUTA = str.maketrans({"ة": "ه"})   # ة → ه
_ALEF_MAQSURA = str.maketrans({"ى": "ي"}) # ى → ي

# Digit tables
_ARABIC_INDIC_TO_WESTERN = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_WESTERN_TO_ARABIC_INDIC = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")


# ── Public API ─────────────────────────────────────────────────────────────────

def normalize_arabic(
    text: str,
    *,
    strip_tashkeel: bool = True,
    normalize_alef: bool = True,
    normalize_ta_marbuta: bool = False,
    normalize_alef_maqsura: bool = False,
    strip_tatweel: bool = True,
    digits: str | None = "western",
) -> str:
    """Normalize Arabic text for fairer OCR metric computation.

    Parameters
    ----------
    text:
        Input Arabic text.
    strip_tashkeel:
        Remove harakat/tashkeel (diacritics, shadda, sukun, quranic marks).
    normalize_alef:
        Map أ إ آ ٱ → ا (different encodings of the same letter).
    normalize_ta_marbuta:
        Map ة → ه. Off by default; changes apparent word meaning.
    normalize_alef_maqsura:
        Map ى → ي. Off by default; changes apparent word meaning.
    strip_tatweel:
        Remove tatweel/kashida (U+0640), used for visual stretching only.
    digits:
        ``"western"``       — convert Arabic-Indic digits (٠-٩) to Western (0-9).
        ``"arabic_indic"``  — convert Western digits (0-9) to Arabic-Indic (٠-٩).
        ``None``            — leave digits unchanged.

    Returns
    -------
    str
        Normalized text.
    """
    if strip_tashkeel:
        text = _TASHKEEL_RE.sub("", text)
    if strip_tatweel:
        text = text.replace(_TATWEEL, "")
    if normalize_alef:
        text = text.translate(_ALEF_VARIANTS)
    if normalize_ta_marbuta:
        text = text.translate(_TA_MARBUTA)
    if normalize_alef_maqsura:
        text = text.translate(_ALEF_MAQSURA)
    if digits == "western":
        text = text.translate(_ARABIC_INDIC_TO_WESTERN)
    elif digits == "arabic_indic":
        text = text.translate(_WESTERN_TO_ARABIC_INDIC)

    return text

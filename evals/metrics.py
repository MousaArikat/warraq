"""CER and WER metrics with an Arabic normalization pre-pass.

Both functions mirror jiwer's interface but apply normalize_arabic() first so
that diacritic variation, alef encoding differences, tatweel, and digit
encoding don't inflate error rates artificially.

Pass normalize=False to get the raw jiwer value, or normalize_kwargs to
override any normalization flag (see arabic_normalize.normalize_arabic).
"""

from jiwer import cer as _jiwer_cer
from jiwer import wer as _jiwer_wer

from evals.arabic_normalize import normalize_arabic


def compute_cer(
    reference: str,
    hypothesis: str,
    *,
    normalize: bool = True,
    normalize_kwargs: dict | None = None,
) -> float:
    """Character Error Rate (CER) between reference and hypothesis.

    Parameters
    ----------
    reference:
        Ground-truth text.
    hypothesis:
        OCR or model output to score.
    normalize:
        Apply Arabic normalization before scoring. Default True.
    normalize_kwargs:
        Keyword overrides forwarded to ``normalize_arabic()``.
        E.g. ``{"strip_tashkeel": False}`` to score diacritics strictly.

    Returns
    -------
    float
        CER in [0, ∞). jiwer does not clamp at 1.0 — insertions can push it
        above 1 when hypothesis is much longer than reference.
    """
    if normalize:
        kw = normalize_kwargs or {}
        reference = normalize_arabic(reference, **kw)
        hypothesis = normalize_arabic(hypothesis, **kw)
    return float(_jiwer_cer(reference, hypothesis))


def compute_wer(
    reference: str,
    hypothesis: str,
    *,
    normalize: bool = True,
    normalize_kwargs: dict | None = None,
) -> float:
    """Word Error Rate (WER) between reference and hypothesis.

    Parameters
    ----------
    reference:
        Ground-truth text.
    hypothesis:
        OCR or model output to score.
    normalize:
        Apply Arabic normalization before scoring. Default True.
    normalize_kwargs:
        Keyword overrides forwarded to ``normalize_arabic()``.

    Returns
    -------
    float
        WER in [0, ∞).
    """
    if normalize:
        kw = normalize_kwargs or {}
        reference = normalize_arabic(reference, **kw)
        hypothesis = normalize_arabic(hypothesis, **kw)
    return float(_jiwer_wer(reference, hypothesis))

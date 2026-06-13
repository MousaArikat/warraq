"""Arabic document eval dataset loader.

KITAB-Bench datasets
--------------------
Published under HuggingFace user ahmedheakl as 13 separate arocrbench_* datasets
(one per domain). Source: github.com/mbzuai-oryx/KITAB-Bench
All use the "train" split (no separate test split — the whole dataset is the bench).

Ground-truth column is "answer" for the first three datasets, "text" for the rest
(confirmed from KITAB eval.py source).

Normalization preset
--------------------
KITAB_NORMALIZE_KWARGS matches the paper's preprocess_arabic_text() exactly:
  - strip tashkeel, alef, ta_marbuta, alef_maqsura, tatweel, whitespace all ON
  - digits → western
Use this dict when computing CER/WER for paper-comparable numbers:
    compute_cer(ref, hyp, normalize_kwargs=KITAB_NORMALIZE_KWARGS)

Non-benchmark suite
-------------------
"misraj-ocr" (Misraj/Misraj-DocOCR) is kept as a temporary smoke-test suite only.
It is NOT a published benchmark — use only to confirm the pipeline works when
KITAB data is unavailable or during quick iteration. Do not report its numbers.

Schema probe (run if columns change):
    python -c "
    from datasets import load_dataset
    ds = load_dataset('ahmedheakl/arocrbench_hindawi', split='train', streaming=True)
    print(list(next(iter(ds)).keys()))
    "
"""
from __future__ import annotations

import io
from typing import Literal

from datasets import load_dataset
from PIL import Image
from pydantic import BaseModel, ConfigDict


TaskType = Literal["ocr", "pdf_to_md", "table", "chart", "unknown"]

# Normalization kwargs that match the KITAB-Bench paper's preprocess_arabic_text().
# Pass to compute_cer / compute_wer for paper-comparable numbers.
# Differences from our defaults: ta_marbuta ON, alef_maqsura ON, whitespace ON.
# Library note: paper uses torchmetrics; we use jiwer. Both are Levenshtein-based
# and produce equivalent scores.
KITAB_NORMALIZE_KWARGS: dict = {
    "strip_tashkeel":       True,
    "normalize_alef":       True,
    "normalize_ta_marbuta": True,   # ON — paper applies this
    "normalize_alef_maqsura": True, # ON — paper applies this
    "strip_tatweel":        True,
    "digits":               "western",
    "normalize_whitespace": True,   # ON — paper collapses whitespace
}

# ── Suite registry ─────────────────────────────────────────────────────────────
# (hf_dataset_name, hf_config_or_None, split, gt_column, task_type)

_SuiteEntry = tuple[str, str | None, str, str, TaskType]

_SUITE_MAP: dict[str, _SuiteEntry] = {
    # ── KITAB-Bench printed / modern text ──
    "kitab-hindawi":    ("ahmedheakl/arocrbench_hindawi",        None, "train", "text",   "ocr"),
    "kitab-adab":       ("ahmedheakl/arocrbench_adab",           None, "train", "text",   "ocr"),
    "kitab-historyar":  ("ahmedheakl/arocrbench_historyar",      None, "train", "text",   "ocr"),
    "kitab-arabicocr":  ("ahmedheakl/arocrbench_arabicocr",      None, "train", "text",   "ocr"),
    "kitab-evarest":    ("ahmedheakl/arocrbench_evarest",        None, "train", "text",   "ocr"),
    "kitab-isippt":     ("ahmedheakl/arocrbench_isippt",         None, "train", "text",   "ocr"),
    "kitab-synthesize": ("ahmedheakl/arocrbench_synthesizear",   None, "train", "text",   "ocr"),

    # ── KITAB-Bench datasets using "answer" column (first 3 in paper's list) ──
    "kitab-patsocr":    ("ahmedheakl/arocrbench_patsocr",        None, "train", "answer", "ocr"),
    "kitab-historical": ("ahmedheakl/arocrbench_historicalbooks", None, "train", "answer", "ocr"),
    "kitab-khattpar":   ("ahmedheakl/arocrbench_khattparagraph", None, "train", "answer", "ocr"),

    # ── KITAB-Bench handwriting ("hard slice" per DESIGN.md §6.1) ──
    "kitab-khatt":      ("ahmedheakl/arocrbench_khatt",          None, "train", "text",   "ocr"),
    "kitab-muharaf":    ("ahmedheakl/arocrbench_muharaf",        None, "train", "text",   "ocr"),
    "kitab-onlinekhatt":("ahmedheakl/arocrbench_onlinekhatt",    None, "train", "text",   "ocr"),

    # ── Temporary smoke-test suite — NOT a benchmark, do not report numbers ──
    "misraj-ocr": ("Misraj/Misraj-DocOCR", None, "train", "markdown", "pdf_to_md"),
}


class EvalSample(BaseModel):
    """One evaluation sample."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    image: Image.Image
    ground_truth: str
    task_type: TaskType
    suite: str


def load_kitab(suite: str = "kitab-hindawi", limit: int | None = None) -> list[EvalSample]:
    """Load eval samples for a given suite.

    Parameters
    ----------
    suite:
        Dataset key. See module docstring for the full list.
        Default is ``"kitab-hindawi"`` — printed Arabic books, easiest to start with.
    limit:
        Cap on returned samples. Use 5–10 for cheap smoke tests.

    Returns
    -------
    list[EvalSample]
    """
    if suite not in _SUITE_MAP:
        raise ValueError(
            f"Unknown suite {suite!r}.\n"
            f"Available: {', '.join(sorted(_SUITE_MAP))}"
        )

    hf_name, hf_config, hf_split, gt_col, task_type = _SUITE_MAP[suite]

    # streaming=True: only fetches rows we iterate — critical for --limit 5.
    ds = load_dataset(hf_name, hf_config, split=hf_split, streaming=True)

    samples: list[EvalSample] = []
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break

        samples.append(
            EvalSample(
                id=str(_pick(row, ["uuid", "id", "sample_id", "idx"], i)),
                image=_extract_image(row, i),
                ground_truth=str(row.get(gt_col, "")),
                task_type=task_type,
                suite=suite,
            )
        )

    if not samples:
        raise RuntimeError(
            f"No samples loaded from {hf_name!r} (split={hf_split!r}). "
            "Check your network connection."
        )

    return samples


# ── helpers ────────────────────────────────────────────────────────────────────

def _pick(row: dict, candidates: list[str], default=None):
    for key in candidates:
        if key in row:
            return row[key]
    return default


def _extract_image(row: dict, idx: int) -> Image.Image:
    for col in ("image", "img", "page_image"):
        val = row.get(col)
        if val is None:
            continue
        if isinstance(val, Image.Image):
            return val
        if isinstance(val, bytes):
            return Image.open(io.BytesIO(val))
        if isinstance(val, dict) and val.get("bytes"):
            return Image.open(io.BytesIO(val["bytes"]))

    raise KeyError(
        f"Sample {idx}: no image column found in {list(row.keys())}. "
        "Run the schema probe in the module docstring."
    )

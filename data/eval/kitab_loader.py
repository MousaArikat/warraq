"""KITAB-Bench dataset loader.

HuggingFace dataset: mbzuai-oryx/KITAB-Bench
Loads the OCR / page-recognition subset and wraps each row in an EvalSample.

If column names ever change upstream, run this one-liner to inspect the schema:
    python -c "
    from datasets import load_dataset
    ds = load_dataset('mbzuai-oryx/KITAB-Bench', split='test', streaming=True, trust_remote_code=True)
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

# Candidate column names — tried in order, first hit wins.
_IMAGE_COLS = ["image", "img", "page_image"]
_GT_COLS    = ["ground_truth", "text", "answer", "transcription", "ocr_text"]
_TASK_COLS  = ["task_type", "task", "category", "type"]
_ID_COLS    = ["id", "sample_id", "idx", "image_id", "file_name"]

# Maps our suite name → (HF dataset name, HF config name or None)
_SUITE_MAP: dict[str, tuple[str, str | None]] = {
    "kitab-ocr": ("mbzuai-oryx/KITAB-Bench", None),
}


class EvalSample(BaseModel):
    """One evaluation sample from KITAB-Bench."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    image: Image.Image
    ground_truth: str
    task_type: TaskType


def load_kitab(suite: str = "kitab-ocr", limit: int | None = None) -> list[EvalSample]:
    """Load KITAB-Bench samples for a given suite.

    Always loads the published test split. Never pass a training split here —
    eval contamination risk (the model may have seen those examples).

    Parameters
    ----------
    suite:
        Which sub-task to load. Supported: ``"kitab-ocr"``.
    limit:
        Cap on returned samples. Use 5–10 for cheap smoke tests.

    Returns
    -------
    list[EvalSample]
        Samples ready for the harness.
    """
    if suite not in _SUITE_MAP:
        raise ValueError(f"Unknown suite {suite!r}. Available: {list(_SUITE_MAP)}")

    hf_name, hf_config = _SUITE_MAP[suite]

    # streaming=True means we never download more rows than we actually iterate,
    # which is important when limit is small.
    ds = load_dataset(
        hf_name,
        hf_config,
        split="test",
        streaming=True,
        trust_remote_code=True,
    )

    samples: list[EvalSample] = []
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break

        samples.append(
            EvalSample(
                id=str(_pick(row, _ID_COLS, i)),
                image=_extract_image(row, i),
                ground_truth=str(_pick(row, _GT_COLS, "")),
                task_type=_normalize_task(str(_pick(row, _TASK_COLS, "ocr"))),
            )
        )

    if not samples:
        raise RuntimeError(
            f"No samples loaded from {hf_name!r}. "
            "Check your HF token and network connection."
        )

    return samples


# ── helpers ────────────────────────────────────────────────────────────────────

def _pick(row: dict, candidates: list[str], default=None):
    for key in candidates:
        if key in row:
            return row[key]
    return default


def _extract_image(row: dict, idx: int) -> Image.Image:
    val = _pick(row, _IMAGE_COLS)
    if val is None:
        raise KeyError(
            f"Sample {idx}: no image column found. "
            f"Row has columns: {list(row.keys())}. "
            f"Expected one of: {_IMAGE_COLS}. "
            "Run the schema probe in the module docstring to inspect."
        )
    if isinstance(val, Image.Image):
        return val
    if isinstance(val, bytes):
        return Image.open(io.BytesIO(val))
    # HF datasets sometimes wraps image bytes as {"bytes": b"...", "path": "..."}
    if isinstance(val, dict) and "bytes" in val and val["bytes"]:
        return Image.open(io.BytesIO(val["bytes"]))
    raise TypeError(
        f"Sample {idx}: cannot convert image column type {type(val)} to PIL Image."
    )


def _normalize_task(raw: str) -> TaskType:
    raw = raw.lower()
    if any(k in raw for k in ("ocr", "page", "recog", "transcri")):
        return "ocr"
    if any(k in raw for k in ("pdf", "markdown", " md")):
        return "pdf_to_md"
    if "table" in raw:
        return "table"
    if "chart" in raw:
        return "chart"
    return "unknown"

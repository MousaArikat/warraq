"""Arabic document eval dataset loader.

Supported suites
----------------
misraj-ocr
    Misraj/Misraj-DocOCR — public Arabic doc images with markdown ground truth.
    Use this for smoke tests and early baselines while KITAB-Bench is unavailable.

kitab-ocr
    mbzuai-oryx/KITAB-Bench — the north-star benchmark (ACL 2025).
    Not yet publicly released on HF. Swap in when it lands.

Schema probe (run if a dataset's columns ever change):
    python -c "
    from datasets import load_dataset
    ds = load_dataset('<hf_name>', split='train', streaming=True)
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
_GT_COLS    = ["ground_truth", "markdown", "text", "answer", "transcription", "ocr_text"]
_TASK_COLS  = ["task_type", "task", "category", "type"]
_ID_COLS    = ["uuid", "id", "sample_id", "idx", "image_id", "file_name"]

# Maps suite name → (HF dataset name, HF config or None, split)
_SUITE_MAP: dict[str, tuple[str, str | None, str]] = {
    "misraj-ocr":  ("Misraj/Misraj-DocOCR",         None, "train"),
    "kitab-ocr":   ("mbzuai-oryx/KITAB-Bench",       None, "test"),   # not yet public
}


class EvalSample(BaseModel):
    """One evaluation sample."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    image: Image.Image
    ground_truth: str
    task_type: TaskType


def load_kitab(suite: str = "misraj-ocr", limit: int | None = None) -> list[EvalSample]:
    """Load eval samples for a given suite.

    Parameters
    ----------
    suite:
        Dataset key. See module docstring for available suites.
    limit:
        Cap on returned samples. Use 5–10 for cheap smoke tests.

    Returns
    -------
    list[EvalSample]
    """
    if suite not in _SUITE_MAP:
        raise ValueError(f"Unknown suite {suite!r}. Available: {list(_SUITE_MAP)}")

    hf_name, hf_config, hf_split = _SUITE_MAP[suite]

    # streaming=True: only downloads rows we actually iterate — critical for --limit 5.
    ds = load_dataset(hf_name, hf_config, split=hf_split, streaming=True)

    samples: list[EvalSample] = []
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break

        samples.append(
            EvalSample(
                id=str(_pick(row, _ID_COLS, i)),
                image=_extract_image(row, i),
                ground_truth=str(_pick(row, _GT_COLS, "")),
                task_type=_normalize_task(str(_pick(row, _TASK_COLS, suite))),
            )
        )

    if not samples:
        raise RuntimeError(
            f"No samples loaded from {hf_name!r} (split={hf_split!r}). "
            "Check your network connection and HF token."
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
            f"Sample {idx}: no image column found in {list(row.keys())}. "
            f"Expected one of: {_IMAGE_COLS}. "
            "Run the schema probe in the module docstring."
        )
    if isinstance(val, Image.Image):
        return val
    if isinstance(val, bytes):
        return Image.open(io.BytesIO(val))
    # HF sometimes wraps image bytes as {"bytes": b"...", "path": "..."}
    if isinstance(val, dict) and "bytes" in val and val["bytes"]:
        return Image.open(io.BytesIO(val["bytes"]))
    raise TypeError(
        f"Sample {idx}: cannot convert image column type {type(val)} to PIL Image."
    )


def _normalize_task(raw: str) -> TaskType:
    raw = raw.lower()
    if any(k in raw for k in ("ocr", "page", "recog", "transcri")):
        return "ocr"
    if any(k in raw for k in ("pdf", "markdown", "md", "misraj")):
        return "pdf_to_md"
    if "table" in raw:
        return "table"
    if "chart" in raw:
        return "chart"
    return "unknown"

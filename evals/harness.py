"""Warraq eval harness — ties loader, runner, and metrics together.

Usage:
    python -m evals.harness --model gemini --suite kitab-ocr --limit 5

Results are saved under evals/results/ as a single JSON file named:
    {model}_{suite}_{timestamp}_{git_sha}.json

Each result file contains: run metadata, per-sample scores, and a summary
table with both normalized and strict (tashkeel-preserved) CER/WER.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from data.eval.kitab_loader import load_kitab
from evals.metrics import compute_cer, compute_wer

_RESULTS_DIR = Path(__file__).parent / "results"

# Registry: add new runners here as they're built.
def _get_runner(model: str, model_variant: str | None = None):
    if model == "gemini":
        from evals.baselines.gemini import GeminiRunner
        return GeminiRunner(**({"model_name": model_variant} if model_variant else {}))
    raise SystemExit(f"Unknown model {model!r}. Available: gemini")


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def run_eval(model: str, suite: str, limit: int | None, model_variant: str | None = None) -> dict:
    print(f"Loading {suite!r}  (limit={limit}) …")
    samples = load_kitab(suite=suite, limit=limit)
    print(f"Loaded {len(samples)} samples.\n")

    runner = _get_runner(model, model_variant)
    print(f"Runner : {runner.name}")
    print(f"Samples: {len(samples)}\n")

    per_sample = []
    for i, sample in enumerate(samples, 1):
        hyp = runner.run(sample.image)

        # Normalized: strip tashkeel, normalize alef, digits → western (our defaults)
        cer_norm = compute_cer(sample.ground_truth, hyp)
        wer_norm = compute_wer(sample.ground_truth, hyp)

        # Strict: keep tashkeel — penalises models that drop diacritics
        cer_strict = compute_cer(
            sample.ground_truth, hyp,
            normalize_kwargs={"strip_tashkeel": False},
        )
        wer_strict = compute_wer(
            sample.ground_truth, hyp,
            normalize_kwargs={"strip_tashkeel": False},
        )

        per_sample.append({
            "id": sample.id,
            "task_type": sample.task_type,
            "cer": round(cer_norm, 4),
            "wer": round(wer_norm, 4),
            "cer_strict": round(cer_strict, 4),
            "wer_strict": round(wer_strict, 4),
            "hypothesis": hyp,
            "ground_truth": sample.ground_truth,
        })
        print(
            f"  [{i:>3}/{len(samples)}]  id={sample.id:<20}  "
            f"CER={cer_norm:.3f}  WER={wer_norm:.3f}"
        )

    n = len(per_sample)
    summary = {
        "n_samples": n,
        "mean_cer":        round(sum(r["cer"]        for r in per_sample) / n, 4),
        "mean_wer":        round(sum(r["wer"]        for r in per_sample) / n, 4),
        "mean_cer_strict": round(sum(r["cer_strict"] for r in per_sample) / n, 4),
        "mean_wer_strict": round(sum(r["wer_strict"] for r in per_sample) / n, 4),
    }

    git_sha   = _git_sha()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id    = f"{model}_{suite}_{timestamp}_{git_sha}"

    output = {
        "run_id":      run_id,
        "model":       model,
        "runner_name": runner.name,
        "suite":       suite,
        "git_sha":     git_sha,
        "timestamp":   timestamp,
        "config":      {"limit": limit},
        "summary":     summary,
        "samples":     per_sample,
    }

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _RESULTS_DIR / f"{run_id}.json"
    out_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    _print_summary(summary, runner.name)
    print(f"Results → {out_path}\n")
    return output


def _print_summary(summary: dict, runner_name: str) -> None:
    w = 44
    print(f"\n{'─' * w}")
    print(f"  {runner_name}")
    print(f"{'─' * w}")
    print(f"  Samples            : {summary['n_samples']}")
    print(f"  CER  (normalized)  : {summary['mean_cer']:.4f}")
    print(f"  WER  (normalized)  : {summary['mean_wer']:.4f}")
    print(f"  CER  (strict)      : {summary['mean_cer_strict']:.4f}")
    print(f"  WER  (strict)      : {summary['mean_wer_strict']:.4f}")
    print(f"{'─' * w}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Warraq eval harness")
    parser.add_argument("--model",  required=True, help="Runner name, e.g. gemini")
    parser.add_argument("--suite",  required=True, help="Eval suite, e.g. kitab-ocr")
    parser.add_argument("--limit",  type=int, default=None,
                        help="Max samples to run (omit for full suite)")
    parser.add_argument("--model-variant", default=None,
                        help="Override the runner's default model name, "
                             "e.g. gemini-1.5-flash or gemini-2.0-flash-lite")
    args = parser.parse_args()
    run_eval(args.model, args.suite, args.limit, args.model_variant)


if __name__ == "__main__":
    main()

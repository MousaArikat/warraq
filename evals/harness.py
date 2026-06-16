"""Warraq eval harness — ties loader, runner, and metrics together.

Usage:
    python -m evals.harness --model gemini --suite kitab-hindawi --limit 100

Incremental saving
------------------
After each sample completes, results are written to a partial file:
    evals/results/{model}_{suite}_{git_sha}_partial.json

The partial file uses a deterministic name (no timestamp) so re-running the
exact same command finds it and resumes. On completion the partial is renamed
to the final timestamped file and deleted.

cer_text_only
-------------
KITAB ground truth contains markdown/HTML formatting (**bold**, <table>,
<page_number>) that inflates CER when the model reads the text correctly but
omits formatting. cer_text_only strips markup from both sides before scoring so
reading errors remain visible but formatting mismatches don't. The headline CER
is unchanged (raw, paper-comparable).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from data.eval.kitab_loader import KITAB_NORMALIZE_KWARGS, load_kitab
from evals.metrics import compute_cer, compute_wer

_RESULTS_DIR = Path(__file__).parent / "results"

# ── Markup stripper for cer_text_only ─────────────────────────────────────────

_HTML_TAG_RE   = re.compile(r"<[^>]+>")
_MD_LINK_RE    = re.compile(r"\[([^\]]*)\]\([^)]*\)")  # [text](url) → text
_MD_FORMAT_RE  = re.compile(r"\*{1,3}|_{1,3}")
_MD_HEADING_RE = re.compile(r"^#{1,6}\s*", re.MULTILINE)


def _strip_markup(text: str) -> str:
    text = _HTML_TAG_RE.sub("", text)
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _MD_FORMAT_RE.sub("", text)
    text = _MD_HEADING_RE.sub("", text)
    return text


# ── Runner registry ────────────────────────────────────────────────────────────

def _get_runner(model: str, model_variant: str | None = None):
    if model == "gemini":
        from evals.baselines.gemini import GeminiRunner
        return GeminiRunner(**({"model_name": model_variant} if model_variant else {}))
    if model == "gpt4o":
        from evals.baselines.gpt4o import GPT4oRunner
        return GPT4oRunner(**({"model_name": model_variant} if model_variant else {}))
    raise SystemExit(f"Unknown model {model!r}. Available: gemini, gpt4o")


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


# ── Partial-file helpers ───────────────────────────────────────────────────────

def _partial_path(model: str, suite: str, git_sha: str) -> Path:
    return _RESULTS_DIR / f"{model}_{suite}_{git_sha}_partial.json"


def _write_partial(path: Path, model: str, suite: str,
                   git_sha: str, config: dict, samples: list) -> None:
    path.write_text(
        json.dumps(
            {"model": model, "suite": suite, "git_sha": git_sha,
             "config": config, "samples": samples},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )


# ── Main eval loop ─────────────────────────────────────────────────────────────

def run_eval(model: str, suite: str, limit: int | None,
             model_variant: str | None = None) -> dict:
    git_sha     = _git_sha()
    norm_kwargs = KITAB_NORMALIZE_KWARGS if suite.startswith("kitab-") else {}
    strict_kw   = {**norm_kwargs, "strip_tashkeel": False}
    config      = {"limit": limit, "normalize_kwargs": norm_kwargs}

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    part_path = _partial_path(model, suite, git_sha)

    # Load any previously completed samples (keyed by sample id).
    completed: dict[str, dict] = {}
    if part_path.exists():
        saved = json.loads(part_path.read_text(encoding="utf-8"))
        for r in saved.get("samples", []):
            completed[r["id"]] = r
        print(f"Resuming: {len(completed)} samples already saved.\n")

    print(f"Loading {suite!r}  (limit={limit}) …")
    samples = load_kitab(suite=suite, limit=limit)
    print(f"Loaded {len(samples)} samples.\n")

    runner = _get_runner(model, model_variant)
    print(f"Runner : {runner.name}\n")

    all_results: list[dict] = []

    for i, sample in enumerate(samples, 1):
        # Resume: skip samples already scored in a previous run.
        if sample.id in completed:
            all_results.append(completed[sample.id])
            print(f"  [{i:>3}/{len(samples)}]  id={sample.id:<20}  (skipped)")
            continue

        hyp = runner.run(sample.image)

        cer_norm   = compute_cer(sample.ground_truth, hyp, normalize_kwargs=norm_kwargs)
        wer_norm   = compute_wer(sample.ground_truth, hyp, normalize_kwargs=norm_kwargs)
        cer_strict = compute_cer(sample.ground_truth, hyp, normalize_kwargs=strict_kw)
        wer_strict = compute_wer(sample.ground_truth, hyp, normalize_kwargs=strict_kw)
        cer_text   = compute_cer(
            _strip_markup(sample.ground_truth),
            _strip_markup(hyp),
            normalize_kwargs=norm_kwargs,
        )

        result = {
            "id":            sample.id,
            "task_type":     sample.task_type,
            "cer":           round(cer_norm,   4),
            "wer":           round(wer_norm,   4),
            "cer_strict":    round(cer_strict, 4),
            "wer_strict":    round(wer_strict, 4),
            "cer_text_only": round(cer_text,   4),
            "hypothesis":    hyp,
            "ground_truth":  sample.ground_truth,
        }
        all_results.append(result)

        # Write after every new result — crash-safe.
        _write_partial(part_path, model, suite, git_sha, config, all_results)

        print(
            f"  [{i:>3}/{len(samples)}]  id={sample.id:<20}  "
            f"CER={cer_norm:.3f}  WER={wer_norm:.3f}  CER_text={cer_text:.3f}"
        )

    n = len(all_results)
    summary = {
        "n_samples":          n,
        "mean_cer":           round(sum(r["cer"]           for r in all_results) / n, 4),
        "mean_wer":           round(sum(r["wer"]           for r in all_results) / n, 4),
        "mean_cer_strict":    round(sum(r["cer_strict"]    for r in all_results) / n, 4),
        "mean_wer_strict":    round(sum(r["wer_strict"]    for r in all_results) / n, 4),
        "mean_cer_text_only": round(sum(r["cer_text_only"] for r in all_results) / n, 4),
    }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id    = f"{model}_{suite}_{timestamp}_{git_sha}"
    out_path  = _RESULTS_DIR / f"{run_id}.json"

    output = {
        "run_id":      run_id,
        "model":       model,
        "runner_name": runner.name,
        "suite":       suite,
        "git_sha":     git_sha,
        "timestamp":   timestamp,
        "config":      config,
        "summary":     summary,
        "samples":     all_results,
    }

    out_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    part_path.unlink(missing_ok=True)

    _print_summary(summary, runner.name)
    print(f"Results → {out_path}\n")
    return output


def _print_summary(summary: dict, runner_name: str) -> None:
    w = 50
    print(f"\n{'─' * w}")
    print(f"  {runner_name}")
    print(f"{'─' * w}")
    print(f"  Samples                : {summary['n_samples']}")
    print(f"  CER  (normalized)      : {summary['mean_cer']:.4f}")
    print(f"  WER  (normalized)      : {summary['mean_wer']:.4f}")
    print(f"  CER  (strict)          : {summary['mean_cer_strict']:.4f}")
    print(f"  WER  (strict)          : {summary['mean_wer_strict']:.4f}")
    print(f"  CER  (text only)       : {summary['mean_cer_text_only']:.4f}")
    print(f"{'─' * w}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Warraq eval harness")
    parser.add_argument("--model",  required=True, help="Runner name, e.g. gemini")
    parser.add_argument("--suite",  required=True, help="Eval suite, e.g. kitab-hindawi")
    parser.add_argument("--limit",  type=int, default=None,
                        help="Max samples to run (omit for full suite)")
    parser.add_argument("--model-variant", default=None,
                        help="Override the runner's default model name")
    args = parser.parse_args()
    run_eval(args.model, args.suite, args.limit, args.model_variant)


if __name__ == "__main__":
    main()

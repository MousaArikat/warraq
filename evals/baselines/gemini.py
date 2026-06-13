"""Gemini baseline runner (google-genai SDK, free tier).

Every raw API response is cached to disk under evals/cache/gemini/ so that:
  - Re-runs don't burn API quota.
  - Every result is auditable: you can diff the cache entry against what the
    harness scored, proving no post-hoc editing.

Cache files are gitignored (large, binary-adjacent). The harness results JSON
records the cache key for each sample so you can always look up the raw response.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from PIL import Image

from evals.baselines.base import BaselineRunner

load_dotenv()

_CACHE_DIR = Path(__file__).parent.parent / "cache" / "gemini"

# Matches the prompt used in the KITAB-Bench eval.py exactly, so our numbers
# are directly comparable to the paper's leaderboard.
_DEFAULT_PROMPT = "Extract the text in the image. Give me the final text, nothing else."

_MAX_RETRIES = 4
_INTER_SAMPLE_DELAY = 13.0       # gemini-2.5-flash free tier = 5 RPM → 12s minimum, 13s for safety
_RETRY_DELAY_RE = re.compile(r"retryDelay['\"]:\s*['\"](\d+)s")


class GeminiRunner(BaselineRunner):
    """Gemini Flash via google-genai.

    Default model: gemini-2.5-flash (free tier). Change model_name in the
    constructor to use a different variant.
    """

    def __init__(self, model_name: str = "gemini-2.5-flash") -> None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY is not set. "
                "Copy .env.example → .env and add your key."
            )
        self._client = genai.Client(api_key=api_key)
        self._model_name = model_name
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:
        return f"gemini/{self._model_name}"

    def run(self, image: Image.Image, prompt: str | None = None) -> str:
        prompt = prompt or _DEFAULT_PROMPT
        cache_key = _make_cache_key(self._model_name, image, prompt)
        cache_path = _CACHE_DIR / f"{cache_key}.json"

        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            return cached["response"]["text"]

        text = self._call_with_retry(image, prompt)

        cache_path.write_text(
            json.dumps(
                {
                    "cache_key": cache_key,
                    "model": self._model_name,
                    "prompt": prompt,
                    "response": {"text": text},
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        # Pause after every real API call to stay under the per-minute request cap.
        time.sleep(_INTER_SAMPLE_DELAY)
        return text

    def _call_with_retry(self, image: Image.Image, prompt: str) -> str:
        for attempt in range(_MAX_RETRIES):
            try:
                response = self._client.models.generate_content(
                    model=self._model_name,
                    contents=[prompt, image],
                )
                return response.text
            except genai_errors.ClientError as e:
                is_rate_limit = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
                is_last_attempt = attempt == _MAX_RETRIES - 1
                if is_rate_limit and not is_last_attempt:
                    m = _RETRY_DELAY_RE.search(str(e))
                    delay = int(m.group(1)) + 2 if m else 60
                    print(
                        f"    rate-limited (429), sleeping {delay}s "
                        f"(attempt {attempt + 1}/{_MAX_RETRIES}) …"
                    )
                    time.sleep(delay)
                else:
                    raise
        # unreachable — the loop always raises on the last attempt
        raise RuntimeError("retry loop exited without returning or raising")


def _make_cache_key(model_name: str, image: Image.Image, prompt: str) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    payload = model_name.encode() + buf.getvalue() + prompt.encode()
    return hashlib.sha256(payload).hexdigest()[:32]

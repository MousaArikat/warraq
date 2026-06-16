"""GPT-4o baseline runner (openai SDK).

Every raw API response is cached to disk under evals/cache/gpt4o/ so that:
  - Re-runs don't burn API quota.
  - Every result is auditable: you can diff the cache entry against what the
    harness scored, proving no post-hoc editing.

Cache files are gitignored (large, binary-adjacent). The harness results JSON
records the cache key for each sample so you can always look up the raw response.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import openai
from dotenv import load_dotenv
from PIL import Image

from evals.baselines.base import BaselineRunner

load_dotenv()

_CACHE_DIR = Path(__file__).parent.parent / "cache" / "gpt4o"

# Matches the prompt used in the KITAB-Bench eval.py exactly, so our numbers
# are directly comparable to the paper's leaderboard. Identical to gemini.py.
_DEFAULT_PROMPT = "Extract the text in the image. Give me the final text, nothing else."

_MAX_RETRIES = 4
_INTER_SAMPLE_DELAY = 13.0  # mirrors gemini.py's pacing; adjust if OpenAI's rate limit differs

# Set True on the first call to dump the raw response + finish_reason once, so we
# can tell a content-filter refusal apart from a malformed-request error. Flip back
# to False once you've confirmed which one you're hitting.
_debug_printed = False


class GPT4oRunner(BaselineRunner):
    """GPT-4o via the openai SDK.

    Default model: gpt-4o. Change model_name in the constructor to use a
    different variant (e.g. gpt-4o-mini).
    """

    def __init__(self, model_name: str = "gpt-4o") -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. "
                "Copy .env.example → .env and add your key."
            )
        self._client = openai.OpenAI(api_key=api_key)
        self._model_name = model_name
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:
        return f"gpt4o/{self._model_name}"

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
        global _debug_printed
        data_url = _make_data_url(image)
        for attempt in range(_MAX_RETRIES):
            try:
                response = self._client.chat.completions.create(
                    model=self._model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": data_url}},
                            ],
                        }
                    ],
                )

                if not _debug_printed:
                    _debug_printed = True
                    choice = response.choices[0]
                    print("\n[gpt4o debug] raw response:")
                    print(response.model_dump_json(indent=2))
                    print(f"[gpt4o debug] finish_reason: {choice.finish_reason}\n")

                return response.choices[0].message.content
            except (openai.RateLimitError, openai.APIStatusError, openai.APIConnectionError) as e:
                status_code = getattr(e, "status_code", None)
                is_rate_limit = isinstance(e, openai.RateLimitError) or status_code == 429
                is_server_error = status_code is not None and 500 <= status_code < 600
                is_retryable = is_rate_limit or is_server_error or isinstance(e, openai.APIConnectionError)
                is_last_attempt = attempt == _MAX_RETRIES - 1

                if is_retryable and not is_last_attempt:
                    delay = _suggested_delay(e)
                    print(
                        f"    rate-limited/server error ({status_code}), sleeping {delay}s "
                        f"(attempt {attempt + 1}/{_MAX_RETRIES}) …"
                    )
                    time.sleep(delay)
                else:
                    raise
        # unreachable — the loop always raises on the last attempt
        raise RuntimeError("retry loop exited without returning or raising")


def _suggested_delay(e: Exception) -> int:
    """Read the suggested retry delay from the error's response headers, if present."""
    response = getattr(e, "response", None)
    headers = getattr(response, "headers", None) if response is not None else None
    if headers:
        retry_after = headers.get("retry-after")
        if retry_after is not None:
            try:
                return int(float(retry_after)) + 2
            except ValueError:
                pass
    return 60


def _to_png_bytes(image: Image.Image) -> bytes:
    # PNG can't encode CMYK (and palette/LA modes are fragile across libraries),
    # so normalize to RGB first. Without this, .save(format="PNG") can raise on
    # some scanned-document modes, or silently shift colors.
    if image.mode not in ("RGB", "RGBA", "L"):
        image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _make_data_url(image: Image.Image) -> str:
    png_bytes = _to_png_bytes(image)
    b64 = base64.b64encode(png_bytes).decode("ascii")
    # Must be exactly this form for OpenAI's vision API: data:image/<fmt>;base64,<DATA>
    return f"data:image/png;base64,{b64}"


def _make_cache_key(model_name: str, image: Image.Image, prompt: str) -> str:
    png_bytes = _to_png_bytes(image)
    payload = model_name.encode() + png_bytes + prompt.encode()
    return hashlib.sha256(payload).hexdigest()[:32]

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
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

from evals.baselines.base import BaselineRunner

load_dotenv()

_CACHE_DIR = Path(__file__).parent.parent / "cache" / "gemini"

_DEFAULT_PROMPT = (
    "Read all the Arabic text in this image exactly as it appears. "
    "Output only the transcribed text, preserving line breaks. "
    "Do not translate, summarise, or add any commentary."
)


class GeminiRunner(BaselineRunner):
    """Gemini Flash via google-genai.

    Default model: gemini-2.0-flash (free tier). Change model_name in the
    constructor to use a different variant.
    """

    def __init__(self, model_name: str = "gemini-2.0-flash") -> None:
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

        response = self._client.models.generate_content(
            model=self._model_name,
            contents=[prompt, image],
        )
        text = response.text

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
        return text


def _make_cache_key(model_name: str, image: Image.Image, prompt: str) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    payload = model_name.encode() + buf.getvalue() + prompt.encode()
    return hashlib.sha256(payload).hexdigest()[:32]

"""Abstract base class for all baseline runners."""
from __future__ import annotations

from abc import ABC, abstractmethod

from PIL import Image


class BaselineRunner(ABC):
    """Every model runner implements this interface so the harness can swap them."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier shown in results, e.g. 'gemini/gemini-1.5-flash'."""

    @abstractmethod
    def run(self, image: Image.Image, prompt: str | None = None) -> str:
        """Run inference on one image and return the model's raw text output.

        Parameters
        ----------
        image:
            The page image to transcribe.
        prompt:
            Override the runner's default OCR prompt. Pass None to use the
            runner's built-in default.

        Returns
        -------
        str
            The model's text output (not yet normalized or scored).
        """

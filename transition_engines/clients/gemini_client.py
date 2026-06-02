"""Google Gemini adapter (google-genai SDK), JSON output mode."""
from __future__ import annotations

from typing import Any

from .base import BaseLLMClient


class GeminiClient(BaseLLMClient):
    """Gemini client requesting JSON output; tolerant parser handles extraction."""

    env_vars = ("GEMINI_API_KEY", "GOOGLE_API_KEY")

    def _call(self, prompt: str) -> Any:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self._api_key)
        config_kwargs: dict[str, Any] = {
            "max_output_tokens": self.max_output_tokens,
            "response_mime_type": "application/json",
        }
        if self.temperature is not None:
            config_kwargs["temperature"] = self.temperature
        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        return response.text


__all__ = ["GeminiClient"]

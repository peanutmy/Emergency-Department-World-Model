"""Anthropic Claude adapter (Messages API), JSON via prompt + tolerant parser."""
from __future__ import annotations

from typing import Any

from .base import BaseLLMClient


class AnthropicClient(BaseLLMClient):
    """Claude client; the prompt already instructs 'Output JSON only'.

    Text blocks are concatenated and handed to the engines' tolerant JSON
    parser, which strips fences / extracts the first object.
    """

    env_vars = ("ANTHROPIC_API_KEY",)

    def _call(self, prompt: str) -> str:
        from anthropic import Anthropic

        client = Anthropic(api_key=self._api_key)
        message = client.messages.create(
            model=self.model,
            max_tokens=self.max_output_tokens,
            temperature=self.temperature if self.temperature is not None else 0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        chunks: list[str] = []
        for block in getattr(message, "content", []) or []:
            if getattr(block, "type", None) == "text":
                text = getattr(block, "text", None)
                if isinstance(text, str):
                    chunks.append(text)
        return "".join(chunks)


__all__ = ["AnthropicClient"]

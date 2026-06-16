"""
OpenRouter LLM provider layer.

A thin wrapper over OpenRouter's OpenAI-compatible chat completions API.
Supports plain chat and native tool/function calling, with automatic
fallback across a list of free models if one is unavailable or rate-limited.

Usage:
    llm = LLMClient()
    text = llm.chat([{"role": "user", "content": "Hello"}])
    msg = llm.chat_with_tools(messages, tools)  # returns the raw message dict
"""
from __future__ import annotations

import logging
from typing import Any

from openai import OpenAI

import config

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Raised when no configured model can serve a request."""


class LLMClient:
    """OpenRouter chat client with multi-model fallback."""

    def __init__(self, api_key: str | None = None, models: list[str] | None = None):
        self.api_key = api_key or config.OPENROUTER_API_KEY
        if not self.api_key:
            raise LLMError(
                "OPENROUTER_API_KEY is not set. Add it to your .env file or "
                "Streamlit secrets. Get a key at https://openrouter.ai/keys"
            )
        self.models = models or config.MODELS
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=config.OPENROUTER_BASE_URL,
        )

    def _create(self, **kwargs: Any):
        """Call the chat API, trying each model until one succeeds."""
        last_error: Exception | None = None
        for model in self.models:
            try:
                return self.client.chat.completions.create(model=model, **kwargs)
            except Exception as e:  # noqa: BLE001 - fall through to next model
                logger.warning("Model %s failed: %s", model, e)
                last_error = e
        raise LLMError(f"All models failed. Last error: {last_error}")

    def chat(self, messages: list[dict]) -> str:
        """Return the assistant's text reply for a list of chat messages."""
        response = self._create(
            messages=messages,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS,
        )
        return response.choices[0].message.content or ""

    def chat_with_tools(self, messages: list[dict], tools: list[dict], tool_choice: str = "auto"):
        """Run a chat turn with tools available.

        ``tool_choice`` is "auto" (model decides) or "required" (force at least
        one tool call). Returns the raw assistant message object; if the model
        called tools, ``message.tool_calls`` is populated.
        """
        response = self._create(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS,
        )
        return response.choices[0].message

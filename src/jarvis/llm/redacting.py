"""Redaction decorator for cloud-bound LLM backends.

When chat runs against a remote provider (OpenAI-compatible endpoints
like OpenRouter, Mistral, OpenAI), the prompt leaves the machine. The
project's privacy contract — the same automatic scrub that protects
on-disk diary writes — should apply before any byte crosses the wire.

``RedactingBackend`` wraps any :class:`LLMBackend` and scrubs the text
arguments on the egress methods (``direct``, ``streaming``, ``chat``,
``embed``) via :func:`jarvis.utils.redact.scrub_secrets` before
delegating to the inner backend. ``list_models`` and ``warm_up`` carry
no user content and pass straight through.

Design choice — decorator, not a constructor flag on the concrete
backend: this keeps the upstream ``OpenAICompatibleBackend`` untouched
(minimal divergence from upstream) and makes redaction a composable
layer the factory can add or drop with one ``if``. It is itself a
first-class ``LLMBackend`` so callers never branch on whether redaction
is active.

``scrub_secrets`` (not ``redact``) is used because it preserves
newlines and applies no length cap — system prompts and multi-turn
message arrays must keep their structure; only the secret-shaped tokens
(emails, API keys, JWTs, password/token/secret keyword pairs) are
masked.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from .backend import LLMBackend
from ..utils.redact import scrub_secrets


def _scrub(text: Any) -> Any:
    """Scrub a value when it's a non-empty string; pass anything else
    through unchanged (structured content blocks, None, etc.)."""
    if isinstance(text, str) and text:
        return scrub_secrets(text)
    return text


def _scrub_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return a shallow copy of ``messages`` with every string
    ``content`` field scrubbed. Tool-call metadata and non-string
    content (e.g. multimodal blocks) are left untouched."""
    out: List[Dict[str, Any]] = []
    for m in messages or []:
        if isinstance(m, dict) and isinstance(m.get("content"), str):
            scrubbed = dict(m)
            scrubbed["content"] = scrub_secrets(m["content"])
            out.append(scrubbed)
        else:
            out.append(m)
    return out


class RedactingBackend(LLMBackend):
    """Wrap an ``LLMBackend`` and scrub user/system text before it is
    sent to the inner (cloud) backend."""

    def __init__(self, inner: LLMBackend) -> None:
        self._inner = inner

    @property
    def inner(self) -> LLMBackend:
        return self._inner

    def direct(
        self,
        chat_model: str,
        system_prompt: str,
        user_content: str,
        timeout_sec: float = 10.0,
        thinking: bool = False,
        num_ctx: int = 4096,
        temperature: Optional[float] = None,
    ) -> Optional[str]:
        return self._inner.direct(
            chat_model,
            _scrub(system_prompt),
            _scrub(user_content),
            timeout_sec=timeout_sec,
            thinking=thinking,
            num_ctx=num_ctx,
            temperature=temperature,
        )

    def streaming(
        self,
        chat_model: str,
        system_prompt: str,
        user_content: str,
        on_token: Optional[Callable[[str], None]] = None,
        timeout_sec: float = 30.0,
        thinking: bool = False,
    ) -> Optional[str]:
        return self._inner.streaming(
            chat_model,
            _scrub(system_prompt),
            _scrub(user_content),
            on_token=on_token,
            timeout_sec=timeout_sec,
            thinking=thinking,
        )

    def chat(
        self,
        chat_model: str,
        messages: List[Dict[str, Any]],
        timeout_sec: float = 30.0,
        extra_options: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        thinking: bool = False,
    ) -> Optional[Dict[str, Any]]:
        return self._inner.chat(
            chat_model,
            _scrub_messages(messages),
            timeout_sec=timeout_sec,
            extra_options=extra_options,
            tools=tools,
            thinking=thinking,
        )

    def embed(
        self,
        text: str,
        model: str,
        timeout_sec: float = 15.0,
    ) -> Optional[List[float]]:
        # Embedding text egresses too when embeddings run in the cloud.
        return self._inner.embed(_scrub(text), model, timeout_sec=timeout_sec)

    def list_models(self, timeout_sec: float = 5.0) -> List[str]:
        return self._inner.list_models(timeout_sec=timeout_sec)

    def warm_up(
        self,
        model: str,
        timeout_sec: float = 60.0,
        keep_alive: str = "30m",
    ) -> bool:
        return self._inner.warm_up(model, timeout_sec=timeout_sec, keep_alive=keep_alive)

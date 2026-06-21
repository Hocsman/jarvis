"""Factories for resolving the active LLM and embedding backends.

Two factories share one provider catalogue:

- :func:`get_llm_backend` — chat / completion path. Dispatches on
  ``settings.llm_provider``.
- :func:`get_embedding_backend` — embeddings path. Dispatches on
  ``settings.embedding_provider``, falling back to the LLM provider
  when unset. The override exists for runtimes that ship chat without
  embeddings (early oMLX builds, some llama.cpp configurations); users
  can keep chat on their preferred runtime and route embeddings
  through Ollama instead.
"""

from __future__ import annotations
import json
from typing import Any, Dict, Optional

from .backend import LLMBackend
from .ollama import OllamaBackend
from .openai_compatible import OpenAICompatibleBackend
from .redacting import RedactingBackend


_OLLAMA = "ollama"
_OPENAI_COMPATIBLE = "openai_compatible"
_DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"

# Backends are cached by their resolved connection parameters so that a
# single instance — and its persistent HTTP session — is reused across
# the many ``get_llm_backend`` calls a single reply makes. Without this
# every call built a fresh backend with a cold connection and paid the
# TLS handshake (and the provider's cold-start) every time; on a remote
# cloud endpoint that dominated the latency of a multi-call reply.
_BACKEND_CACHE: dict = {}


def clear_backend_cache() -> None:
    """Drop all cached backends.

    Call after a configuration reload so the next ``get_llm_backend``
    rebuilds against the new settings instead of returning a backend
    wired to the old endpoint/key. Also used by tests to keep instances
    isolated.
    """
    _BACKEND_CACHE.clear()


def _resolve_provider(value: Any) -> str:
    if isinstance(value, str):
        v = value.strip().lower()
        if v in (_OLLAMA, _OPENAI_COMPATIBLE):
            return v
    return _OLLAMA


def _str_attr(settings: Any, name: str, default: str = "") -> str:
    val = getattr(settings, name, None)
    return val if isinstance(val, str) and val else default


def _build(
    provider: str,
    base_url: str,
    api_key: Optional[str],
    redact: bool = False,
    extra_body: Optional[Dict[str, Any]] = None,
) -> LLMBackend:
    key = (
        provider, base_url, api_key, redact,
        json.dumps(extra_body or {}, sort_keys=True),
    )
    cached = _BACKEND_CACHE.get(key)
    if cached is not None:
        return cached
    backend = _build_uncached(provider, base_url, api_key, redact, extra_body)
    _BACKEND_CACHE[key] = backend
    return backend


def _build_uncached(
    provider: str,
    base_url: str,
    api_key: Optional[str],
    redact: bool = False,
    extra_body: Optional[Dict[str, Any]] = None,
) -> LLMBackend:
    if provider == _OPENAI_COMPATIBLE:
        backend: LLMBackend = OpenAICompatibleBackend(
            base_url, api_key=api_key, extra_body=extra_body
        )
        # Scrub secrets before remote egress. Only the OpenAI-compatible
        # path is wrapped — the Ollama path is local, so its prompts
        # never leave the machine and redaction would be wasted work.
        if redact:
            backend = RedactingBackend(backend)
        return backend
    return OllamaBackend(base_url)


def get_llm_backend(settings: Any) -> LLMBackend:
    """Return the configured chat backend.

    ``llm_base_url`` is the OpenAI-compatible server's URL; the Ollama path
    uses ``ollama_base_url``. Keeping each provider on its own URL field
    means toggling ``llm_provider`` back to Ollama can never leave the
    backend pointed at a stale OpenAI-compatible URL.
    """
    provider = _resolve_provider(getattr(settings, "llm_provider", None))
    if provider == _OPENAI_COMPATIBLE:
        base_url = _str_attr(settings, "llm_base_url") or _str_attr(
            settings, "ollama_base_url", _DEFAULT_OLLAMA_URL
        )
    else:
        base_url = _str_attr(settings, "ollama_base_url", _DEFAULT_OLLAMA_URL)
    api_key = _str_attr(settings, "llm_api_key") or None
    # Fallback False (not True) only matters for partial cfg objects
    # that lack the field — i.e. test mocks. The real ``Settings`` always
    # carries ``auto_redact_before_cloud`` (required field, default True
    # in the parser), so production egress is always redaction-wrapped;
    # this fallback just keeps the factory's type contract intact for the
    # backend-resolution unit tests that pass bare mocks.
    redact = bool(getattr(settings, "auto_redact_before_cloud", False))
    extra_body = getattr(settings, "llm_extra_body", None)
    if not isinstance(extra_body, dict):
        extra_body = None
    return _build(provider, base_url, api_key, redact=redact, extra_body=extra_body)


def get_embedding_backend(settings: Any) -> LLMBackend:
    """Return the configured embedding backend.

    Falls through ``embedding_provider`` → ``llm_provider`` → ``"ollama"``.
    Users can pin embeddings to Ollama (recommended when chat runs on a
    runtime without embedding support) by setting
    ``embedding_provider: "ollama"`` in their config.
    """
    raw = getattr(settings, "embedding_provider", None)
    if isinstance(raw, str) and raw.strip():
        provider = _resolve_provider(raw)
    else:
        provider = _resolve_provider(getattr(settings, "llm_provider", None))

    base_url = _str_attr(settings, "embedding_base_url")
    if not base_url:
        if provider == _OPENAI_COMPATIBLE:
            base_url = _str_attr(settings, "llm_base_url")
        else:
            base_url = _str_attr(settings, "ollama_base_url", _DEFAULT_OLLAMA_URL)
    if not base_url:
        base_url = _DEFAULT_OLLAMA_URL

    api_key = _str_attr(settings, "embedding_api_key") or _str_attr(
        settings, "llm_api_key"
    ) or None
    # Fallback False (not True) only matters for partial cfg objects
    # that lack the field — i.e. test mocks. The real ``Settings`` always
    # carries ``auto_redact_before_cloud`` (required field, default True
    # in the parser), so production egress is always redaction-wrapped;
    # this fallback just keeps the factory's type contract intact for the
    # backend-resolution unit tests that pass bare mocks.
    redact = bool(getattr(settings, "auto_redact_before_cloud", False))
    return _build(provider, base_url, api_key, redact=redact)

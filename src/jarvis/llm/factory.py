"""Factories for resolving the active LLM and embedding backends.

Two factories share one provider catalogue:

- :func:`get_llm_backend` — chat / completion path. Dispatches on
  ``settings.llm_provider``.
- :func:`get_embedding_backend` — embeddings path. Dispatches on
  ``settings.embedding_provider``, falling back to the LLM provider
  when unset. The override exists for runtimes that ship chat without
  embeddings (early oMLX builds, some llama.cpp configurations, and
  Anthropic, which always embeds on Ollama); users can keep chat on
  their preferred runtime and route embeddings through Ollama instead.
"""

from __future__ import annotations
import json
from typing import Any, Dict, Optional

from ..debug import debug_log
from .anthropic_backend import AnthropicBackend
from .backend import LLMBackend
from .ollama import OllamaBackend
from .openai_compatible import OpenAICompatibleBackend
from .redacting import RedactingBackend
from ..config import (
    ANTHROPIC_BASE_URL,
    LLM_PROVIDERS,
    is_local_endpoint,
    names_provider_model,
)


_OLLAMA = "ollama"
_OPENAI_COMPATIBLE = "openai_compatible"
_ANTHROPIC = "anthropic"
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
        if v in LLM_PROVIDERS:
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
    if provider == _OLLAMA:
        return OllamaBackend(base_url)
    if provider == _ANTHROPIC:
        backend: LLMBackend = AnthropicBackend(api_key=api_key)
    else:
        backend = OpenAICompatibleBackend(
            base_url, api_key=api_key, extra_body=extra_body
        )
    # Scrub secrets before remote egress. The Ollama path is not wrapped:
    # its prompts never leave the machine and redaction would be wasted work.
    if redact:
        backend = RedactingBackend(backend)
    return backend


def get_llm_backend(settings: Any, model: Optional[str] = None) -> LLMBackend:
    """Return the configured chat backend.

    ``llm_base_url`` is the OpenAI-compatible server's URL; the Ollama path
    uses ``ollama_base_url`` and the Anthropic path its fixed endpoint.
    Keeping each provider on its own URL means toggling ``llm_provider``
    can never leave the backend pointed at a stale OpenAI-compatible URL.

    When ``model`` is a local tag rather than a name the remote provider
    serves (:func:`names_provider_model`), the model cannot run on the
    remote endpoint. It routes to local Ollama instead, eliminating cloud
    network latency for auxiliary classification and routing passes.
    """
    model_str = (model or "").strip()
    provider = _resolve_provider(getattr(settings, "llm_provider", None))
    if provider == _OLLAMA:
        base_url = _str_attr(settings, "ollama_base_url", _DEFAULT_OLLAMA_URL)
    else:
        if provider == _ANTHROPIC:
            base_url = ANTHROPIC_BASE_URL
        else:
            base_url = _str_attr(settings, "llm_base_url") or _str_attr(
                settings, "ollama_base_url", _DEFAULT_OLLAMA_URL
            )
        if (model_str and not is_local_endpoint(base_url)
                and not names_provider_model(provider, model_str)):
            return get_llm_backend(_SurLaMachine(settings))
    api_key = _str_attr(settings, "llm_api_key") or None
    # Fallback False (not True) only matters for partial cfg objects
    # that lack the field — i.e. test mocks. The real ``Settings`` always
    # carries ``auto_redact_before_cloud`` (required field, default True
    # in the parser), so production egress is always redaction-wrapped;
    # this fallback just keeps the factory's type contract intact for the
    # backend-resolution unit tests that pass bare mocks.
    redact = bool(getattr(settings, "auto_redact_before_cloud", False))
    extra_body = getattr(settings, "llm_extra_body", None)
    if not isinstance(extra_body, dict) or provider != _OPENAI_COMPATIBLE:
        extra_body = None
    return _build(provider, base_url, api_key, redact=redact, extra_body=extra_body)


def get_embedding_backend(settings: Any) -> LLMBackend:
    """Return the configured embedding backend.

    Falls through ``embedding_provider`` → ``llm_provider`` → ``"ollama"``.
    Users can pin embeddings to Ollama (recommended when chat runs on a
    runtime without embedding support) by setting
    ``embedding_provider: "ollama"`` in their config. Anthropic serves no
    embeddings, so a chat provider of ``anthropic`` embeds on Ollama.
    """
    chat_provider = _resolve_provider(getattr(settings, "llm_provider", None))
    raw = getattr(settings, "embedding_provider", None)
    if isinstance(raw, str) and raw.strip():
        provider = _resolve_provider(raw)
    else:
        provider = chat_provider
    if provider == _ANTHROPIC:
        provider = _OLLAMA

    # The chat settings' URL and key are never inherited from the Anthropic
    # provider: its key belongs to Anthropic alone, and ``llm_base_url`` is
    # not its endpoint.
    inherits = chat_provider != _ANTHROPIC
    base_url = _str_attr(settings, "embedding_base_url")
    if not base_url:
        if provider == _OPENAI_COMPATIBLE:
            base_url = _str_attr(settings, "llm_base_url") if inherits else ""
        else:
            base_url = _str_attr(settings, "ollama_base_url", _DEFAULT_OLLAMA_URL)
    if not base_url:
        base_url = _DEFAULT_OLLAMA_URL

    api_key = _str_attr(settings, "embedding_api_key") or (
        _str_attr(settings, "llm_api_key") if inherits else ""
    ) or None
    # Fallback False (not True) only matters for partial cfg objects
    # that lack the field — i.e. test mocks. The real ``Settings`` always
    # carries ``auto_redact_before_cloud`` (required field, default True
    # in the parser), so production egress is always redaction-wrapped;
    # this fallback just keeps the factory's type contract intact for the
    # backend-resolution unit tests that pass bare mocks.
    redact = bool(getattr(settings, "auto_redact_before_cloud", False))
    return _build(provider, base_url, api_key, redact=redact)


class _SurLaMachine:
    """`settings`, read as if the provider were the local one.

    A thin view rather than a copy: `Settings` is frozen, callers pass
    mocks, and the factory reads a dozen attributes it should keep
    reading from the original.
    """

    __slots__ = ("_vrai",)

    def __init__(self, vrai: Any):
        object.__setattr__(self, "_vrai", vrai)

    def __getattr__(self, nom: str) -> Any:
        if nom == "llm_provider":
            return "ollama"
        return getattr(object.__getattribute__(self, "_vrai"), nom)


def get_private_backend(settings: Any, pinned: str, model: str) -> LLMBackend:
    """The backend for a prompt carrying the user's own life.

    Four contexts read his own sentences: the time he said out loud, the
    routine he described, the goal he is working towards, and the
    fortnight of diary the learning step mines. Each specifies that
    pinning a model is how he keeps that sentence off the network.

    A name alone does not do that. Called without a model,
    :func:`get_llm_backend` picks the endpoint from ``llm_provider``
    alone, which on a cloud provider sends a pinned local tag, and the
    sentence with it, to the cloud.

    So a pin decides the destination too, whatever its shape: an
    ``hf.co/...`` Ollama tag has a slash and is still his machine. With
    nothing pinned, ``model`` (the one the context resolved and is about
    to send) decides it the way it does for every other call: a bare tag
    on a remote provider names a model that lives on local Ollama, and
    goes there. A namespaced model stays on the provider he chose: the
    chat model at the end of the chain, a router given as
    ``vendor/model``, and a slashed local tag too, which only a pin keeps
    here.

    ``pinned`` and ``model`` are separate because a context's chain runs
    past its pin: the model it sends may be a fallback's, and the request
    has to go where that model lives.
    """
    is_pinned = bool((pinned or "").strip())
    if is_pinned:
        backend = get_llm_backend(_SurLaMachine(settings))
    else:
        backend = get_llm_backend(settings, model=model)
    # The class rather than the URL: a configured URL can carry an
    # account-bearing query string, and `OllamaBackend` already reads as
    # "this machine".
    debug_log(
        f"🔒 private prompt for {model!r} → {type(backend).__name__}"
        f"{' (pinned)' if is_pinned else ''}",
        "llm",
    )
    return backend


def get_auxiliary_backend(settings: Any, model: Optional[str] = None) -> LLMBackend:
    """The backend for auxiliary classification and routing tasks.

    Delegates to :func:`get_llm_backend` with the requested model name so
    local tags are routed to local Ollama while names the provider serves
    (and everything on a local endpoint) stay on the configured chat
    provider.
    """
    return get_llm_backend(settings, model=model)



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
import time
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

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
_ANTHROPIC_KEY_PREFIX = "sk-ant-"

# Backends are cached by their resolved connection parameters so that a
# single instance — and its persistent HTTP session — is reused across
# the many ``get_llm_backend`` calls a single reply makes. Without this
# every call built a fresh backend with a cold connection and paid the
# TLS handshake (and the provider's cold-start) every time; on a remote
# cloud endpoint that dominated the latency of a multi-call reply.
_BACKEND_CACHE: dict = {}

# What local Ollama answered to ``/api/tags``, by its URL, with when.
# Only a private pin behind a local endpoint asks; a model pulled
# meanwhile, or Ollama started, is picked up within the TTL.
_OLLAMA_TAGS_CACHE: Dict[str, tuple] = {}
_OLLAMA_TAGS_TTL_SEC = 30.0
_OLLAMA_TAGS_TIMEOUT_SEC = 1.0


def clear_backend_cache() -> None:
    """Drop all cached backends, and Ollama's cached tag list.

    Call after a configuration reload so the next ``get_llm_backend``
    rebuilds against the new settings instead of returning a backend
    wired to the old endpoint/key. Also used by tests to keep instances
    isolated.
    """
    _BACKEND_CACHE.clear()
    _OLLAMA_TAGS_CACHE.clear()


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


def _key_for_host(api_key: Optional[str], base_url: str) -> Optional[str]:
    """``api_key``, unless it is an Anthropic key bound for another host.

    The key fields are shared by the remote providers, so a provider
    switch can leave the Anthropic key configured for an OpenAI-compatible
    server or an embedding endpoint. Anthropic keys carry a fixed prefix,
    so the key itself says where it belongs; Anthropic's own
    OpenAI-compatible endpoint still receives it.
    """
    if not api_key or not api_key.startswith(_ANTHROPIC_KEY_PREFIX):
        return api_key
    host = (urlparse(base_url if "//" in base_url else f"//{base_url}").hostname or "").lower()
    if host == urlparse(ANTHROPIC_BASE_URL).hostname:
        return api_key
    print("⚠️ The configured key is an Anthropic key; not sent to this provider", flush=True)
    print("   🔑 set the key of the provider in use, or switch llm_provider to anthropic", flush=True)
    return None


def _anthropic_key(api_key: Optional[str]) -> Optional[str]:
    """``api_key`` if it is an Anthropic key; another provider's key left in
    the shared field is not sent to Anthropic."""
    if not api_key or api_key.startswith(_ANTHROPIC_KEY_PREFIX):
        return api_key
    print("⚠️ The configured key is not an Anthropic key; not sent to Anthropic", flush=True)
    print("   🔑 set llm_api_key_env to the variable that holds your Anthropic key", flush=True)
    return None


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
        backend: LLMBackend = AnthropicBackend(api_key=_anthropic_key(api_key))
    else:
        backend = OpenAICompatibleBackend(
            base_url, api_key=_key_for_host(api_key, base_url), extra_body=extra_body
        )
    # Scrub secrets before remote egress. The Ollama path is not wrapped:
    # its prompts never leave the machine and redaction would be wasted work.
    if redact:
        backend = RedactingBackend(backend)
    return backend


def _chat_endpoint(settings: Any) -> Tuple[str, str]:
    """The configured provider, and the URL its chat requests go to."""
    provider = _resolve_provider(getattr(settings, "llm_provider", None))
    if provider == _OLLAMA:
        return provider, _str_attr(settings, "ollama_base_url", _DEFAULT_OLLAMA_URL)
    if provider == _ANTHROPIC:
        return provider, ANTHROPIC_BASE_URL
    return provider, _str_attr(settings, "llm_base_url") or _str_attr(
        settings, "ollama_base_url", _DEFAULT_OLLAMA_URL
    )


def _on_a_remote_provider(settings: Any) -> bool:
    """Whether the provider is not Ollama and its endpoint is off this
    machine and its network, as ``is_local_endpoint`` reads the URL."""
    provider, base_url = _chat_endpoint(settings)
    return provider != _OLLAMA and not is_local_endpoint(base_url)


def chat_is_remote(settings: Any) -> bool:
    """Whether the reply goes to a provider off home.

    The reading :func:`get_llm_backend` makes for the reply, which passes
    no model (:func:`_on_a_remote_provider`): Anthropic, or an
    OpenAI-compatible server at an address ``is_local_endpoint`` does not
    recognise. The Ollama provider counts as home whatever its URL, as it
    does for :func:`get_private_backend`. A prompt carrying only what the
    reply sends there anyway can go there too without anything new
    leaving.
    """
    return _on_a_remote_provider(settings)


def _names_the_providers_chat_model(settings: Any, model: str) -> bool:
    """Whether ``model`` is the chat model, named for the provider.

    The chat model is the one name known to exist on the provider,
    whatever :func:`names_provider_model` makes of its spelling:
    api.openai.com, DeepSeek and Groq spell theirs without a slash. Except
    the Ollama chat model, which an empty ``llm_chat_model`` inherits:
    that name is a local tag whatever field carries it.
    """
    return model == _str_attr(settings, "llm_chat_model") and model != _str_attr(
        settings, "ollama_chat_model"
    )


def _lives_on_ollama(settings: Any, model: str) -> bool:
    """Whether ``model``, named on a provider other than Ollama, lives on
    local Ollama instead.

    Behind a remote endpoint that is a local tag, a name the provider does
    not serve (:func:`names_provider_model`). Behind a local one it is a
    tag local Ollama lists: the configured server may be LM Studio beside
    Ollama, or a gateway that forwards elsewhere. The provider's own chat
    model never is.
    """
    if _names_the_providers_chat_model(settings, model):
        return False
    if _on_a_remote_provider(settings):
        return not names_provider_model(_chat_endpoint(settings)[0], model)
    return _ollama_has(settings, model)


def get_llm_backend(settings: Any, model: Optional[str] = None) -> LLMBackend:
    """Return the configured chat backend.

    ``llm_base_url`` is the OpenAI-compatible server's URL; the Ollama path
    uses ``ollama_base_url`` and the Anthropic path its fixed endpoint.
    Keeping each provider on its own URL means toggling ``llm_provider``
    can never leave the backend pointed at a stale OpenAI-compatible URL.

    When ``model`` lives on local Ollama rather than on the configured
    server (:func:`_lives_on_ollama`), the request goes there: a local tag
    behind a remote endpoint, which the remote endpoint does not have, and
    behind a local endpoint a tag Ollama lists. Auxiliary classification
    and routing passes are what name such models.

    The provider's own chat model is the exception: it stays on the
    provider however it is spelled, so a call that lands on it, by default
    or because ``_cloud_safe_model`` substituted it, reaches the one
    server known to have it.
    """
    model_str = (model or "").strip()
    provider, base_url = _chat_endpoint(settings)
    if model_str and provider != _OLLAMA and _lives_on_ollama(settings, model_str):
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


def _ollama_has(settings: Any, tag: str) -> bool:
    """Whether local Ollama answers and lists ``tag``.

    Ollama runs an untagged name as ``name:latest``, so that counts too.
    Nothing answering is False, as is anything else going wrong.
    """
    url = _str_attr(settings, "ollama_base_url", _DEFAULT_OLLAMA_URL)
    now = time.monotonic()
    cached = _OLLAMA_TAGS_CACHE.get(url)
    if cached is not None and now - cached[0] < _OLLAMA_TAGS_TTL_SEC:
        names = cached[1]
    else:
        names = frozenset(
            get_llm_backend(_SurLaMachine(settings)).list_models(
                timeout_sec=_OLLAMA_TAGS_TIMEOUT_SEC
            )
        )
        _OLLAMA_TAGS_CACHE[url] = (now, names)
        debug_log(f"🦙 local Ollama lists {len(names)} model(s)", "llm")
    return tag in names or (":" not in tag and f"{tag}:latest" in names)


def _pin_runs_on_ollama(settings: Any, pin: str) -> bool:
    """Whether a private pin runs on local Ollama rather than the
    configured server. On the Ollama provider the two are the same.

    Stricter than :func:`_lives_on_ollama`: a pin Ollama lists runs there
    even when it is also the chat model's name, so a gateway at a local
    address never receives it.
    """
    if _chat_endpoint(settings)[0] == _OLLAMA:
        return True
    return _on_a_remote_provider(settings) or _ollama_has(settings, pin)


def get_private_backend(settings: Any, pinned: str, model: str) -> LLMBackend:
    """The backend for a prompt carrying the user's own life.

    Four contexts read his own sentences: the goal he is working towards
    and the fortnight of diary the learning step mines, wherever the chat
    runs, and the time he said out loud and the routine he described when
    the chat is at home (with the chat remote, those two are read where
    the chat model lives; see :func:`chat_is_remote`). Each specifies that
    pinning a model is how he keeps that sentence off the network here.

    A name alone does not do that. Called without a model,
    :func:`get_llm_backend` picks the endpoint from ``llm_provider``
    alone, which on a cloud provider sends a pinned local tag, and the
    sentence with it, to the cloud.

    So a pin decides the destination too. Behind a remote provider it runs
    on local Ollama whatever its shape: an ``hf.co/...`` Ollama tag has a
    slash and is still his machine. Behind an endpoint on his own machine
    or network it runs on local Ollama when Ollama lists that tag, and on
    the configured server otherwise (LM Studio, ``llama-server``), where
    its ID lives. A local address is not proof of a local model: a gateway
    or a tunnel there forwards wherever it is set to, so a tag Ollama has
    never goes to it.

    With nothing pinned, ``model`` (the one the context resolved and is
    about to send) decides it the way it does for every other call
    (:func:`_lives_on_ollama`): a local tag on a remote provider names a
    model that lives on local Ollama, and goes there, as does a tag
    Ollama lists behind a local endpoint. A name the provider serves
    stays on the provider he chose: its own chat model whatever its
    spelling, a router given as ``vendor/model`` or as a Claude ID, and on
    a remote OpenAI-compatible endpoint a slashed local tag too, which
    only a pin keeps here.

    ``pinned`` and ``model`` are separate because a context's chain runs
    past its pin: the model it sends may be a fallback's, and the request
    has to go where that model lives.
    """
    pin = (pinned or "").strip()
    if pin and _pin_runs_on_ollama(settings, pin):
        backend = get_llm_backend(_SurLaMachine(settings))
    else:
        backend = get_llm_backend(settings, model=model)
    # The class and a word rather than the URL: a configured URL can carry
    # an account-bearing query string. `OllamaBackend` is this machine;
    # any other backend is local or remote as its endpoint is.
    ici = isinstance(backend, OllamaBackend) or not _on_a_remote_provider(settings)
    debug_log(
        f"🔒 private prompt for {model!r} → {type(backend).__name__}, "
        f"{'local' if ici else 'remote'}{' (pinned)' if pin else ''}",
        "llm",
    )
    return backend


def get_auxiliary_backend(settings: Any, model: Optional[str] = None) -> LLMBackend:
    """The backend for auxiliary classification and routing tasks.

    Delegates to :func:`get_llm_backend` with the requested model name so
    a model that lives on local Ollama runs there (a local tag behind a
    remote endpoint, a tag Ollama lists behind a local one) while names
    the provider serves, its own chat model however it is spelled, and
    anything else behind a local endpoint stay on the configured chat
    provider.
    """
    return get_llm_backend(settings, model=model)

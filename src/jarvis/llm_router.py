"""Hybrid LLM router.

Routes LLM calls between local Ollama (default, privacy-preserving) and
optional cloud providers (Anthropic). Mode is opt-in via config; the
default ``local_only`` mode bypasses the router entirely so users who
stay 100% local pay zero overhead.

Public surface
--------------
- ``Intent`` enum and ``RouterDecision`` dataclass.
- ``classify_intent(prompt, recent_context, cfg)`` returns a ``RouterDecision``
  for the current query. Memoised via an LRU(256) keyed on prompt+context
  hash and classifier model name so the same query inside one conversation
  never pays the classification cost twice.
- ``route(prompt, context, cfg, call_kind)`` returns ``(provider_callable,
  model_name)``. The callable mirrors the signature of the corresponding
  ``llm.py`` function (``"direct"``, ``"streaming"``, ``"chat"``).
- ``redact_for_cloud(prompt)`` delegates to the project's existing
  redaction (``jarvis.utils.redact.redact``). The router never
  re-implements scrubbing rules; cloud payloads ride the same pipeline
  as on-disk writes.

Design contract
---------------
- ``local_only`` short-circuits before any classification. Zero LLM
  call, zero cache touch, zero provider import. This is the privacy
  promise.
- The classifier rides the *warm small model* chain via
  ``jarvis.reply.engine.resolve_tool_router_model`` (``tool_router_model``
  -> ``intent_judge_model`` -> ``ollama_chat_model``), so the project's
  default warm model is reused with no extra weight pull.
- Fail-open: on classifier error, route locally. The cloud path is
  always optional; a broken classifier must never block a reply.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Any, Callable, List, Optional, Tuple

from .debug import debug_log
from .utils.redact import redact


# ── Modes ───────────────────────────────────────────────────────────────
MODE_LOCAL_ONLY = "local_only"
MODE_HYBRID = "hybrid"
MODE_CLOUD_ONLY = "cloud_only"
_VALID_MODES = frozenset({MODE_LOCAL_ONLY, MODE_HYBRID, MODE_CLOUD_ONLY})


# ── Intents ─────────────────────────────────────────────────────────────
class Intent(str, Enum):
    CASUAL_CHAT = "casual_chat"
    SIMPLE_QUERY = "simple_query"
    CODE_COMPLEX = "code_complex"
    MULTI_STEP_REASONING = "multi_step_reasoning"
    TOOL_USE_CHAIN = "tool_use_chain"
    STOP_COMMAND = "stop_command"


_VALID_INTENT_VALUES = frozenset(i.value for i in Intent)


@dataclass(frozen=True)
class RouterDecision:
    provider: str  # "local" or "cloud"
    model: str
    reason: str  # short trace line, includes intent label
    intent_score: float  # 0.0..1.0, classifier confidence


# Default cloud-eligible intents when ``cfg.llm_router.cloud_intents`` is
# unset or empty. ``casual_chat`` and ``simple_query`` deliberately stay
# local: they're the cheap path the small warm model handles fine, and
# routing them to cloud would burn budget for no quality gain.
_DEFAULT_CLOUD_INTENTS = frozenset({
    Intent.CODE_COMPLEX.value,
    Intent.MULTI_STEP_REASONING.value,
    Intent.TOOL_USE_CHAIN.value,
})


_CLASSIFIER_SYSTEM_PROMPT = (
    "You are an intent classifier for a voice assistant. Read the user "
    "query and any recent dialogue, then label the request with EXACTLY "
    "one of these intents:\n"
    "- casual_chat: greetings, small-talk, persona questions, mood, jokes\n"
    "- simple_query: one short factual question (time, weather, definition, "
    "single lookup)\n"
    "- code_complex: anything involving code reading, writing, debugging, "
    "refactoring, architecture, or API design\n"
    "- multi_step_reasoning: tasks needing planning, option comparison, "
    "multi-hop logic, or trade-off analysis\n"
    "- tool_use_chain: queries that chain multiple tools (search then "
    "summarise, fetch then analyse, query then act)\n"
    "- stop_command: a literal stop, cancel, or abort instruction\n"
    "Output STRICTLY a single JSON object: "
    "{\"intent\": \"<one_of_the_six>\", \"confidence\": <0..1>}. "
    "No prose, no markdown, no extra fields."
)


# ── Helpers ─────────────────────────────────────────────────────────────


def redact_for_cloud(prompt: str) -> str:
    """Apply the project's auto-redaction before any cloud egress.

    Delegates to ``jarvis.utils.redact.redact`` so cloud-bound payloads
    share the exact same scrub rules as memory writes. Never inline
    bespoke patterns here.
    """
    if not prompt:
        return prompt
    return redact(prompt)


def _resolve_classifier_model(cfg: Any) -> str:
    """Pick the small warm model for classification.

    Reuses the canonical resolver chain from ``reply/engine.py`` so the
    router automatically follows whatever the rest of the project warms
    up: ``tool_router_model`` -> ``intent_judge_model`` -> ``ollama_chat_model``.
    Imported lazily to avoid a circular import at module load (engine
    imports plenty downstream).
    """
    try:
        from .reply.engine import resolve_tool_router_model
        return resolve_tool_router_model(cfg)
    except Exception:
        # Fall back to chat model directly if the resolver is unavailable
        # for any reason. The router must not crash on import quirks.
        return str(getattr(cfg, "ollama_chat_model", "") or "")


def _hash_prompt(prompt: str, context: Optional[List[str]]) -> str:
    """Stable hash of prompt + ordered context lines.

    The unit separator (\\x1e) prevents two adjacent strings from
    colliding with one longer string. SHA-256 is overkill for a 256-slot
    cache but cheap and avoids cross-key contamination.
    """
    h = hashlib.sha256()
    h.update((prompt or "").encode("utf-8", errors="ignore"))
    for line in context or []:
        h.update(b"\x1e")
        h.update((line or "").encode("utf-8", errors="ignore"))
    return h.hexdigest()


@lru_cache(maxsize=256)
def _cached_classify(prompt_hash: str, classifier_model: str, base_url: str, timeout_sec: float) -> Optional[str]:
    """LRU(256) classifier cache (membership + eviction bookkeeping).

    Keyed on ``(prompt_hash, classifier_model, base_url, timeout_sec)``
    so a config change naturally invalidates entries. The real JSON
    payload lives in ``_decision_store`` keyed on the same tuple; this
    decorator exists so the eviction policy is the standard LRU one,
    sharing a single 256-entry budget with the side store.
    """
    return None


def clear_decision_cache() -> None:
    """Reset the classifier LRU cache. Public for tests."""
    _cached_classify.cache_clear()
    _decision_store.clear()


# Side dict keeping the actual JSON payload for each cached key. We
# can't put the payload directly in lru_cache because the cache key
# is the prompt *hash*, and we need to reconstruct a RouterDecision
# without the prompt itself. The LRU above tracks recency for the
# same tuple, and ``_set_cached_decision_json`` keeps the side dict
# bounded to 256 entries in lock-step.
_decision_store: dict[Tuple[str, str, str, float], str] = {}


def _get_cached_decision_json(key: Tuple[str, str, str, float]) -> Optional[str]:
    return _decision_store.get(key)


def _set_cached_decision_json(key: Tuple[str, str, str, float], payload: str) -> None:
    _cached_classify(*key)  # touch LRU so eviction tracks this key too
    _decision_store[key] = payload
    if len(_decision_store) > 256:
        oldest = next(iter(_decision_store))
        if oldest != key:
            _decision_store.pop(oldest, None)


def _call_classifier_llm(
    prompt: str,
    context: Optional[List[str]],
    cfg: Any,
    classifier_model: str,
) -> Optional[dict]:
    """Run the small warm model to label the prompt.

    Returns the parsed JSON dict ``{"intent": ..., "confidence": ...}``
    on success, or ``None`` on any error (timeout, malformed JSON,
    unknown intent value). Separated from ``classify_intent`` so tests
    can replace just the LLM call without touching cache logic.
    """
    if not classifier_model:
        debug_log("router: no classifier model resolved, skipping", "router")
        return None

    base_url = str(getattr(cfg, "ollama_base_url", "") or "")
    timeout_sec = float(getattr(cfg, "llm_digest_timeout_sec", 8.0))
    if not base_url:
        debug_log("router: no ollama_base_url, skipping classification", "router")
        return None

    user_content_parts: list[str] = []
    if context:
        joined = "\n".join(str(c) for c in context if c)
        if joined:
            user_content_parts.append(f"Recent dialogue:\n{joined}")
    user_content_parts.append(f"Query: {prompt}")
    user_content = "\n\n".join(user_content_parts)

    try:
        from .llm import call_llm_direct
        raw = call_llm_direct(
            base_url=base_url,
            chat_model=classifier_model,
            system_prompt=_CLASSIFIER_SYSTEM_PROMPT,
            user_content=user_content,
            timeout_sec=timeout_sec,
            temperature=0.0,
        )
    except Exception as e:
        debug_log(f"router: classifier call failed - {e}", "router")
        return None

    if not raw:
        return None

    # Extract first balanced {...} - small models sometimes wrap the
    # JSON in commentary even when told not to.
    match = re.search(r"\{[^{}]*\}", raw)
    candidate = match.group(0) if match else raw.strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        debug_log(f"router: classifier returned non-JSON: {raw[:120]!r}", "router")
        return None

    if not isinstance(parsed, dict):
        return None
    intent_val = parsed.get("intent")
    if not isinstance(intent_val, str) or intent_val not in _VALID_INTENT_VALUES:
        debug_log(f"router: classifier returned invalid intent {intent_val!r}", "router")
        return None
    return parsed


def classify_intent(
    prompt: str,
    recent_context: Optional[List[str]],
    cfg: Any,
) -> RouterDecision:
    """Label ``prompt`` with one of the six ``Intent`` values.

    Cached on the (prompt+context, classifier model, base_url, timeout)
    tuple via an LRU(256). Two calls with the same query inside one
    conversation never pay the classifier twice.

    Fail-open: on any classifier error returns a ``casual_chat`` decision
    pointing at the local Ollama chat model, with ``intent_score=0.0``
    so callers can tell the label was not validated.
    """
    classifier_model = _resolve_classifier_model(cfg)
    base_url = str(getattr(cfg, "ollama_base_url", "") or "")
    timeout_sec = float(getattr(cfg, "llm_digest_timeout_sec", 8.0))

    prompt_hash = _hash_prompt(prompt, recent_context)
    cache_key = (prompt_hash, classifier_model, base_url, timeout_sec)

    cached = _get_cached_decision_json(cache_key)
    if cached is not None:
        try:
            payload = json.loads(cached)
            return RouterDecision(**payload)
        except Exception:
            _decision_store.pop(cache_key, None)

    parsed = _call_classifier_llm(prompt, recent_context, cfg, classifier_model)
    fallback_model = str(getattr(cfg, "ollama_chat_model", "") or classifier_model)

    if parsed is None:
        decision = RouterDecision(
            provider="local",
            model=fallback_model,
            reason=f"{Intent.CASUAL_CHAT.value} (fallback)",
            intent_score=0.0,
        )
    else:
        intent = parsed["intent"]
        try:
            confidence = float(parsed.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        decision = RouterDecision(
            provider="local",
            model=fallback_model,
            reason=intent,
            intent_score=confidence,
        )

    _set_cached_decision_json(
        cache_key,
        json.dumps({
            "provider": decision.provider,
            "model": decision.model,
            "reason": decision.reason,
            "intent_score": decision.intent_score,
        }),
    )
    return decision


# ── Routing ─────────────────────────────────────────────────────────────


def _get_cloud_intents(cfg: Any) -> frozenset[str]:
    router_cfg = getattr(cfg, "llm_router", None)
    if router_cfg is None:
        return _DEFAULT_CLOUD_INTENTS
    configured = getattr(router_cfg, "cloud_intents", None)
    if not configured:
        return _DEFAULT_CLOUD_INTENTS
    valid = {c for c in configured if c in _VALID_INTENT_VALUES}
    return frozenset(valid) if valid else _DEFAULT_CLOUD_INTENTS


def _get_router_mode(cfg: Any) -> str:
    router_cfg = getattr(cfg, "llm_router", None)
    if router_cfg is None:
        return MODE_LOCAL_ONLY
    mode = str(getattr(router_cfg, "mode", MODE_LOCAL_ONLY) or MODE_LOCAL_ONLY).lower()
    return mode if mode in _VALID_MODES else MODE_LOCAL_ONLY


def _get_cloud_model(cfg: Any) -> str:
    router_cfg = getattr(cfg, "llm_router", None)
    if router_cfg is None:
        return "claude-sonnet-4-6"
    return str(getattr(router_cfg, "anthropic_model", "claude-sonnet-4-6") or "claude-sonnet-4-6")


def _get_local_callable(call_kind: str) -> Callable[..., Any]:
    """Lazy lookup for the local Ollama function for ``call_kind``."""
    from . import llm as _llm
    mapping = {
        "direct": getattr(_llm, "_call_llm_direct_local", None) or _llm.call_llm_direct,
        "streaming": getattr(_llm, "_call_llm_streaming_local", None) or _llm.call_llm_streaming,
        "chat": getattr(_llm, "_chat_with_messages_local", None) or _llm.chat_with_messages,
    }
    if call_kind not in mapping:
        raise ValueError(f"unknown call_kind: {call_kind!r}")
    return mapping[call_kind]


def _get_cloud_callable(call_kind: str) -> Callable[..., Any]:
    """Resolve the Anthropic-backed callable for ``call_kind``.

    Imported lazily so ``local_only`` users never import the cloud
    provider (privacy contract).
    """
    from .providers import anthropic_provider as _cloud
    mapping = {
        "direct": _cloud.call_direct,
        "streaming": _cloud.call_streaming,
        "chat": _cloud.chat_with_messages,
    }
    if call_kind not in mapping:
        raise ValueError(f"unknown call_kind: {call_kind!r}")
    return mapping[call_kind]


def route(
    prompt: str,
    context: Optional[List[str]],
    cfg: Any,
    call_kind: str = "chat",
) -> Tuple[Callable[..., Any], str]:
    """Pick the LLM backend for this call.

    Returns ``(callable, model_name)``. The callable mirrors the
    signature of the underlying ``llm.py`` function for ``call_kind``
    so the caller can dispatch transparently. ``model_name`` is the
    *resolved* model (Anthropic ID for cloud, Ollama tag for local).

    Mode contract:
    - ``local_only`` (default): zero classifier call. Returns local
      callable + ``cfg.ollama_chat_model``. This is the privacy promise.
    - ``cloud_only``: zero classifier call. Returns cloud callable +
      ``cfg.llm_router.anthropic_model``.
    - ``hybrid``: classifies, then maps intent to provider.
    """
    mode = _get_router_mode(cfg)
    local_model = str(getattr(cfg, "ollama_chat_model", "") or "")

    if mode == MODE_LOCAL_ONLY:
        return _get_local_callable(call_kind), local_model

    if mode == MODE_CLOUD_ONLY:
        return _get_cloud_callable(call_kind), _get_cloud_model(cfg)

    # Hybrid: classify, then map intent to provider.
    decision = classify_intent(prompt, context, cfg)
    cloud_intents = _get_cloud_intents(cfg)
    intent_label = decision.reason.split(" ", 1)[0]
    if intent_label in cloud_intents:
        debug_log(
            f"router: hybrid -> cloud (intent={intent_label}, score={decision.intent_score:.2f})",
            "router",
        )
        return _get_cloud_callable(call_kind), _get_cloud_model(cfg)
    debug_log(
        f"router: hybrid -> local (intent={intent_label}, score={decision.intent_score:.2f})",
        "router",
    )
    return _get_local_callable(call_kind), local_model

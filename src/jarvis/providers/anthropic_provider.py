"""Anthropic provider.

Mirrors the three public functions of ``jarvis.llm`` so the hybrid router
can substitute Anthropic for Ollama transparently. The Ollama-shaped
response is reconstructed at the boundary (Anthropic returns content
blocks; Ollama returns ``message.content`` + ``message.tool_calls``)
so the agentic loop in ``reply/engine.py`` consumes either provider
without branching.

Resilience contract
-------------------
- The Anthropic SDK auto-retries 429 and 5xx with exponential backoff,
  default ``max_retries=2`` — that satisfies the spec's "max 2 retries
  exponentiels". We do not layer a second retry loop on top.
- On failure (after the SDK retries are exhausted, the API key is
  absent, or the SDK is not installed), the provider falls back to the
  local Ollama function when ``cfg.llm_router.fallback_to_local_on_error``
  is true. Otherwise it returns ``None``.
- The path actually taken is recorded in a thread-local that the
  integration layer reads when writing telemetry. See
  ``last_provider_used()``.

Redaction
---------
- Cloud-bound payloads are scrubbed via ``redact_for_cloud`` when
  ``cfg.llm_router.auto_redact_before_cloud`` is true (default). The
  caller can disable redaction for advanced use cases by setting that
  flag false. Local fallback re-uses the original (un-redacted) prompt
  so on-device replies retain full fidelity.

Prompt caching
--------------
- The Anthropic provider applies an ``ephemeral`` cache_control marker
  on the system prompt when its length exceeds
  ``cfg.llm_router.anthropic_cache_threshold_chars`` (default 8000).
  The threshold is exposed via config so users can tune it per model
  (Sonnet 4.6 caches above 2048 tokens; longer thresholds avoid
  wasting cache_control budget on prompts the API won't cache).
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Callable, Dict, List, Optional

from ..debug import debug_log
from ..llm_router import redact_for_cloud


# Default cloud model. The user explicitly chose Sonnet 4.6 for this
# project; the claude-api skill defaults to Opus 4.7 globally but the
# spec overrides for this repo.
DEFAULT_CLOUD_MODEL = "claude-sonnet-4-6"

# Reasonable per-call ceiling. Cloud responses can be much longer than
# small-model Ollama replies; we cap at 4096 to keep voice latency sane.
_DEFAULT_MAX_TOKENS = 4096

# Default cache threshold (chars) if cfg doesn't carry one. Mirrors the
# default in LLMRouterSettings.anthropic_cache_threshold_chars.
_DEFAULT_CACHE_THRESHOLD_CHARS = 8000


# Thread-local that records which path the most recent provider call
# actually used: "cloud" if Anthropic answered, "local_fallback" if
# we degraded to Ollama after a cloud failure. The integration layer
# reads this when writing telemetry so the local-fallback rows are
# distinguishable from native local calls.
_path_local = threading.local()


def last_provider_used() -> Optional[str]:
    """Return the path used by the most recent provider call on this thread.

    One of: ``"cloud"``, ``"local_fallback"``, or ``None`` if no call
    has run on this thread yet.
    """
    return getattr(_path_local, "value", None)


def _set_path(value: str) -> None:
    _path_local.value = value


# ── SDK loader ──────────────────────────────────────────────────────────


_sdk_lock = threading.Lock()
_sdk_cache: Dict[str, Any] = {}


def _load_sdk():
    """Import the ``anthropic`` SDK lazily.

    Cached so we don't pay the import cost on every call. Returns the
    module on success or ``None`` when the SDK is not installed. The
    router's local_only mode never reaches this code, so a 100%-local
    install can run without ``anthropic`` on the dependency list — the
    line in ``requirements.txt`` only matters for users who opt into
    hybrid mode.
    """
    with _sdk_lock:
        if "module" in _sdk_cache:
            return _sdk_cache["module"]
        try:
            import anthropic  # type: ignore
            _sdk_cache["module"] = anthropic
            return anthropic
        except Exception as e:
            debug_log(f"anthropic SDK not importable: {e}", "router")
            _sdk_cache["module"] = None
            return None


def _build_client(cfg: Any, timeout_sec: float):
    """Build a fresh Anthropic client with the configured timeout.

    Returns ``None`` when the API key env var is unset or empty; this
    is the trigger for the local fallback path.
    """
    sdk = _load_sdk()
    if sdk is None:
        return None
    env_name = "ANTHROPIC_API_KEY"
    router_cfg = getattr(cfg, "llm_router", None)
    if router_cfg is not None:
        env_name = getattr(router_cfg, "anthropic_api_key_env", env_name) or env_name
    api_key = os.environ.get(env_name, "").strip()
    if not api_key:
        debug_log(f"anthropic: env var {env_name} unset or empty", "router")
        return None
    try:
        return sdk.Anthropic(api_key=api_key, timeout=timeout_sec, max_retries=2)
    except Exception as e:
        debug_log(f"anthropic: client construction failed - {e}", "router")
        return None


def _redaction_enabled(cfg: Any) -> bool:
    router_cfg = getattr(cfg, "llm_router", None)
    if router_cfg is None:
        return True
    return bool(getattr(router_cfg, "auto_redact_before_cloud", True))


def _fallback_enabled(cfg: Any) -> bool:
    router_cfg = getattr(cfg, "llm_router", None)
    if router_cfg is None:
        return True
    return bool(getattr(router_cfg, "fallback_to_local_on_error", True))


def _cache_threshold(cfg: Any) -> int:
    """Read the user-tunable threshold for prompt-cache markers."""
    router_cfg = getattr(cfg, "llm_router", None)
    if router_cfg is None:
        return _DEFAULT_CACHE_THRESHOLD_CHARS
    try:
        value = int(getattr(router_cfg, "anthropic_cache_threshold_chars",
                             _DEFAULT_CACHE_THRESHOLD_CHARS))
    except (TypeError, ValueError):
        return _DEFAULT_CACHE_THRESHOLD_CHARS
    return max(0, value)


# ── Message / tool shape translation ────────────────────────────────────


def _split_system_from_messages(messages: List[Dict[str, Any]]) -> tuple[str, List[Dict[str, Any]]]:
    """Pull leading ``system`` messages into a single string.

    Anthropic accepts a top-level ``system`` parameter and rejects
    ``role: "system"`` inside ``messages``. Ollama prompts often start
    with one system message followed by user/assistant turns; we
    concatenate every ``system``-role message we find (preserving order)
    and strip them from the returned message list.
    """
    system_parts: List[str] = []
    rest: List[Dict[str, Any]] = []
    for m in messages or []:
        role = (m.get("role") or "").lower()
        content = m.get("content")
        if role == "system" and isinstance(content, str):
            system_parts.append(content)
        else:
            rest.append(m)
    return "\n\n".join(system_parts), rest


def _translate_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Translate Ollama-style chat messages to Anthropic shape.

    Ollama uses ``{role: "user"|"assistant"|"tool", content: str}`` with
    optional ``tool_calls`` on assistant turns. Anthropic uses
    user/assistant turns with content blocks; tool results live inside
    user turns as ``tool_result`` blocks. The translation walks the
    list and emits the equivalent block shape.
    """
    out: List[Dict[str, Any]] = []
    for m in messages:
        role = (m.get("role") or "").lower()
        content = m.get("content")
        if role == "user":
            if isinstance(content, str):
                out.append({"role": "user", "content": content})
            else:
                out.append({"role": "user", "content": str(content or "")})
        elif role == "assistant":
            blocks: List[Dict[str, Any]] = []
            if isinstance(content, str) and content.strip():
                blocks.append({"type": "text", "text": content})
            tool_calls = m.get("tool_calls") or []
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") or {}
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args_obj = json.loads(args) if args else {}
                    except json.JSONDecodeError:
                        args_obj = {}
                elif isinstance(args, dict):
                    args_obj = args
                else:
                    args_obj = {}
                blocks.append({
                    "type": "tool_use",
                    "id": tc.get("id") or "call_unknown",
                    "name": fn.get("name") or "",
                    "input": args_obj,
                })
            if blocks:
                out.append({"role": "assistant", "content": blocks})
        elif role == "tool":
            # Ollama tool-role message carries a single tool_result.
            out.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": m.get("tool_call_id") or m.get("id") or "call_unknown",
                    "content": content if isinstance(content, str) else json.dumps(content),
                }],
            })
        # Silently drop unknown roles (defensive; same policy as Ollama).
    return out


def _translate_tools(tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    """Translate Ollama OpenAI-style tool definitions to Anthropic shape.

    Ollama: ``{"type": "function", "function": {"name", "description", "parameters"}}``
    Anthropic: ``{"name", "description", "input_schema"}``
    """
    if not tools:
        return None
    out: List[Dict[str, Any]] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        if "function" in t and isinstance(t["function"], dict):
            fn = t["function"]
            out.append({
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            })
        elif "name" in t:
            out.append({
                "name": t.get("name", ""),
                "description": t.get("description", ""),
                "input_schema": t.get("input_schema", t.get("parameters", {"type": "object", "properties": {}})),
            })
    return out or None


def _response_to_ollama_shape(msg: Any, chat_model: str) -> Dict[str, Any]:
    """Map an Anthropic ``Message`` object back to Ollama's response shape.

    The agentic loop in ``reply/engine.py`` reads ``message.content``
    and ``message.tool_calls`` (Ollama field names). We assemble that
    shape from Anthropic's content blocks so the caller is agnostic to
    which provider answered.
    """
    text_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    blocks = getattr(msg, "content", None) or []
    for block in blocks:
        btype = getattr(block, "type", None)
        if btype == "text":
            t = getattr(block, "text", "")
            if isinstance(t, str):
                text_parts.append(t)
        elif btype == "tool_use":
            args = getattr(block, "input", {}) or {}
            try:
                args_json = json.dumps(args)
            except (TypeError, ValueError):
                args_json = "{}"
            tool_calls.append({
                "id": getattr(block, "id", "call_unknown"),
                "type": "function",
                "function": {
                    "name": getattr(block, "name", ""),
                    "arguments": args_json,
                },
            })
    message_dict: Dict[str, Any] = {
        "role": "assistant",
        "content": "".join(text_parts),
    }
    if tool_calls:
        message_dict["tool_calls"] = tool_calls
    return {
        "model": chat_model,
        "message": message_dict,
        "done": True,
        "stop_reason": getattr(msg, "stop_reason", None),
        "usage": _usage_to_dict(getattr(msg, "usage", None)),
    }


def _usage_to_dict(usage: Any) -> Dict[str, int]:
    if usage is None:
        return {}
    out: Dict[str, int] = {}
    for key in ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"):
        v = getattr(usage, key, None)
        if isinstance(v, int):
            out[key] = v
    return out


# ── Local fallback dispatch ─────────────────────────────────────────────


def _local_callable(name: str) -> Callable[..., Any]:
    """Return the in-process Ollama function for fallback.

    Imported lazily to avoid pulling ``llm`` at module import time
    (``llm`` imports the router back, which would loop).
    """
    from .. import llm as _llm
    mapping = {
        "direct": getattr(_llm, "_call_llm_direct_local", None) or _llm.call_llm_direct,
        "streaming": getattr(_llm, "_call_llm_streaming_local", None) or _llm.call_llm_streaming,
        "chat": getattr(_llm, "_chat_with_messages_local", None) or _llm.chat_with_messages,
    }
    return mapping[name]


# ── Public API ──────────────────────────────────────────────────────────


def call_direct(
    base_url: str,
    chat_model: str,
    system_prompt: str,
    user_content: str,
    timeout_sec: float = 10.0,
    thinking: bool = False,
    num_ctx: int = 4096,
    temperature: Optional[float] = None,
    cfg: Any = None,
) -> Optional[str]:
    """Anthropic equivalent of ``llm.call_llm_direct``.

    Signature mirrors the Ollama function so the router can swap them
    transparently. ``base_url`` and ``num_ctx`` are accepted for shape
    compatibility but ignored (the Anthropic SDK manages its own
    endpoint and the model's context window is fixed). On cloud
    failure, falls back to the local Ollama function when configured.
    """
    client = _build_client(cfg, timeout_sec)
    if client is None:
        if _fallback_enabled(cfg):
            debug_log("anthropic.call_direct: no client, falling back to local", "router")
            _set_path("local_fallback")
            return _local_callable("direct")(
                base_url, chat_model, system_prompt, user_content,
                timeout_sec=timeout_sec, thinking=thinking, num_ctx=num_ctx,
                temperature=temperature,
            )
        _set_path("cloud")
        return None

    prompt = redact_for_cloud(user_content) if _redaction_enabled(cfg) else user_content
    system_blocks = _build_system_blocks(system_prompt, cfg)

    try:
        message = client.messages.create(
            model=chat_model,
            max_tokens=_DEFAULT_MAX_TOKENS,
            system=system_blocks,
            messages=[{"role": "user", "content": prompt}],
            **_optional_temperature(temperature, chat_model),
        )
    except Exception as e:
        debug_log(f"anthropic.call_direct failed: {type(e).__name__}: {e}", "router")
        if _fallback_enabled(cfg):
            _set_path("local_fallback")
            return _local_callable("direct")(
                base_url, chat_model, system_prompt, user_content,
                timeout_sec=timeout_sec, thinking=thinking, num_ctx=num_ctx,
                temperature=temperature,
            )
        _set_path("cloud")
        return None

    _set_path("cloud")
    parts: List[str] = []
    for block in getattr(message, "content", []) or []:
        if getattr(block, "type", None) == "text":
            t = getattr(block, "text", "")
            if isinstance(t, str):
                parts.append(t)
    return "".join(parts).strip() or None


def call_streaming(
    base_url: str,
    chat_model: str,
    system_prompt: str,
    user_content: str,
    on_token: Optional[Callable[[str], None]] = None,
    timeout_sec: float = 30.0,
    thinking: bool = False,
    cfg: Any = None,
) -> Optional[str]:
    """Anthropic equivalent of ``llm.call_llm_streaming``."""
    client = _build_client(cfg, timeout_sec)
    if client is None:
        if _fallback_enabled(cfg):
            _set_path("local_fallback")
            return _local_callable("streaming")(
                base_url, chat_model, system_prompt, user_content,
                on_token=on_token, timeout_sec=timeout_sec, thinking=thinking,
            )
        _set_path("cloud")
        return None

    prompt = redact_for_cloud(user_content) if _redaction_enabled(cfg) else user_content
    system_blocks = _build_system_blocks(system_prompt, cfg)

    try:
        collected: List[str] = []
        with client.messages.stream(
            model=chat_model,
            max_tokens=_DEFAULT_MAX_TOKENS,
            system=system_blocks,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for chunk in stream.text_stream:
                if not chunk:
                    continue
                collected.append(chunk)
                if on_token:
                    try:
                        on_token(chunk)
                    except Exception as cb_err:
                        debug_log(f"on_token callback raised: {cb_err}", "router")
        _set_path("cloud")
        result = "".join(collected)
        return result if result.strip() else None
    except Exception as e:
        debug_log(f"anthropic.call_streaming failed: {type(e).__name__}: {e}", "router")
        if _fallback_enabled(cfg):
            _set_path("local_fallback")
            return _local_callable("streaming")(
                base_url, chat_model, system_prompt, user_content,
                on_token=on_token, timeout_sec=timeout_sec, thinking=thinking,
            )
        _set_path("cloud")
        return None


def chat_with_messages(
    base_url: str,
    chat_model: str,
    messages: List[Dict[str, Any]],
    timeout_sec: float = 30.0,
    extra_options: Optional[Dict[str, Any]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    thinking: bool = False,
    cfg: Any = None,
) -> Optional[Dict[str, Any]]:
    """Anthropic equivalent of ``llm.chat_with_messages``.

    Returns an Ollama-shaped dict ({"message": {"content", "tool_calls"}, ...})
    so the agentic loop in ``reply/engine.py`` consumes it without
    branching on provider.
    """
    client = _build_client(cfg, timeout_sec)
    if client is None:
        if _fallback_enabled(cfg):
            _set_path("local_fallback")
            return _local_callable("chat")(
                base_url, chat_model, messages,
                timeout_sec=timeout_sec, extra_options=extra_options, tools=tools,
                thinking=thinking,
            )
        _set_path("cloud")
        return None

    system_str, rest = _split_system_from_messages(messages or [])
    if _redaction_enabled(cfg):
        rest = _redact_messages(rest)
        if system_str:
            system_str = redact_for_cloud(system_str)
    a_messages = _translate_messages(rest)
    a_tools = _translate_tools(tools)
    system_blocks = _build_system_blocks(system_str, cfg)

    kwargs: Dict[str, Any] = {
        "model": chat_model,
        "max_tokens": _DEFAULT_MAX_TOKENS,
        "messages": a_messages,
    }
    if system_blocks:
        kwargs["system"] = system_blocks
    if a_tools:
        kwargs["tools"] = a_tools

    try:
        message = client.messages.create(**kwargs)
    except Exception as e:
        debug_log(f"anthropic.chat_with_messages failed: {type(e).__name__}: {e}", "router")
        if _fallback_enabled(cfg):
            _set_path("local_fallback")
            return _local_callable("chat")(
                base_url, chat_model, messages,
                timeout_sec=timeout_sec, extra_options=extra_options, tools=tools,
                thinking=thinking,
            )
        _set_path("cloud")
        return None

    _set_path("cloud")
    return _response_to_ollama_shape(message, chat_model)


# ── Helpers ─────────────────────────────────────────────────────────────


def _build_system_blocks(system_prompt: str, cfg: Any) -> Any:
    """Render the system prompt with optional prompt-cache markers.

    The project's system prompt is stable across turns (cf.
    ``system_prompt.py``) so an ``ephemeral`` cache_control on it saves
    repeated input tokens at ~90% discount when the same prompt is
    reused. Below the configured threshold we return a plain string —
    Anthropic's minimum cacheable prefix is 2048 tokens on Sonnet 4.6,
    so marking shorter prompts is wasted overhead.
    """
    if not system_prompt:
        return ""
    threshold = _cache_threshold(cfg)
    if len(system_prompt) < threshold:
        return system_prompt
    return [{
        "type": "text",
        "text": system_prompt,
        "cache_control": {"type": "ephemeral"},
    }]


def _redact_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Apply ``redact_for_cloud`` to every textual content in the list.

    Walks both string-shaped ``content`` and block-shaped content (only
    redacting ``text`` blocks; ``tool_result`` payloads are passed
    through, because they originate from tools, not user input — and
    the project's tools own their own redaction at write time).
    """
    out: List[Dict[str, Any]] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        c = m.get("content")
        if isinstance(c, str):
            m = {**m, "content": redact_for_cloud(c)}
        elif isinstance(c, list):
            new_blocks = []
            for b in c:
                if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str):
                    new_blocks.append({**b, "text": redact_for_cloud(b["text"])})
                else:
                    new_blocks.append(b)
            m = {**m, "content": new_blocks}
        out.append(m)
    return out


def _optional_temperature(temperature: Optional[float], model: str) -> Dict[str, Any]:
    """Forward ``temperature`` only when set and supported.

    Opus 4.7 removed ``temperature``/``top_p``/``top_k`` (sending any
    returns HTTP 400). For other Claude 4 models, ``temperature`` is
    accepted. We pass it through except on Opus 4.7+.
    """
    if temperature is None:
        return {}
    if model.startswith("claude-opus-4-7"):
        return {}
    return {"temperature": float(temperature)}

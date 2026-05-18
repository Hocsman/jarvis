"""Direct LLM interaction utilities without extra features like temporal context."""

from __future__ import annotations
from typing import Optional, Any, Dict, List, Generator, Callable
import requests
import json

from .debug import debug_log


class ToolsNotSupportedError(Exception):
    """Raised when the model returns HTTP 400 because native tool calling is not supported."""
    pass


def _call_llm_direct_local(base_url: str, chat_model: str, system_prompt: str, user_content: str, timeout_sec: float = 10.0, thinking: bool = False, num_ctx: int = 4096, temperature: Optional[float] = None) -> Optional[str]:
    """Direct LLM call without temporal context, location, or other ask_coach features.

    ``num_ctx`` controls Ollama's context window for this call. Default 4096 is
    fine for small classification-shaped passes; callers that assemble richer
    prompts (planner with dialogue + memory + tool catalogue) should pass a
    larger value to avoid silent truncation.

    ``temperature`` is forwarded to Ollama when set. Pass ``0.0`` for
    classification / extraction calls where determinism beats creativity —
    Ollama defaults to ~0.8 otherwise, which can flake small models on
    rule-following tasks (e.g. the knowledge extractor's banned-form list).
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    options: Dict[str, Any] = {"num_ctx": num_ctx}
    if temperature is not None:
        options["temperature"] = temperature

    payload: Dict[str, Any] = {
        "model": chat_model,
        "messages": messages,
        "stream": False,
        "options": options,
        "think": thinking,
    }
    
    try:
        with requests.post(f"{base_url.rstrip('/')}/api/chat", json=payload, timeout=timeout_sec) as resp:
            resp.raise_for_status()
            data = resp.json()

        if isinstance(data, dict):
            content = extract_text_from_response(data)
            if isinstance(content, str) and content.strip():
                return content
            debug_log(f"call_llm_direct: empty content from response keys={list(data.keys())}", "llm")
    except requests.exceptions.Timeout:
        debug_log(f"call_llm_direct: timeout after {timeout_sec}s", "llm")
        return None
    except Exception as e:
        debug_log(f"call_llm_direct: request failed — {e}", "llm")
        return None

    return None


def _call_llm_streaming_local(
    base_url: str,
    chat_model: str,
    system_prompt: str,
    user_content: str,
    on_token: Optional[Callable[[str], None]] = None,
    timeout_sec: float = 30.0,
    thinking: bool = False,
) -> Optional[str]:
    """
    Streaming LLM call that invokes on_token callback for each token received.

    Args:
        base_url: Ollama base URL
        chat_model: Model name
        system_prompt: System prompt
        user_content: User message
        on_token: Callback invoked with each token as it arrives
        timeout_sec: Request timeout
        thinking: Enable thinking/reasoning mode

    Returns:
        Complete response text, or None on error
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    payload: Dict[str, Any] = {
        "model": chat_model,
        "messages": messages,
        "stream": True,
        "options": {"num_ctx": 4096},
        "think": thinking,
    }

    # Use ``with`` so the streaming response (and the underlying TCP
    # connection) is released even if iter_lines exits early via an
    # exception or the caller stops consuming. Without this an aborted
    # stream pinned the connection until GC, which could happen many
    # turns later under sustained reply load.
    try:
        with requests.post(
            f"{base_url.rstrip('/')}/api/chat",
            json=payload,
            timeout=timeout_sec,
            stream=True,
        ) as resp:
            resp.raise_for_status()

            full_response = []
            for line in resp.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if "message" in data and isinstance(data["message"], dict):
                            content = data["message"].get("content", "")
                            if content:
                                full_response.append(content)
                                if on_token:
                                    on_token(content)
                    except json.JSONDecodeError:
                        continue

            result = "".join(full_response)
            return result if result.strip() else None

    except requests.exceptions.Timeout:
        return None
    except Exception:
        return None


def extract_text_from_response(data: Dict[str, Any]) -> Optional[str]:
    """Extract text from LLM response - supports multiple response formats."""
    # Preferred: Ollama chat non-stream format
    if "message" in data and isinstance(data["message"], dict):
        content = data["message"].get("content")
        if isinstance(content, str):
            return content
    
    # Fallback: OpenAI-style format
    if "choices" in data and isinstance(data["choices"], list) and len(data["choices"]) > 0:
        choice = data["choices"][0]
        if isinstance(choice, dict):
            if "message" in choice and isinstance(choice["message"], dict):
                content = choice["message"].get("content")
                if isinstance(content, str):
                    return content
            elif "text" in choice:
                content = choice["text"]
                if isinstance(content, str):
                    return content
    
    # Another fallback: direct "content" field
    if "content" in data:
        content = data["content"]
        if isinstance(content, str):
            return content
    
    return None


def _chat_with_messages_local(
    base_url: str,
    chat_model: str,
    messages: List[Dict[str, str]],
    timeout_sec: float = 30.0,
    extra_options: Optional[Dict[str, Any]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    thinking: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Send an arbitrary messages array to the LLM and return the raw response JSON.
    Caller is responsible for interpreting assistant content (including JSON/tool calls).

    Args:
        base_url: Ollama base URL
        chat_model: Model name
        messages: Conversation messages
        timeout_sec: Request timeout
        extra_options: Additional model options
        tools: Optional list of tools in OpenAI-compatible JSON schema format for native tool calling
        thinking: Enable thinking/reasoning mode

    Returns the parsed JSON response dict on success, or None on error/timeout.
    """
    # Main agentic chat uses 8192 so the system prompt (tool list + protocol
    # guidance + memory context) doesn't overflow and force ollama to truncate
    # — which previously dropped the tool schema on smaller models like
    # gemma4:e2b, tipping them into their pre-trained tool_code scaffolding.
    payload: Dict[str, Any] = {
        "model": chat_model,
        "messages": messages,
        "stream": False,
        "options": {"num_ctx": 8192},
        "think": thinking,
    }
    if extra_options and isinstance(extra_options, dict):
        # Merge shallowly into options
        payload["options"].update(extra_options)

    # Add tools for native tool calling support (Ollama 0.4+)
    if tools and isinstance(tools, list) and len(tools) > 0:
        payload["tools"] = tools

    try:
        with requests.post(f"{base_url.rstrip('/')}/api/chat", json=payload, timeout=timeout_sec) as resp:
            resp.raise_for_status()
            data = resp.json()
        if isinstance(data, dict):
            return data
    except requests.exceptions.Timeout:
        print("  ⏱️ LLM request timed out", flush=True)
        return None
    except requests.exceptions.ConnectionError as e:
        print(f"  ❌ LLM connection error: {e}", flush=True)
        return None
    except requests.exceptions.HTTPError as e:
        # Raise a specific error when the model rejects the tools parameter (HTTP 400).
        # This lets the caller fall back to text-based tool calling automatically.
        if e.response is not None and e.response.status_code == 400 and tools:
            raise ToolsNotSupportedError(
                f"Model {chat_model!r} returned HTTP 400 — native tools API not supported"
            )
        print(f"  ❌ LLM HTTP error: {e}", flush=True)
        return None
    except Exception as e:
        print(f"  ❌ LLM error: {e}", flush=True)
        return None

    return None


# ────────────────────────────────────────────────────────────────────────
# Hybrid router integration
# ────────────────────────────────────────────────────────────────────────
#
# The three functions above (``_call_llm_direct_local``,
# ``_call_llm_streaming_local``, ``_chat_with_messages_local``) keep
# the original Ollama implementations verbatim. The public surface
# below (``call_llm_direct``, ``call_llm_streaming``,
# ``chat_with_messages``) wraps them and routes through the hybrid
# router. The privacy contract: when ``cfg.llm_router.mode ==
# "local_only"`` (the default), the wrapper short-circuits to the
# local implementation *before* any router code runs — zero overhead,
# zero classifier call, zero cloud import.


import time as _time


_cfg_lock: Any = None
_cfg_cache: Any = None
_cfg_test_override: Any = None


def _get_router_cfg() -> Any:
    """Return the cached ``Settings`` instance, or a test override.

    Reading ``load_settings()`` on every call would re-read config.json
    from disk each time; this lazy-cached singleton matches the
    daemon's startup-time load. Tests can inject a fake cfg via
    ``set_router_cfg_for_tests`` to avoid touching disk.
    """
    global _cfg_lock, _cfg_cache, _cfg_test_override
    if _cfg_test_override is not None:
        return _cfg_test_override
    if _cfg_lock is None:
        import threading as _threading
        _cfg_lock = _threading.Lock()
    if _cfg_cache is not None:
        return _cfg_cache
    with _cfg_lock:
        if _cfg_cache is None:
            try:
                from .config import load_settings as _load_settings
                _cfg_cache = _load_settings()
            except Exception as _e:
                debug_log(f"router cfg load failed (assuming local_only): {_e}", "router")
                _cfg_cache = None
    return _cfg_cache


def set_router_cfg_for_tests(cfg: Any) -> None:
    """Override the cached cfg in tests. Pass ``None`` to clear."""
    global _cfg_test_override
    _cfg_test_override = cfg


def _is_local_only(cfg: Any) -> bool:
    if cfg is None:
        return True
    router_cfg = getattr(cfg, "llm_router", None)
    if router_cfg is None:
        return True
    return str(getattr(router_cfg, "mode", "local_only") or "local_only").lower() == "local_only"


def _record_telemetry_safe(
    cfg: Any,
    *,
    provider: str,
    model: str,
    intent: Optional[str],
    response: Any,
    latency_ms: int,
) -> None:
    """Best-effort telemetry recording for non-local_only modes."""
    try:
        from . import llm_router_telemetry as _telem
    except Exception:
        return
    db_path = getattr(cfg, "db_path", None) if cfg is not None else None
    tokens_in, tokens_out, cache_create, cache_read = _extract_tokens(response)
    # Ollama and local-fallback paths are always billed at zero — the
    # pricing table only covers cloud models, and even when a
    # cloud-bound call degraded into local, the user paid nothing for
    # the bytes that came back from the local model.
    if provider in ("local", "local_fallback"):
        cost = 0.0
    else:
        cost = _telem.estimate_cost_usd(
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cache_creation_input_tokens=cache_create,
            cache_read_input_tokens=cache_read,
        )
    _telem.record(
        db_path,
        provider=provider,
        model=model,
        intent=intent,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_estimate_usd=cost,
        latency_ms=latency_ms,
    )


def _extract_tokens(response: Any) -> tuple[int, int, int, int]:
    """Return ``(tokens_in, tokens_out, cache_create, cache_read)``."""
    if isinstance(response, dict):
        usage = response.get("usage")
        if isinstance(usage, dict):
            return (
                int(usage.get("input_tokens") or 0),
                int(usage.get("output_tokens") or 0),
                int(usage.get("cache_creation_input_tokens") or 0),
                int(usage.get("cache_read_input_tokens") or 0),
            )
        return (
            int(response.get("prompt_eval_count") or 0),
            int(response.get("eval_count") or 0),
            0,
            0,
        )
    return (0, 0, 0, 0)


def _routed_call_kind_for(prompt: str, kind: str, cfg: Any) -> tuple[Any, str, str]:
    """Classify (when in hybrid mode) and return the chosen callable.

    Returns ``(callable, resolved_model, intent_label)``.
    """
    from . import llm_router as _r
    callable_, resolved_model = _r.route(prompt, None, cfg, call_kind=kind)
    intent_label = "n/a"
    mode = str(getattr(getattr(cfg, "llm_router", None), "mode", "local_only") or "local_only").lower()
    if mode == "hybrid":
        try:
            decision = _r.classify_intent(prompt, None, cfg)
            intent_label = (decision.reason or "").split(" ", 1)[0] or "n/a"
        except Exception:
            pass
    elif mode == "cloud_only":
        intent_label = "cloud_only_bypass"
    return callable_, resolved_model, intent_label


def _actual_provider_used(callable_: Any) -> str:
    """Map the chosen callable + provider state back to a telemetry label."""
    module = getattr(callable_, "__module__", "") or ""
    if "anthropic_provider" in module:
        try:
            from .providers.anthropic_provider import last_provider_used
            value = last_provider_used()
            return value or "cloud"
        except Exception:
            return "cloud"
    return "local"


def _cloud_usage_payload() -> Optional[Dict[str, Any]]:
    """Return the Anthropic provider's last usage block wrapped in an
    Ollama-shaped dict so ``_extract_tokens`` can read it. ``None`` when
    no cloud usage is available (local call, streaming/direct that
    fell back, or the call failed before ``_set_usage``)."""
    try:
        from .providers.anthropic_provider import last_usage as _last_usage
    except Exception:
        return None
    usage = _last_usage()
    if not isinstance(usage, dict) or not usage:
        return None
    return {"usage": usage}


def call_llm_direct(
    base_url: str,
    chat_model: str,
    system_prompt: str,
    user_content: str,
    timeout_sec: float = 10.0,
    thinking: bool = False,
    num_ctx: int = 4096,
    temperature: Optional[float] = None,
) -> Optional[str]:
    """Hybrid-routed direct LLM call. See ``_call_llm_direct_local``."""
    cfg = _get_router_cfg()
    if _is_local_only(cfg):
        return _call_llm_direct_local(
            base_url, chat_model, system_prompt, user_content,
            timeout_sec=timeout_sec, thinking=thinking, num_ctx=num_ctx,
            temperature=temperature,
        )
    callable_, resolved_model, intent = _routed_call_kind_for(user_content, "direct", cfg)
    started = _time.monotonic()
    try:
        result = callable_(
            base_url, resolved_model, system_prompt, user_content,
            timeout_sec=timeout_sec, thinking=thinking, num_ctx=num_ctx,
            temperature=temperature, cfg=cfg,
        )
    except TypeError:
        result = callable_(
            base_url, resolved_model, system_prompt, user_content,
            timeout_sec=timeout_sec, thinking=thinking, num_ctx=num_ctx,
            temperature=temperature,
        )
    latency_ms = int((_time.monotonic() - started) * 1000)
    provider = _actual_provider_used(callable_)
    telemetry_response = _cloud_usage_payload() if provider == "cloud" else None
    _record_telemetry_safe(
        cfg,
        provider=provider,
        model=resolved_model,
        intent=intent,
        response=telemetry_response,
        latency_ms=latency_ms,
    )
    return result


def call_llm_streaming(
    base_url: str,
    chat_model: str,
    system_prompt: str,
    user_content: str,
    on_token: Optional[Callable[[str], None]] = None,
    timeout_sec: float = 30.0,
    thinking: bool = False,
) -> Optional[str]:
    """Hybrid-routed streaming LLM call. See ``_call_llm_streaming_local``."""
    cfg = _get_router_cfg()
    if _is_local_only(cfg):
        return _call_llm_streaming_local(
            base_url, chat_model, system_prompt, user_content,
            on_token=on_token, timeout_sec=timeout_sec, thinking=thinking,
        )
    callable_, resolved_model, intent = _routed_call_kind_for(user_content, "streaming", cfg)
    started = _time.monotonic()
    try:
        result = callable_(
            base_url, resolved_model, system_prompt, user_content,
            on_token=on_token, timeout_sec=timeout_sec, thinking=thinking, cfg=cfg,
        )
    except TypeError:
        result = callable_(
            base_url, resolved_model, system_prompt, user_content,
            on_token=on_token, timeout_sec=timeout_sec, thinking=thinking,
        )
    latency_ms = int((_time.monotonic() - started) * 1000)
    provider = _actual_provider_used(callable_)
    telemetry_response = _cloud_usage_payload() if provider == "cloud" else None
    _record_telemetry_safe(
        cfg,
        provider=provider,
        model=resolved_model,
        intent=intent,
        response=telemetry_response,
        latency_ms=latency_ms,
    )
    return result


def chat_with_messages(
    base_url: str,
    chat_model: str,
    messages: List[Dict[str, str]],
    timeout_sec: float = 30.0,
    extra_options: Optional[Dict[str, Any]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    thinking: bool = False,
) -> Optional[Dict[str, Any]]:
    """Hybrid-routed messages-array LLM call. See ``_chat_with_messages_local``.

    The router classifies on the *last user message* and uses the
    preceding turns as context.
    """
    cfg = _get_router_cfg()
    if _is_local_only(cfg):
        return _chat_with_messages_local(
            base_url, chat_model, messages,
            timeout_sec=timeout_sec, extra_options=extra_options, tools=tools,
            thinking=thinking,
        )
    classification_prompt = ""
    for m in reversed(messages or []):
        if (m.get("role") or "").lower() == "user":
            c = m.get("content")
            classification_prompt = c if isinstance(c, str) else str(c or "")
            break
    callable_, resolved_model, intent = _routed_call_kind_for(classification_prompt, "chat", cfg)
    started = _time.monotonic()
    try:
        result = callable_(
            base_url, resolved_model, messages,
            timeout_sec=timeout_sec, extra_options=extra_options, tools=tools,
            thinking=thinking, cfg=cfg,
        )
    except TypeError:
        result = callable_(
            base_url, resolved_model, messages,
            timeout_sec=timeout_sec, extra_options=extra_options, tools=tools,
            thinking=thinking,
        )
    latency_ms = int((_time.monotonic() - started) * 1000)
    _record_telemetry_safe(
        cfg,
        provider=_actual_provider_used(callable_),
        model=resolved_model,
        intent=intent,
        response=result,
        latency_ms=latency_ms,
    )
    return result

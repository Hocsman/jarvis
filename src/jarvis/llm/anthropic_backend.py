"""Anthropic implementation of :class:`LLMBackend`.

Talks to Claude through the official ``anthropic`` SDK and the native
Messages API (``POST /v1/messages``), not through an OpenAI-compatible
shim: that entry point rejects the sampling values Jarvis sends and has no
prompt caching. The differences from the OpenAI shape stay inside this
module, so callers see the same response shape as from every other
backend:

1. **Messages.** The system prompt is a top-level field, tool calls are
   ``tool_use`` blocks in the assistant turn and their results are
   ``tool_result`` blocks in the next user turn. :func:`_to_turns` and
   :func:`_well_formed` rewrite the engine's OpenAI-style array, and
   :func:`_normalise` lifts the answer back into
   ``{"message": {"content", "tool_calls"}}``.
2. **Sampling.** API version 2026-10-21 rejects ``temperature``,
   ``top_p`` and ``top_k`` on every model, so they are never sent.
3. **Thinking.** ``thinking=False`` is ``{"type": "disabled"}``; a model
   that refuses to turn it off is asked for adaptive thinking at low
   effort instead, and remembered. ``thinking=True`` is adaptive thinking.
4. **Caching.** The system prompt carries a cache breakpoint, and a call
   that offers tools (the agentic loop) also caches the growing
   conversation.

The key is the configured one and nothing else: the SDK is built with an
explicit ``api_key``, so ``ANTHROPIC_API_KEY`` and stored profiles are
never read, headers set in ``ANTHROPIC_CUSTOM_HEADERS`` for other tools
are removed, and with no key nothing is sent.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

from ..config import ANTHROPIC_BASE_URL
from ..debug import debug_log
from .backend import LLMBackend, ToolsNotSupportedError

# Generation cap when the caller sets none. The API requires one; this is
# far above any reply Jarvis speaks, and the call's timeout bounds it.
_DEFAULT_MAX_TOKENS = 8192

_EPHEMERAL = {"type": "ephemeral"}

# Lets a replayed thinking block be dropped, instead of failing the
# request, when the history before it changed (the engine widens the tool
# list mid-loop when the model asks toolSearchTool for more).
_BINDING_BETA = "thinking-binding-controls-2026-08-01"

# Assistant turns whose thinking blocks are kept for replay, keyed by the
# id of their first tool call. One reply's loop needs a handful.
_REPLAY_KEEP = 32

# One retry on a rate limit or an overloaded server, when the wait is short
# and the caller's deadline still has room after it. A longer wait is not
# spent: the voice callers have a fallback that answers now.
_RETRY_WAIT_SEC = 0.25
_RETRY_MAX_WAIT_SEC = 1.0
_RETRY_MIN_LEFT_SEC = 1.0
_RETRYABLE_ERRORS = ("overloaded_error", "rate_limit_error", "api_error")

# How long an idle connection to the API stays open. The SDK's pool closes
# it after 5 s, so every voice turn after a pause would pay a fresh TLS
# handshake; utterances are usually further apart than that.
_KEEPALIVE_SEC = 300.0

# Headers the SDK adds that describe this machine (operating system,
# processor, Python build). The API needs none of them.
_FINGERPRINT_HEADERS = (
    "X-Stainless-OS", "X-Stainless-Arch", "X-Stainless-Runtime", "X-Stainless-Runtime-Version",
)

# When tools are offered, text is held back until it is known not to be the
# preamble of a tool call (a tool-calling turn speaks nothing,
# streaming.spec.md). Past this many characters it is taken for the answer
# and streams from there on.
_HELD_BEFORE_TOOLS_CHARS = 240


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    return ""


def _tool_use_block(call: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
        return None
    func = call["function"]
    name = func.get("name")
    if not isinstance(name, str) or not name:
        return None
    args = func.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except ValueError:
            args = {}
    if not isinstance(args, dict):
        args = {}
    call_id = call.get("id") or f"toolu_{uuid.uuid4().hex[:12]}"
    return {"type": "tool_use", "id": call_id, "name": name, "input": args}


def _tool_result_block(call_id: str, text: str) -> Dict[str, Any]:
    block: Dict[str, Any] = {"type": "tool_result", "tool_use_id": call_id}
    if text:
        block["content"] = text
    return block


def _to_turns(
    messages: List[Dict[str, Any]],
    kept_turn: Callable[[Any], Optional[List[Dict[str, Any]]]],
) -> Tuple[str, List[Dict[str, Any]], bool]:
    """The engine's messages as ``(system, turns, replayed)``.

    Every system message joins the top-level system prompt. A tool result
    answers the call it names in the turn just before it, and a user
    message the engine tagged with a tool's name answers that tool's call
    (the approval path replays a granted call that way, see
    ``is_tool_message`` in ``memory/conversation.py``); any other result is
    plain text. ``kept_turn(call_id)`` returns an assistant turn to send
    back exactly as the API returned it (thinking blocks and signatures
    included), or ``None``.
    """
    system_parts: List[str] = []
    turns: List[Dict[str, Any]] = []
    pending: Dict[str, str] = {}
    replayed = False

    def push(role: str, blocks: List[Dict[str, Any]]) -> None:
        if not blocks:
            return
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"].extend(blocks)
        else:
            turns.append({"role": role, "content": list(blocks)})

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        text = _text_of(msg.get("content"))
        if role == "system":
            if text.strip():
                system_parts.append(text)
        elif role == "assistant":
            calls = msg.get("tool_calls") if isinstance(msg.get("tool_calls"), list) else []
            blocks = kept_turn(calls[0].get("id")) if calls and isinstance(calls[0], dict) else None
            if blocks:
                replayed = True
            else:
                blocks = [{"type": "text", "text": text}] if text.strip() else []
                blocks += [b for b in map(_tool_use_block, calls) if b]
            pending = {b["id"]: b["name"] for b in blocks if b.get("type") == "tool_use"}
            push("assistant", blocks)
        elif role == "tool":
            call_id = msg.get("tool_call_id")
            if call_id in pending:
                pending.pop(call_id)
                push("user", [_tool_result_block(call_id, text)])
            elif text.strip():
                push("user", [{"type": "text", "text": text}])
        else:
            name = msg.get("tool_name")
            answered = next((cid for cid, n in pending.items() if n == name), None) if name else None
            if answered:
                pending.pop(answered)
                push("user", [_tool_result_block(answered, text)])
            else:
                pending = {}
                if text.strip():
                    push("user", [{"type": "text", "text": text}])
    return "\n\n".join(system_parts), turns, replayed


def _well_formed(turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Turns the API accepts: user first and last, roles alternating, every
    call answered in the next turn and every result answering a call in
    the turn before, results ahead of any text in their turn.

    Trimming a turn can orphan its neighbour (a history that opens on an
    old tool call loses the call and keeps its result), so the passes
    repeat until nothing changes: a call left unanswered is not sent, and a
    result whose call is gone is sent as text.
    """
    turns = [{"role": t["role"], "content": list(t["content"])} for t in turns]
    while True:
        before = json.dumps(turns, sort_keys=True, default=str)
        while turns and turns[0]["role"] != "user":
            turns.pop(0)
        while turns and turns[-1]["role"] != "user":
            turns.pop()
        for i, turn in enumerate(turns):
            if turn["role"] == "user":
                prev = turns[i - 1] if i > 0 and turns[i - 1]["role"] == "assistant" else None
                calls = {b.get("id") for b in (prev["content"] if prev else []) if b.get("type") == "tool_use"}
                results, rest = [], []
                for block in turn["content"]:
                    if block.get("type") != "tool_result":
                        rest.append(block)
                    elif block.get("tool_use_id") in calls:
                        results.append(block)
                    elif _text_of(block.get("content")).strip():
                        rest.append({"type": "text", "text": _text_of(block.get("content"))})
                turn["content"] = results + rest
            else:
                nxt = turns[i + 1] if i + 1 < len(turns) else None
                answered = {b.get("tool_use_id") for b in (nxt["content"] if nxt else [])
                            if b.get("type") == "tool_result"}
                turn["content"] = [b for b in turn["content"]
                                   if b.get("type") != "tool_use" or b.get("id") in answered]
        merged: List[Dict[str, Any]] = []
        for turn in turns:
            if not turn["content"]:
                continue
            if merged and merged[-1]["role"] == turn["role"]:
                merged[-1]["content"].extend(turn["content"])
            else:
                merged.append(turn)
        turns = merged
        if json.dumps(turns, sort_keys=True, default=str) == before:
            return turns


def _tools(tools: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """OpenAI ``{"type": "function", "function": {...}}`` → Messages tools."""
    out: List[Dict[str, Any]] = []
    for tool in tools or []:
        func = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(func, dict) or not func.get("name"):
            continue
        entry: Dict[str, Any] = {"name": func["name"]}
        if func.get("description"):
            entry["description"] = func["description"]
        entry["input_schema"] = func.get("parameters") or {"type": "object", "properties": {}}
        out.append(entry)
    return out


def _generation_cap(extra_options: Optional[Dict[str, Any]]) -> Optional[int]:
    """The caller's cap, under either backend's name for it."""
    opts = extra_options if isinstance(extra_options, dict) else {}
    nested = opts.get("options") if isinstance(opts.get("options"), dict) else {}
    for source in (opts, nested):
        for key in ("max_tokens", "num_predict"):
            value = source.get(key)
            if isinstance(value, int) and value > 0:
                return value
    return None


def _api_error(error: Any) -> Tuple[str, str]:
    """``(type, message)`` of the API's error body, empty when there is none."""
    body = getattr(error, "body", None)
    detail = body.get("error") if isinstance(body, dict) else None
    if isinstance(detail, dict):
        return str(detail.get("type") or ""), str(detail.get("message") or "")
    return "", str(getattr(error, "message", "") or "")


def _retry_after(error: Any) -> float:
    response = getattr(error, "response", None)
    try:
        return max(float(response.headers.get("retry-after")), 0.0)
    except (AttributeError, TypeError, ValueError):
        return _RETRY_WAIT_SEC


def _usage(message: Any) -> Dict[str, int]:
    usage = getattr(message, "usage", None)
    return {
        key: int(getattr(usage, key, 0) or 0)
        for key in ("input_tokens", "output_tokens",
                    "cache_creation_input_tokens", "cache_read_input_tokens")
    }


def _normalise(message: Any) -> Dict[str, Any]:
    """The Messages answer in the shape every backend returns.

    A tool call cut off by the token cap is dropped rather than run on a
    truncated input. A refusal comes back empty, reasoning included: the
    engine reads reasoning with no answer as a step to continue, which
    would re-send the refused request.
    """
    text: List[str] = []
    thoughts: List[str] = []
    calls: List[Dict[str, Any]] = []
    for block in getattr(message, "content", None) or []:
        kind = getattr(block, "type", "")
        if kind == "text":
            text.append(block.text or "")
        elif kind == "tool_use":
            calls.append({"id": block.id, "type": "function", "function": {
                "name": block.name,
                "arguments": dict(block.input) if isinstance(block.input, dict) else {},
            }})
        elif kind == "thinking":
            thoughts.append(block.thinking or "")

    stop = getattr(message, "stop_reason", None)
    if stop == "max_tokens" and calls:
        debug_log("AnthropicBackend: tool call cut off by the token cap, dropped", "llm")
        calls = []
    if stop == "refusal":
        details = getattr(message, "stop_details", None)
        debug_log(f"AnthropicBackend: refusal ({getattr(details, 'category', None)})", "llm")
        text, thoughts = [], []

    usage = _usage(message)
    debug_log(
        f"AnthropicBackend: {getattr(message, 'model', '?')} stop={stop} "
        f"in={usage['input_tokens']} cache_read={usage['cache_read_input_tokens']} "
        f"cache_write={usage['cache_creation_input_tokens']} out={usage['output_tokens']}",
        "llm",
    )
    out: Dict[str, Any] = {"role": "assistant", "content": "".join(text)}
    if calls:
        out["tool_calls"] = calls
    thought = "".join(thoughts).strip()
    if thought:
        out["thinking"] = thought
    return {"message": out, "stop_reason": stop, "usage": usage}


def _headers_to_send(api_key: str) -> Dict[str, Any]:
    """``default_headers`` that strip what the SDK would add on its own.

    The SDK copies every header named in ``ANTHROPIC_CUSTOM_HEADERS`` onto
    each request, after its own: a variable set for a gateway used by
    another tool would send that gateway's secrets here, and an
    ``Authorization`` line in it would replace the configured key.
    """
    import anthropic

    headers: Dict[str, Any] = {name: anthropic.omit for name in _FINGERPRINT_HEADERS}
    for line in os.environ.get("ANTHROPIC_CUSTOM_HEADERS", "").split("\n"):
        name = line.partition(":")[0].strip()
        if not name:
            continue
        headers[name] = f"Bearer {api_key}" if name.lower() == "authorization" else anthropic.omit
    return headers


class AnthropicBackend(LLMBackend):
    """:class:`LLMBackend` implementation for the Anthropic Messages API."""

    def __init__(self, api_key: Optional[str] = None, http_client: Any = None) -> None:
        self._api_key = api_key or None
        # The SDK's transport; tests hand in an in-memory one.
        self._http_client = http_client
        self._client: Any = None
        self._lock = threading.Lock()
        self._warned_no_key = False
        # Models that answered "thinking cannot be disabled".
        self._always_thinks: set = set()
        self._replay: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()

    @property
    def base_url(self) -> str:
        return ANTHROPIC_BASE_URL

    def _sdk(self) -> Any:
        if not self._api_key:
            if not self._warned_no_key:
                self._warned_no_key = True
                print("⚠️ Anthropic provider: no API key, nothing sent", flush=True)
                print("   🔑 set llm_api_key_env to the name of the variable that holds it",
                      flush=True)
            return None
        with self._lock:
            if self._client is None:
                import anthropic

                http_client = self._http_client
                if http_client is None:
                    import httpx2

                    http_client = anthropic.DefaultHttpxClient(limits=httpx2.Limits(
                        max_connections=100, max_keepalive_connections=20,
                        keepalive_expiry=_KEEPALIVE_SEC,
                    ))
                # One persistent client: its connection pool keeps the TLS
                # session to the API alive between calls. No SDK retries:
                # they would let a timeout run several times over the
                # caller's deadline; ``_send`` retries within it instead.
                self._client = anthropic.Anthropic(
                    api_key=self._api_key,
                    base_url=ANTHROPIC_BASE_URL,
                    max_retries=0,
                    default_headers=_headers_to_send(self._api_key),
                    open_telemetry=False,
                    http_client=http_client,
                )
            return self._client

    def _kept_turn(self, call_id: Any) -> Optional[List[Dict[str, Any]]]:
        with self._lock:
            kept = self._replay.get(call_id)
        return [dict(b) for b in kept] if kept else None

    def _keep_for_replay(self, message: Any, normalised: Dict[str, Any]) -> None:
        """Remember an assistant turn that reasoned before calling a tool,
        so the next request of the loop can send it back unchanged."""
        calls = normalised["message"].get("tool_calls")
        content = getattr(message, "content", None) or []
        if not calls or not any(getattr(b, "type", "") in ("thinking", "redacted_thinking")
                                for b in content):
            return
        blocks = [block.to_dict(exclude_none=True) for block in content]
        with self._lock:
            self._replay[calls[0]["id"]] = blocks
            while len(self._replay) > _REPLAY_KEEP:
                self._replay.popitem(last=False)

    # ── sending ───────────────────────────────────────────────────────

    def _request_kwargs(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        *,
        thinks: bool,
        thinking_requested: bool,
        max_tokens: int,
        tools: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        system, turns, replayed = _to_turns(messages, self._kept_turn if thinks else lambda _id: None)
        kwargs: Dict[str, Any] = {"model": model, "max_tokens": max_tokens,
                                  "messages": _well_formed(turns)}
        if system:
            kwargs["system"] = [{"type": "text", "text": system, "cache_control": dict(_EPHEMERAL)}]
        if tools:
            kwargs["tools"] = tools
            # The engine runs the first call of a turn and answers it; a
            # second parallel call would be left without a result.
            kwargs["tool_choice"] = {"type": "auto", "disable_parallel_tool_use": True}
            # Tool loops re-send the whole conversation: cache its tail too.
            kwargs["cache_control"] = dict(_EPHEMERAL)
        if thinks:
            thinking: Dict[str, Any] = {"type": "adaptive"}
            if replayed:
                thinking["block_binding"] = {"prefix_mismatch_behavior": "drop_block"}
                kwargs["extra_headers"] = {"anthropic-beta": _BINDING_BETA}
            kwargs["thinking"] = thinking
            if not thinking_requested:
                kwargs["output_config"] = {"effort": "low"}
        else:
            kwargs["thinking"] = {"type": "disabled"}
        return kwargs

    @staticmethod
    def _stream(
        client: Any, kwargs: Dict[str, Any], timeout: float,
        on_token: Callable[[str], None], hold: bool,
    ) -> Any:
        """Forward text deltas, return the final message.

        With ``hold``, text waits until it is known not to precede a tool
        call: dropped if a ``tool_use`` block follows, released at the end
        of the message, or from the moment it outgrows
        ``_HELD_BEFORE_TOOLS_CHARS``.
        """
        held: List[str] = []
        silenced = False
        live = not hold

        def emit(text: str) -> None:
            try:
                on_token(text)
            except Exception as e:
                debug_log(f"AnthropicBackend: on_token raised — {e}", "llm")

        with client.messages.stream(**kwargs, timeout=timeout) as stream:
            for event in stream:
                if event.type == "content_block_start" and getattr(event.content_block, "type", "") == "tool_use":
                    held.clear()
                    silenced = True
                if (event.type != "content_block_delta" or silenced
                        or getattr(event.delta, "type", "") != "text_delta"):
                    continue
                if live:
                    emit(event.delta.text)
                    continue
                held.append(event.delta.text)
                if sum(len(t) for t in held) > _HELD_BEFORE_TOOLS_CHARS:
                    live = True
                    for text in held:
                        emit(text)
                    held.clear()
            message = stream.get_final_message()
        for text in held:
            emit(text)
        return message

    def _send(
        self,
        client: Any,
        model: str,
        messages: List[Dict[str, Any]],
        *,
        timeout_sec: float,
        thinking_requested: bool,
        max_tokens: int,
        tools: List[Dict[str, Any]],
        on_token: Optional[Callable[[str], None]],
    ) -> Any:
        import anthropic

        retryable = (anthropic.RateLimitError, anthropic.OverloadedError,
                     anthropic.InternalServerError, anthropic.ServiceUnavailableError)
        deadline = time.monotonic() + timeout_sec
        retried = False
        while True:
            thinks = thinking_requested or model in self._always_thinks
            kwargs = self._request_kwargs(model, messages, thinks=thinks,
                                          thinking_requested=thinking_requested,
                                          max_tokens=max_tokens, tools=tools)
            timeout = max(deadline - time.monotonic(), 0.1)
            emitted = False

            def sink(text: str) -> None:
                nonlocal emitted
                emitted = True
                on_token(text)

            try:
                if on_token is None:
                    return client.messages.create(**kwargs, timeout=timeout)
                return self._stream(client, kwargs, timeout, sink, hold=bool(tools))
            except anthropic.APIStatusError as e:
                kind, text = _api_error(e)
                if isinstance(e, anthropic.BadRequestError):
                    if not thinks and "thinking" in text.lower():
                        self._always_thinks.add(model)
                        debug_log(f"AnthropicBackend: {model} keeps thinking on; asking low effort", "llm")
                        continue
                    raise
                if not (isinstance(e, retryable) or kind in _RETRYABLE_ERRORS):
                    raise
                wait = _retry_after(e)
                if (retried or emitted or wait > _RETRY_MAX_WAIT_SEC
                        or deadline - time.monotonic() - wait < _RETRY_MIN_LEFT_SEC):
                    raise
                retried = True
                debug_log(f"AnthropicBackend: {kind or e.status_code}, one retry in {wait:.2f}s", "llm")
                time.sleep(wait)

    def _complete(
        self,
        client: Any,
        model: str,
        messages: List[Dict[str, Any]],
        *,
        timeout_sec: float,
        thinking: bool,
        max_tokens: int,
        tools: Optional[List[Dict[str, Any]]] = None,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        message = self._send(client, model, messages, timeout_sec=timeout_sec,
                             thinking_requested=thinking, max_tokens=max_tokens,
                             tools=_tools(tools), on_token=on_token)
        normalised = _normalise(message)
        self._keep_for_replay(message, normalised)
        return normalised

    def _text_answer(self, where: str, model: str, system_prompt: str, user_content: str, *,
                     timeout_sec: float, thinking: bool, max_tokens: int,
                     on_token: Optional[Callable[[str], None]] = None) -> Optional[str]:
        client = self._sdk()
        if client is None:
            return None
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}]
        try:
            resp = self._complete(client, model, messages, timeout_sec=timeout_sec,
                                  thinking=thinking, max_tokens=max_tokens, on_token=on_token)
        except Exception as e:
            kind, text = _api_error(e)
            status = getattr(e, "status_code", None)
            reason = f"HTTP {status} {kind}: {text[:300]}" if status else type(e).__name__
            debug_log(f"AnthropicBackend.{where}: {reason}", "llm")
            return None
        content = resp["message"]["content"]
        return content if content.strip() else None

    # ── LLMBackend ────────────────────────────────────────────────────

    def direct(
        self,
        chat_model: str,
        system_prompt: str,
        user_content: str,
        timeout_sec: float = 10.0,
        thinking: bool = False,
        num_ctx: int = 4096,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Optional[str]:
        # ``num_ctx`` has no equivalent here, and ``temperature`` is never
        # sent: both are accepted for signature parity only.
        return self._text_answer("direct", chat_model, system_prompt, user_content,
                                 timeout_sec=timeout_sec, thinking=thinking,
                                 max_tokens=max_tokens or _DEFAULT_MAX_TOKENS)

    def streaming(
        self,
        chat_model: str,
        system_prompt: str,
        user_content: str,
        on_token: Optional[Callable[[str], None]] = None,
        timeout_sec: float = 30.0,
        thinking: bool = False,
    ) -> Optional[str]:
        return self._text_answer("streaming", chat_model, system_prompt, user_content,
                                 timeout_sec=timeout_sec, thinking=thinking,
                                 max_tokens=_DEFAULT_MAX_TOKENS,
                                 on_token=on_token or (lambda _text: None))

    def chat(
        self,
        chat_model: str,
        messages: List[Dict[str, Any]],
        timeout_sec: float = 30.0,
        extra_options: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        thinking: bool = False,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> Optional[Dict[str, Any]]:
        # From ``extra_options`` only the generation cap applies: sampling
        # values are rejected by the API and the Ollama knobs
        # (``keep_alive``, ``num_ctx``) have no equivalent.
        client = self._sdk()
        if client is None:
            return None
        import anthropic

        try:
            return self._complete(client, chat_model, messages, timeout_sec=timeout_sec,
                                  thinking=thinking,
                                  max_tokens=_generation_cap(extra_options) or _DEFAULT_MAX_TOKENS,
                                  tools=tools, on_token=on_token)
        except anthropic.APITimeoutError:
            print("  ⏱️ LLM request timed out", flush=True)
            return None
        except anthropic.APIConnectionError:
            # Re-raised as the type callers already handle (the intent judge
            # backs off on it), without the SDK's message: it names the host.
            print("  ❌ LLM connection error", flush=True)
            raise requests.ConnectionError("LLM connection error") from None
        except anthropic.APIStatusError as e:
            kind, text = _api_error(e)
            # Only an error about the tool definitions is one the engine's
            # text-tools fallback can get past: it re-sends the same
            # conversation without them.
            if e.status_code == 400 and tools and text.lstrip().startswith(("tools", "tool_choice")):
                raise ToolsNotSupportedError(
                    f"Model {chat_model!r} rejected the tools: HTTP 400"
                ) from None
            print(f"  ❌ LLM HTTP error (status {e.status_code}{', ' + kind if kind else ''})",
                  flush=True)
            # The body is the API's own explanation; it carries no key.
            debug_log(f"AnthropicBackend.chat: HTTP {e.status_code} {kind}: {text[:600]}", "llm")
            return None
        except Exception as e:
            print(f"  ❌ LLM error ({type(e).__name__})", flush=True)
            return None

    def embed(
        self,
        text: str,
        model: str,
        timeout_sec: float = 15.0,
    ) -> Optional[List[float]]:
        # The Messages API has no embeddings: the factory sends them to
        # Ollama, so a call here is a routing mistake, not a provider fault.
        debug_log("AnthropicBackend.embed: Anthropic serves no embeddings", "llm")
        return None

    def list_models(self, timeout_sec: float = 5.0) -> List[str]:
        client = self._sdk()
        if client is None:
            return []
        try:
            page = client.models.list(limit=1000, timeout=timeout_sec)
            return [m.id for m in page.data if getattr(m, "id", None)]
        except Exception as e:
            debug_log(f"AnthropicBackend.list_models: {type(e).__name__}", "llm")
            return []

    def warm_up(
        self,
        model: str,
        timeout_sec: float = 60.0,
        keep_alive: str = "30m",
    ) -> bool:
        """Open the client's connection to the API ahead of the first query.

        There is nothing to page into memory; the first request on a cold
        connection pays the TLS handshake, which one 1-token request at
        startup absorbs. The connection then stays open for
        ``_KEEPALIVE_SEC`` of idleness. The request is billed, a few
        tokens' worth.
        """
        client = self._sdk()
        if not model or client is None:
            return False
        try:
            self._complete(client, model, [{"role": "user", "content": "hi"}],
                           timeout_sec=timeout_sec, thinking=False, max_tokens=1)
            return True
        except Exception as e:
            debug_log(f"AnthropicBackend.warm_up: {type(e).__name__}", "llm")
            return False

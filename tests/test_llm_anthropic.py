"""Behaviour tests for :class:`AnthropicBackend`.

The backend talks to the Messages API through the official ``anthropic``
SDK. These tests run the real SDK against an in-memory transport
(``httpx2.MockTransport``), so what they check is the request that would
leave the machine and the shape the reply engine gets back — never a mock
call count. No socket is opened.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

import httpx2
import pytest
import requests


MODEL = "claude-haiku-5-5"


def _message(content: List[Dict[str, Any]], *, stop_reason: str = "end_turn",
             usage: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": MODEL,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": usage or {"input_tokens": 12, "output_tokens": 3},
    }


def _text(text: str) -> Dict[str, Any]:
    return _message([{"type": "text", "text": text}])


def _sse(events: List[Dict[str, Any]]) -> bytes:
    lines = []
    for event in events:
        lines.append(f"event: {event['type']}")
        lines.append(f"data: {json.dumps(event)}")
        lines.append("")
    return ("\n".join(lines) + "\n").encode("utf-8")


def _streamed_text(chunks: List[str]) -> bytes:
    events: List[Dict[str, Any]] = [
        {"type": "message_start", "message": {**_message([]), "stop_reason": None}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    ]
    for chunk in chunks:
        events.append({"type": "content_block_delta", "index": 0,
                       "delta": {"type": "text_delta", "text": chunk}})
    events += [
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
         "usage": {"output_tokens": len(chunks)}},
        {"type": "message_stop"},
    ]
    return _sse(events)


def _streamed_tool_call(name: str, call_id: str, arguments_json: str,
                        preamble: Optional[List[str]] = None) -> bytes:
    half = len(arguments_json) // 2
    events: List[Dict[str, Any]] = [
        {"type": "message_start", "message": {**_message([]), "stop_reason": None}},
    ]
    index = 0
    if preamble:
        events.append({"type": "content_block_start", "index": 0,
                       "content_block": {"type": "text", "text": ""}})
        events += [{"type": "content_block_delta", "index": 0,
                    "delta": {"type": "text_delta", "text": chunk}} for chunk in preamble]
        events.append({"type": "content_block_stop", "index": 0})
        index = 1
    events += [
        {"type": "content_block_start", "index": index,
         "content_block": {"type": "tool_use", "id": call_id, "name": name, "input": {}}},
        {"type": "content_block_delta", "index": index,
         "delta": {"type": "input_json_delta", "partial_json": arguments_json[:half]}},
        {"type": "content_block_delta", "index": index,
         "delta": {"type": "input_json_delta", "partial_json": arguments_json[half:]}},
        {"type": "content_block_stop", "index": index},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": None},
         "usage": {"output_tokens": 9}},
        {"type": "message_stop"},
    ]
    return _sse(events)


def _streamed_error(kind: str) -> bytes:
    """An error the API reports inside a stream that had already answered 200."""
    return _sse([
        {"type": "message_start", "message": {**_message([]), "stop_reason": None}},
        {"type": "error", "error": {"type": kind, "message": "Overloaded"}},
    ])


class _Server:
    """An in-memory Messages API: records every request, answers from a queue."""

    def __init__(self, *responses: Any):
        self.requests: List[httpx2.Request] = []
        self._responses = list(responses)

    def answer(self, *responses: Any) -> None:
        self._responses.extend(responses)

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        if not self._responses:
            return httpx2.Response(500, json={"type": "error", "error": {
                "type": "api_error", "message": "no answer queued"}})
        nxt = self._responses.pop(0)
        if callable(nxt):
            nxt = nxt(request)
        if isinstance(nxt, httpx2.Response):
            return nxt
        if isinstance(nxt, bytes):
            return httpx2.Response(200, content=nxt, headers={"content-type": "text/event-stream"})
        return httpx2.Response(200, json=nxt)

    def body(self, index: int = -1) -> Dict[str, Any]:
        return json.loads(self.requests[index].content)


def _error(status: int, message: str = "nope", kind: str = "invalid_request_error",
           headers: Optional[Dict[str, str]] = None) -> httpx2.Response:
    return httpx2.Response(status, json={"type": "error", "error": {"type": kind, "message": message}},
                           headers=headers or {})


def _raise(exc: Exception) -> Callable[[httpx2.Request], httpx2.Response]:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise exc
    return handler


def _backend(server: _Server, api_key: Optional[str] = "sk-ant-test"):
    from jarvis.llm import AnthropicBackend

    return AnthropicBackend(api_key=api_key,
                            http_client=httpx2.Client(transport=httpx2.MockTransport(server)))


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "getWeather",
        "description": "Current weather for a place",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                       "required": ["city"]},
    },
}


# ── where the request goes, and with which key ─────────────────────────────

class TestDestination:
    def test_sends_to_the_messages_endpoint_of_anthropic(self):
        server = _Server(_text("ok"))

        _backend(server).direct(MODEL, "sys", "hello")

        assert server.requests[0].url == "https://api.anthropic.com/v1/messages"

    def test_sends_the_configured_key_and_ignores_the_environment(self, monkeypatch):
        """``ANTHROPIC_API_KEY`` in the environment must not decide which
        key leaves the machine: the configured one does."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-environment")
        server = _Server(_text("ok"))

        _backend(server, api_key="sk-ant-configured").direct(MODEL, "sys", "hello")

        headers = server.requests[0].headers
        sent = headers.get("x-api-key", "") + headers.get("authorization", "")
        assert "sk-ant-configured" in sent
        assert "sk-ant-from-environment" not in sent

    def test_without_a_key_nothing_is_sent(self, monkeypatch):
        """No configured key means no request: the SDK would otherwise pick
        up ``ANTHROPIC_API_KEY`` or a stored profile the user never pointed
        Jarvis at."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-environment")
        server = _Server(_text("ok"))
        backend = _backend(server, api_key=None)

        assert backend.direct(MODEL, "sys", "hello") is None
        assert backend.chat(MODEL, [{"role": "user", "content": "hi"}]) is None
        assert server.requests == []

    def test_headers_set_for_other_tools_are_not_sent(self, monkeypatch):
        """``ANTHROPIC_CUSTOM_HEADERS`` serves gateways set up for other
        tools; its secrets must not ride along, nor replace the key."""
        monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS",
                           "Authorization: Bearer someone-else\nx-gateway-key: s3cret")
        server = _Server(_text("ok"))

        _backend(server, api_key="sk-ant-configured").direct(MODEL, "sys", "hello")

        headers = server.requests[0].headers
        assert "x-gateway-key" not in headers
        assert "someone-else" not in headers.get("authorization", "")
        assert "sk-ant-configured" in headers.get("x-api-key", "") + headers.get("authorization", "")

    def test_no_description_of_the_machine_is_sent(self):
        server = _Server(_text("ok"))

        _backend(server).direct(MODEL, "sys", "hello")

        sent = {name.lower() for name in server.requests[0].headers}
        for fingerprint in ("x-stainless-os", "x-stainless-arch", "x-stainless-runtime",
                            "x-stainless-runtime-version"):
            assert fingerprint not in sent

    def test_the_key_never_reaches_standard_output(self, capsys):
        server = _Server(_error(401, "invalid x-api-key", "authentication_error"))

        _backend(server, api_key="sk-ant-secret-value").chat(MODEL, [{"role": "user", "content": "hi"}])

        out = capsys.readouterr()
        assert "sk-ant-secret-value" not in out.out + out.err


# ── direct() ────────────────────────────────────────────────────────────────

class TestDirect:
    def test_returns_the_answer_text(self):
        server = _Server(_text("Bonjour"))

        assert _backend(server).direct(MODEL, "sys", "hello") == "Bonjour"

    def test_system_prompt_is_a_cached_block_and_the_user_text_follows(self):
        server = _Server(_text("ok"))

        _backend(server).direct(MODEL, "You classify.", "Is it raining?")

        body = server.body()
        assert body["system"] == [{"type": "text", "text": "You classify.",
                                   "cache_control": {"type": "ephemeral"}}]
        assert body["messages"] == [{"role": "user", "content": [{"type": "text", "text": "Is it raining?"}]}]

    def test_sampling_temperature_is_never_sent(self):
        """API version 2026-10-21 rejects ``temperature`` on every model."""
        server = _Server(_text("ok"))

        _backend(server).direct(MODEL, "sys", "hello", temperature=0.0)

        assert "temperature" not in server.body()

    def test_thinking_is_turned_off_unless_asked_for(self):
        server = _Server(_text("ok"), _text("ok"))
        backend = _backend(server)

        backend.direct(MODEL, "sys", "hello")
        backend.direct(MODEL, "sys", "hello", thinking=True)

        assert server.body(0)["thinking"] == {"type": "disabled"}
        assert server.body(1)["thinking"] == {"type": "adaptive"}

    def test_max_tokens_is_the_callers_cap_when_given(self):
        server = _Server(_text("ok"), _text("ok"))
        backend = _backend(server)

        backend.direct(MODEL, "sys", "hello", max_tokens=300)
        backend.direct(MODEL, "sys", "hello")

        assert server.body(0)["max_tokens"] == 300
        assert server.body(1)["max_tokens"] > 300

    def test_an_http_error_returns_none(self):
        server = _Server(_error(404, "model not found", "not_found_error"))

        assert _backend(server).direct(MODEL, "sys", "hello") is None

    def test_a_refusal_returns_none(self):
        server = _Server(_message([], stop_reason="refusal"))

        assert _backend(server).direct(MODEL, "sys", "hello") is None

    def test_a_timeout_returns_none(self):
        server = _Server(_raise(httpx2.ReadTimeout("slow")))

        assert _backend(server).direct(MODEL, "sys", "hello", timeout_sec=0.5) is None


class TestThinkingThatCannotBeTurnedOff:
    def test_a_model_that_refuses_disabled_thinking_is_asked_for_low_effort_instead(self):
        """Some Claude models reject ``{"type": "disabled"}``. The caller
        asked for no reasoning, so the closest the model allows is adaptive
        thinking at low effort — and the model is remembered, so the next
        call does not pay the refused round-trip again."""
        server = _Server(
            _error(400, "thinking.type: 'disabled' is not supported for this model"),
            _text("first"),
            _text("second"),
        )
        backend = _backend(server)

        assert backend.direct("claude-opus-5-5", "sys", "hello") == "first"
        assert backend.direct("claude-opus-5-5", "sys", "hello") == "second"

        assert server.body(0)["thinking"] == {"type": "disabled"}
        assert server.body(1)["thinking"] == {"type": "adaptive"}
        assert server.body(1)["output_config"] == {"effort": "low"}
        assert len(server.requests) == 3
        assert server.body(2)["thinking"] == {"type": "adaptive"}

    def test_an_unrelated_bad_request_is_not_retried(self):
        server = _Server(_error(400, "messages: at least one message is required"))

        assert _backend(server).direct(MODEL, "sys", "hello") is None
        assert len(server.requests) == 1


class TestRetry:
    def test_an_overloaded_answer_is_retried_once_within_the_deadline(self):
        server = _Server(_error(529, "Overloaded", "overloaded_error"), _text("ok"))

        assert _backend(server).direct(MODEL, "sys", "hello", timeout_sec=10.0) == "ok"
        assert len(server.requests) == 2

    def test_a_rate_limit_is_retried_once_and_then_given_up(self):
        server = _Server(_error(429, "rate", "rate_limit_error"), _error(429, "rate", "rate_limit_error"))

        assert _backend(server).direct(MODEL, "sys", "hello", timeout_sec=10.0) is None
        assert len(server.requests) == 2

    def test_no_retry_when_the_wait_would_overrun_the_deadline(self):
        server = _Server(_error(429, "rate", "rate_limit_error", headers={"retry-after": "30"}), _text("ok"))

        assert _backend(server).direct(MODEL, "sys", "hello", timeout_sec=5.0) is None
        assert len(server.requests) == 1

    def test_a_long_wait_is_not_spent_even_inside_the_deadline(self):
        """A caller with a fallback (the judge's is the text wake word) is
        better served by an answer of None now than by one in seconds."""
        server = _Server(_error(429, "rate", "rate_limit_error", headers={"retry-after": "4"}), _text("ok"))

        assert _backend(server).direct(MODEL, "sys", "hello", timeout_sec=30.0) is None
        assert len(server.requests) == 1

    def test_an_overload_reported_inside_the_stream_is_retried(self):
        server = _Server(_streamed_error("overloaded_error"), _streamed_text(["ok"]))
        seen: List[str] = []

        resp = _backend(server).chat(MODEL, [{"role": "user", "content": "?"}], timeout_sec=10.0,
                                     on_token=seen.append)

        assert resp["message"]["content"] == "ok"
        assert len(server.requests) == 2


# ── streaming() ─────────────────────────────────────────────────────────────

class TestStreaming:
    def test_tokens_arrive_one_by_one_and_the_whole_text_is_returned(self):
        server = _Server(_streamed_text(["Bon", "jour", " !"]))
        seen: List[str] = []

        result = _backend(server).streaming(MODEL, "sys", "hello", on_token=seen.append)

        assert seen == ["Bon", "jour", " !"]
        assert result == "Bonjour !"
        assert server.body()["stream"] is True


# ── chat() ──────────────────────────────────────────────────────────────────

class TestChat:
    def test_returns_the_answer_in_the_engines_shape(self):
        server = _Server(_text("Il fait beau."))

        resp = _backend(server).chat(MODEL, [{"role": "user", "content": "Quel temps ?"}])

        assert resp["message"]["role"] == "assistant"
        assert resp["message"]["content"] == "Il fait beau."
        assert "tool_calls" not in resp["message"]

    def test_every_system_message_becomes_the_cached_system_prompt(self):
        server = _Server(_text("ok"))

        _backend(server).chat(MODEL, [
            {"role": "system", "content": "Persona.", "_is_context_injected": True},
            {"role": "user", "content": "hi"},
        ])

        body = server.body()
        assert body["system"] == [{"type": "text", "text": "Persona.",
                                   "cache_control": {"type": "ephemeral"}}]
        assert [m["role"] for m in body["messages"]] == ["user"]

    def test_tools_are_sent_in_the_messages_shape_one_call_per_turn(self):
        server = _Server(_text("ok"))

        _backend(server).chat(MODEL, [{"role": "user", "content": "weather?"}], tools=[WEATHER_TOOL])

        body = server.body()
        assert body["tools"] == [{
            "name": "getWeather",
            "description": "Current weather for a place",
            "input_schema": WEATHER_TOOL["function"]["parameters"],
        }]
        # The engine runs the first tool call of a turn and answers it; a
        # second parallel call would be left without a result.
        assert body["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}

    def test_a_tool_loop_caches_the_growing_conversation(self):
        server = _Server(_text("ok"), _text("ok"))
        backend = _backend(server)

        backend.chat(MODEL, [{"role": "user", "content": "weather?"}], tools=[WEATHER_TOOL])
        backend.chat(MODEL, [{"role": "system", "content": "s"}, {"role": "user", "content": "x"}])

        assert server.body(0)["cache_control"] == {"type": "ephemeral"}
        assert "cache_control" not in server.body(1)

    def test_a_tool_call_comes_back_with_decoded_arguments(self):
        server = _Server(_message([
            {"type": "text", "text": "Je regarde."},
            {"type": "tool_use", "id": "toolu_1", "name": "getWeather", "input": {"city": "Paris"}},
        ], stop_reason="tool_use"))

        resp = _backend(server).chat(MODEL, [{"role": "user", "content": "weather?"}], tools=[WEATHER_TOOL])

        assert resp["message"]["content"] == "Je regarde."
        assert resp["message"]["tool_calls"] == [{
            "id": "toolu_1", "type": "function",
            "function": {"name": "getWeather", "arguments": {"city": "Paris"}},
        }]

    def test_a_tool_call_cut_off_by_the_token_cap_is_not_returned(self):
        server = _Server(_message([
            {"type": "tool_use", "id": "toolu_1", "name": "getWeather", "input": {"ci": ""}},
        ], stop_reason="max_tokens"))

        resp = _backend(server).chat(MODEL, [{"role": "user", "content": "weather?"}], tools=[WEATHER_TOOL])

        assert "tool_calls" not in resp["message"]

    def test_the_engines_tool_round_trip_becomes_tool_use_and_tool_result(self):
        server = _Server(_text("Il fait 18 degrés."))

        _backend(server).chat(MODEL, [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "toolu_1", "type": "function",
                "function": {"name": "getWeather", "arguments": {"city": "Paris"}}}]},
            {"role": "tool", "tool_call_id": "toolu_1", "content": "18°C", "tool_name": "getWeather"},
        ], tools=[WEATHER_TOOL])

        assert server.body()["messages"] == [
            {"role": "user", "content": [{"type": "text", "text": "weather?"}]},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "getWeather", "input": {"city": "Paris"}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "18°C"}]},
        ]

    def test_json_encoded_tool_arguments_are_decoded(self):
        server = _Server(_text("ok"))

        _backend(server).chat(MODEL, [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "toolu_1", "type": "function",
                "function": {"name": "getWeather", "arguments": "{\"city\": \"Paris\"}"}}]},
            {"role": "tool", "tool_call_id": "toolu_1", "content": "18°C"},
        ], tools=[WEATHER_TOOL])

        assert server.body()["messages"][1]["content"][0]["input"] == {"city": "Paris"}

    def test_a_result_the_engine_tagged_as_a_user_message_answers_its_call(self):
        """The approval path replays a granted call as an assistant tool
        call followed by a user message tagged with the tool's name. On this
        API a call with no result is a 400, so that message is its result."""
        server = _Server(_text("ok"))

        _backend(server).chat(MODEL, [
            {"role": "user", "content": "envoie-le"},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "call_grant_1", "type": "function",
                "function": {"name": "sendMail", "arguments": {"to": "x"}}}]},
            {"role": "user", "content": "[Tool result: sendMail]\nsent", "tool_name": "sendMail",
             "tool_failed": False},
        ], tools=[WEATHER_TOOL])

        assert server.body()["messages"][2] == {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "call_grant_1",
             "content": "[Tool result: sendMail]\nsent"}]}

    def test_a_call_left_without_a_result_is_not_sent(self):
        server = _Server(_text("ok"))

        _backend(server).chat(MODEL, [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "Je regarde.", "tool_calls": [{
                "id": "toolu_9", "type": "function",
                "function": {"name": "getWeather", "arguments": {}}}]},
            {"role": "user", "content": "second"},
        ], tools=[WEATHER_TOOL])

        assert server.body()["messages"] == [
            {"role": "user", "content": [{"type": "text", "text": "first"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Je regarde."}]},
            {"role": "user", "content": [{"type": "text", "text": "second"}]},
        ]

    def test_a_result_whose_call_is_gone_is_sent_as_text(self):
        server = _Server(_text("ok"))

        _backend(server).chat(MODEL, [
            {"role": "user", "content": "first"},
            {"role": "tool", "tool_call_id": "toolu_gone", "content": "18°C"},
        ])

        assert server.body()["messages"] == [
            {"role": "user", "content": [{"type": "text", "text": "first"}, {"type": "text", "text": "18°C"}]},
        ]

    def test_a_history_opening_on_an_old_tool_round_trip_keeps_its_result_as_text(self):
        """Tool carry-over outlives the text it followed: past five minutes
        the history can open on an assistant call. The turn before the
        first user message goes, so its result has no call left to answer."""
        server = _Server(_text("ok"))

        _backend(server).chat(MODEL, [
            {"role": "system", "content": "s"},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "toolu_old", "type": "function",
                "function": {"name": "getWeather", "arguments": {"city": "Paris"}}}]},
            {"role": "tool", "tool_call_id": "toolu_old", "content": "18°C"},
            {"role": "user", "content": "and tomorrow?"},
            {"role": "assistant", "content": "Let me see."},
            {"role": "user", "content": "well?"},
        ], tools=[WEATHER_TOOL])

        assert server.body()["messages"] == [
            {"role": "user", "content": [{"type": "text", "text": "18°C"},
                                         {"type": "text", "text": "and tomorrow?"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Let me see."}]},
            {"role": "user", "content": [{"type": "text", "text": "well?"}]},
        ]

    def test_the_conversation_starts_with_the_user_and_ends_with_the_user(self):
        server = _Server(_text("ok"))

        _backend(server).chat(MODEL, [
            {"role": "assistant", "content": "earlier answer"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": ""},
        ])

        assert server.body()["messages"] == [
            {"role": "user", "content": [{"type": "text", "text": "question"}]},
        ]

    def test_ollama_knobs_and_sampling_are_dropped_and_the_cap_is_kept(self):
        server = _Server(_text("ok"))

        _backend(server).chat(MODEL, [{"role": "user", "content": "hi"}], extra_options={
            "temperature": 0.0, "max_tokens": 1500, "num_ctx": 8192, "keep_alive": "30m",
            "options": {"top_p": 0.9, "num_predict": 99}})

        body = server.body()
        for absent in ("temperature", "top_p", "num_ctx", "keep_alive", "options"):
            assert absent not in body
        assert body["max_tokens"] == 1500

    def test_a_bad_request_with_tools_hands_the_engine_its_text_fallback(self):
        from jarvis.llm import ToolsNotSupportedError

        server = _Server(_error(400, "tools.0.custom.name: String should match pattern"))

        with pytest.raises(ToolsNotSupportedError):
            _backend(server).chat(MODEL, [{"role": "user", "content": "hi"}], tools=[WEATHER_TOOL])

    def test_a_bad_request_about_the_conversation_is_not_blamed_on_the_tools(self):
        """Falling back to text tools re-sends the same conversation: it
        cannot fix an error in it."""
        server = _Server(_error(400, "messages.0.content.0: unexpected `tool_use_id` found in `tool_result` blocks"))

        assert _backend(server).chat(MODEL, [{"role": "user", "content": "hi"}], tools=[WEATHER_TOOL]) is None

    def test_an_unreachable_server_raises_a_connection_error_without_the_url(self):
        """The intent judge backs off on ``requests.ConnectionError``."""
        server = _Server(_raise(httpx2.ConnectError("[Errno 11001] api.anthropic.com")))

        with pytest.raises(requests.ConnectionError) as info:
            _backend(server).chat(MODEL, [{"role": "user", "content": "hi"}])

        assert "anthropic.com" not in str(info.value)

    def test_a_timeout_returns_none(self):
        server = _Server(_raise(httpx2.ReadTimeout("slow")))

        assert _backend(server).chat(MODEL, [{"role": "user", "content": "hi"}], timeout_sec=0.5) is None


class TestChatStreaming:
    def test_streamed_text_matches_the_buffered_result(self):
        server = _Server(_streamed_text(["Il ", "fait ", "beau."]), _text("Il fait beau."))
        backend = _backend(server)
        seen: List[str] = []

        streamed = backend.chat(MODEL, [{"role": "user", "content": "?"}], on_token=seen.append)
        buffered = backend.chat(MODEL, [{"role": "user", "content": "?"}])

        assert seen == ["Il ", "fait ", "beau."]
        assert streamed["message"] == buffered["message"]

    def test_a_streamed_tool_call_is_reassembled(self):
        server = _Server(_streamed_tool_call("getWeather", "toolu_7", "{\"city\": \"Lyon\"}"))

        resp = _backend(server).chat(MODEL, [{"role": "user", "content": "?"}], tools=[WEATHER_TOOL],
                                     on_token=lambda _t: None)

        assert resp["message"]["tool_calls"] == [{
            "id": "toolu_7", "type": "function",
            "function": {"name": "getWeather", "arguments": {"city": "Lyon"}},
        }]

    def test_text_written_before_a_tool_call_is_not_streamed(self):
        """A tool-calling turn speaks nothing: what Claude writes before
        the call is not the answer, and the voice only says the answer."""
        server = _Server(_streamed_tool_call("getWeather", "toolu_8", "{\"city\": \"Lyon\"}",
                                             preamble=["Je ", "regarde."]))
        seen: List[str] = []

        resp = _backend(server).chat(MODEL, [{"role": "user", "content": "?"}], tools=[WEATHER_TOOL],
                                     on_token=seen.append)

        assert seen == []
        assert resp["message"]["content"] == "Je regarde."
        assert resp["message"]["tool_calls"][0]["function"]["name"] == "getWeather"

    def test_with_tools_offered_the_answer_is_still_streamed(self):
        server = _Server(_streamed_text(["Il ", "fait ", "beau."]))
        seen: List[str] = []

        resp = _backend(server).chat(MODEL, [{"role": "user", "content": "?"}], tools=[WEATHER_TOOL],
                                     on_token=seen.append)

        assert "".join(seen) == resp["message"]["content"] == "Il fait beau."

    def test_a_raising_callback_never_costs_the_reply(self):
        server = _Server(_streamed_text(["a", "b"]))

        def boom(_t: str) -> None:
            raise RuntimeError("ui fault")

        resp = _backend(server).chat(MODEL, [{"role": "user", "content": "?"}], on_token=boom)

        assert resp["message"]["content"] == "ab"


class TestThinkingAcrossToolTurns:
    def _thinking_tool_call(self) -> Dict[str, Any]:
        return _message([
            {"type": "thinking", "thinking": "", "signature": "sig-abc"},
            {"type": "tool_use", "id": "toolu_t", "name": "getWeather", "input": {"city": "Nice"}},
        ], stop_reason="tool_use")

    def _follow_up(self, call: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": call["content"], "tool_calls": call["tool_calls"]},
            {"role": "tool", "tool_call_id": "toolu_t", "content": "21°C"},
        ]

    def test_the_reasoning_behind_a_tool_call_is_sent_back_unchanged(self):
        server = _Server(self._thinking_tool_call(), _text("21 degrés."))
        backend = _backend(server)

        first = backend.chat(MODEL, [{"role": "user", "content": "weather?"}], tools=[WEATHER_TOOL],
                             thinking=True)
        backend.chat(MODEL, self._follow_up(first["message"]), tools=[WEATHER_TOOL], thinking=True)

        replayed = server.body(1)["messages"][1]["content"]
        assert replayed[0] == {"type": "thinking", "thinking": "", "signature": "sig-abc"}
        assert replayed[1]["type"] == "tool_use"
        # A replayed block degrades instead of failing when the history
        # changed under it (the tool list can widen mid-loop).
        assert server.body(1)["thinking"]["block_binding"] == {"prefix_mismatch_behavior": "drop_block"}
        assert "thinking-binding-controls" in server.requests[1].headers["anthropic-beta"]

    def test_a_refusal_carries_no_reasoning_forward(self):
        """The engine reads reasoning with no answer as a step to continue;
        after a refusal that would re-send the refused request."""
        server = _Server(_message([{"type": "thinking", "thinking": "hmm", "signature": "s"}],
                                  stop_reason="refusal"))

        resp = _backend(server).chat(MODEL, [{"role": "user", "content": "?"}], thinking=True)

        assert resp["message"]["content"] == ""
        assert "thinking" not in resp["message"]

    def test_no_reasoning_is_replayed_into_a_request_that_does_not_think(self):
        server = _Server(self._thinking_tool_call(), _text("ok"))
        backend = _backend(server)

        first = backend.chat(MODEL, [{"role": "user", "content": "weather?"}], tools=[WEATHER_TOOL],
                             thinking=True)
        backend.chat(MODEL, self._follow_up(first["message"]), tools=[WEATHER_TOOL], thinking=False)

        types = [block["type"] for block in server.body(1)["messages"][1]["content"]]
        assert types == ["tool_use"]
        assert "anthropic-beta" not in server.requests[1].headers


# ── embeddings, listing, warm-up ───────────────────────────────────────────

class TestOtherMethods:
    def test_embeddings_are_not_offered(self):
        server = _Server()

        assert _backend(server).embed("text", "any-model") is None
        assert server.requests == []

    def test_lists_the_model_ids(self):
        server = _Server({"data": [
            {"type": "model", "id": "claude-haiku-5-5", "display_name": "Claude Haiku 5.5",
             "created_at": "2026-01-01T00:00:00Z"},
            {"type": "model", "id": "claude-opus-5-5", "display_name": "Claude Opus 5.5",
             "created_at": "2026-01-01T00:00:00Z"},
        ], "has_more": False, "first_id": "claude-haiku-5-5", "last_id": "claude-opus-5-5"})

        assert _backend(server).list_models() == ["claude-haiku-5-5", "claude-opus-5-5"]
        assert server.requests[0].url.path == "/v1/models"

    def test_list_models_is_empty_on_error(self):
        server = _Server(_error(401, "bad key", "authentication_error"))

        assert _backend(server).list_models() == []

    def test_warm_up_sends_one_minimal_request(self):
        server = _Server(_text("h"))

        assert _backend(server).warm_up(MODEL) is True
        assert server.body()["max_tokens"] == 1

    def test_warm_up_reports_failure(self):
        server = _Server(_error(404, "no such model", "not_found_error"))

        assert _backend(server).warm_up("claude-nope") is False

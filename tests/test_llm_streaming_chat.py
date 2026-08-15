"""Behaviour tests for streamed ``chat()`` on the OpenAI-compatible backend.

``chat()`` gains an optional ``on_token`` callback. When supplied the
request is streamed (SSE) and each content delta is handed to the
callback as it arrives, so the UI can render a reply while it is still
being generated. The important guarantee is that the *return value* is
unchanged: callers still get the same normalised ``{"message": {...}}``
dict, tool calls included, whether or not they stream. That keeps the
reply engine's tool loop working identically on both paths.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from jarvis.llm.openai_compatible import OpenAICompatibleBackend


def _sse(*chunks) -> list:
    """Encode dicts as SSE ``data:`` lines, terminated by [DONE]."""
    lines = [f"data: {json.dumps(c)}".encode() for c in chunks]
    lines.append(b"data: [DONE]")
    return lines


def _stream_response(lines):
    resp = MagicMock()
    resp.status_code = 200
    resp.iter_lines.return_value = lines
    resp.raise_for_status = MagicMock()
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=None)
    return resp


def _delta(**d):
    return {"choices": [{"delta": d}]}


class TestStreamedContent:
    @patch("requests.Session.post")
    def test_tokens_are_delivered_as_they_arrive(self, mock_post):
        mock_post.return_value = _stream_response(_sse(
            _delta(content="Bon"), _delta(content="jour"), _delta(content=" !"),
        ))
        seen = []
        be = OpenAICompatibleBackend("http://x/v1")
        be.chat("m", [{"role": "user", "content": "hi"}], on_token=seen.append)
        assert seen == ["Bon", "jour", " !"]

    @patch("requests.Session.post")
    def test_return_value_matches_the_non_streaming_shape(self, mock_post):
        mock_post.return_value = _stream_response(_sse(
            _delta(role="assistant", content="Bonjour"), _delta(content=" !"),
        ))
        be = OpenAICompatibleBackend("http://x/v1")
        out = be.chat("m", [{"role": "user", "content": "hi"}], on_token=lambda t: None)
        # Same contract as the buffered path: top-level "message" with the
        # full concatenated content.
        assert out["message"]["content"] == "Bonjour !"
        assert out["message"]["role"] == "assistant"

    @patch("requests.Session.post")
    def test_requests_a_stream_when_streaming(self, mock_post):
        mock_post.return_value = _stream_response(_sse(_delta(content="x")))
        be = OpenAICompatibleBackend("http://x/v1")
        be.chat("m", [{"role": "user", "content": "hi"}], on_token=lambda t: None)
        assert mock_post.call_args.kwargs["json"]["stream"] is True

    @patch("requests.Session.post")
    def test_without_callback_the_request_is_not_streamed(self, mock_post):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "hi"}}]
        }
        resp.raise_for_status = MagicMock()
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=None)
        mock_post.return_value = resp
        be = OpenAICompatibleBackend("http://x/v1")
        out = be.chat("m", [{"role": "user", "content": "hi"}])
        assert mock_post.call_args.kwargs["json"]["stream"] is False
        assert out["message"]["content"] == "hi"


class TestStreamedToolCalls:
    """Tool calls arrive as indexed deltas whose arguments are split across
    chunks; they must be reassembled into the same shape the engine's tool
    loop already consumes (arguments decoded to a dict)."""

    @patch("requests.Session.post")
    def test_tool_call_deltas_are_reassembled(self, mock_post):
        mock_post.return_value = _stream_response(_sse(
            _delta(tool_calls=[{
                "index": 0, "id": "call_1", "type": "function",
                "function": {"name": "getWeather", "arguments": '{"loc'},
            }]),
            _delta(tool_calls=[{
                "index": 0, "function": {"arguments": 'ation": "Paris"}'},
            }]),
        ))
        be = OpenAICompatibleBackend("http://x/v1")
        out = be.chat("m", [{"role": "user", "content": "météo"}], on_token=lambda t: None)
        calls = out["message"]["tool_calls"]
        assert len(calls) == 1
        assert calls[0]["id"] == "call_1"
        assert calls[0]["function"]["name"] == "getWeather"
        # Decoded to a dict, exactly like the buffered path.
        assert calls[0]["function"]["arguments"] == {"location": "Paris"}

    @patch("requests.Session.post")
    def test_multiple_tool_calls_keep_their_own_arguments(self, mock_post):
        mock_post.return_value = _stream_response(_sse(
            _delta(tool_calls=[
                {"index": 0, "id": "a", "function": {"name": "one", "arguments": '{"x":'}},
                {"index": 1, "id": "b", "function": {"name": "two", "arguments": '{"y":'}},
            ]),
            _delta(tool_calls=[
                {"index": 0, "function": {"arguments": "1}"}},
                {"index": 1, "function": {"arguments": "2}"}},
            ]),
        ))
        be = OpenAICompatibleBackend("http://x/v1")
        out = be.chat("m", [{"role": "user", "content": "hi"}], on_token=lambda t: None)
        calls = {c["id"]: c for c in out["message"]["tool_calls"]}
        assert calls["a"]["function"]["arguments"] == {"x": 1}
        assert calls["b"]["function"]["arguments"] == {"y": 2}

    @patch("requests.Session.post")
    def test_tool_call_deltas_do_not_fire_the_token_callback(self, mock_post):
        # Only user-visible text should reach the UI.
        mock_post.return_value = _stream_response(_sse(
            _delta(tool_calls=[{"index": 0, "id": "a",
                                "function": {"name": "one", "arguments": "{}"}}]),
        ))
        seen = []
        be = OpenAICompatibleBackend("http://x/v1")
        be.chat("m", [{"role": "user", "content": "hi"}], on_token=seen.append)
        assert seen == []


class TestStreamingThroughDecorators:
    """The redaction decorator wraps every cloud call. It must pass
    ``on_token`` through, or streaming silently dies in production while
    every backend-level test still passes."""

    @patch("requests.Session.post")
    def test_redacting_backend_forwards_on_token(self, mock_post):
        from jarvis.llm.redacting import RedactingBackend

        mock_post.return_value = _stream_response(_sse(
            _delta(content="Bon"), _delta(content="jour"),
        ))
        seen = []
        be = RedactingBackend(OpenAICompatibleBackend("http://x/v1"))
        out = be.chat("m", [{"role": "user", "content": "hi"}], on_token=seen.append)
        assert seen == ["Bon", "jour"]
        assert out["message"]["content"] == "Bonjour"

    @patch("requests.Session.post")
    def test_factory_backend_streams_end_to_end(self, mock_post):
        # The backend the app actually resolves (redaction on) must stream.
        from types import SimpleNamespace
        from jarvis.llm import clear_backend_cache, get_llm_backend

        clear_backend_cache()
        try:
            mock_post.return_value = _stream_response(_sse(_delta(content="ok")))
            cfg = SimpleNamespace(
                llm_provider="openai_compatible",
                llm_base_url="http://cloud/v1",
                llm_api_key="sk-x",
                llm_chat_model="vendor/model",
                auto_redact_before_cloud=True,
                llm_extra_body={},
                ollama_base_url="http://127.0.0.1:11434",
            )
            seen = []
            out = get_llm_backend(cfg).chat(
                "vendor/model", [{"role": "user", "content": "hi"}], on_token=seen.append
            )
            assert seen == ["ok"]
            assert out["message"]["content"] == "ok"
        finally:
            clear_backend_cache()


class TestStreamedFailures:
    @patch("requests.Session.post")
    def test_returns_none_on_error(self, mock_post):
        mock_post.side_effect = RuntimeError("boom")
        be = OpenAICompatibleBackend("http://x/v1")
        assert be.chat("m", [{"role": "user", "content": "hi"}], on_token=lambda t: None) is None

    @patch("requests.Session.post")
    def test_malformed_sse_lines_are_skipped(self, mock_post):
        mock_post.return_value = _stream_response(
            [b": ping", b"data: not-json"] + _sse(_delta(content="ok"))
        )
        seen = []
        be = OpenAICompatibleBackend("http://x/v1")
        out = be.chat("m", [{"role": "user", "content": "hi"}], on_token=seen.append)
        assert seen == ["ok"]
        assert out["message"]["content"] == "ok"

    @patch("requests.Session.post")
    def test_a_failing_callback_does_not_break_the_reply(self, mock_post):
        # A UI hiccup must not lose the reply the user already paid for.
        mock_post.return_value = _stream_response(_sse(
            _delta(content="a"), _delta(content="b"),
        ))
        def boom(_):
            raise RuntimeError("ui died")
        be = OpenAICompatibleBackend("http://x/v1")
        out = be.chat("m", [{"role": "user", "content": "hi"}], on_token=boom)
        assert out["message"]["content"] == "ab"

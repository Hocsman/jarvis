"""Behaviour tests for the cloud-latency features of the OpenAI-compatible
backend and its factory.

These pin the mechanisms that keep a remote cloud endpoint responsive:

- ``extra_body`` provider-specific request fields (e.g. OpenRouter's
  ``{"provider": {"sort": "throughput"}}``) are merged into every chat
  payload, without clobbering the keys the backend owns.
- ``warm_up`` fires a real minimal request so the persistent session's
  connection and the provider are warm before the first user query.
- ``get_llm_backend`` returns the same instance for the same config, so
  the warm session is reused across the many calls a reply makes, and
  rebuilds after ``clear_backend_cache``.
- the config's ``llm_extra_body`` reaches the backend through the factory.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from jarvis.llm import clear_backend_cache, get_llm_backend
from jarvis.llm.openai_compatible import OpenAICompatibleBackend


def _resp(json_data):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=None)
    return resp


THROUGHPUT = {"provider": {"sort": "throughput"}}


class TestExtraBodyMerge:
    @patch("requests.Session.post")
    def test_extra_body_merged_into_direct_payload(self, mock_post):
        mock_post.return_value = _resp(
            {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}
        )
        be = OpenAICompatibleBackend("http://x/v1", extra_body=THROUGHPUT)
        be.direct("m", "sys", "user", timeout_sec=5)
        sent = mock_post.call_args.kwargs["json"]
        assert sent["provider"] == {"sort": "throughput"}

    @patch("requests.Session.post")
    def test_extra_body_merged_into_chat_payload(self, mock_post):
        mock_post.return_value = _resp(
            {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}
        )
        be = OpenAICompatibleBackend("http://x/v1", extra_body=THROUGHPUT)
        be.chat("m", [{"role": "user", "content": "hi"}], timeout_sec=5)
        sent = mock_post.call_args.kwargs["json"]
        assert sent["provider"] == {"sort": "throughput"}

    @patch("requests.Session.post")
    def test_extra_body_never_overrides_backend_owned_keys(self, mock_post):
        mock_post.return_value = _resp(
            {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}
        )
        # A hostile/confused extra_body must not be able to swap the model
        # or messages the backend set itself.
        be = OpenAICompatibleBackend(
            "http://x/v1", extra_body={"model": "evil", "messages": []}
        )
        be.direct("real-model", "sys", "user", timeout_sec=5)
        sent = mock_post.call_args.kwargs["json"]
        assert sent["model"] == "real-model"
        assert sent["messages"] and sent["messages"][-1]["content"] == "user"

    @patch("requests.Session.post")
    def test_no_extra_body_leaves_payload_standard(self, mock_post):
        mock_post.return_value = _resp(
            {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}
        )
        be = OpenAICompatibleBackend("http://x/v1")
        be.direct("m", "sys", "user", timeout_sec=5)
        sent = mock_post.call_args.kwargs["json"]
        assert "provider" not in sent


class TestWarmUp:
    @patch("requests.Session.post")
    def test_warm_up_fires_minimal_request_and_returns_true(self, mock_post):
        mock_post.return_value = _resp(
            {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        )
        be = OpenAICompatibleBackend("http://x/v1")
        assert be.warm_up("m", timeout_sec=5) is True
        sent = mock_post.call_args.kwargs["json"]
        assert sent["model"] == "m"
        assert sent["max_tokens"] == 1  # minimal: don't pay for generation

    @patch("requests.Session.post")
    def test_warm_up_returns_false_on_error(self, mock_post):
        mock_post.side_effect = RuntimeError("boom")
        be = OpenAICompatibleBackend("http://x/v1")
        assert be.warm_up("m", timeout_sec=5) is False

    def test_warm_up_returns_false_for_empty_model(self):
        be = OpenAICompatibleBackend("http://x/v1")
        assert be.warm_up("", timeout_sec=5) is False


def _cfg(**over):
    base = dict(
        llm_provider="openai_compatible",
        llm_base_url="http://cloud/v1",
        llm_api_key="sk-x",
        llm_chat_model="vendor/model",
        auto_redact_before_cloud=False,
        llm_extra_body={},
        ollama_base_url="http://127.0.0.1:11434",
    )
    base.update(over)
    return SimpleNamespace(**base)


class TestFactoryMemoisationAndWiring:
    def setup_method(self):
        clear_backend_cache()

    def teardown_method(self):
        clear_backend_cache()

    def test_same_config_returns_same_instance(self):
        cfg = _cfg()
        assert get_llm_backend(cfg) is get_llm_backend(cfg)

    def test_changed_extra_body_rebuilds(self):
        a = get_llm_backend(_cfg(llm_extra_body={}))
        b = get_llm_backend(_cfg(llm_extra_body=THROUGHPUT))
        assert a is not b

    def test_clear_cache_forces_rebuild(self):
        a = get_llm_backend(_cfg())
        clear_backend_cache()
        b = get_llm_backend(_cfg())
        assert a is not b

    def test_extra_body_from_config_reaches_backend(self):
        be = get_llm_backend(_cfg(llm_extra_body=THROUGHPUT))
        # No redaction in this cfg, so the backend is the bare OpenAI one.
        assert isinstance(be, OpenAICompatibleBackend)
        assert be._extra_body == THROUGHPUT

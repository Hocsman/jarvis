"""The tool router waits for its model on a deadline of its own.

The router is a classification call that stands between the user's sentence
and the first token of the reply. It waited on ``llm_tools_timeout_sec``, a
ceiling of minutes shared with other contexts, so a provider that stalled for
fifteen seconds held the whole turn for fifteen seconds. These tests pin the
properties that keep the wait bounded:

1. the deadline is its own setting, read from the user's file, kept inside
   sane bounds, and far below the shared tools ceiling by default;
2. the reply engine and ``toolSearchTool`` both hand that deadline to the
   router, not the shared one;
3. a router whose server accepts the request and never answers is given up
   on at its deadline, and the selection degrades to the keyword strategy;
4. the same holds for the intent judge, whose give-up leaves the listener on
   its no-verdict path instead of waiting out a stalled provider.
"""

from __future__ import annotations

import contextlib
import json
import socket
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest


# ── Helpers ───────────────────────────────────────────────────────────


def _settings_from(tmp_path, monkeypatch, values: dict):
    from jarvis.config import load_settings

    path = tmp_path / "config.json"
    path.write_text(json.dumps(values), encoding="utf-8")
    monkeypatch.setenv("JARVIS_CONFIG_PATH", str(path))
    return load_settings()


class _Tool:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description


def _catalogue():
    return {
        "getWeather": _Tool("getWeather", "Get current weather conditions."),
        "fetchMeals": _Tool("fetchMeals", "Retrieve meals from the database."),
        "stop": _Tool("stop", "End the current conversation."),
    }


@contextlib.contextmanager
def _server_that_never_answers():
    """A local port that accepts the connection and then says nothing.

    That is what a stalled upstream looks like from the client: the request
    is delivered, the socket stays open, no byte comes back.
    """
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(8)
    listener.settimeout(0.05)
    held: list[socket.socket] = []
    stop = threading.Event()

    def accept_and_hold():
        while not stop.is_set():
            try:
                conn, _ = listener.accept()
            except (socket.timeout, OSError):
                continue
            held.append(conn)

    thread = threading.Thread(target=accept_and_hold, daemon=True)
    thread.start()
    try:
        yield listener.getsockname()[1]
    finally:
        stop.set()
        thread.join(timeout=2)
        for conn in held:
            conn.close()
        listener.close()


def _stalled_backend(port: int):
    from jarvis.llm.openai_compatible import OpenAICompatibleBackend

    return OpenAICompatibleBackend(f"http://127.0.0.1:{port}/v1", api_key="test-key")


# ── 1. The setting ────────────────────────────────────────────────────


@pytest.mark.unit
def test_the_router_deadline_is_its_own_and_far_below_the_shared_tools_one():
    from jarvis.config import get_default_config

    defaults = get_default_config()

    assert 0 < defaults["tool_router_timeout_sec"] < defaults["llm_tools_timeout_sec"]


@pytest.mark.unit
def test_the_loader_reads_the_router_deadline_from_the_users_file(tmp_path, monkeypatch):
    settings = _settings_from(tmp_path, monkeypatch, {"tool_router_timeout_sec": 6.5})

    assert settings.tool_router_timeout_sec == 6.5


@pytest.mark.unit
@pytest.mark.parametrize("asked", [0, -3, 0.01, 100000])
def test_the_router_deadline_stays_inside_sane_bounds(tmp_path, monkeypatch, asked):
    """Zero or a negative number would fail every routing call at once; a huge
    one is the shared ceiling again under another name."""
    from jarvis.config import get_default_config

    settings = _settings_from(tmp_path, monkeypatch, {"tool_router_timeout_sec": asked})

    assert 1.0 <= settings.tool_router_timeout_sec
    assert settings.tool_router_timeout_sec < get_default_config()["llm_tools_timeout_sec"]


@pytest.mark.unit
def test_the_settings_window_offers_the_router_deadline_within_the_loaders_bounds(
    tmp_path, monkeypatch
):
    """What the window lets the user type must be what the loader keeps."""
    from desktop_app.settings_window import FIELD_METADATA

    field = next((fm for fm in FIELD_METADATA if fm.key == "tool_router_timeout_sec"), None)
    assert field is not None, "the router deadline has no field in the settings window"
    assert field.field_type == "float"

    low = _settings_from(tmp_path, monkeypatch, {"tool_router_timeout_sec": field.min_val})
    high = _settings_from(tmp_path, monkeypatch, {"tool_router_timeout_sec": field.max_val})
    assert low.tool_router_timeout_sec == field.min_val
    assert high.tool_router_timeout_sec == field.max_val


# ── 2. The deadline reaches the router ────────────────────────────────


@pytest.mark.unit
def test_the_resolver_reads_the_dedicated_setting_not_the_shared_one():
    from jarvis.tools.selection import router_timeout_sec

    cfg = SimpleNamespace(tool_router_timeout_sec=3.5, llm_tools_timeout_sec=77.0)

    assert router_timeout_sec(cfg) == 3.5


@pytest.mark.unit
def test_the_resolver_stays_bounded_when_the_setting_is_missing():
    from jarvis.config import get_default_config
    from jarvis.tools.selection import router_timeout_sec

    value = router_timeout_sec(SimpleNamespace(llm_tools_timeout_sec=300.0))

    assert 0 < value < get_default_config()["llm_tools_timeout_sec"]


@pytest.mark.unit
def test_tool_search_hands_the_router_deadline_to_select_tools(mock_config, monkeypatch):
    captured: dict = {}

    def fake_select_tools(**kwargs):
        captured.update(kwargs)
        return ["stop"]

    monkeypatch.setattr("jarvis.tools.builtin.tool_search.select_tools", fake_select_tools)
    mock_config.tool_router_timeout_sec = 3.5
    mock_config.llm_tools_timeout_sec = 77.0

    from jarvis.tools.base import ToolContext
    from jarvis.tools.builtin.tool_search import ToolSearchTool

    ToolSearchTool().run(
        {"query": "weather"},
        ToolContext(
            db=None, cfg=mock_config, system_prompt="", original_prompt="anything",
            redacted_text="anything", max_retries=0, user_print=lambda *_: None,
        ),
    )

    assert captured.get("llm_timeout_sec") == 3.5


@pytest.mark.unit
def test_the_reply_engine_hands_the_router_deadline_to_select_tools(
    mock_config, db, dialogue_memory
):
    from jarvis.reply.engine import run_reply_engine

    captured: dict = {}

    def fake_select_tools(**kwargs):
        captured.update(kwargs)
        return ["stop"]

    mock_config.tool_selection_strategy = "llm"
    mock_config.tool_router_timeout_sec = 3.5
    mock_config.llm_tools_timeout_sec = 77.0

    with patch("jarvis.reply.engine.plan_query", return_value=[]), \
         patch("jarvis.reply.engine.chat_with_messages",
               return_value={"message": {"content": "Bonjour.", "role": "assistant"}}), \
         patch("jarvis.reply.engine.extract_search_params_for_memory",
               return_value={"keywords": []}), \
         patch("jarvis.reply.engine.select_tools", side_effect=fake_select_tools):
        run_reply_engine(
            db=db, cfg=mock_config, tts=None, text="bonjour",
            dialogue_memory=dialogue_memory,
        )

    assert captured.get("llm_timeout_sec") == 3.5


# ── 3. A router that never answers is given up on ─────────────────────


@pytest.mark.unit
def test_a_router_that_never_answers_is_given_up_on_at_its_deadline():
    from jarvis.tools.selection import ToolSelectionStrategy, select_tools

    with _server_that_never_answers() as port:
        started = time.monotonic()
        selected = select_tools(
            "weather in London", _catalogue(), {},
            strategy=ToolSelectionStrategy.LLM,
            llm_backend=_stalled_backend(port),
            llm_model="router-model",
            llm_timeout_sec=0.4,
        )
        waited = time.monotonic() - started

    assert waited < 5.0, f"the router held the turn for {waited:.1f}s past its 0.4s deadline"
    # The keyword strategy answered: it narrows on the query, it does not
    # hand the model the whole catalogue.
    assert "getWeather" in selected
    assert "fetchMeals" not in selected


# ── 4. A judge that never answers leaves the no-verdict path ──────────


@pytest.mark.unit
def test_a_judge_that_never_answers_gives_no_verdict_at_its_deadline():
    from jarvis.listening.intent_judge import create_intent_judge
    from jarvis.listening.transcript_buffer import TranscriptSegment

    cfg = SimpleNamespace(
        wake_word="yuba", wake_aliases=[], intent_judge_model="judge-model",
        intent_judge_timeout_sec=0.4, intent_judge_thinking_enabled=False,
        low_power_mode=False,
    )
    now = time.time()
    segments = [TranscriptSegment("yuba, quelle heure est-il", now - 3.0, now - 1.0)]

    with _server_that_never_answers() as port:
        backend = _stalled_backend(port)
        with patch("jarvis.listening.intent_judge.get_auxiliary_backend", return_value=backend):
            judge = create_intent_judge(cfg)
            started = time.monotonic()
            verdict = judge.judge(
                segments=segments, wake_timestamp=now - 3.0,
                current_text="yuba, quelle heure est-il",
            )
            waited = time.monotonic() - started

    assert verdict is None
    assert waited < 5.0, f"the judge held the audio loop for {waited:.1f}s past its 0.4s deadline"
    assert judge.last_failure_reason, "the listener prints why the judge was unavailable"

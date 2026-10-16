"""The latency journal, as the pipeline feeds it.

``test_latency_journal.py`` pins the journal on its own. These tests pin the
places that mark it: the listener (a transcript, the intent judge, the
dispatch), the reply engine (the router giving up) and the daemon (the
retention at start-up).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from test_hot_window_input import _create_listener, _install_intent_judge, _make_judgment


class _Clock:
    def __init__(self, now: float = 1000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def journal(db):
    from jarvis.latency import get_latency_journal

    clock = _Clock()
    j = get_latency_journal()
    j.reset(clock=clock)
    j.configure(db, SimpleNamespace(latency_journal_enabled=True))
    j.test_clock = clock  # the tests move time through this
    yield j
    j.reset()


def _only_row(db) -> dict:
    rows = db.recent_latency()
    assert len(rows) == 1, f"expected one row, found {len(rows)}"
    return rows[0]


def _a_listener_timed_against(db, journal):
    """A listener whose journal writes to ``db``.

    A listener binds the journal to its own database when it is built, so the
    journal is pointed back at the test's database right after.
    """
    listener, tts = _create_listener()
    journal.configure(db, SimpleNamespace(latency_journal_enabled=True))
    return listener, tts


# ── The listener ──────────────────────────────────────────────────────


@pytest.mark.unit
@patch("builtins.print")
def test_a_spoken_query_becomes_one_row_with_the_judge_and_the_engine_stages(_print, db, journal):
    clock = journal.test_clock
    listener, _tts = _a_listener_timed_against(db, journal)
    listener.cfg.tool_router_model = "router-m"
    listener.cfg.llm_chat_model = "chat-m"
    judge = _install_intent_judge(listener, _make_judgment(directed=True, query="what time is it"))

    # The wake-word utterance ended at +0 and its transcript is ready at +400 ms.
    clock.now = 1000.4
    listener._process_transcript(
        "jarvis what time is it", utterance_energy=0.01,
        utterance_start_time=999.0, utterance_end_time=1000.0,
    )
    assert judge.judge.called, "the wake word should have reached the intent judge"

    def fake_engine(*args, **kwargs):
        clock.now += 0.5
        kwargs["on_stage"]("routing")
        clock.now += 0.5
        kwargs["on_stage"]("generating")
        clock.now += 2.0
        return "It is noon."

    clock.now = 1006.0                         # dispatched 6 s after the end of speech
    with patch("jarvis.reply.engine.run_reply_engine", side_effect=fake_engine), \
         patch.object(listener, "_speak_reply"):
        listener._dispatch_query("what time is it")

    row = _only_row(db)
    assert row["transcribed_ms"] == 400
    assert row["judge_outcome"] == "directed"
    assert row["judge_ms"] is not None
    assert row["dispatch_ms"] == 6000
    assert (row["route_ms"], row["generate_ms"]) == (500, 2000)
    assert row["total_ms"] == 6000 + 500 + 500 + 2000
    assert (row["judge_model"], row["router_model"], row["chat_model"]) == (
        "gemma4:e2b", "router-m", "chat-m",
    )
    listener.state_manager.stop()


@pytest.mark.unit
@patch("builtins.print")
def test_a_judge_that_cannot_answer_is_recorded_as_no_verdict(_print, db, journal):
    clock = journal.test_clock
    listener, _tts = _a_listener_timed_against(db, journal)
    _install_intent_judge(listener, None)

    clock.now = 1000.4
    listener._process_transcript(
        "jarvis what time is it", utterance_energy=0.01,
        utterance_start_time=999.0, utterance_end_time=1000.0,
    )
    clock.now = 1001.0
    with patch("jarvis.reply.engine.run_reply_engine", return_value="ok"), \
         patch.object(listener, "_speak_reply"):
        listener._dispatch_query("what time is it")

    assert _only_row(db)["judge_outcome"] == "no_verdict"
    listener.state_manager.stop()


@pytest.mark.unit
@patch("builtins.print")
def test_an_engine_that_raises_still_closes_the_turn(_print, db, journal):
    clock = journal.test_clock
    listener, _tts = _a_listener_timed_against(db, journal)
    clock.now = 1000.4
    listener._process_transcript(
        "jarvis hello", utterance_energy=0.01,
        utterance_start_time=999.0, utterance_end_time=1000.0,
    )
    clock.now = 1001.0
    with patch("jarvis.reply.engine.run_reply_engine", side_effect=RuntimeError("boom")):
        listener._dispatch_query("hello")

    assert len(db.recent_latency()) == 1
    listener.state_manager.stop()


@pytest.mark.unit
@patch("builtins.print")
def test_the_listener_binds_the_journal_to_its_database(_print):
    from jarvis.latency import get_latency_journal

    journal = get_latency_journal()
    journal.reset()
    try:
        listener, _tts = _create_listener()

        assert journal.enabled is True
        listener.state_manager.stop()
    finally:
        journal.reset()


# ── The reply engine ──────────────────────────────────────────────────


@pytest.mark.unit
def test_a_router_that_gave_up_is_flagged_in_the_journal(mock_config, db, dialogue_memory, journal):
    from jarvis.reply.engine import run_reply_engine

    gave_up = MagicMock()
    gave_up.direct.return_value = None
    mock_config.tool_selection_strategy = "llm"
    mock_config.tool_router_model = "router-model"

    journal.begin_turn({})
    with patch("jarvis.reply.engine.plan_query", return_value=[]), \
         patch("jarvis.reply.engine.chat_with_messages",
               return_value={"message": {"content": "Il fait beau.", "role": "assistant"}}), \
         patch("jarvis.reply.engine.extract_search_params_for_memory",
               return_value={"keywords": []}), \
         patch("jarvis.reply.engine.get_auxiliary_backend", return_value=gave_up):
        run_reply_engine(
            db=db, cfg=mock_config, tts=None, text="what is the weather in Paris",
            dialogue_memory=dialogue_memory,
        )
    journal.end_turn()

    assert _only_row(db)["router_degraded"] == 1


@pytest.mark.unit
def test_a_router_that_answered_is_not_flagged(mock_config, db, dialogue_memory, journal):
    from jarvis.reply.engine import run_reply_engine

    answered = MagicMock()
    answered.direct.return_value = "getWeather"
    mock_config.tool_selection_strategy = "llm"
    mock_config.tool_router_model = "router-model"

    journal.begin_turn({})
    with patch("jarvis.reply.engine.plan_query", return_value=[]), \
         patch("jarvis.reply.engine.chat_with_messages",
               return_value={"message": {"content": "Il fait beau.", "role": "assistant"}}), \
         patch("jarvis.reply.engine.extract_search_params_for_memory",
               return_value={"keywords": []}), \
         patch("jarvis.reply.engine.get_auxiliary_backend", return_value=answered):
        run_reply_engine(
            db=db, cfg=mock_config, tts=None, text="what is the weather in Paris",
            dialogue_memory=dialogue_memory,
        )
    journal.end_turn()

    assert _only_row(db)["router_degraded"] == 0


# ── The daemon ────────────────────────────────────────────────────────


@pytest.mark.unit
def test_start_up_drops_the_rows_older_than_the_retention(db):
    from jarvis.daemon import _apply_latency_retention

    now = datetime.now(timezone.utc)
    db.record_latency({"ts_utc": (now - timedelta(days=40)).isoformat(), "origin": "voice"})
    db.record_latency({"ts_utc": (now - timedelta(days=3)).isoformat(), "origin": "voice"})

    _apply_latency_retention(db, SimpleNamespace(latency_journal_days=30))

    assert len(db.recent_latency()) == 1


@pytest.mark.unit
def test_start_up_retention_never_raises(db):
    from jarvis.daemon import _apply_latency_retention

    broken = MagicMock()
    broken.prune_latency.side_effect = RuntimeError("locked")

    _apply_latency_retention(broken, SimpleNamespace(latency_journal_days=30))

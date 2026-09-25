"""Behaviour tests for the text-chat submission path in the daemon.

These verify the contract in ``src/desktop_app/chat_window.spec.md``:

- ``submit_text_query`` runs the reply engine with ``tts=None`` and the shared
  global dialogue memory (one conversation for voice + text).
- It is fire-and-forget; results arrive via callbacks, not the return value.
- It fires ``on_start`` (with the redacted query) and ``on_complete`` (with the
  reply or ``None`` on failure).
- It rejects a concurrent submission via ``on_busy`` (one query at a time).
- In IPC mode it emits ``__CHAT__:`` JSON events to stdout.
- It never passes unredacted user text to the reply engine or to IPC.

Tests patch ``jarvis.reply.engine.run_reply_engine`` (the canonical location
the daemon imports from at call time) per the conftest note about module
instance identity.
"""

import json
import sys
import threading
import time

import pytest

from jarvis import daemon
from jarvis.memory.conversation import DialogueMemory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reset_daemon_globals():
    """Restore daemon module globals between tests."""
    daemon._global_dialogue_memory = None
    daemon._global_cfg = None
    daemon._global_db = None
    daemon._global_stop_requested = False
    daemon._global_skip_shutdown_diary_update = False
    daemon._chat_query_lock = threading.Lock()


def _install_dialogue_memory(cfg=None, db=None):
    """Install a DialogueMemory plus optional cfg/db into the daemon globals.

    The contract tests pass mock cfg/db so ``submit_text_query`` can hand
    them to the (patched) reply engine without touching the filesystem.
    """
    dm = DialogueMemory(inactivity_timeout=300, max_interactions=20)
    daemon._global_dialogue_memory = dm
    daemon._global_cfg = cfg
    daemon._global_db = db
    return dm


def _wait_for_complete(events, timeout=5.0):
    """Block until an ``on_complete`` event lands, or time out."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if any(e[0] == "complete" for e in events):
            return
        time.sleep(0.01)
    raise AssertionError("on_complete was not fired within timeout")


def _wait_for_ipc_complete(capsys, timeout=5.0):
    """Block until a ``__CHAT__:`` ``complete`` event appears on stdout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        out = capsys.readouterr().out
        chat_lines = [
            ln for ln in out.splitlines()
            if ln.startswith(daemon.CHAT_IPC_PREFIX)
        ]
        for ln in chat_lines:
            try:
                payload = json.loads(ln[len(daemon.CHAT_IPC_PREFIX):])
            except json.JSONDecodeError:
                continue
            if payload.get("type") == "complete":
                return chat_lines
        time.sleep(0.02)
    raise AssertionError("__CHAT__: complete event was not emitted within timeout")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSubmitTextQueryContract:
    """Core contract: shares memory, no TTS, callbacks fire."""

    def setup_method(self, _method):
        _reset_daemon_globals()

    def teardown_method(self, _method):
        _reset_daemon_globals()

    def test_runs_engine_with_tts_none_and_shared_memory(self, monkeypatch):
        """The worker must call run_reply_engine with tts=None and the global
        DialogueMemory, so text and voice share one conversation."""
        dm = _install_dialogue_memory(cfg=object(), db=object())
        captured = {}

        def fake_engine(db, cfg, tts, text, dialogue_memory, language=None,
                        origin=None):
            captured["tts"] = tts
            captured["dialogue_memory"] = dialogue_memory
            captured["text"] = text
            captured["language"] = language
            captured["origin"] = origin
            return "hello from the engine"

        monkeypatch.setattr("jarvis.reply.engine.run_reply_engine", fake_engine)

        events = []
        daemon.submit_text_query(
            "hi there",
            on_start=lambda q: events.append(("start", q)),
            on_complete=lambda r: events.append(("complete", r)),
        )
        _wait_for_complete(events)

        assert captured["tts"] is None
        assert captured["dialogue_memory"] is dm
        assert captured["language"] is None
        assert captured["text"] == "hi there"
        # And it says so, so the action ledger can tell a typed request
        # from a spoken one or from something that ran unattended.
        assert captured["origin"] == "chat"

    def test_fires_on_start_with_query_then_on_complete_with_reply(self, monkeypatch):
        """on_start fires first with the query, on_complete fires last with
        the reply text. Ordering matters for the UI."""
        _install_dialogue_memory(cfg=object(), db=object())
        monkeypatch.setattr(
            "jarvis.reply.engine.run_reply_engine",
            lambda *a, **k: "the reply",
        )

        events = []
        daemon.submit_text_query(
            "what is 2+2",
            on_start=lambda q: events.append(("start", q)),
            on_complete=lambda r: events.append(("complete", r)),
        )
        _wait_for_complete(events)

        start_idx = next(i for i, e in enumerate(events) if e[0] == "start")
        complete_idx = next(i for i, e in enumerate(events) if e[0] == "complete")
        assert start_idx < complete_idx
        assert events[start_idx][1] == "what is 2+2"
        assert events[complete_idx][1] == "the reply"

    def test_on_complete_none_when_engine_returns_none(self, monkeypatch):
        """An empty/stop reply surfaces as on_complete(None), not silence."""
        _install_dialogue_memory(cfg=object(), db=object())
        monkeypatch.setattr(
            "jarvis.reply.engine.run_reply_engine", lambda *a, **k: None
        )
        events = []
        daemon.submit_text_query(
            "hi",
            on_complete=lambda r: events.append(("complete", r)),
        )
        _wait_for_complete(events)
        assert events[-1] == ("complete", None)

    def test_on_complete_none_when_engine_raises(self, monkeypatch):
        """An engine exception must not crash the worker; on_complete(None)
        fires so the UI can recover."""
        _install_dialogue_memory(cfg=object(), db=object())
        def boom(*a, **k):
            raise RuntimeError("engine exploded")

        monkeypatch.setattr("jarvis.reply.engine.run_reply_engine", boom)
        events = []
        daemon.submit_text_query(
            "hi",
            on_complete=lambda r: events.append(("complete", r)),
        )
        _wait_for_complete(events)
        assert events[-1] == ("complete", None)

    def test_callbacks_are_optional(self, monkeypatch):
        """With no callbacks registered, submit_text_query still runs the
        engine and returns without error."""
        monkeypatch.setattr(
            "jarvis.reply.engine.run_reply_engine", lambda *a, **k: "ok"
        )
        _install_dialogue_memory(cfg=object(), db=object())
        daemon.submit_text_query("hi")
        time.sleep(0.5)
        # No assertion needed — reaching here without hanging means it worked.

    def test_daemon_not_initialised_fires_complete_none(self, monkeypatch):
        """When the daemon globals are None (daemon not booted), submit_text_query
        fails open with complete(None) so the UI doesn't hang. The engine must
        never be called."""
        called = []
        monkeypatch.setattr(
            "jarvis.reply.engine.run_reply_engine",
            lambda *a, **k: called.append(True),
        )
        # Globals left as None by _reset_daemon_globals.
        events = []
        daemon.submit_text_query(
            "hi", on_complete=lambda r: events.append(("complete", r)),
        )
        _wait_for_complete(events)
        assert events[-1] == ("complete", None)
        assert called == []

    def test_empty_or_whitespace_does_not_run_engine(self, monkeypatch):
        """Empty / whitespace input is dropped before the worker spawns; no
        callbacks fire and the engine is never called."""
        called = []
        monkeypatch.setattr(
            "jarvis.reply.engine.run_reply_engine",
            lambda *a, **k: called.append(True),
        )
        _install_dialogue_memory(cfg=object(), db=object())
        events = []
        daemon.submit_text_query(
            "   ", on_start=lambda q: events.append(("start", q)),
            on_complete=lambda r: events.append(("complete", r)),
        )
        time.sleep(0.3)
        assert events == []
        assert called == []

    def test_cancel_drops_reply_and_emits_complete_none(self, monkeypatch):
        """cancel_active_chat_query sets the per-query flag so the worker drops
        the reply (complete(None)) instead of displaying it."""
        _install_dialogue_memory(cfg=object(), db=object())
        started = threading.Event()

        def slow_engine(*a, **k):
            started.set()
            time.sleep(0.3)  # let cancel fire mid-run
            return "the reply that should be dropped"

        monkeypatch.setattr("jarvis.reply.engine.run_reply_engine", slow_engine)
        events = []
        daemon.submit_text_query(
            "hi", on_complete=lambda r: events.append(("complete", r)),
        )
        assert started.wait(timeout=2)
        daemon.cancel_active_chat_query()
        _wait_for_complete(events)
        assert events[-1] == ("complete", None)

    def test_start_event_carries_redacted_query(self, monkeypatch):
        """on_start receives the redacted query, not the raw input. Verifies the
        privacy boundary with a redactable pattern (email)."""
        _install_dialogue_memory(cfg=object(), db=object())
        monkeypatch.setattr(
            "jarvis.reply.engine.run_reply_engine", lambda *a, **k: "ok"
        )
        events = []
        daemon.submit_text_query(
            "my email is test@example.com",
            on_start=lambda q: events.append(("start", q)),
            on_complete=lambda r: events.append(("complete", r)),
        )
        _wait_for_complete(events)
        start_query = next((e[1] for e in events if e[0] == "start"), None)
        assert start_query is not None
        assert "test@example.com" not in start_query
        assert "[REDACTED_EMAIL]" in start_query


@pytest.mark.unit
class TestSubmitTextQueryConcurrency:
    """One query at a time: a second submission is rejected, not queued."""

    def setup_method(self, _method):
        _reset_daemon_globals()

    def teardown_method(self, _method):
        _reset_daemon_globals()

    def test_second_submission_fires_on_busy(self, monkeypatch):
        """While a query is running, a second submission fires on_busy and
        does NOT call the reply engine a second time."""
        _install_dialogue_memory(cfg=object(), db=object())
        call_count = {"n": 0}
        slow_done = threading.Event()

        def slow_engine(*a, **k):
            call_count["n"] += 1
            slow_done.wait(timeout=5)
            return "first reply"

        monkeypatch.setattr("jarvis.reply.engine.run_reply_engine", slow_engine)

        events = []
        on_complete = lambda r: events.append(("complete", r))  # noqa: E731
        on_busy = lambda: events.append(("busy", None))  # noqa: E731

        daemon.submit_text_query("first", on_complete=on_complete, on_busy=on_busy)
        # Give the worker a moment to acquire the lock.
        time.sleep(0.1)
        daemon.submit_text_query("second", on_complete=on_complete, on_busy=on_busy)

        assert ("busy", None) in events
        assert call_count["n"] == 1  # second submission did not run the engine

        # Let the first query finish so the worker thread exits cleanly.
        slow_done.set()
        _wait_for_complete(events)


@pytest.mark.unit
class TestSubmitTextQueryIPC:
    """Subprocess mode: __CHAT__: JSON events on stdout."""

    def setup_method(self, _method):
        _reset_daemon_globals()

    def teardown_method(self, _method):
        _reset_daemon_globals()

    def test_emits_chat_ipc_events(self, monkeypatch, capsys):
        """In IPC mode, start + complete events are emitted as __CHAT__: lines
        containing JSON with the redacted query and the reply."""
        monkeypatch.setattr(
            "jarvis.reply.engine.run_reply_engine", lambda *a, **k: "ipc reply"
        )
        _install_dialogue_memory(cfg=object(), db=object())

        daemon.submit_text_query("hello world", use_ipc=True)
        chat_lines = _wait_for_ipc_complete(capsys)

        assert chat_lines, f"no __CHAT__: lines in stdout"
        types = []
        for ln in chat_lines:
            payload = json.loads(ln[len(daemon.CHAT_IPC_PREFIX):])
            assert "type" in payload
            assert "data" in payload
            types.append(payload["type"])
        assert "start" in types
        assert "complete" in types
        start_payload = json.loads(
            next(ln for ln in chat_lines if '"start"' in ln)[len(daemon.CHAT_IPC_PREFIX):]
        )
        assert start_payload["data"] == "hello world"

    def test_ipc_start_event_carries_redacted_query(self, monkeypatch, capsys):
        """The start event carries the redacted query (the daemon redacts before
        the worker starts), so a redactable pattern never appears in the IPC
        stream. Verifies the spec's privacy boundary for the subprocess path."""
        _install_dialogue_memory(cfg=object(), db=object())
        monkeypatch.setattr(
            "jarvis.reply.engine.run_reply_engine", lambda *a, **k: "ok"
        )
        daemon.submit_text_query("my email is test@example.com", use_ipc=True)
        chat_lines = _wait_for_ipc_complete(capsys)
        start_line = next(ln for ln in chat_lines if '"start"' in ln)
        payload = json.loads(start_line[len(daemon.CHAT_IPC_PREFIX):])
        assert "test@example.com" not in json.dumps(payload["data"])
        assert "@" not in json.dumps(payload["data"])


@pytest.mark.unit
class TestChatQueryStdinHandler:
    """The stdin monitor parses ``__CHAT_QUERY__:`` lines (subprocess mode)."""

    def setup_method(self, _method):
        _reset_daemon_globals()

    def teardown_method(self, _method):
        _reset_daemon_globals()

    def test_non_chat_line_returns_false(self):
        """Lines without the prefix are not consumed so SHUTDOWN/EOF still work."""
        assert daemon.handle_chat_query_stdin_line("SHUTDOWN") is False
        assert daemon.handle_chat_query_stdin_line("some random log line") is False
        assert daemon.handle_chat_query_stdin_line("") is False

    def test_chat_query_line_submits_and_returns_true(self, monkeypatch, capsys):
        """A valid __CHAT_QUERY__ line submits the query (via use_ipc=True) and
        returns True so the caller knows not to treat it as shutdown."""
        monkeypatch.setattr(
            "jarvis.reply.engine.run_reply_engine", lambda *a, **k: "stdin reply"
        )
        _install_dialogue_memory(cfg=object(), db=object())
        line = f'{daemon.CHAT_QUERY_IPC_PREFIX}{{"text":"hello from stdin"}}'
        assert daemon.handle_chat_query_stdin_line(line) is True
        chat_lines = _wait_for_ipc_complete(capsys)
        start_payload = json.loads(
            next(ln for ln in chat_lines if '"start"' in ln)[len(daemon.CHAT_IPC_PREFIX):]
        )
        assert start_payload["data"] == "hello from stdin"

    def test_malformed_chat_query_line_is_swallowed(self, monkeypatch):
        """A malformed JSON payload must not crash the monitor; it returns True
        (the line was addressed to the chat handler) and submits nothing."""
        submitted = []
        monkeypatch.setattr(
            daemon, "submit_text_query",
            lambda *a, **k: submitted.append(k),
        )
        _install_dialogue_memory(cfg=object(), db=object())
        line = f'{daemon.CHAT_QUERY_IPC_PREFIX}not valid json'
        assert daemon.handle_chat_query_stdin_line(line) is True
        assert submitted == []

    def test_non_string_chat_query_payload_is_swallowed(self, monkeypatch):
        """A non-string text payload is consumed but not submitted."""
        submitted = []
        monkeypatch.setattr(
            daemon,
            "submit_text_query",
            lambda *a, **k: submitted.append((a, k)),
        )
        line = f'{daemon.CHAT_QUERY_IPC_PREFIX}{{"text":["not", "text"]}}'
        assert daemon.handle_chat_query_stdin_line(line) is True
        assert submitted == []


@pytest.mark.unit
class TestDaemonShutdownMode:
    """Shutdown requests can skip the final diary LLM pass when explicitly asked."""

    def setup_method(self, _method):
        _reset_daemon_globals()

    def teardown_method(self, _method):
        _reset_daemon_globals()

    def test_normal_stop_keeps_shutdown_diary_update_enabled(self):
        daemon.request_stop()

        assert daemon.is_stop_requested() is True
        assert daemon.is_shutdown_diary_update_skipped() is False

    def test_fast_stop_marks_shutdown_diary_update_skipped(self):
        daemon.request_stop(skip_diary_update=True)

        assert daemon.is_stop_requested() is True
        assert daemon.is_shutdown_diary_update_skipped() is True

    def test_shutdown_skip_diary_command_is_not_a_chat_query(self):
        assert daemon.handle_chat_query_stdin_line(daemon.SHUTDOWN_SKIP_DIARY_COMMAND) is False


@pytest.mark.unit
class TestGetHotWindowMessages:
    """``get_hot_window_messages`` backs the chat window's first-show replay."""

    def setup_method(self, _method):
        _reset_daemon_globals()

    def teardown_method(self, _method):
        _reset_daemon_globals()

    def test_empty_when_daemon_not_booted(self):
        assert daemon.get_hot_window_messages() == []

    def test_returns_recent_turns_in_order(self):
        dm = _install_dialogue_memory()
        dm.add_message("user", "what is the weather")
        dm.add_message("assistant", "It is sunny.")

        messages = daemon.get_hot_window_messages()

        assert [m["role"] for m in messages] == ["user", "assistant"]
        assert messages[0]["content"] == "what is the weather"
        assert messages[1]["content"] == "It is sunny."

    def test_empty_when_hot_window_has_aged_out(self):
        """Turns older than the recent window are not replayed."""
        import time as _time
        dm = _install_dialogue_memory()
        # A 1-second recent window makes anything added now age out after sleep.
        dm.RECENT_WINDOW_SEC = 1
        dm.add_message("user", "old turn")
        _time.sleep(1.2)

        assert daemon.get_hot_window_messages() == []


def _chat_events(capsys):
    """The ``__CHAT__:`` events printed since the last read, as (type, data)."""
    events = []
    for ln in capsys.readouterr().out.splitlines():
        if ln.startswith(daemon.CHAT_IPC_PREFIX):
            payload = json.loads(ln[len(daemon.CHAT_IPC_PREFIX):])
            events.append((payload.get("type"), payload.get("data")))
    return events


@pytest.mark.unit
class TestChatRewindControl:
    """Rewind on the daemon side: the memory anchors on the message text,
    a held question is closed with the turn that asked it, and in
    subprocess mode the verdict travels back on the chat bus."""

    def setup_method(self, _method):
        _reset_daemon_globals()

    def teardown_method(self, _method):
        _reset_daemon_globals()
        daemon.set_confirmation_callbacks()

    def _seeded_memory(self):
        dm = _install_dialogue_memory(cfg=object(), db=object())
        dm.add_message("user", "remind me to buy oat milk")
        dm.add_message("assistant", "Noted.")
        dm.add_message("user", "what about eggs?")
        dm.add_message("assistant", "Eggs too.")
        return dm

    def _held_question(self, dm, origin="chat"):
        from jarvis.tools.confirmation import CHANNEL_GESTE, PendingAction

        dm.begin_turn()
        action = PendingAction.create(
            tool="localFiles", args={"operation": "delete", "path": "/a"},
            risk="destructif", channel=CHANNEL_GESTE, origin=origin,
            query_redacted="supprime", raised_at_turn=dm.current_turn(),
            ttl_sec=180.0,
        )
        dm.raise_pending(action)
        return action

    # --- bundled-mode function ---

    def test_rewind_truncates_shared_memory_from_the_named_turn(self):
        dm = self._seeded_memory()
        assert daemon.rewind_chat_to_user(2, "what about eggs?") is True
        assert dm.get_recent_messages() == [
            {"role": "user", "content": "remind me to buy oat milk"},
            {"role": "assistant", "content": "Noted."},
        ]

    def test_rewind_is_anchored_on_the_text_not_the_ordinal(self):
        """A voice exchange the window never saw shifts every ordinal the
        window counts; the text still names the right turn."""
        dm = self._seeded_memory()
        # The window opened after the first exchange happened by voice, so
        # it counts "what about eggs?" as its first message.
        assert daemon.rewind_chat_to_user(1, "what about eggs?") is True
        assert [m["content"] for m in dm.get_recent_messages()] == [
            "remind me to buy oat milk", "Noted.",
        ]

    def test_rewind_refuses_a_turn_the_memory_no_longer_holds(self):
        """A turn pruned by a diary pass, or one of a memory that was
        rebuilt, is refused rather than approximated by position."""
        dm = self._seeded_memory()
        assert daemon.rewind_chat_to_user(1, "a message of a pruned memory") is False
        assert len(dm.get_recent_messages()) == 4

    def test_rewind_matches_the_redacted_form_of_what_was_typed(self):
        from jarvis.utils.redact import redact

        dm = _install_dialogue_memory(cfg=object(), db=object())
        typed = "write to someone@example.com about it"
        dm.add_message("user", redact(typed))
        dm.add_message("assistant", "Done.")

        assert daemon.rewind_chat_to_user(1, typed) is True
        assert dm.get_recent_messages() == []

    def test_rewind_closes_a_waiting_question_as_expired(self, capsys):
        from unittest.mock import MagicMock

        dm = self._seeded_memory()
        daemon._global_db = MagicMock()
        action = self._held_question(dm)
        settled = []
        daemon.set_confirmation_callbacks(
            on_confirm_settled=lambda rid, outcome: settled.append((rid, outcome))
        )
        capsys.readouterr()

        assert daemon.rewind_chat_to_user(2, "what about eggs?") is True

        assert dm.peek_pending() is None
        assert daemon._global_db.record_action.call_args.kwargs["outcome"] == "expiré"
        assert daemon._global_db.record_action.call_args.kwargs["request_id"] == action.request_id
        assert settled == [(action.request_id, "expiré")]
        assert ("confirm_settled", {"request_id": action.request_id, "outcome": "expiré"}) in _chat_events(capsys)

    def test_a_refused_rewind_leaves_a_waiting_question_open(self):
        from unittest.mock import MagicMock

        dm = self._seeded_memory()
        daemon._global_db = MagicMock()
        action = self._held_question(dm)

        assert daemon.rewind_chat_to_user(1, "never said") is False

        assert dm.peek_pending() is action
        daemon._global_db.record_action.assert_not_called()

    def test_rewind_noops_when_daemon_not_booted(self):
        assert daemon.rewind_chat_to_user(1, "anything") is False

    # --- subprocess stdin handlers ---

    def test_cancel_stdin_line(self, monkeypatch):
        cancelled = []
        monkeypatch.setattr(
            daemon, "cancel_active_chat_query", lambda: cancelled.append(True)
        )
        assert daemon.handle_chat_cancel_stdin_line(
            daemon.CHAT_CANCEL_IPC_PREFIX
        ) is True
        assert cancelled == [True]
        assert daemon.handle_chat_cancel_stdin_line("SHUTDOWN") is False
        assert daemon.handle_chat_cancel_stdin_line("") is False

    def test_rewind_stdin_line_answers_rewound(self, capsys):
        dm = self._seeded_memory()
        capsys.readouterr()
        line = f'{daemon.CHAT_REWIND_IPC_PREFIX}{{"user_index": 2, "content": "what about eggs?"}}'

        assert daemon.handle_chat_rewind_stdin_line(line) is True

        assert len(dm.get_recent_messages()) == 2
        assert _chat_events(capsys) == [("rewound", {"user_index": 2})]

    def test_rewind_stdin_line_answers_nack_when_refused(self, capsys):
        dm = self._seeded_memory()
        capsys.readouterr()
        line = f'{daemon.CHAT_REWIND_IPC_PREFIX}{{"user_index": 2, "content": "never said"}}'

        assert daemon.handle_chat_rewind_stdin_line(line) is True

        assert len(dm.get_recent_messages()) == 4
        assert _chat_events(capsys) == [("rewind_nack", {"user_index": 2})]

    def test_rewind_stdin_line_without_a_text_is_refused_not_guessed(self, capsys):
        dm = self._seeded_memory()
        capsys.readouterr()

        assert daemon.handle_chat_rewind_stdin_line(
            f'{daemon.CHAT_REWIND_IPC_PREFIX}{{"user_index": 2}}'
        ) is True
        assert _chat_events(capsys) == [("rewind_nack", {"user_index": 2})]

        assert daemon.handle_chat_rewind_stdin_line(
            f'{daemon.CHAT_REWIND_IPC_PREFIX}{{"user_index": 0, "content": "what about eggs?"}}'
        ) is True
        assert _chat_events(capsys) == [("rewind_nack", {"user_index": 0})]
        assert len(dm.get_recent_messages()) == 4

    def test_malformed_rewind_stdin_line_is_swallowed(self, capsys):
        dm = self._seeded_memory()
        capsys.readouterr()
        assert daemon.handle_chat_rewind_stdin_line(
            f"{daemon.CHAT_REWIND_IPC_PREFIX}not json"
        ) is True
        assert _chat_events(capsys) == []
        assert len(dm.get_recent_messages()) == 4
        assert daemon.handle_chat_rewind_stdin_line("SHUTDOWN") is False


@pytest.mark.unit
class TestChatRewindLockGuard:
    """A rewind must not run while a query is in flight: the engine appends
    its turns after the truncation, which would resurrect the conversation
    the rewind just dropped."""

    def setup_method(self, _method):
        _reset_daemon_globals()

    def teardown_method(self, _method):
        _reset_daemon_globals()

    def test_rewind_is_rejected_while_the_query_lock_is_held(self):
        dm = _install_dialogue_memory(cfg=object(), db=object())
        dm.add_message("user", "q1")
        dm.add_message("assistant", "a1")
        dm.add_message("user", "q2")
        dm.add_message("assistant", "a2")

        # Simulate an in-flight query (voice or text) holding the lock.
        assert daemon._chat_query_lock.acquire(blocking=False)
        try:
            assert daemon.rewind_chat_to_user(2, "q2") is False
            assert len(dm.get_recent_messages()) == 4, "memory must be untouched"
        finally:
            daemon._chat_query_lock.release()

        # Once the lock is free the same call applies.
        assert daemon.rewind_chat_to_user(2, "q2") is True
        assert len(dm.get_recent_messages()) == 2


@pytest.mark.unit
class TestWaitForChatWorker:
    """Shutdown waits, bounded, for a chat worker that still holds the
    database the diary pass is about to use."""

    def setup_method(self, _method):
        _reset_daemon_globals()

    def teardown_method(self, _method):
        _reset_daemon_globals()

    def test_returns_at_once_when_no_worker_runs(self):
        started = time.monotonic()
        assert daemon.wait_for_chat_worker(timeout_sec=2.0) is True
        assert time.monotonic() - started < 1.0
        assert not daemon._chat_query_lock.locked(), "the wait must not keep the lock"

    def test_gives_up_after_the_timeout_while_a_worker_holds_the_lock(self):
        assert daemon._chat_query_lock.acquire(blocking=False)
        try:
            started = time.monotonic()
            assert daemon.wait_for_chat_worker(timeout_sec=0.05) is False
            elapsed = time.monotonic() - started
            assert 0.04 <= elapsed < 1.0
        finally:
            daemon._chat_query_lock.release()

    def test_a_worker_finishing_in_time_releases_the_wait(self):
        assert daemon._chat_query_lock.acquire(blocking=False)
        threading.Timer(0.05, daemon._chat_query_lock.release).start()

        assert daemon.wait_for_chat_worker(timeout_sec=2.0) is True
        assert not daemon._chat_query_lock.locked()

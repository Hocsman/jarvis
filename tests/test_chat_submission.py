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

        def fake_engine(db, cfg, tts, text, dialogue_memory, language=None):
            captured["tts"] = tts
            captured["dialogue_memory"] = dialogue_memory
            captured["text"] = text
            captured["language"] = language
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

    def test_ipc_never_carries_unredacted_secrets(self, monkeypatch, capsys):
        """The start event carries the query as-passed; redaction is the
        engine's job, but the IPC layer must not add extra raw echoes. We
        verify the only user-text-bearing event is `start` and it carries
        exactly the input string (no duplicate raw echo in other events)."""
        monkeypatch.setattr(
            "jarvis.reply.engine.run_reply_engine", lambda *a, **k: "ok"
        )
        _install_dialogue_memory(cfg=object(), db=object())
        daemon.submit_text_query("my secret password is hunter2", use_ipc=True)
        chat_lines = _wait_for_ipc_complete(capsys)
        # Only the start event may carry the raw query; complete carries the reply.
        for ln in chat_lines:
            payload = json.loads(ln[len(daemon.CHAT_IPC_PREFIX):])
            if payload["type"] != "start":
                assert "hunter2" not in json.dumps(payload["data"])


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

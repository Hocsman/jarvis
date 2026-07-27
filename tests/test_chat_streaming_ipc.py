"""Behaviour tests for streamed chat tokens over the daemon's IPC.

``chat_window.spec.md`` already reserves a ``token`` event on the
``__CHAT__:`` stream; these tests pin its activation. The daemon now hands
``run_reply_engine`` an ``on_token`` callback and forwards each delta as a
``token`` event, so the desktop can render the reply while it generates.

The ordering guarantee matters: ``start`` → ``token``* → ``complete``, with
``complete`` still carrying the whole reply. Streaming is progressive
display only; the completed string stays the source of truth.
"""

from __future__ import annotations

import json
import time

import pytest

from src.jarvis import daemon


def _install_memory():
    from src.jarvis.memory.conversation import DialogueMemory
    daemon._global_dialogue_memory = DialogueMemory(inactivity_timeout=300, max_interactions=20)
    daemon._global_cfg = object()
    daemon._global_db = object()
    daemon._global_stop_requested = False


def _reset():
    daemon._global_dialogue_memory = None
    daemon._global_cfg = None
    daemon._global_db = None
    daemon._global_stop_requested = False


def _chat_events(capsys, timeout=5.0):
    """Collect __CHAT__ events until ``complete`` arrives."""
    deadline = time.time() + timeout
    collected = []
    while time.time() < deadline:
        out = capsys.readouterr().out
        for ln in out.splitlines():
            if ln.startswith(daemon.CHAT_IPC_PREFIX):
                try:
                    collected.append(json.loads(ln[len(daemon.CHAT_IPC_PREFIX):]))
                except json.JSONDecodeError:
                    pass
        if any(e.get("type") == "complete" for e in collected):
            return collected
        time.sleep(0.02)
    return collected


class TestStreamedChatIPC:
    def setup_method(self):
        _reset()

    def teardown_method(self):
        _reset()

    def test_tokens_are_emitted_as_events_before_complete(self, monkeypatch, capsys):
        _install_memory()

        def fake_engine(*a, on_token=None, **k):
            for piece in ["Bon", "jour", " !"]:
                if on_token:
                    on_token(piece)
            return "Bonjour !"

        monkeypatch.setattr("src.jarvis.reply.engine.run_reply_engine", fake_engine)
        daemon.submit_text_query("salut", use_ipc=True)
        events = _chat_events(capsys)

        kinds = [e["type"] for e in events]
        assert "token" in kinds, f"no token event in {kinds}"
        # start precedes tokens, which precede complete
        assert kinds.index("start") < kinds.index("token")
        assert kinds.index("token") < kinds.index("complete")

        tokens = [e["data"] for e in events if e["type"] == "token"]
        assert "".join(tokens) == "Bonjour !"

    def test_complete_still_carries_the_full_reply(self, monkeypatch, capsys):
        # Streaming is display-only: the completed reply remains authoritative.
        _install_memory()

        def fake_engine(*a, on_token=None, **k):
            if on_token:
                on_token("partiel")
            return "réponse finale"

        monkeypatch.setattr("src.jarvis.reply.engine.run_reply_engine", fake_engine)
        daemon.submit_text_query("salut", use_ipc=True)
        events = _chat_events(capsys)
        complete = [e for e in events if e["type"] == "complete"]
        assert complete and complete[-1]["data"] == "réponse finale"

    def test_engine_without_on_token_support_still_replies(self, monkeypatch, capsys):
        # An engine that doesn't accept on_token must not break the chat path.
        _install_memory()
        monkeypatch.setattr(
            "src.jarvis.reply.engine.run_reply_engine",
            lambda db, cfg, tts, text, dialogue_memory, language=None, origin=None: "ok",
        )
        daemon.submit_text_query("salut", use_ipc=True)
        events = _chat_events(capsys)
        complete = [e for e in events if e["type"] == "complete"]
        assert complete and complete[-1]["data"] == "ok"

    def test_callbacks_receive_tokens_too(self, monkeypatch):
        # Non-IPC (bundled) callers get the same stream via on_token.
        _install_memory()

        def fake_engine(*a, on_token=None, **k):
            if on_token:
                on_token("a")
                on_token("b")
            return "ab"

        monkeypatch.setattr("src.jarvis.reply.engine.run_reply_engine", fake_engine)
        seen, done = [], []
        daemon.submit_text_query(
            "salut", use_ipc=False,
            on_token=seen.append, on_complete=done.append,
        )
        deadline = time.time() + 5.0
        while time.time() < deadline and not done:
            time.sleep(0.02)
        assert seen == ["a", "b"]
        assert done == ["ab"]

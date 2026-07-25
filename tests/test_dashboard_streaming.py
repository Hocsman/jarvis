"""Tests for progressive (streamed) reply display on the dashboard.

The daemon emits ``token`` events while the model generates; the bridge
re-emits them so the page can fill a reply bubble as it arrives. The
contract that matters: tokens are forwarded in order, the orb stays in its
"thinking" state while they stream (the reply isn't finished yet), and the
final ``complete`` reply still arrives — it is authoritative and settles
the bubble, so a partial stream can never be mistaken for the answer.
"""

from __future__ import annotations

import os

import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def _qapp():
    from PyQt6.QtWidgets import QApplication
    import sys

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


def _make_bridge(submit_fn=None):
    from desktop_app.dashboard.bridge import DashboardBridge
    return DashboardBridge(submit_fn=submit_fn)


class TestBridgeTokens:
    def test_deliver_token_emits_in_order(self, _qapp):
        bridge = _make_bridge()
        seen = []
        bridge.tokenReceived.connect(seen.append)
        for chunk in ["Bon", "jour", " !"]:
            bridge.deliver_token(chunk)
        assert seen == ["Bon", "jour", " !"]

    def test_empty_token_is_ignored(self, _qapp):
        bridge = _make_bridge()
        seen = []
        bridge.tokenReceived.connect(seen.append)
        bridge.deliver_token("")
        bridge.deliver_token(None)
        assert seen == []

    def test_streaming_does_not_end_the_thinking_state(self, _qapp):
        # The orb must keep pulsing while tokens stream; only the completed
        # reply settles it.
        bridge = _make_bridge(submit_fn=lambda t: None)
        bridge.submitQuery("salut")
        assert bridge._chat_pending is True
        bridge.deliver_token("Bon")
        assert bridge._chat_pending is True, "tokens must not clear the pending state"
        bridge.deliver_reply("Bonjour !")
        assert bridge._chat_pending is False

    def test_complete_reply_still_delivered_after_tokens(self, _qapp):
        bridge = _make_bridge()
        replies = []
        bridge.replyReceived.connect(replies.append)
        bridge.deliver_token("part")
        bridge.deliver_reply("réponse finale")
        assert replies == ["réponse finale"]


class _Recorder:
    """Stand-in bridge that records what it was asked to do."""

    def __init__(self):
        self.calls = []

    def deliver_token(self, chunk): self.calls.append(("token", chunk))
    def deliver_reply(self, text): self.calls.append(("reply", text))
    def deliver_busy(self): self.calls.append(("busy", None))


def _line(**payload):
    import json as _json
    from jarvis.daemon import CHAT_IPC_PREFIX
    return CHAT_IPC_PREFIX + _json.dumps(payload)


class TestIpcRouting:
    """The daemon's wire events must reach the right bridge calls. Tested
    against the Qt-free dispatcher (importing the window pulls in
    QtWebEngine, which cannot be constructed headlessly)."""

    def test_token_then_complete_are_routed_in_order(self):
        from desktop_app.dashboard.bridge import dispatch_chat_ipc_line

        rec = _Recorder()
        assert dispatch_chat_ipc_line(_line(type="token", data="Bon"), rec) is True
        dispatch_chat_ipc_line(_line(type="token", data="jour"), rec)
        dispatch_chat_ipc_line(_line(type="complete", data="Bonjour !"), rec)
        assert rec.calls == [
            ("token", "Bon"), ("token", "jour"), ("reply", "Bonjour !"),
        ]

    def test_busy_is_routed(self):
        from desktop_app.dashboard.bridge import dispatch_chat_ipc_line

        rec = _Recorder()
        dispatch_chat_ipc_line(_line(type="busy", data=None), rec)
        assert rec.calls == [("busy", None)]

    def test_start_event_triggers_nothing(self):
        # submitQuery already set the thinking state locally.
        from desktop_app.dashboard.bridge import dispatch_chat_ipc_line

        rec = _Recorder()
        assert dispatch_chat_ipc_line(_line(type="start", data="q"), rec) is True
        assert rec.calls == []

    def test_non_chat_line_is_not_consumed(self):
        from desktop_app.dashboard.bridge import dispatch_chat_ipc_line

        rec = _Recorder()
        assert dispatch_chat_ipc_line("some unrelated log line", rec) is False
        assert rec.calls == []

    def test_malformed_chat_payload_is_consumed_but_harmless(self):
        from desktop_app.dashboard.bridge import dispatch_chat_ipc_line
        from jarvis.daemon import CHAT_IPC_PREFIX

        rec = _Recorder()
        # Consumed (True) so the caller doesn't re-handle it as a log line.
        assert dispatch_chat_ipc_line(CHAT_IPC_PREFIX + "not json", rec) is True
        assert rec.calls == []

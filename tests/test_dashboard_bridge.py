"""Tests for the dashboard Python<->JS bridge.

The bridge exposes chat (submitQuery + reply/echo/busy signals), live
system stats, and a JarvisState->orb mapping to the HUD page. These
tests exercise the Python side directly (no web view needed).
"""

from __future__ import annotations

import json
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


class TestChatRoundTrip:

    @pytest.mark.unit
    def test_submit_forwards_and_echoes(self, _qapp) -> None:
        sent = {}
        b = _make_bridge(submit_fn=lambda t: sent.setdefault("q", t))
        echoes = []
        b.userEcho.connect(echoes.append)

        b.submitQuery("quelle météo à Paris")
        assert sent["q"] == "quelle météo à Paris"
        assert echoes == ["quelle météo à Paris"]
        assert b._chat_pending is True  # orb held at THINKING

    @pytest.mark.unit
    def test_reply_clears_pending(self, _qapp) -> None:
        b = _make_bridge(submit_fn=lambda t: None)
        replies = []
        b.replyReceived.connect(replies.append)

        b.submitQuery("x")
        assert b._chat_pending is True
        b.deliver_reply("réponse")
        assert replies == ["réponse"]
        assert b._chat_pending is False

    @pytest.mark.unit
    def test_busy_clears_pending(self, _qapp) -> None:
        b = _make_bridge(submit_fn=lambda t: None)
        busies = []
        b.busy.connect(lambda: busies.append(True))
        b.submitQuery("x")
        b.deliver_busy()
        assert busies == [True]
        assert b._chat_pending is False

    @pytest.mark.unit
    def test_empty_query_ignored(self, _qapp) -> None:
        sent = []
        b = _make_bridge(submit_fn=lambda t: sent.append(t))
        b.submitQuery("   ")
        assert sent == []

    @pytest.mark.unit
    def test_no_submit_fn_preview(self, _qapp) -> None:
        """Standalone (no daemon): submit echoes + returns a preview
        line instead of forwarding."""
        b = _make_bridge(submit_fn=None)
        replies = []
        b.replyReceived.connect(replies.append)
        b.submitQuery("hello")
        assert replies and "aperçu" in replies[0]
        assert b._chat_pending is False  # nothing pending without a daemon


class TestStateMapping:

    @pytest.mark.unit
    def test_pending_forces_thinking(self, _qapp) -> None:
        b = _make_bridge(submit_fn=lambda t: None)
        states = []
        b.stateChanged.connect(states.append)
        b.submitQuery("x")
        # last emitted state during pending must be THINKING (amber)
        last = json.loads(states[-1])
        assert last["status"] == "Réflexion…"

    @pytest.mark.unit
    def test_voice_state_maps_to_accent(self, _qapp) -> None:
        from desktop_app.face_widget import get_jarvis_state, JarvisState
        b = _make_bridge()
        states = []
        b.stateChanged.connect(states.append)

        get_jarvis_state().set_state(JarvisState.LISTENING)
        b._emit_state(force=True)
        listening = json.loads(states[-1])
        assert listening["status"] == "Je vous écoute…"
        assert isinstance(listening["accent"], list) and len(listening["accent"]) == 3


class TestStats:

    @pytest.mark.unit
    def test_stats_payload_shape(self, _qapp) -> None:
        b = _make_bridge()
        stats = []
        b.statsUpdated.connect(stats.append)
        b._emit_stats()
        assert stats, "stats should emit"
        o = json.loads(stats[-1])
        for k in ("cpu", "ram", "disk_used", "disk_total", "disk_pct"):
            assert k in o
            assert isinstance(o[k], (int, float))


# Note: DashboardWindow.process_ipc_line is a thin parser that delegates
# to bridge.deliver_reply / deliver_busy (both covered above). It isn't
# unit-tested here because instantiating QWebEngineView under headless
# pytest is unstable; the parse + delegate path is verified manually via
# the standalone round-trip and exercised live in the app.

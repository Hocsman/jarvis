"""Tests for the orb embedded in the chat window.

The orb is the chat window's visual heartbeat: THINKING while a reply
is in flight, IDLE otherwise. It must also degrade gracefully (the
chat still works if the orb stack can't load) and pause its 60 fps
render loop while the window is hidden (battery/heat).
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


def _orb_state(window):
    """Current OrbState of the embedded orb (tick with dt=0 to read
    the live target without advancing the animation)."""
    return window._orb.state_controller().tick(0.0).state


class TestOrbEmbeddedInChat:

    @pytest.mark.unit
    def test_orb_is_embedded(self, _qapp) -> None:
        from desktop_app.chat_window import ChatWindow

        w = ChatWindow(daemon_available=True)
        try:
            assert w._orb is not None, "orb should be embedded in the chat window"
            assert w._orb.height() == 150, "orb hero band should be the fixed hero height"
        finally:
            w.deleteLater()

    @pytest.mark.unit
    def test_thinking_drives_orb_state(self, _qapp) -> None:
        from desktop_app.chat_window import ChatWindow
        from desktop_app.orb.state_controller import OrbState

        w = ChatWindow(daemon_available=True)
        try:
            assert _orb_state(w) == OrbState.IDLE
            w._set_thinking(True)
            assert _orb_state(w) == OrbState.THINKING, (
                "a query in flight must pulse the orb THINKING"
            )
            w._set_thinking(False)
            assert _orb_state(w) == OrbState.IDLE, (
                "the orb must settle to IDLE once the reply lands"
            )
        finally:
            w.deleteLater()

    @pytest.mark.unit
    def test_complete_signal_settles_orb(self, _qapp) -> None:
        """The completed callback (reply landed) routes through
        _set_thinking(False) -> IDLE."""
        from desktop_app.chat_window import ChatWindow
        from desktop_app.orb.state_controller import OrbState

        w = ChatWindow(daemon_available=True)
        try:
            w._set_thinking(True)
            assert _orb_state(w) == OrbState.THINKING
            w._on_complete("réponse de DeepSeek")
            assert _orb_state(w) == OrbState.IDLE
        finally:
            w.deleteLater()

    @pytest.mark.unit
    def test_daemon_stopped_settles_orb_to_idle(self, _qapp) -> None:
        """A stopped daemon is an expected user action, not a fault —
        the orb goes IDLE, not ERROR."""
        from desktop_app.chat_window import ChatWindow
        from desktop_app.orb.state_controller import OrbState

        w = ChatWindow(daemon_available=True)
        try:
            w._set_thinking(True)
            assert _orb_state(w) == OrbState.THINKING
            w.set_daemon_status("stopped")
            assert _orb_state(w) == OrbState.IDLE
        finally:
            w.deleteLater()

    @pytest.mark.unit
    def test_thinking_ignored_when_daemon_down(self, _qapp) -> None:
        """No query can be in flight against a stopped daemon, so
        _set_thinking(True) must not light the orb THINKING."""
        from desktop_app.chat_window import ChatWindow
        from desktop_app.orb.state_controller import OrbState

        w = ChatWindow(daemon_available=False)
        try:
            w._set_thinking(True)
            assert _orb_state(w) == OrbState.IDLE
        finally:
            w.deleteLater()


class TestOrbRenderPauseOnHide:

    @pytest.mark.unit
    def test_pause_resume_methods(self, _qapp) -> None:
        from desktop_app.orb.orb_widget import OrbWidget

        orb = OrbWidget()
        try:
            assert orb._timer.isActive()
            orb.pause_rendering()
            assert not orb._timer.isActive()
            orb.resume_rendering()
            assert orb._timer.isActive()
            # Idempotent.
            orb.resume_rendering()
            assert orb._timer.isActive()
            orb.pause_rendering()
            orb.pause_rendering()
            assert not orb._timer.isActive()
        finally:
            orb.deleteLater()

    @pytest.mark.unit
    def test_hide_pauses_orb_show_resumes(self, _qapp) -> None:
        from desktop_app.chat_window import ChatWindow

        w = ChatWindow(daemon_available=True)
        try:
            w.show()
            assert w._orb._timer.isActive(), "orb should render while window is visible"
            w.hide()
            assert not w._orb._timer.isActive(), "orb should pause while window is hidden"
            w.show()
            assert w._orb._timer.isActive(), "orb should resume when window is shown again"
        finally:
            w.deleteLater()


class TestGracefulDegradation:

    @pytest.mark.unit
    def test_chat_works_without_orb(self, _qapp, monkeypatch) -> None:
        """If the orb stack fails to build, the chat window must still
        construct and the state hooks must no-op safely."""
        from desktop_app import chat_window as cw

        # Force _build_orb to fail by making the import raise.
        orig = cw.ChatWindow._build_orb
        monkeypatch.setattr(cw.ChatWindow, "_build_orb", lambda self: None)

        w = cw.ChatWindow(daemon_available=True)
        try:
            assert w._orb is None
            # These must not raise even with no orb.
            w._set_thinking(True)
            w._set_thinking(False)
            w.set_daemon_status("stopped")
            w.show()
            w.hide()
        finally:
            w.deleteLater()
            cw.ChatWindow._build_orb = orig

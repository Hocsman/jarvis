"""Tests for the OrbWindow surface.

We assert observable behaviour: the public toggle method flips
visibility. The pynput global hotkey is *not* exercised here because
it would require a real keyboard event injected at the OS level
(which CI cannot provide). Toggle is wired through a Qt signal that
the hotkey calls into, so testing the signal-connected slot validates
the same code path the hotkey traverses.
"""

from __future__ import annotations

import os

import pytest


# QOpenGLWidget requires an OpenGL context; the offscreen Qt platform
# plugin works for instantiation + show/hide on macOS as long as
# QT_QPA_PLATFORM is set before QApplication.__init__.
@pytest.fixture(scope="module")
def qt_app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app
    # Don't quit; another test module may need the same app.


class TestToggleVisibility:
    """The hotkey-targeted public method must flip the window's
    visible state and be idempotent."""

    @pytest.mark.unit
    def test_hotkey_toggles_window_show_hide(self, qt_app):
        from desktop_app.orb.orb_window import OrbWindow

        win = OrbWindow()
        try:
            # Newly constructed window is hidden.
            assert win.isVisible() is False

            win.toggle_visibility()
            qt_app.processEvents()
            assert win.isVisible() is True

            win.toggle_visibility()
            qt_app.processEvents()
            assert win.isVisible() is False
        finally:
            win.close()
            qt_app.processEvents()

    @pytest.mark.unit
    def test_show_orb_then_hide_orb(self, qt_app):
        """The explicit API (show_orb/hide_orb) must agree with
        toggle_visibility on observable state."""
        from desktop_app.orb.orb_window import OrbWindow

        win = OrbWindow()
        try:
            win.show_orb()
            qt_app.processEvents()
            assert win.isVisible() is True

            win.hide_orb()
            qt_app.processEvents()
            assert win.isVisible() is False
        finally:
            win.close()
            qt_app.processEvents()


class TestDevBadge:
    """The DEV badge surfaces when the dev-mode env var is set."""

    @pytest.mark.unit
    def test_dev_badge_present_when_forced(self, qt_app, monkeypatch):
        monkeypatch.setenv("JARVIS_ORB_FORCE_DEV", "1")
        from desktop_app.orb.orb_window import OrbWindow

        win = OrbWindow()
        try:
            assert win._dev_badge is not None
            assert win._dev_badge.text() == "DEV"
        finally:
            win.close()
            qt_app.processEvents()

    @pytest.mark.unit
    def test_dev_badge_absent_in_prod(self, qt_app, monkeypatch):
        monkeypatch.setenv("JARVIS_ORB_FORCE_PROD", "1")
        monkeypatch.delenv("JARVIS_ORB_FORCE_DEV", raising=False)
        from desktop_app.orb.orb_window import OrbWindow

        win = OrbWindow()
        try:
            assert win._dev_badge is None
        finally:
            win.close()
            qt_app.processEvents()

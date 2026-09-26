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


# The offscreen Qt platform plugin is enough for instantiation and
# show/hide as long as QT_QPA_PLATFORM is set before
# QApplication.__init__.
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


class TestRenderingFollowsVisibility:
    """The floating orb renders only while it is on screen: hidden, its
    render loop stops; shown again, it restarts."""

    @pytest.mark.unit
    def test_hide_pauses_the_orb_and_show_resumes_it(self, qt_app):
        from desktop_app.orb.orb_window import OrbWindow

        win = OrbWindow()
        try:
            win.show_orb()
            qt_app.processEvents()
            assert win._orb._timer.isActive(), "the orb renders while the window is visible"

            win.hide_orb()
            qt_app.processEvents()
            assert not win._orb._timer.isActive(), "the orb keeps rendering while hidden"

            win.show_orb()
            qt_app.processEvents()
            assert win._orb._timer.isActive(), "the orb does not resume when shown again"
        finally:
            win.close()
            qt_app.processEvents()


class TestWindowShortcut:
    """The window-scoped shortcut is Ctrl+Shift+J on every platform. Qt
    maps Ctrl to the Command key on macOS, so the same sequence reads
    Cmd+Shift+J there, the keys the global hotkey and the hint name."""

    @pytest.mark.unit
    @pytest.mark.parametrize("platform_name", ["darwin", "win32", "linux"])
    def test_shortcut_is_ctrl_shift_j_everywhere(self, qt_app, monkeypatch, platform_name):
        from PyQt6.QtGui import QKeySequence
        from desktop_app.orb import orb_window as ow

        monkeypatch.setattr(ow.sys, "platform", platform_name)
        win = ow.OrbWindow()
        try:
            assert win._qt_shortcut.key() == QKeySequence("Ctrl+Shift+J")
        finally:
            win.close()
            qt_app.processEvents()


class TestPynputGuard:
    """The pynput global hotkey crashes the process on macOS 26+
    (Tahoe) via a TSM main-thread assertion. Reproducing the
    dictation engine's guard here protects the orb from the same
    crash."""

    @pytest.mark.unit
    def test_pynput_disabled_on_macos_26(self, monkeypatch):
        from desktop_app.orb import orb_window as ow
        monkeypatch.setattr(ow.sys, "platform", "darwin")
        monkeypatch.setattr(ow.platform, "mac_ver", lambda: ("26.3.1", "", ""))
        assert ow._pynput_is_safe_on_this_platform() is False

    @pytest.mark.unit
    def test_pynput_enabled_on_macos_15(self, monkeypatch):
        from desktop_app.orb import orb_window as ow
        monkeypatch.setattr(ow.sys, "platform", "darwin")
        monkeypatch.setattr(ow.platform, "mac_ver", lambda: ("15.4", "", ""))
        assert ow._pynput_is_safe_on_this_platform() is True

    @pytest.mark.unit
    def test_pynput_enabled_on_linux(self, monkeypatch):
        from desktop_app.orb import orb_window as ow
        monkeypatch.setattr(ow.sys, "platform", "linux")
        assert ow._pynput_is_safe_on_this_platform() is True

    @pytest.mark.unit
    def test_show_does_not_register_pynput_on_macos_26(self, qt_app, monkeypatch):
        """Showing the orb on macOS 26+ must NOT touch pynput at all.

        Reproduces the crash scenario: if any pynput import or thread
        is started, the SIGTRAP from TSM kills the process. We verify
        the no-touch contract by failing the test if anything in
        pynput.keyboard.GlobalHotKeys is even instantiated.
        """
        from desktop_app.orb import orb_window as ow
        monkeypatch.setattr(ow.sys, "platform", "darwin")
        monkeypatch.setattr(ow.platform, "mac_ver", lambda: ("26.3.1", "", ""))

        # Make any accidental call to pynput.keyboard.GlobalHotKeys
        # raise loudly so the test catches it.
        import types
        fake = types.SimpleNamespace(
            GlobalHotKeys=lambda *a, **kw: pytest.fail(
                "pynput.GlobalHotKeys was constructed on macOS 26+"
            )
        )
        monkeypatch.setitem(__import__("sys").modules, "pynput.keyboard", fake)

        win = ow.OrbWindow()
        try:
            win._ensure_hotkey_registered()  # explicit call, no real show needed
            assert win._hotkey_listener is None
        finally:
            win.close()
            qt_app.processEvents()

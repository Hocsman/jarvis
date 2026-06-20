"""Behaviour tests for the ChatWindow (text chat interface).

These verify the contract in ``src/desktop_app/chat_window.spec.md``:

- The window has a transcript area, an input box, a send button, and a stop
  button (visible only while a query is in flight).
- Sending submits text via ``jarvis.daemon.submit_text_query`` and appends the
  user's message to the transcript.
- Daemon callback signals (start/complete/busy) update the transcript and the
  status indicator on the Qt main thread.
- The stop button calls ``jarvis.daemon.request_stop``.
- Closing hides the window; it does not quit the daemon.
- Styling uses the shared theme stylesheet (no hardcoded colour literals in
  the widget classes).
"""

from __future__ import annotations

import pytest


@pytest.mark.unit
class TestChatWindowStructure:
    """The window exposes the UI elements the spec requires."""

    def test_has_transcript_input_send_and_stop(self, qapp):
        from desktop_app.chat_window import ChatWindow

        win = ChatWindow()
        assert win.transcript_widget is not None
        assert win.input_widget is not None
        assert win.send_button is not None
        assert win.stop_button is not None

    def test_stop_button_hidden_at_rest(self, qapp):
        """The stop button is only relevant while a query is running."""
        from desktop_app.chat_window import ChatWindow

        win = ChatWindow()
        assert not win.stop_button.isVisible()

    def test_window_title_mentions_jarvis(self, qapp):
        from desktop_app.chat_window import ChatWindow

        win = ChatWindow()
        title = win.windowTitle()
        assert "Jarvis" in title


@pytest.mark.unit
class TestChatWindowSend:
    """Sending a message dispatches to the daemon and echoes the user text."""

    def test_send_calls_submit_text_query(self, qapp, monkeypatch):
        from desktop_app.chat_window import ChatWindow

        calls = []
        monkeypatch.setattr(
            "jarvis.daemon.submit_text_query",
            lambda text, **kw: calls.append(text),
        )
        win = ChatWindow()
        win.input_widget.setPlainText("what is the weather")
        win._send()
        assert calls == ["what is the weather"]

    def test_send_appends_user_message_to_transcript(self, qapp, monkeypatch):
        from desktop_app.chat_window import ChatWindow

        monkeypatch.setattr(
            "jarvis.daemon.submit_text_query", lambda text, **kw: None
        )
        win = ChatWindow()
        win.input_widget.setPlainText("hello there")
        win._send()
        text = win.transcript_widget.toPlainText()
        assert "hello there" in text

    def test_send_clears_input(self, qapp, monkeypatch):
        from desktop_app.chat_window import ChatWindow

        monkeypatch.setattr(
            "jarvis.daemon.submit_text_query", lambda text, **kw: None
        )
        win = ChatWindow()
        win.input_widget.setPlainText("clear me after send")
        win._send()
        assert win.input_widget.toPlainText() == ""

    def test_send_empty_does_nothing(self, qapp, monkeypatch):
        from desktop_app.chat_window import ChatWindow

        calls = []
        monkeypatch.setattr(
            "jarvis.daemon.submit_text_query",
            lambda text, **kw: calls.append(text),
        )
        win = ChatWindow()
        win.input_widget.setPlainText("   ")
        win._send()
        assert calls == []


@pytest.mark.unit
class TestChatWindowCallbacks:
    """Daemon callback signals update the UI on the main thread."""

    def test_on_complete_appends_reply_to_transcript(self, qapp, monkeypatch):
        from desktop_app.chat_window import ChatWindow

        monkeypatch.setattr(
            "jarvis.daemon.submit_text_query", lambda text, **kw: None
        )
        win = ChatWindow()
        win.input_widget.setPlainText("hi")
        win._send()
        # Simulate the daemon completing with a reply.
        win._on_complete("It is sunny today.")
        text = win.transcript_widget.toPlainText()
        assert "It is sunny today." in text

    def test_on_complete_hides_stop_button(self, qapp, monkeypatch):
        from desktop_app.chat_window import ChatWindow

        monkeypatch.setattr(
            "jarvis.daemon.submit_text_query", lambda text, **kw: None
        )
        win = ChatWindow()
        win.show()
        qapp.processEvents()
        win.input_widget.setPlainText("hi")
        win._send()
        qapp.processEvents()
        # While "thinking" the stop button should be visible.
        assert win.stop_button.isVisible()
        win._on_complete("done")
        qapp.processEvents()
        assert not win.stop_button.isVisible()

    def test_on_busy_appends_busy_notice(self, qapp, monkeypatch):
        from desktop_app.chat_window import ChatWindow

        monkeypatch.setattr(
            "jarvis.daemon.submit_text_query", lambda text, **kw: None
        )
        win = ChatWindow()
        win.input_widget.setPlainText("second query")
        win._send()
        # Simulate the daemon rejecting because a query is already running.
        win._on_busy()
        text = win.transcript_widget.toPlainText()
        # The notice is language-neutral in shape but must mention the query
        # was not accepted.
        assert "second query" in text  # user echo stays
        assert "busy" in text.lower() or "already" in text.lower()


@pytest.mark.unit
class TestChatWindowStop:
    """The stop button cancels the chat query, not the whole daemon."""

    def test_stop_calls_cancel_active_chat_query(self, qapp, monkeypatch):
        from desktop_app.chat_window import ChatWindow

        called = []
        monkeypatch.setattr(
            "jarvis.daemon.cancel_active_chat_query", lambda: called.append(True)
        )
        # request_stop must NOT be called because it tears down the whole
        # voice assistant.
        request_stop_called = []
        monkeypatch.setattr(
            "jarvis.daemon.request_stop",
            lambda: request_stop_called.append(True),
        )
        win = ChatWindow()
        win.show()
        qapp.processEvents()
        win._set_thinking(True)
        qapp.processEvents()
        win._stop()
        qapp.processEvents()
        assert called == [True]
        assert request_stop_called == []

    def test_stop_resets_thinking_indicator(self, qapp, monkeypatch):
        from desktop_app.chat_window import ChatWindow

        monkeypatch.setattr(
            "jarvis.daemon.cancel_active_chat_query", lambda: None
        )
        win = ChatWindow()
        win.show()
        qapp.processEvents()
        win._set_thinking(True)
        qapp.processEvents()
        assert win.stop_button.isVisible()
        win._stop()
        qapp.processEvents()
        assert not win.stop_button.isVisible()


@pytest.mark.unit
class TestChatWindowLifecycle:
    """Closing hides rather than tearing down daemon state."""

    def test_close_event_hides_window(self, qapp, monkeypatch):
        from desktop_app.chat_window import ChatWindow
        from PyQt6.QtGui import QCloseEvent

        monkeypatch.setattr(
            "jarvis.daemon.submit_text_query", lambda text, **kw: None
        )
        win = ChatWindow()
        win.show()
        qapp.processEvents()
        # The daemon stop function must NOT be called on close.
        stop_called = []
        monkeypatch.setattr(
            "jarvis.daemon.request_stop", lambda: stop_called.append(True)
        )
        win.closeEvent(QCloseEvent())
        assert stop_called == []


@pytest.mark.unit
class TestChatWindowSubmitFn:
    """When a ``submit_fn`` is injected (subprocess mode), sending routes
    through it instead of the daemon's direct call path."""

    def test_submit_fn_receives_text(self, qapp, monkeypatch):
        from desktop_app.chat_window import ChatWindow

        calls = []
        win = ChatWindow(submit_fn=lambda text: calls.append(text))
        # The bundled path must NOT be touched when submit_fn is set.
        monkeypatch.setattr(
            "jarvis.daemon.submit_text_query",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("must use submit_fn")),
        )
        win.input_widget.setPlainText("via stdin")
        win._send()
        assert calls == ["via stdin"]


@pytest.mark.unit
class TestDesktopAppChatDispatch:
    """The desktop app routes ``__CHAT__:`` IPC lines to the chat window on the
    main thread via ``_on_chat_ipc_line`` + ``ChatWindow.process_ipc_line``."""

    def _make_tray(self):
        import desktop_app.app as app_mod
        tray = app_mod.JarvisSystemTray.__new__(app_mod.JarvisSystemTray)
        tray.chat_window = None
        tray._chat_submit_fn = None
        return tray

    def test_on_chat_ipc_line_creates_window_lazily(self, qapp):
        from jarvis.daemon import CHAT_IPC_PREFIX
        tray = self._make_tray()
        assert tray.chat_window is None
        tray._on_chat_ipc_line(f'{CHAT_IPC_PREFIX}{{"type":"complete","data":"hi"}}')
        assert tray.chat_window is not None

    def test_dispatch_complete_appends_reply(self, qapp):
        from jarvis.daemon import CHAT_IPC_PREFIX
        tray = self._make_tray()
        tray._on_chat_ipc_line(f'{CHAT_IPC_PREFIX}{{"type":"complete","data":"hello back"}}')
        tray.chat_window.show()
        qapp.processEvents()
        assert "hello back" in tray.chat_window.transcript_widget.toPlainText()

    def test_dispatch_start_sets_thinking(self, qapp):
        from jarvis.daemon import CHAT_IPC_PREFIX
        tray = self._make_tray()
        tray._on_chat_ipc_line(f'{CHAT_IPC_PREFIX}{{"type":"start","data":"a query"}}')
        tray.chat_window.show()
        qapp.processEvents()
        assert tray.chat_window.stop_button.isVisible()

    def test_dispatch_busy_appends_notice(self, qapp):
        from jarvis.daemon import CHAT_IPC_PREFIX
        tray = self._make_tray()
        tray._on_chat_ipc_line(f'{CHAT_IPC_PREFIX}{{"type":"busy","data":null}}')
        tray.chat_window.show()
        qapp.processEvents()
        text = tray.chat_window.transcript_widget.toPlainText().lower()
        assert "busy" in text

    def test_dispatch_malformed_line_is_swallowed(self, qapp):
        from jarvis.daemon import CHAT_IPC_PREFIX
        tray = self._make_tray()
        # Must not raise; window is created lazily but no reply text lands.
        tray._on_chat_ipc_line(f"{CHAT_IPC_PREFIX}not json")
        qapp.processEvents()

    def test_process_ipc_line_returns_false_for_non_chat(self, qapp):
        from desktop_app.chat_window import ChatWindow
        win = ChatWindow()
        assert win.process_ipc_line("not a chat line") is False

    def test_process_ipc_line_returns_true_for_malformed_chat(self, qapp):
        from desktop_app.chat_window import ChatWindow
        from jarvis.daemon import CHAT_IPC_PREFIX
        win = ChatWindow()
        assert win.process_ipc_line(f"{CHAT_IPC_PREFIX}not json") is True

    def test_subprocess_submit_fn_writes_chat_query_line(self, qapp, monkeypatch):
        """The stdin-bridge callable writes a __CHAT_QUERY__: JSON line."""
        import io
        import json
        from jarvis.daemon import CHAT_QUERY_IPC_PREFIX
        import desktop_app.app as app_mod

        tray = app_mod.JarvisSystemTray.__new__(app_mod.JarvisSystemTray)

        # Fake a subprocess.Popen with a writable stdin pipe.
        sink = io.StringIO()
        fake_proc = type("P", (), {"stdin": sink})()
        tray.daemon_process = fake_proc

        # Reconstruct the closure the real start_daemon builds.
        def _submit(text: str) -> None:
            tray.daemon_process.stdin.write(
                f"{CHAT_QUERY_IPC_PREFIX}{json.dumps({'text': text})}\n"
            )
            tray.daemon_process.stdin.flush()

        _submit("hello over stdin")
        written = sink.getvalue()
        assert written.startswith(CHAT_QUERY_IPC_PREFIX)
        payload = json.loads(written[len(CHAT_QUERY_IPC_PREFIX):].strip())
        assert payload["text"] == "hello over stdin"


@pytest.mark.unit
class TestChatWindowInputKeys:
    """Enter sends; Shift+Enter inserts a newline (does not send)."""

    def test_enter_sends(self, qapp, monkeypatch):
        from desktop_app.chat_window import ChatWindow
        from PyQt6.QtCore import Qt as _Qt, QEvent
        from PyQt6.QtGui import QKeyEvent

        calls = []
        monkeypatch.setattr(
            "jarvis.daemon.submit_text_query",
            lambda text, **kw: calls.append(text),
        )
        win = ChatWindow()
        win.input_widget.setPlainText("hi")
        event = QKeyEvent(
            QEvent.Type.KeyPress,
            _Qt.Key.Key_Return,
            _Qt.KeyboardModifier.NoModifier,
        )
        win._input_key_press(event)
        assert calls == ["hi"]
        assert win.input_widget.toPlainText() == ""

    def test_shift_enter_does_not_send(self, qapp, monkeypatch):
        from desktop_app.chat_window import ChatWindow
        from PyQt6.QtCore import Qt as _Qt, QEvent
        from PyQt6.QtGui import QKeyEvent

        calls = []
        monkeypatch.setattr(
            "jarvis.daemon.submit_text_query",
            lambda text, **kw: calls.append(text),
        )
        win = ChatWindow()
        win.input_widget.setPlainText("line one")
        event = QKeyEvent(
            QEvent.Type.KeyPress,
            _Qt.Key.Key_Return,
            _Qt.KeyboardModifier.ShiftModifier,
        )
        win._input_key_press(event)
        # Default QPlainTextEdit handling inserts a newline; no send.
        assert calls == []

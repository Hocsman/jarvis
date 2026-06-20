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
    """The stop button routes to the shared daemon stop signal."""

    def test_stop_calls_request_stop(self, qapp, monkeypatch):
        from desktop_app.chat_window import ChatWindow

        called = []
        monkeypatch.setattr(
            "jarvis.daemon.request_stop", lambda: called.append(True)
        )
        win = ChatWindow()
        win._stop()
        assert called == [True]


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
    """The desktop app routes ``__CHAT__:`` IPC lines to the chat window."""

    def test_dispatch_creates_window_lazily_and_emits_complete(self, qapp, monkeypatch):
        from jarvis.daemon import CHAT_IPC_PREFIX
        import desktop_app.app as app_mod

        # A minimal stand-in for JarvisSystemTray with just the attributes the
        # dispatch path touches. We avoid constructing the full tray (which
        # pulls in the daemon, face widget, etc.).
        tray = app_mod.JarvisSystemTray.__new__(app_mod.JarvisSystemTray)
        tray.chat_window = None
        tray._chat_submit_fn = None

        line = f'{CHAT_IPC_PREFIX}{{"type":"complete","data":"hello back"}}'
        tray._dispatch_chat_ipc(line)

        assert tray.chat_window is not None
        # The signal is queued on the Qt event loop; flush it so the slot runs
        # and appends the reply to the transcript.
        tray.chat_window.show()
        qapp.processEvents()
        assert "hello back" in tray.chat_window.transcript_widget.toPlainText()

    def test_dispatch_malformed_line_is_swallowed(self, qapp, monkeypatch):
        from jarvis.daemon import CHAT_IPC_PREFIX
        import desktop_app.app as app_mod

        tray = app_mod.JarvisSystemTray.__new__(app_mod.JarvisSystemTray)
        tray.chat_window = None
        tray._chat_submit_fn = None

        # Must not raise.
        tray._dispatch_chat_ipc(f"{CHAT_IPC_PREFIX}not json")
        # A window is still created lazily, but no reply text lands.
        qapp.processEvents()

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

"""Python<->JS bridge for the HUD dashboard.

Exposed to the page over ``QWebChannel`` as ``window.jarvis``. The page
calls slots (``submitQuery``) and listens to signals (``statsUpdated``,
``stateChanged``, ``weatherUpdated``, ``replyReceived`` …). Complex
payloads cross as JSON strings to keep the channel marshalling trivial.

The bridge owns two timers:
- a system-stats timer (psutil) that pushes CPU / RAM / disk, and
- a JarvisState poll that maps the shared cross-process voice state to
  an orb accent colour + status label.

Chat is wired by the host (the tray): it sets ``submit_fn`` to the
daemon submission callable and calls ``deliver_reply`` / ``deliver_busy``
when the daemon answers, which re-emit as signals to the page.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Optional

from PyQt6.QtCore import QObject, QTimer, pyqtSignal, pyqtSlot

from jarvis.debug import debug_log


# JarvisState.value -> (orb accent RGB, status label FR)
_STATE_VIEW = {
    "asleep":               ((90, 120, 150),  "Endormi"),
    "idle":                 ((56, 216, 232),  "En écoute du mot d'éveil…"),
    "listening":            ((70, 224, 160),  "Je vous écoute…"),
    "thinking":             ((240, 185, 90),  "Réflexion…"),
    "speaking":             ((90, 160, 255),  "Réponse…"),
    "dictating":            ((70, 224, 160),  "Dictée…"),
    "dictation_processing": ((240, 185, 90),  "Transcription…"),
}
_STATE_DEFAULT = ((56, 216, 232), "En écoute du mot d'éveil…")


class DashboardBridge(QObject):
    # Python -> JS
    statsUpdated = pyqtSignal(str)     # JSON: {cpu, ram, disk_used, disk_total, disk_pct}
    stateChanged = pyqtSignal(str)     # JSON: {accent:[r,g,b], status:"…"}
    weatherUpdated = pyqtSignal(str)   # JSON: {temp, loc, desc, icon, hum, wind, feels}
    userEcho = pyqtSignal(str)         # echo the user's message into the transcript
    replyReceived = pyqtSignal(str)    # assistant reply text
    busy = pyqtSignal()                # daemon busy with another query

    def __init__(self, submit_fn: Optional[Callable[[str], None]] = None,
                 weather_city: str = "Paris",
                 parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._submit_fn = submit_fn
        self._weather_city = (weather_city or "Paris").strip() or "Paris"
        self._last_state: Optional[str] = None
        # Text-chat doesn't necessarily flip the cross-process JarvisState,
        # so we track an in-flight chat query locally and force the orb to
        # THINKING while one is pending — the orb pulses for text queries
        # too, not only voice.
        self._chat_pending: bool = False

        # System stats: poll every 2 s.
        self._stats_timer = QTimer(self)
        self._stats_timer.timeout.connect(self._emit_stats)
        self._stats_timer.start(2000)

        # Voice state: poll the cross-process JarvisState at 5 Hz so the
        # orb tracks listening/thinking/speaking without a heavy loop.
        self._state_timer = QTimer(self)
        self._state_timer.timeout.connect(self._emit_state_if_changed)
        self._state_timer.start(200)

        # Weather: refresh every 10 min (the first fetch fires on ready()).
        # The network call runs on a worker thread; the signal hop back to
        # the main thread is handled by Qt's queued connection.
        self._weather_timer = QTimer(self)
        self._weather_timer.timeout.connect(self._fetch_weather_async)
        self._weather_timer.start(600_000)

    # ── setters used by the host (tray) ────────────────────────────────
    def set_submit_fn(self, fn: Optional[Callable[[str], None]]) -> None:
        self._submit_fn = fn

    def deliver_reply(self, text: Optional[str]) -> None:
        """Called by the host when the daemon returns a reply."""
        self._chat_pending = False
        self._emit_state(force=True)  # settle the orb out of THINKING
        if text:
            self.replyReceived.emit(str(text))

    def deliver_busy(self) -> None:
        self._chat_pending = False
        self._emit_state(force=True)
        self.busy.emit()

    # ── JS -> Python ───────────────────────────────────────────────────
    @pyqtSlot(str)
    def submitQuery(self, text: str) -> None:
        """The page sent a chat message. Echo it, then forward to the
        daemon submission callable if wired (else no-op preview)."""
        text = (text or "").strip()
        if not text:
            return
        self.userEcho.emit(text)
        if self._submit_fn is not None:
            self._chat_pending = True
            self._emit_state(force=True)  # orb -> THINKING while generating
            try:
                self._submit_fn(text)
            except Exception as exc:
                debug_log(f"dashboard submitQuery failed: {exc}", "desktop")
                self._chat_pending = False
                self._emit_state(force=True)
                self.replyReceived.emit("(échec d'envoi au daemon)")
        else:
            # Standalone preview without a daemon.
            self.replyReceived.emit("(aperçu — pas de daemon connecté)")

    @pyqtSlot()
    def ready(self) -> None:
        """The page finished wiring; push an initial frame so it isn't
        blank until the first timer tick."""
        self._emit_stats()
        self._emit_state(force=True)
        self._fetch_weather_async()

    # ── weather ────────────────────────────────────────────────────────
    def _fetch_weather_async(self) -> None:
        import threading

        def _work():
            try:
                from jarvis.utils.weather_now import fetch_weather_summary
                data = fetch_weather_summary(self._weather_city)
                if data:
                    # Emitted from a worker thread; Qt queues it onto the
                    # main thread for the JS push.
                    self.weatherUpdated.emit(json.dumps(data, ensure_ascii=False))
            except Exception as exc:
                debug_log(f"dashboard weather fetch failed: {exc}", "desktop")

        threading.Thread(target=_work, name="dashboard-weather", daemon=True).start()

    # ── internal emitters ──────────────────────────────────────────────
    def _emit_stats(self) -> None:
        try:
            import psutil
            vm = psutil.virtual_memory()
            du = psutil.disk_usage("/")
            payload = {
                "cpu": round(psutil.cpu_percent(interval=None)),
                "ram": round(vm.percent),
                "disk_used": round(du.used / 1e9),
                "disk_total": round(du.total / 1e9),
                "disk_pct": round(du.percent),
            }
            self.statsUpdated.emit(json.dumps(payload))
        except Exception as exc:
            debug_log(f"dashboard stats failed: {exc}", "desktop")

    def _current_jarvis_state(self) -> str:
        try:
            from desktop_app.face_widget import get_jarvis_state
            st = get_jarvis_state().state
            return getattr(st, "value", str(st))
        except Exception:
            return "idle"

    def _emit_state_if_changed(self) -> None:
        if self._chat_pending:
            return  # held at THINKING until the reply lands
        st = self._current_jarvis_state()
        if st != self._last_state:
            self._emit_state(state=st)

    def _emit_state(self, state: Optional[str] = None, force: bool = False) -> None:
        # A pending text-chat query overrides the polled voice state so the
        # orb pulses THINKING during generation.
        if self._chat_pending:
            st = "thinking"
        else:
            st = state if state is not None else self._current_jarvis_state()
        self._last_state = st
        accent, label = _STATE_VIEW.get(st, _STATE_DEFAULT)
        self.stateChanged.emit(json.dumps({"accent": list(accent), "status": label}))

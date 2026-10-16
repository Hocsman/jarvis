"""How long a spoken exchange takes, stage by stage.

The journal keeps durations and enumerations and nothing else: no text the
user said, none the assistant answered, none a tool returned. That is what
lets it stay on by default, and it is enforced by the table's shape (see
``Database.record_latency``) and by the way the marks below take their
arguments, not by care at the call sites. The contract is in
``latency.spec.md``.

The pipeline calls a handful of cheap marks. Every one is a no-op when the
journal is off or no turn is open, and none ever raises: the journal is a
side channel and must not cost the user a reply.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from .debug import debug_log

# What the intent judge may be recorded as saying. Anything else is dropped
# rather than stored, so a free-text value can never reach the table.
_JUDGE_OUTCOMES = frozenset({"directed", "not_directed", "no_verdict"})

# A first sound this long after the reply was written belongs to something
# else (a reminder, a routine), not to the reply.
_LATE_SOUND_SECONDS = 30.0

# A heard utterance older than this cannot be the one a query came from; the
# longest a query is collected for is far shorter.
_STALE_HEARD_SECONDS = 300.0

# The reply engine's own stage announcements, and the column each one fills.
_STAGE_COLUMNS = {
    "routing": "route_ms",
    "planning": "plan_ms",
    "memory": "memory_ms",
    "tool": "tools_ms",
    "generating": "generate_ms",
}

_MODEL_NAME_MAX = 120


def _ms(seconds: float) -> int:
    return int(round(seconds * 1000.0))


class _Turn:
    """One exchange being timed. Internal: only the journal touches it."""

    def __init__(self, zero: Optional[float], dispatched: float, heard: Optional[Tuple[float, float]],
                 judge: Optional[Tuple[int, Optional[str]]], models: Mapping[str, Any]):
        # Clock zero: the end of the user's speech when it is known, the
        # dispatch otherwise (a query that did not come from a heard
        # utterance, such as a click-approved resume).
        self.speech_known = zero is not None
        self.zero = zero if zero is not None else dispatched
        self.dispatched = dispatched
        self.transcribed_ms = _ms(heard[1] - heard[0]) if heard else None
        self.judge_ms, self.judge_outcome = judge if judge else (None, None)
        self.models = {k: _clean_model(models.get(k)) for k in ("judge", "router", "chat")}
        self.seconds: Dict[str, float] = {col: 0.0 for col in _STAGE_COLUMNS.values()}
        self.tool_calls = 0
        self.stage_column: Optional[str] = None
        self.stage_since = dispatched
        self.first_sound: Optional[float] = None
        self.frozen = False
        self.router_degraded = False

    def close_stage(self, now: float) -> None:
        """Add the time since the current stage began to that stage."""
        if self.stage_column is not None and not self.frozen:
            self.seconds[self.stage_column] += max(0.0, now - self.stage_since)
        self.stage_since = now


def _clean_model(value: Any) -> Optional[str]:
    if not value or not isinstance(value, str):
        return None
    return value.strip()[:_MODEL_NAME_MAX] or None


class LatencyJournal:
    """The process-wide recorder of how long spoken exchanges take."""

    def __init__(self, clock: Callable[[], float] = time.time):
        self._clock = clock
        self._lock = threading.RLock()
        self._db: Any = None
        self._enabled = False
        self._heard: Optional[Tuple[float, float]] = None       # (speech end, transcript ready)
        self._judge: Optional[Tuple[int, Optional[str]]] = None  # (ms, outcome)
        self._turn: Optional[_Turn] = None
        self._last_row: Optional[Tuple[int, float, float]] = None  # (id, zero, written at)

    # ── Setup ─────────────────────────────────────────────────────────

    def configure(self, db: Any, cfg: Any) -> None:
        """Bind the journal to a database and the user's settings."""
        with self._lock:
            self._db = db
            self._enabled = bool(getattr(cfg, "latency_journal_enabled", True)) and db is not None
            if not self._enabled:
                self._forget()
        debug_log(f"latency journal {'on' if self._enabled else 'off'}", "jarvis")

    def reset(self, clock: Optional[Callable[[], float]] = None) -> None:
        """Back to an unconfigured, empty journal (tests, shutdown)."""
        with self._lock:
            self._clock = clock if clock is not None else time.time
            self._db = None
            self._enabled = False
            self._forget()

    def _forget(self) -> None:
        self._heard = None
        self._judge = None
        self._turn = None
        self._last_row = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ── Marks from the listener ───────────────────────────────────────

    def note_heard(self, speech_end: float) -> None:
        """An utterance ended at ``speech_end`` and its transcript is ready now."""
        if not self._enabled or not speech_end or speech_end <= 0:
            return
        with self._lock:
            self._heard = (float(speech_end), self._clock())
            # A judge verdict belongs to the utterance it judged.
            self._judge = None

    def note_judge(self, seconds: float, outcome: Optional[str]) -> None:
        """The intent judge was asked about the last utterance."""
        if not self._enabled:
            return
        with self._lock:
            self._judge = (_ms(max(0.0, seconds)), outcome if outcome in _JUDGE_OUTCOMES else None)

    def begin_turn(self, models: Optional[Mapping[str, Any]] = None) -> None:
        """The query was handed to the reply engine: open the turn."""
        if not self._enabled:
            return
        with self._lock:
            if self._turn is not None:
                debug_log("latency journal: a turn was left open, dropped", "jarvis")
            now = self._clock()
            heard = self._heard
            if heard is not None and now - heard[0] > _STALE_HEARD_SECONDS:
                heard = None
            zero = heard[0] if heard else None
            self._turn = _Turn(zero, now, heard, self._judge if heard else None, models or {})
            self._heard = None
            self._judge = None

    # ── Marks from the reply engine and the speech engines ────────────

    def on_stage(self, stage_id: str, detail: Any = None) -> None:
        """The reply engine announced a stage. ``detail`` is never read:
        it can carry the name of a tool or text, and neither is recorded."""
        with self._lock:
            turn = self._turn
            if turn is None:
                return
            now = self._clock()
            turn.close_stage(now)
            turn.stage_column = _STAGE_COLUMNS.get(stage_id)
            if stage_id == "tool":
                turn.tool_calls += 1

    def note_router_degraded(self) -> None:
        """The tool router gave up this turn and the keyword strategy answered."""
        with self._lock:
            if self._turn is not None:
                self._turn.router_degraded = True

    def mark_first_sound(self) -> None:
        """Speech began. Only the first sound of a turn counts."""
        with self._lock:
            now = self._clock()
            turn = self._turn
            if turn is not None:
                if turn.first_sound is None:
                    turn.close_stage(now)
                    turn.first_sound = now
                    turn.frozen = True
                return
            self._late_sound(now)

    def _late_sound(self, now: float) -> None:
        """A sound after the reply was written: fill in the row it belongs to."""
        last = self._last_row
        if last is None:
            return
        row_id, zero, written_at = last
        self._last_row = None
        if now - written_at > _LATE_SOUND_SECONDS:
            return
        try:
            self._db.set_latency_first_sound(row_id, _ms(now - zero))
        except Exception as e:  # noqa: BLE001
            debug_log(f"latency journal: first sound not recorded: {e}", "jarvis")

    def end_turn(self) -> None:
        """The reply engine returned: write the row."""
        with self._lock:
            turn = self._turn
            if turn is None:
                return
            self._turn = None
            try:
                self._write(turn)
            except Exception as e:  # noqa: BLE001
                debug_log(f"latency journal: row not written: {e}", "jarvis")

    def _write(self, turn: _Turn) -> None:
        now = self._clock()
        turn.close_stage(now)
        row: Dict[str, Any] = {
            "ts_utc": datetime.fromtimestamp(turn.zero, timezone.utc).isoformat(),
            "origin": "voice",
            "transcribed_ms": turn.transcribed_ms,
            "judge_ms": turn.judge_ms,
            "judge_outcome": turn.judge_outcome,
            "dispatch_ms": _ms(turn.dispatched - turn.zero) if turn.speech_known else None,
            "first_sound_ms": _ms(turn.first_sound - turn.zero) if turn.first_sound is not None else None,
            "total_ms": _ms(now - turn.zero),
            "router_degraded": 1 if turn.router_degraded else 0,
            "tool_calls": turn.tool_calls,
            "judge_model": turn.models["judge"],
            "router_model": turn.models["router"],
            "chat_model": turn.models["chat"],
        }
        for column, seconds in turn.seconds.items():
            row[column] = _ms(seconds)
        row_id = self._db.record_latency(row)
        if row_id is not None and turn.first_sound is None:
            self._last_row = (row_id, turn.zero, now)


_journal = LatencyJournal()


def get_latency_journal() -> LatencyJournal:
    """The process-wide journal."""
    return _journal

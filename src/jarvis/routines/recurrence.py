"""
📆 When a recurring thing is next owed, and when it is simply skipped.

Pure functions over wall-clock local time, reprojected through ZoneInfo.
No UTC arithmetic anywhere: twenty-four hours is not always a day, and a
routine that advances by adding 86400 seconds drifts an hour every spring
and never comes back.

The catch-up rule is the **opposite** of the reminder rule, deliberately.
A reminder is a one-off promise: owed since Thursday, it is still said on
Monday, late and saying so, because dropping it silently breaks the
promise. A routine is a standing habit: "summarise my mail every morning"
owed three times over a long weekend does not mean three summaries on
Monday afternoon — it means one at most, because a summary of Saturday's
mail delivered on Monday is noise wearing the costume of diligence.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - Python < 3.9
    ZoneInfo = None  # type: ignore[assignment]

# What `should_run` concluded. Sentinels rather than booleans so a caller
# cannot mistake "not yet" for "too late" — they are the same False and
# very different situations.
RUN = "run"
SKIP = "skip"

_DAILY = "daily"
_WEEKLY = "weekly"
_KINDS = (_DAILY, _WEEKLY)


@dataclass(frozen=True)
class Regle:
    """How often, and at what o'clock.

    Deliberately small. Anything a cron expression can say and this
    cannot is something the user would have to be taught to say, and
    every such thing is a way for a routine to run at a moment they did
    not picture.
    """

    kind: str
    hour: int
    minute: int
    weekday: Optional[int]  # 0 = Monday, for weekly only

    def to_json(self) -> Dict[str, Any]:
        return {
            "kind": self.kind, "hour": self.hour, "minute": self.minute,
            "weekday": self.weekday,
        }

    @classmethod
    def from_json(cls, raw: Any) -> "Regle":
        """Build one, or refuse. A routine that cannot say when it runs
        must not exist."""
        if not isinstance(raw, dict):
            raise ValueError("la règle n'est pas un objet")

        kind = raw.get("kind")
        if kind not in _KINDS:
            raise ValueError(f"récurrence inconnue : {kind!r}")

        hour, minute = raw.get("hour"), raw.get("minute")
        for name, value, top in (("hour", hour, 23), ("minute", minute, 59)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} n'est pas un entier")
            if not 0 <= value <= top:
                raise ValueError(f"{name} hors bornes : {value}")

        weekday = raw.get("weekday")
        if kind == _WEEKLY:
            if isinstance(weekday, bool) or not isinstance(weekday, int):
                raise ValueError("une règle hebdomadaire a besoin d'un jour")
            if not 0 <= weekday <= 6:
                raise ValueError(f"jour hors bornes : {weekday}")
        else:
            weekday = None

        return cls(kind=kind, hour=hour, minute=minute, weekday=weekday)


def period_seconds(regle: Regle) -> int:
    """How long between two occurrences, for the staleness window."""
    return 7 * 86400 if regle.kind == _WEEKLY else 86400


def next_occurrence(regle: Regle, after_local: datetime, tz_name: str) -> datetime:
    """The next wall-clock moment the rule names, strictly after
    ``after_local``.

    Strictly, because a moment exactly on the hour must move forward or
    the row fires twice for one occurrence.

    Computed on the calendar rather than by adding a period: that is what
    keeps 07:00 at 07:00 across a clock change instead of drifting an
    hour twice a year.
    """
    candidate = after_local.replace(
        hour=regle.hour, minute=regle.minute, second=0, microsecond=0,
    )

    if regle.kind == _WEEKLY:
        ahead = (regle.weekday - candidate.weekday()) % 7
        candidate += timedelta(days=ahead)
        if candidate <= after_local:
            candidate += timedelta(days=7)
        return _real_instant(candidate, tz_name)

    if candidate <= after_local:
        candidate += timedelta(days=1)
    return _real_instant(candidate, tz_name)


def _real_instant(candidate: datetime, tz_name: str) -> datetime:
    """Nudge a wall-clock reading that does not exist onto one that does.

    On a spring-forward night 02:30 is not a time. A routine set for it
    must run at the first moment that does exist, not vanish for a year.
    """
    if not tz_name or ZoneInfo is None:
        return candidate
    try:
        zone = ZoneInfo(tz_name)
    except Exception:
        return candidate

    # A non-existent local time round-trips to a different wall clock.
    aware = candidate.replace(tzinfo=zone)
    if aware.astimezone(timezone.utc).astimezone(zone).replace(tzinfo=None) == candidate:
        return candidate
    for minutes in range(1, 121):
        moved = candidate + timedelta(minutes=minutes)
        aware = moved.replace(tzinfo=zone)
        if aware.astimezone(timezone.utc).astimezone(zone).replace(tzinfo=None) == moved:
            return moved
    return candidate


def staleness_window(regle: Regle, cap: float) -> float:
    """How late a run may be and still be worth doing.

    Bounded by the period, so a daily routine never fires for yesterday
    while today's slot is already approaching.
    """
    return min(float(cap), float(period_seconds(regle)))


def should_run(due_utc: str, now_utc: str, regle: Regle, *, cap: float) -> str:
    """RUN if this occurrence is owed and still worth doing, else SKIP.

    Skipping is not dropping: the caller advances the row to its next
    occurrence either way, so a skipped morning costs one summary rather
    than the routine.
    """
    try:
        due = datetime.fromisoformat(due_utc)
        now = datetime.fromisoformat(now_utc)
    except Exception:
        # A row nobody can place must not fire at an arbitrary moment.
        return SKIP

    behind = (now - due).total_seconds()
    if behind < 0:
        return SKIP
    return RUN if behind <= staleness_window(regle, cap) else SKIP

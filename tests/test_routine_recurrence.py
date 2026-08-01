"""When a recurring thing is next owed, and when it is simply skipped.

The catch-up rule here is the **opposite** of the reminder rule, and the
reversal is deliberate. A reminder is a one-off promise: owed since
Thursday, it is still said on Monday, late and saying so, because
dropping it silently breaks the promise. A routine is a standing habit:
"summarise my mail every morning" owed three times over a long weekend
does not mean three summaries at 14:00 on Monday — it means one, if
that, because a summary of Saturday's mail delivered on Monday afternoon
is noise wearing the costume of diligence.

So a run that is too stale is skipped, and the next occurrence is
computed from the calendar rather than by adding a period to a missed
slot. Adding periods is how a daily 07:00 routine drifts to 08:00 across
a daylight-saving change and never comes back.

Everything here is a pure function over wall-clock local time,
reprojected through ZoneInfo. No UTC arithmetic: 24 hours is not always
a day.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from src.jarvis.routines.recurrence import (
    RUN,
    SKIP,
    Regle,
    next_occurrence,
    period_seconds,
    should_run,
)

PARIS = "Europe/Paris"


def _daily(hour=7, minute=0):
    return Regle(kind="daily", hour=hour, minute=minute, weekday=None)


def _weekly(weekday=0, hour=7, minute=0):
    return Regle(kind="weekly", hour=hour, minute=minute, weekday=weekday)


# ── The next time it is owed ──────────────────────────────────────────


def test_a_daily_rule_lands_at_its_hour_tomorrow():
    after = datetime(2026, 8, 1, 14, 0)  # Saturday afternoon

    assert next_occurrence(_daily(7), after, PARIS) == datetime(2026, 8, 2, 7, 0)


def test_a_daily_rule_lands_today_when_its_hour_has_not_come():
    after = datetime(2026, 8, 1, 5, 0)

    assert next_occurrence(_daily(7), after, PARIS) == datetime(2026, 8, 1, 7, 0)


def test_the_next_occurrence_is_strictly_after_now():
    """Exactly on the hour must move forward, or the row fires twice."""
    after = datetime(2026, 8, 1, 7, 0)

    assert next_occurrence(_daily(7), after, PARIS) == datetime(2026, 8, 2, 7, 0)


def test_a_weekly_rule_lands_on_its_weekday():
    after = datetime(2026, 8, 1, 14, 0)  # Saturday

    # weekday 0 = Monday
    assert next_occurrence(_weekly(0, 7), after, PARIS) == datetime(2026, 8, 3, 7, 0)


def test_a_weekly_rule_on_today_before_its_hour_stays_today():
    after = datetime(2026, 8, 3, 5, 0)  # Monday morning

    assert next_occurrence(_weekly(0, 7), after, PARIS) == datetime(2026, 8, 3, 7, 0)


def test_a_weekly_rule_on_today_after_its_hour_waits_a_week():
    after = datetime(2026, 8, 3, 9, 0)

    assert next_occurrence(_weekly(0, 7), after, PARIS) == datetime(2026, 8, 10, 7, 0)


# ── It stays at its hour across a clock change ────────────────────────


def test_a_daily_rule_is_still_at_seven_two_hundred_days_later():
    """Adding 86400 seconds repeatedly is how 07:00 becomes 08:00 and
    never comes back. Europe/Paris changes twice a year."""
    when = datetime(2026, 8, 1, 5, 0)
    for _ in range(200):
        when = next_occurrence(_daily(7), when, PARIS)

    assert (when.hour, when.minute) == (7, 0)


def test_it_crosses_the_spring_change_without_drifting():
    # 2027-03-28 is the spring change in Europe/Paris: 02:00 → 03:00.
    after = datetime(2027, 3, 27, 12, 0)
    first = next_occurrence(_daily(7), after, PARIS)
    second = next_occurrence(_daily(7), first, PARIS)

    assert (first.hour, second.hour) == (7, 7)


def test_it_crosses_the_autumn_change_without_repeating():
    # 2026-10-25, when 02:00–03:00 happens twice.
    after = datetime(2026, 10, 24, 12, 0)
    first = next_occurrence(_daily(7), after, PARIS)
    second = next_occurrence(_daily(7), first, PARIS)

    assert first == datetime(2026, 10, 25, 7, 0)
    assert second == datetime(2026, 10, 26, 7, 0)


def test_an_hour_that_does_not_exist_lands_on_the_first_one_that_does():
    """02:30 on a spring-forward night is not a time. A routine set for
    it must run, not vanish for a year."""
    after = datetime(2027, 3, 27, 12, 0)

    got = next_occurrence(Regle(kind="daily", hour=2, minute=30, weekday=None),
                          after, PARIS)

    # It exists as an instant, whatever wall-clock reading it takes.
    assert got.replace(tzinfo=ZoneInfo(PARIS)).astimezone(timezone.utc)


def test_an_unknown_timezone_still_produces_an_occurrence():
    """An empty zone is documented as naive about far-off changes, not as
    a reason to stop scheduling."""
    after = datetime(2026, 8, 1, 14, 0)

    assert next_occurrence(_daily(7), after, "") == datetime(2026, 8, 2, 7, 0)


# ── A long weekend does not become three summaries ────────────────────


def _utc(when_local: datetime, tz=PARIS) -> str:
    return when_local.replace(tzinfo=ZoneInfo(tz)).astimezone(timezone.utc).isoformat()


def test_a_run_missed_by_hours_still_happens():
    """Late is not the same as pointless. The morning summary at 09:30
    is still the morning."""
    due = _utc(datetime(2026, 8, 3, 7, 0))
    now = _utc(datetime(2026, 8, 3, 9, 30))

    assert should_run(due, now, _daily(7), cap=86400) is RUN


def test_a_daily_run_missed_by_a_whole_weekend_is_skipped():
    """Three summaries at Monday 14:00, two of them of Saturday's mail,
    is noise wearing the costume of diligence."""
    due = _utc(datetime(2026, 7, 31, 7, 0))  # Friday
    now = _utc(datetime(2026, 8, 3, 14, 0))  # Monday afternoon

    assert should_run(due, now, _daily(7), cap=86400) is SKIP


def test_a_weekly_run_has_a_longer_memory_than_a_daily_one():
    """Missing Monday's weekly review by a day still matters; missing
    Monday's daily summary by a day does not."""
    due = _utc(datetime(2026, 8, 3, 7, 0))
    now = _utc(datetime(2026, 8, 4, 9, 0))  # a day later

    assert should_run(due, now, _weekly(0, 7), cap=7 * 86400) is RUN
    assert should_run(due, now, _daily(7), cap=86400) is SKIP


def test_a_future_run_is_not_due_at_all():
    due = _utc(datetime(2026, 8, 4, 7, 0))
    now = _utc(datetime(2026, 8, 3, 14, 0))

    assert should_run(due, now, _daily(7), cap=86400) is SKIP


def test_the_cap_can_narrow_the_window(tmp_path):
    """A user who wants less catch-up gets less."""
    due = _utc(datetime(2026, 8, 3, 7, 0))
    now = _utc(datetime(2026, 8, 3, 9, 30))

    assert should_run(due, now, _daily(7), cap=3600) is SKIP


def test_the_cap_cannot_widen_it_past_one_period():
    """Raising the cap must not let a daily routine fire for yesterday
    while today's slot is a few hours away — that is two runs in a day,
    which is the thing the window exists to prevent."""
    from src.jarvis.routines.recurrence import staleness_window

    due = _utc(datetime(2026, 7, 31, 7, 0))
    now = _utc(datetime(2026, 8, 3, 14, 0))

    assert staleness_window(_daily(7), cap=30 * 86400) == 86400
    assert should_run(due, now, _daily(7), cap=30 * 86400) is SKIP


def test_an_unreadable_due_time_is_skipped_rather_than_run():
    """A row nobody can place must not fire at an arbitrary moment."""
    assert should_run("n'importe quoi", _utc(datetime(2026, 8, 3, 9, 0)),
                      _daily(7), cap=86400) is SKIP


# ── The period, for the staleness window ──────────────────────────────


def test_a_daily_period_is_a_day():
    assert period_seconds(_daily(7)) == 86400


def test_a_weekly_period_is_a_week():
    assert period_seconds(_weekly(0, 7)) == 7 * 86400


# ── The rule itself ───────────────────────────────────────────────────


def test_a_rule_round_trips_through_json():
    """It lives in `payload`, so it has to survive being written and read
    back with nothing lost."""
    import json

    rule = _weekly(2, 18, 30)
    again = Regle.from_json(json.loads(json.dumps(rule.to_json())))

    assert again == rule


@pytest.mark.parametrize("bad", [
    {"kind": "hourly", "hour": 7, "minute": 0},
    {"kind": "weekly", "hour": 7, "minute": 0},          # no weekday
    {"kind": "daily", "hour": 25, "minute": 0},
    {"kind": "daily", "hour": 7, "minute": 61},
    {"kind": "weekly", "hour": 7, "minute": 0, "weekday": 9},
    {},
])
def test_an_impossible_rule_is_refused(bad):
    """A routine that cannot say when it runs must not be created."""
    with pytest.raises(ValueError):
        Regle.from_json(bad)

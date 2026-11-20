"""Agenda tool: what the user's calendar holds on the days asked about.

Read-only by construction: the calendar comes from a file or an HTTPS
address the user named in the settings, and nothing here can write to
either. Only each event's time, title and place come back, fenced as
untrusted data, since an invitation is written by somebody else.
See `read_agenda.spec.md`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, tzinfo
from typing import Any, Dict, List, Optional, Tuple

from ....debug import debug_log
from ....utils import fence
from ....utils.time_context import machine_zone, spell_day
from ...base import Tool, ToolContext
from ...types import ToolExecutionResult
from . import ics
from .source import AgendaUnreadable, kind_of, read_source

# A month at most: past that the answer to "what do I have" is a wall.
MAX_DAYS = 31
# A full week of a busy calendar is a few thousand characters. Past this a
# small model has no context left for the question.
MAX_AGENDA_CHARS = 6_000
# For a title or a place, each on its one line.
MAX_FIELD_CHARS = 120

_WEEKDAYS = (
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
)
# Every word `day` takes. They are the tool's own vocabulary, like the name
# of an argument: the model maps what was said, in whatever language, onto
# one of them, and the schema lists them so it is held to that.
DAY_WORDS = ("today", "tomorrow", "yesterday") + _WEEKDAYS
_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})(?:[t ].*)?$")
_WHOLE_NUMBER_RE = re.compile(r"^[+-]?\d+$")

# What a meeting link looks like, wherever it sits: `scheme://…`, `www.…`,
# or a host followed by a path with no scheme written (`host.example/j/1`).
# The last form also takes a few things that are no link (two words joined
# by a full stop and a slash); a word lost from a title costs less than a
# link that lets someone into the meeting.
_ADDRESS_RE = re.compile(
    r"(?:[a-z][a-z0-9+.\-]*://|www\.)\S*"
    r"|(?<![\w.\-])(?:[a-z0-9\-]+\.)+[a-z]{2,}/\S*",
    re.IGNORECASE,
)
# Whitespace of every kind and the control characters: what could break a
# line or hide in one.
_BREAKS_RE = re.compile(r"[\s\x00-\x1f\x7f-\x9f]+")

_ONE_DAY = timedelta(days=1)

_DAY_HELP = (
    "The day was not understood. Give `day` as `today`, `tomorrow`, `yesterday` "
    "or a weekday in English (`monday` to `sunday`: the next one after today), "
    "or give `date` as `YYYY-MM-DD`."
)
_DAYS_HELP = f"`days` must be a whole number of days (1 to {MAX_DAYS})."


class BadDays(ValueError):
    """The arguments name no days. The message says what is accepted."""


@dataclass(frozen=True)
class Days:
    first: date
    last: date
    # More days were asked for than are read at a time.
    capped: bool


def resolve_days(day: Any, days: Any, today: date) -> Days:
    """The days a call names, worked out from `today`.

    `day` is a word of the tool's own small vocabulary (`DAY_WORDS`) or a
    date; no sentence of the user's is read here. A weekday is the first such day
    after today, never today itself. Anything not understood raises: read
    as today, it would answer another question than the one asked.
    """
    if day is None:
        word = ""
    elif isinstance(day, str):
        word = day.strip().lower()
    else:
        raise BadDays(_DAY_HELP)

    if word in ("", "today"):
        first = today
    elif word == "tomorrow":
        first = today + _ONE_DAY
    elif word == "yesterday":
        first = today - _ONE_DAY
    elif word in _WEEKDAYS:
        ahead = (_WEEKDAYS.index(word) - today.weekday() - 1) % 7 + 1
        first = today + timedelta(days=ahead)
    else:
        match = _DATE_RE.match(word)
        if not match:
            raise BadDays(_DAY_HELP)
        try:
            first = date(*(int(g) for g in match.groups()))
        except ValueError:
            raise BadDays(_DAY_HELP) from None

    if days is None or (isinstance(days, str) and not days.strip()):
        count = 1
    elif isinstance(days, bool):
        raise BadDays(_DAYS_HELP)
    elif isinstance(days, int):
        count = days
    elif isinstance(days, float) and days.is_integer():
        count = int(days)
    elif isinstance(days, str) and _WHOLE_NUMBER_RE.match(days.strip()):
        count = int(days.strip())
    else:
        raise BadDays(_DAYS_HELP)

    capped = count > MAX_DAYS
    count = max(1, min(count, MAX_DAYS))
    try:
        last = first + timedelta(days=count - 1)
    except OverflowError:
        raise BadDays(_DAY_HELP) from None
    return Days(first, last, capped)


def _zone() -> Tuple[tzinfo, str]:
    """The zone the days are resolved and the times shown in, and its
    name when it has one."""
    return machine_zone()


def _today(zone: tzinfo) -> date:
    return datetime.now(zone).date()


# ── What comes back ───────────────────────────────────────────────────


def _shown(text: str) -> str:
    """A title or a place as it is returned: no address in it, on one
    line, and no longer than `MAX_FIELD_CHARS`.

    A place is often a meeting link, and the link is what lets someone
    join. A line break would let an invitation write a day heading or an
    event of its own below its title.
    """
    text = _BREAKS_RE.sub(" ", _ADDRESS_RE.sub(" ", text)).strip()
    if len(text) > MAX_FIELD_CHARS:
        text = text[:MAX_FIELD_CHARS].rstrip() + "…"
    return text


def _clock(moment: datetime) -> str:
    return f"{moment.hour:02d}:{moment.minute:02d}"


def _day_lines(day: date, zone: tzinfo, occurrences: List[ics.Occurrence]) -> List[str]:
    """One line per event that touches `day`, clipped to it: all-day
    events first, then by start."""
    opens = datetime.combine(day, time.min, tzinfo=zone)
    closes = datetime.combine(day + _ONE_DAY, time.min, tzinfo=zone)
    rows: List[Tuple[int, datetime, str]] = []
    for event in occurrences:
        if not (event.start < closes and (event.end > opens or event.start >= opens)):
            continue
        began_before = event.start < opens
        goes_on = event.end > closes
        whole_day = event.all_day or (event.start <= opens and event.end >= closes)
        if whole_day:
            when = "all day"
        else:
            begins = "00:00" if began_before else _clock(event.start)
            ends = "24:00" if event.end >= closes else _clock(event.end)
            when = begins if event.end == event.start else f"{begins} to {ends}"
            if goes_on:
                when += ", goes on the next day"
            elif began_before:
                when += ", began the day before"
        line = f"- {when}: {_shown(event.title) or '(no title)'}"
        place = _shown(event.place)
        if place:
            line += f" [place: {place}]"
        rows.append((0 if whole_day else 1, event.start, line))
    return [line for _rank, _start, line in sorted(rows)]


def _span(days: Days) -> str:
    if days.first == days.last:
        return f"on {spell_day(days.first)}"
    return f"from {spell_day(days.first)} to {spell_day(days.last)}"


def _result_text(days: Days, zone: tzinfo, zone_name: str, reading: ics.Reading) -> str:
    """What the model reads: the events of each day inside one fence
    (`fence.block`), or a plain sentence when the days hold none, then
    what the reading left out. No blank line anywhere: the tool-result
    digest splits on them, and a piece without its markers would read as
    this tool's own words."""
    if reading.occurrences:
        body: List[str] = []
        day = days.first
        while day <= days.last:
            body.append(spell_day(day))
            body.extend(_day_lines(day, zone, reading.occurrences) or ["(nothing)"])
            day += _ONE_DAY
        times = zone_name or "the computer's local time"
        text = fence.block(
            fence.AGENDA_TEXT,
            f"Events in the user's agenda {_span(days)}, times in {times}",
            "\n".join(body),
            holder="agenda",
            max_chars=MAX_AGENDA_CHARS,
        )
    else:
        text = f"The user's agenda was read and holds no event {_span(days)}."

    if reading.unreadable:
        one = reading.unreadable == 1
        text += (
            f"\n[{reading.unreadable} event{'' if one else 's'} of the agenda could "
            "not be read (a date, a time zone or a recurrence this reader does not "
            f"understand) and {'is' if one else 'are'} missing from this list.]"
        )
    if days.capped:
        text += (
            f"\n[Only {MAX_DAYS} days are read at a time: the reading stops on "
            f"{spell_day(days.last)}.]"
        )
    return text


class ReadAgendaTool(Tool):
    """Reads the user's agenda. It has no way to change it."""

    def risk_for(self, args):
        """Not `lecture`, though it changes nothing. It takes his days to
        the reply model, and with a remote one that cannot be taken back.
        As `libre` any turn could do that unasked, one that has just read
        a web page included; as `action` he is asked first, and a spoken
        yes is enough."""
        from ...policy import RISK_ACTION

        return RISK_ACTION

    def offered(self, cfg) -> bool:
        """Only once a source is named: off, every call could only say
        that no agenda is configured."""
        return bool(kind_of(getattr(cfg, "agenda_source", "")))

    @property
    def name(self) -> str:
        return "readAgenda"

    @property
    def description(self) -> str:
        # The planner reads the first 120 characters, the router the first
        # sentence, the chat model all of it: what the tool is for comes
        # first, in the words he asks with. The last sentence is for a
        # model that is shown the arguments' names and types and no more.
        return (
            "Read the user's own agenda (calendar) to answer what they have "
            "planned or scheduled on a day, today, tomorrow or any other, and "
            "whether they are free: their appointments, meetings and events. "
            "Use it when they ask what they have planned, what is on their "
            "agenda or calendar, or whether they are free or busy at some time. "
            "Not when they only mention a meeting or a calendar without asking "
            "about their own, and not to set a reminder. It only reads: it "
            "cannot create, move or answer an event. Returns each event's time, "
            "title and place. `day` is one of " + ", ".join(DAY_WORDS) + " "
            "(always these English words); a calendar date goes in `date` as "
            "YYYY-MM-DD."
        )

    @property
    def inputSchema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "day": {
                    "type": "string",
                    "enum": list(DAY_WORDS),
                    "description": (
                        "OPTIONAL. The first day to read, when the user names it "
                        "by a word: today, tomorrow, yesterday, or a weekday (the "
                        "next one after today). Default: today. For a part of a "
                        "day (an afternoon), give the day."
                    ),
                },
                "date": {
                    "type": "string",
                    "description": (
                        "OPTIONAL. The first day to read, when the user names a "
                        "calendar date: `YYYY-MM-DD`. Use it instead of `day`."
                    ),
                },
                "days": {
                    "type": "integer",
                    "description": (
                        "OPTIONAL. How many days to read from that day on, "
                        f"1 to {MAX_DAYS}. Default: 1. A week is 7; a weekend is "
                        "`saturday` with 2."
                    ),
                },
            },
            "required": [],
        }

    def run(self, args: Optional[Dict[str, Any]], context: ToolContext) -> ToolExecutionResult:
        args = args if isinstance(args, dict) else {}
        setting = str(getattr(context.cfg, "agenda_source", "") or "").strip()
        if not kind_of(setting):
            debug_log("readAgenda: no source configured", "agenda")
            return ToolExecutionResult(
                success=False,
                reply_text="No agenda is configured: the `agenda_source` setting is empty.",
            )

        zone, zone_name = _zone()
        try:
            # One resolver reads both: a date named beside a word wins,
            # being the more exact of the two.
            days = resolve_days(
                args.get("date") or args.get("day"), args.get("days"), _today(zone),
            )
        except BadDays as why:
            debug_log("readAgenda: the arguments name no days", "agenda")
            return ToolExecutionResult(success=False, reply_text=str(why))

        context.user_print("📅 Reading the agenda…")
        debug_log(
            f"readAgenda: first={days.first} last={days.last} capped={days.capped} "
            f"source={kind_of(setting)}",
            "agenda",
        )
        try:
            reading = ics.read_events(read_source(setting), days.first, days.last, zone)
        except AgendaUnreadable as why:
            return self._not_read(context, str(why))
        except ics.NotACalendar:
            return self._not_read(
                context,
                "The agenda source did not hold a calendar: no iCalendar data was "
                "found in what it returned.",
            )
        except ics.TooLong:
            return self._not_read(
                context, "Reading the agenda took too long and was given up."
            )
        except Exception as e:
            # Never the message: it may quote an event or the address.
            return self._not_read(
                context, f"The agenda could not be read ({type(e).__name__})."
            )

        text = _result_text(days, zone, zone_name, reading)
        debug_log(
            f"readAgenda: events={len(reading.occurrences)} "
            f"unreadable={reading.unreadable} chars={len(text)}",
            "agenda",
        )
        count = len(reading.occurrences)
        context.user_print(
            f"  ✅ Agenda read: {count} event{'' if count == 1 else 's'}."
        )
        return ToolExecutionResult(success=True, reply_text=text)

    @staticmethod
    def _not_read(context: ToolContext, why: str) -> ToolExecutionResult:
        """A failure that says why. `why` is written by this code, never
        taken from the source."""
        debug_log(f"readAgenda: nothing read: {why}", "agenda")
        context.user_print(f"  ⚠️ Agenda not read: {why}")
        return ToolExecutionResult(success=False, reply_text=why)

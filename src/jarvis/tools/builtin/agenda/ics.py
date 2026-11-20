"""Reading a calendar's text (iCalendar, RFC 5545) into the events of the
days asked about.

What is kept of an event is when it is, its title and its place. Every
other property (the description, the attendees, links, alarms) is dropped
as the text is read: `Occurrence` has no field for it, so nothing
downstream can return it. See `read_agenda.spec.md`.

Nothing here logs: this module handles the events' own text, and no debug
line may hold it.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone, tzinfo
from time import monotonic
from typing import Dict, Iterator, List, Optional, Set, Tuple
from zoneinfo import ZoneInfo

from dateutil import rrule as dateutil_rrule
from dateutil import tz as dateutil_tz

# How many occurrences of one recurring event are walked before the days
# asked about are reached. A daily event begun forty years ago is under
# fifteen thousand; a rule firing every second since 1970, which an
# invitation is free to carry, would otherwise hold the call.
MAX_OCCURRENCES = 50_000

# For the whole reading. A rule that never yields (the 30th of February,
# every second) is walked by the recurrence library to the end of its
# calendar, which takes it a few seconds, and nothing can stop it midway:
# the clock is looked at between two events, so a calendar carrying many
# such rules costs this long plus one rule, not their sum.
_READ_WALL_CLOCK_SEC = 10.0

_LINE_BREAK_RE = re.compile(r"\r\n|\n|\r")
_DATE_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})$")
_DATETIME_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})[Tt](\d{2})(\d{2})(\d{2})([Zz]?)$")
_DURATION_RE = re.compile(
    r"^([+-])?P(?:(\d+)W)?(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?$",
    re.IGNORECASE,
)

_ONE_DAY = timedelta(days=1)


class NotACalendar(Exception):
    """The text holds no calendar at all (a sign-in page, an empty file)."""


class TooLong(Exception):
    """The reading passed its time limit and was given up: a partial
    list would read as the whole agenda."""


def _clock() -> float:
    return monotonic()


class _Unreadable(Exception):
    """One event that cannot be placed in time. It is counted, never
    shown, and the rest of the calendar is still read."""


@dataclass(frozen=True)
class Occurrence:
    """One event on the days asked about: `start` and `end` in the zone
    the reading was asked in, `end` excluded."""

    start: datetime
    end: datetime
    all_day: bool
    title: str
    place: str


@dataclass
class Reading:
    occurrences: List[Occurrence]
    # Events that could not be placed, so the caller can say the list is
    # not the whole agenda.
    unreadable: int


# ── Lines ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _Line:
    name: str
    params: Dict[str, str]
    value: str


def _unfolded(text: str) -> List[str]:
    """The content lines, each long one put back together (a line that
    opens with a space or a tab continues the one before)."""
    lines: List[str] = []
    for raw in _LINE_BREAK_RE.split(text):
        if raw[:1] in (" ", "\t") and lines:
            lines[-1] += raw[1:]
        else:
            lines.append(raw)
    return lines


def _split_outside_quotes(text: str, separator: str) -> List[str]:
    parts: List[str] = []
    current: List[str] = []
    quoted = False
    for ch in text:
        if ch == '"':
            quoted = not quoted
        if ch == separator and not quoted:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    parts.append("".join(current))
    return parts


def _content_line(raw: str) -> Optional[_Line]:
    """`NAME;PARAM=value:value` taken apart, or None for a line that is
    not one."""
    quoted = False
    for at, ch in enumerate(raw):
        if ch == '"':
            quoted = not quoted
        elif ch == ":" and not quoted:
            head, value = raw[:at], raw[at + 1:]
            break
    else:
        return None
    pieces = _split_outside_quotes(head, ";")
    name = pieces[0].strip().upper()
    if not name:
        return None
    params: Dict[str, str] = {}
    for piece in pieces[1:]:
        key, _, param_value = piece.partition("=")
        params[key.strip().upper()] = param_value.strip().strip('"')
    return _Line(name, params, value)


def _unescaped(value: str) -> str:
    """A TEXT value as it reads: `\\n` is a line break, a backslash
    before any other character gives that character."""
    out: List[str] = []
    at = 0
    while at < len(value):
        ch = value[at]
        if ch == "\\" and at + 1 < len(value):
            following = value[at + 1]
            out.append("\n" if following in "nN" else following)
            at += 2
        else:
            out.append(ch)
            at += 1
    return "".join(out)


def _components(lines: List[str]) -> Tuple[List[List[_Line]], List[str], bool]:
    """Each event's own lines, the raw lines of the time zones the file
    defines, and whether the text holds a calendar at all.

    What is nested inside an event (an alarm has a `SUMMARY` and a
    `DESCRIPTION` of its own) is not read as the event's.
    """
    events: List[List[_Line]] = []
    zone_lines: List[str] = []
    is_calendar = False
    stack: List[str] = []
    current: Optional[List[_Line]] = None

    for raw in lines:
        line = _content_line(raw)
        if line is None:
            continue
        if line.name == "BEGIN":
            kind = line.value.strip().upper()
            stack.append(kind)
            if kind == "VCALENDAR":
                is_calendar = True
            if "VTIMEZONE" in stack:
                zone_lines.append(raw)
            elif kind == "VEVENT" and "VCALENDAR" in stack:
                current = []
            continue
        if line.name == "END":
            kind = line.value.strip().upper()
            if "VTIMEZONE" in stack:
                zone_lines.append(raw)
            if kind == "VEVENT" and current is not None and stack and stack[-1] == "VEVENT":
                events.append(current)
                current = None
            if kind in stack:
                while stack and stack.pop() != kind:
                    pass
            continue
        if "VTIMEZONE" in stack:
            zone_lines.append(raw)
        elif current is not None and stack and stack[-1] == "VEVENT":
            current.append(line)
    return events, zone_lines, is_calendar


# ── Time zones ────────────────────────────────────────────────────────


class _Zones:
    """What a `TZID` means: an IANA zone when it names one, otherwise the
    `VTIMEZONE` the file carries under that name."""

    def __init__(self, zone_lines: List[str], local: tzinfo):
        self.local = local
        self._zone_lines = zone_lines
        self._of_the_file = None
        self._known: Dict[str, tzinfo] = {}

    def get(self, tzid: str) -> tzinfo:
        if tzid not in self._known:
            self._known[tzid] = self._resolve(tzid)
        return self._known[tzid]

    def _resolve(self, tzid: str) -> tzinfo:
        # Some writers prefix the IANA name with a path of their own
        # (`/example.org/20260101/Europe/Paris`): the name is its tail.
        pieces = [p for p in tzid.strip().split("/") if p]
        candidates = [tzid.strip()]
        if tzid.strip().startswith("/"):
            candidates += ["/".join(pieces[at:]) for at in range(1, len(pieces))]
        for candidate in candidates:
            try:
                return ZoneInfo(candidate)
            except Exception:
                continue
        zone = self._from_the_file(tzid)
        if zone is None:
            raise _Unreadable(f"no zone for a TZID of {len(tzid)} characters")
        return zone

    def _from_the_file(self, tzid: str) -> Optional[tzinfo]:
        if self._of_the_file is None:
            try:
                self._of_the_file = dateutil_tz.tzical(io.StringIO("\n".join(self._zone_lines)))
            except Exception:
                self._of_the_file = False
        if not self._of_the_file:
            return None
        try:
            return self._of_the_file.get(tzid)
        except Exception:
            return None


# ── When ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _When:
    """A date or a date-time as the file writes it: the wall clock it
    names (midnight for a date) and the zone that clock is in (none for a
    date, which is the same day everywhere)."""

    naive: datetime
    zone: Optional[tzinfo]

    @property
    def is_date(self) -> bool:
        return self.zone is None

    def instant(self) -> datetime:
        return self.naive.replace(tzinfo=self.zone)


def _when(value: str, params: Dict[str, str], zones: _Zones) -> _When:
    value = value.strip()
    try:
        as_date = _DATE_RE.match(value)
        if as_date:
            year, month, day = (int(g) for g in as_date.groups())
            return _When(datetime(year, month, day), None)
        as_datetime = _DATETIME_RE.match(value)
        if not as_datetime:
            raise _Unreadable("a date that does not parse")
        year, month, day, hour, minute, second = (int(g) for g in as_datetime.groups()[:6])
        naive = datetime(year, month, day, hour, minute, second)
    except ValueError:
        raise _Unreadable("a date that does not exist") from None
    if as_datetime.group(7):
        return _When(naive, timezone.utc)
    if params.get("TZID"):
        return _When(naive, zones.get(params["TZID"]))
    return _When(naive, zones.local)


def _whens(line: _Line, zones: _Zones) -> List[_When]:
    """Every value of an `EXDATE` or `RDATE` line. A period (`start/end`)
    counts for its start."""
    found: List[_When] = []
    for value in line.value.split(","):
        value = value.strip().split("/")[0]
        if value:
            found.append(_when(value, line.params, zones))
    return found


def _duration(value: str) -> timedelta:
    match = _DURATION_RE.match(value.strip())
    if not match:
        raise _Unreadable("a duration that does not parse")
    sign, weeks, days, hours, minutes, seconds = match.groups()
    length = timedelta(
        weeks=int(weeks or 0), days=int(days or 0), hours=int(hours or 0),
        minutes=int(minutes or 0), seconds=int(seconds or 0),
    )
    return -length if sign == "-" else length


# ── Events ────────────────────────────────────────────────────────────


@dataclass
class _Event:
    uid: str
    start: _When
    duration: timedelta
    title: str
    place: str
    cancelled: bool
    rule: Optional[str] = None
    added: List[_When] = field(default_factory=list)
    excluded: List[_When] = field(default_factory=list)
    # Set on an event that replaces one occurrence of a recurring one.
    replaces: Optional[_When] = None


def _event(lines: List[_Line], zones: _Zones) -> _Event:
    """One event from its lines. Only the properties named here are
    looked at; the description, the attendees and the rest are never
    read."""
    first: Dict[str, _Line] = {}
    added: List[_When] = []
    excluded: List[_When] = []
    for line in lines:
        if line.name == "RDATE":
            added.extend(_whens(line, zones))
        elif line.name == "EXDATE":
            excluded.extend(_whens(line, zones))
        elif line.name in (
            "UID", "DTSTART", "DTEND", "DURATION", "SUMMARY", "LOCATION",
            "STATUS", "RRULE", "RECURRENCE-ID",
        ):
            first.setdefault(line.name, line)

    if "DTSTART" not in first:
        raise _Unreadable("no start")
    start = _when(first["DTSTART"].value, first["DTSTART"].params, zones)

    if "DTEND" in first:
        end = _when(first["DTEND"].value, first["DTEND"].params, zones)
        if end.is_date != start.is_date:
            length = _ONE_DAY if start.is_date else timedelta(0)
        elif start.is_date:
            length = end.naive - start.naive
        else:
            length = end.instant() - start.instant()
    elif "DURATION" in first:
        length = _duration(first["DURATION"].value)
    else:
        length = _ONE_DAY if start.is_date else timedelta(0)
    if length < timedelta(0):
        length = timedelta(0)
    if start.is_date and length < _ONE_DAY:
        length = _ONE_DAY

    replaces = None
    if "RECURRENCE-ID" in first:
        replaces = _when(first["RECURRENCE-ID"].value, first["RECURRENCE-ID"].params, zones)

    return _Event(
        uid=first["UID"].value.strip() if "UID" in first else "",
        start=start,
        duration=length,
        title=_unescaped(first["SUMMARY"].value) if "SUMMARY" in first else "",
        place=_unescaped(first["LOCATION"].value) if "LOCATION" in first else "",
        cancelled=(
            "STATUS" in first and first["STATUS"].value.strip().upper() == "CANCELLED"
        ),
        rule=first["RRULE"].value.strip() if "RRULE" in first else None,
        added=added,
        excluded=excluded,
        replaces=replaces,
    )


def _key(when: _When) -> Tuple[str, datetime]:
    """One occurrence's name, the same however its time is written: the
    day for a date, the instant in UTC for a date-time."""
    if when.is_date:
        return ("day", when.naive)
    return ("instant", when.instant().astimezone(timezone.utc).replace(tzinfo=None))


def _on_the_clock_of(event: _Event, when: _When) -> datetime:
    """`when` on the wall clock the event recurs on."""
    if event.start.is_date:
        return datetime.combine(when.naive.date(), time.min)
    if when.is_date:
        return datetime.combine(when.naive.date(), event.start.naive.time())
    return when.instant().astimezone(event.start.zone).replace(tzinfo=None)


def _until(value: str, event: _Event, zones: _Zones) -> datetime:
    """A rule's `UNTIL` on the event's own wall clock, whichever of its
    three forms it is written in. A bare date includes that whole day."""
    until = _when(value, {}, zones)
    if until.is_date:
        if event.start.is_date:
            return until.naive
        return datetime.combine(until.naive.date(), time(23, 59, 59))
    if event.start.is_date:
        return until.naive
    if value.strip()[-1:] in ("Z", "z"):
        return until.instant().astimezone(event.start.zone).replace(tzinfo=None)
    return until.naive


def _rule_starts(event: _Event, zones: _Zones, not_before: datetime, before: datetime) -> Iterator[datetime]:
    """The starts the rule gives in `[not_before, before)`, on the event's
    own wall clock, so an hour stays the same hour across a clock change."""
    parts: List[str] = []
    for part in event.rule.split(";"):
        key, _, value = part.partition("=")
        if not part.strip():
            continue
        if key.strip().upper() == "UNTIL":
            parts.append("UNTIL=" + _until(value, event, zones).strftime("%Y%m%dT%H%M%S"))
        else:
            parts.append(part.strip())
    try:
        rule = dateutil_rrule.rrulestr(
            "RRULE:" + ";".join(parts), dtstart=event.start.naive, cache=False,
        )
        walked = 0
        for start in rule:
            walked += 1
            if walked > MAX_OCCURRENCES:
                raise _Unreadable("a rule that does not reach the days asked about")
            if start >= before:
                return
            if start >= not_before:
                yield start
    except _Unreadable:
        raise
    except Exception:
        raise _Unreadable("a rule that cannot be read") from None


def _occurrences(
    event: _Event, zones: _Zones, opens: datetime, closes: datetime,
    replaced: Set[Tuple[str, Tuple[str, datetime]]],
) -> List[Occurrence]:
    """The event's occurrences that touch `[opens, closes)`."""
    local = zones.local
    if event.start.is_date:
        frame_opens = opens.replace(tzinfo=None)
        frame_closes = closes.replace(tzinfo=None)
    else:
        frame_opens = opens.astimezone(event.start.zone).replace(tzinfo=None)
        frame_closes = closes.astimezone(event.start.zone).replace(tzinfo=None)
    # A day of margin on each side: the exact test is made below, on
    # instants, and a clock change must not push a start out of the walk.
    not_before = frame_opens - event.duration - _ONE_DAY
    before = frame_closes + _ONE_DAY

    starts: Set[datetime] = {event.start.naive}
    if event.replaces is None:
        starts.update(_on_the_clock_of(event, when) for when in event.added)
        if event.rule:
            starts.update(_rule_starts(event, zones, not_before, before))
        excluded_days = {when.naive.date() for when in event.excluded if when.is_date}
        excluded = {
            _on_the_clock_of(event, when) for when in event.excluded if not when.is_date
        }
        starts = {
            start for start in starts
            if start not in excluded
            and start.date() not in excluded_days
            and (event.uid, _key(_When(start, event.start.zone))) not in replaced
        }

    found: List[Occurrence] = []
    for start in starts:
        if event.start.is_date:
            begins = start.replace(tzinfo=local)
            ends = (start + event.duration).replace(tzinfo=local)
        else:
            instant = start.replace(tzinfo=event.start.zone).astimezone(timezone.utc)
            begins = instant.astimezone(local)
            ends = (instant + event.duration).astimezone(local)
        if begins < closes and (ends > opens or begins >= opens):
            found.append(Occurrence(
                start=begins, end=ends, all_day=event.start.is_date,
                title=event.title, place=event.place,
            ))
    return found


def read_events(text: str, first: date, last: date, zone: tzinfo) -> Reading:
    """The events of `text` that touch the days `first` to `last`
    (included), with their times in `zone`, in the order of the days.

    Raises `NotACalendar` when the text holds no calendar: an empty list
    would read as a free day. Raises `TooLong` past the time limit.
    """
    give_up_at = _clock() + _READ_WALL_CLOCK_SEC
    event_lines, zone_lines, is_calendar = _components(_unfolded(text))
    if not is_calendar:
        raise NotACalendar("no VCALENDAR in the text")

    zones = _Zones(zone_lines, zone)
    unreadable = 0
    events: List[_Event] = []
    for lines in event_lines:
        try:
            events.append(_event(lines, zones))
        except _Unreadable:
            unreadable += 1

    replaced = {
        (event.uid, _key(event.replaces))
        for event in events if event.replaces is not None and event.uid
    }
    opens = datetime.combine(first, time.min, tzinfo=zone)
    closes = datetime.combine(last + _ONE_DAY, time.min, tzinfo=zone)

    found: List[Occurrence] = []
    for event in events:
        if event.cancelled:
            continue
        if _clock() >= give_up_at:
            raise TooLong("the reading passed its time limit")
        try:
            found.extend(_occurrences(event, zones, opens, closes, replaced))
        except _Unreadable:
            unreadable += 1
    found.sort(key=lambda o: (o.start, 0 if o.all_day else 1, o.title, o.place))
    return Reading(found, unreadable)

"""Reading a calendar's text into the events of the days asked about.

Every calendar here is invented: the names, the places and the events
belong to nobody.
"""

import dataclasses
import time
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from src.jarvis.tools.builtin.agenda import ics

PARIS = ZoneInfo("Europe/Paris")

THURSDAY = date(2026, 11, 26)


def _calendar(*events, before=()):
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//Invented//EN", *before]
    for event in events:
        lines += ["BEGIN:VEVENT", *event, "END:VEVENT"]
    lines.append("END:VCALENDAR")
    return "\r\n".join(lines) + "\r\n"


def _read(text, first=THURSDAY, last=None, zone=PARIS):
    return ics.read_events(text, first, last or first, zone)


def _only(text, first=THURSDAY, last=None):
    reading = _read(text, first, last)
    assert len(reading.occurrences) == 1, reading
    return reading.occurrences[0]


def _at(hour, minute=0, day=THURSDAY):
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=PARIS)


# ── What is a calendar ────────────────────────────────────────────────


@pytest.mark.parametrize("text", [
    "",
    "<html><body>Sign in to continue</body></html>",
    "BEGIN:VEVENT\r\nSUMMARY:Seul\r\nEND:VEVENT\r\n",
])
def test_a_text_that_is_no_calendar_is_said_so(text):
    with pytest.raises(ics.NotACalendar):
        _read(text)


def test_a_calendar_with_no_event_reads_as_empty():
    reading = _read(_calendar())

    assert reading.occurrences == []
    assert reading.unreadable == 0


# ── When ──────────────────────────────────────────────────────────────


def test_a_time_in_utc_is_shown_in_the_zone_asked_for():
    event = _only(_calendar([
        "UID:a@invented", "DTSTART:20261126T080000Z", "DTEND:20261126T093000Z",
        "SUMMARY:Point budget",
    ]))

    assert event.start == _at(9)
    assert event.end == _at(10, 30)
    assert event.start.utcoffset() == timedelta(hours=1)
    assert event.all_day is False


def test_a_time_in_a_named_zone_is_brought_to_the_zone_asked_for():
    event = _only(_calendar([
        "UID:a@invented", "DTSTART;TZID=America/New_York:20261126T090000",
        "DTEND;TZID=America/New_York:20261126T100000", "SUMMARY:Appel Quai Nord",
    ]))

    assert event.start == _at(15)
    assert event.end == _at(16)


def test_a_time_with_no_zone_is_read_in_the_zone_asked_for():
    event = _only(_calendar([
        "UID:a@invented", "DTSTART:20261126T140000", "DTEND:20261126T150000",
        "SUMMARY:Détartrage",
    ]))

    assert event.start == _at(14)


def test_a_zone_the_file_defines_itself_is_read_from_the_file():
    """Outlook names zones as Windows does, and the file says what each
    one means. This zone exists nowhere but in this calendar."""
    port_vellin = [
        "BEGIN:VTIMEZONE", "TZID:Heure de Port-Vellin",
        "BEGIN:STANDARD", "DTSTART:16010101T000000",
        "TZOFFSETFROM:+0500", "TZOFFSETTO:+0500", "END:STANDARD",
        "END:VTIMEZONE",
    ]
    event = _only(_calendar([
        "UID:a@invented", 'DTSTART;TZID="Heure de Port-Vellin":20261126T140000',
        'DTEND;TZID="Heure de Port-Vellin":20261126T150000', "SUMMARY:Marée haute",
    ], before=port_vellin))

    # 14:00 at UTC+5 is 09:00 UTC, 10:00 in Paris in winter.
    assert event.start == _at(10)
    assert event.end == _at(11)


def test_a_duration_gives_the_end():
    event = _only(_calendar([
        "UID:a@invented", "DTSTART:20261126T140000", "DURATION:PT1H30M",
        "SUMMARY:Répétition",
    ]))

    assert event.end == _at(15, 30)


def test_a_timed_event_with_no_end_ends_when_it_starts():
    event = _only(_calendar([
        "UID:a@invented", "DTSTART:20261126T140000", "SUMMARY:Départ du car",
    ]))

    assert event.end == event.start == _at(14)


def test_an_all_day_event_covers_its_days_and_not_the_end_day():
    text = _calendar([
        "UID:a@invented", "DTSTART;VALUE=DATE:20261126", "DTEND;VALUE=DATE:20261127",
        "SUMMARY:Salon du modélisme",
    ])
    event = _only(text)

    assert event.all_day is True
    assert event.start == _at(0)
    assert event.end == _at(0, day=date(2026, 11, 27))
    assert _read(text, date(2026, 11, 27)).occurrences == []


def test_an_all_day_event_with_no_end_lasts_its_day():
    event = _only(_calendar([
        "UID:a@invented", "DTSTART;VALUE=DATE:20261126", "SUMMARY:Jour de marché",
    ]))

    assert event.all_day is True
    assert event.end - event.start == timedelta(days=1)


def test_an_event_over_several_days_is_there_on_a_day_in_the_middle():
    text = _calendar([
        "UID:a@invented", "DTSTART;VALUE=DATE:20261125", "DTEND;VALUE=DATE:20261128",
        "SUMMARY:Stage de voile",
    ])

    assert _only(text).title == "Stage de voile"


def test_an_event_on_another_day_is_not_returned():
    text = _calendar([
        "UID:a@invented", "DTSTART:20261127T140000", "DTEND:20261127T150000",
        "SUMMARY:Le lendemain",
    ])

    assert _read(text).occurrences == []


def test_an_event_that_crosses_midnight_is_there_on_both_days():
    text = _calendar([
        "UID:a@invented", "DTSTART:20261125T220000", "DTEND:20261126T013000",
        "SUMMARY:Veillée",
    ])

    assert _only(text, date(2026, 11, 25)).title == "Veillée"
    assert _only(text, date(2026, 11, 26)).title == "Veillée"


def test_an_event_that_ends_at_midnight_is_not_on_the_next_day():
    text = _calendar([
        "UID:a@invented", "DTSTART:20261125T220000", "DTEND:20261126T000000",
        "SUMMARY:Veillée",
    ])

    assert _read(text, date(2026, 11, 26)).occurrences == []


def test_events_come_back_in_the_order_of_the_day():
    reading = _read(_calendar(
        ["UID:b@invented", "DTSTART:20261126T140000", "DTEND:20261126T150000", "SUMMARY:Second"],
        ["UID:a@invented", "DTSTART:20261126T090000", "DTEND:20261126T100000", "SUMMARY:Premier"],
    ))

    assert [e.title for e in reading.occurrences] == ["Premier", "Second"]


# ── What is kept of an event ──────────────────────────────────────────


def test_a_title_and_a_place_are_read_with_their_escapes_and_folds():
    event = _only(_calendar([
        "UID:a@invented", "DTSTART:20261126T140000", "DTEND:20261126T150000",
        "SUMMARY:Point budget\\, troisième trimes",
        " tre\\; suite",
        "LOCATION:Salle Cassiopée\\n2e étage",
    ]))

    assert event.title == "Point budget, troisième trimestre; suite"
    assert event.place == "Salle Cassiopée\n2e étage"


def test_nothing_but_the_time_the_title_and_the_place_is_kept():
    """No field for a description, an attendee or a link: what is not
    kept cannot be returned by a later change."""
    assert {f.name for f in dataclasses.fields(ics.Occurrence)} == {
        "start", "end", "all_day", "title", "place",
    }


def test_what_an_alarm_says_is_not_the_events_own():
    event = _only(_calendar([
        "UID:a@invented", "DTSTART:20261126T140000", "DTEND:20261126T150000",
        "SUMMARY:Détartrage",
        "BEGIN:VALARM", "ACTION:DISPLAY", "SUMMARY:Rappel de l'alarme",
        "TRIGGER:-PT15M", "END:VALARM",
    ]))

    assert event.title == "Détartrage"


def test_a_cancelled_event_is_left_out():
    text = _calendar([
        "UID:a@invented", "DTSTART:20261126T140000", "DTEND:20261126T150000",
        "SUMMARY:Annulé", "STATUS:CANCELLED",
    ])

    assert _read(text).occurrences == []
    assert _read(text).unreadable == 0


def test_property_names_are_read_whatever_their_case():
    event = _only(_calendar([
        "uid:a@invented", "dtstart;tzid=Europe/Paris:20261126T140000",
        "dtend;tzid=Europe/Paris:20261126T150000", "summary:Minuscules",
    ]))

    assert event.title == "Minuscules"
    assert event.start == _at(14)


# ── Recurring events ──────────────────────────────────────────────────


WEEKLY = [
    "UID:hebdo@invented", "DTSTART;TZID=Europe/Paris:20261001T090000",
    "DTEND;TZID=Europe/Paris:20261001T100000", "SUMMARY:Atelier vannerie",
    "RRULE:FREQ=WEEKLY;BYDAY=TH",
]


def test_a_weekly_event_keeps_its_hour_across_a_clock_change():
    """Clocks go back on 25 October 2026: 09:00 stays 09:00."""
    before = _only(_calendar(WEEKLY), date(2026, 10, 8))
    after = _only(_calendar(WEEKLY))

    assert before.start == datetime(2026, 10, 8, 9, tzinfo=PARIS)
    assert before.start.utcoffset() == timedelta(hours=2)
    assert after.start == _at(9)
    assert after.start.utcoffset() == timedelta(hours=1)
    assert after.end == _at(10)


def test_a_weekly_event_is_not_there_on_another_weekday():
    assert _read(_calendar(WEEKLY), date(2026, 11, 25)).occurrences == []


def test_a_recurring_event_is_not_there_before_it_starts():
    assert _read(_calendar(WEEKLY), date(2026, 9, 24)).occurrences == []


def test_a_count_ends_the_series():
    text = _calendar([
        "UID:a@invented", "DTSTART:20261124T090000", "DTEND:20261124T093000",
        "SUMMARY:Arrosage", "RRULE:FREQ=DAILY;COUNT=3",
    ])

    assert _only(text).title == "Arrosage"
    assert _read(text, date(2026, 11, 27)).occurrences == []


def test_an_until_written_in_utc_ends_the_series_in_the_events_zone():
    """23:59:59 in Paris on the 25th, written as UTC: the 26th is out."""
    text = _calendar([
        "UID:a@invented", "DTSTART;TZID=Europe/Paris:20261101T090000",
        "DTEND;TZID=Europe/Paris:20261101T093000", "SUMMARY:Arrosage",
        "RRULE:FREQ=DAILY;UNTIL=20261125T225959Z",
    ])

    assert _only(text, date(2026, 11, 25)).title == "Arrosage"
    assert _read(text).occurrences == []


def test_an_until_written_as_a_date_includes_that_day():
    text = _calendar([
        "UID:a@invented", "DTSTART;TZID=Europe/Paris:20261101T090000",
        "DTEND;TZID=Europe/Paris:20261101T093000", "SUMMARY:Arrosage",
        "RRULE:FREQ=DAILY;UNTIL=20261126",
    ])

    assert _only(text).start == _at(9)
    assert _read(text, date(2026, 11, 27)).occurrences == []


def test_a_monthly_event_on_the_first_tuesday():
    text = _calendar([
        "UID:a@invented", "DTSTART:20260106T183000", "DTEND:20260106T200000",
        "SUMMARY:Conseil de quartier", "RRULE:FREQ=MONTHLY;BYDAY=1TU",
    ])

    assert _only(text, date(2026, 12, 1)).start == datetime(2026, 12, 1, 18, 30, tzinfo=PARIS)
    assert _read(text, date(2026, 12, 8)).occurrences == []


def test_a_yearly_all_day_event_comes_back_each_year():
    text = _calendar([
        "UID:a@invented", "DTSTART;VALUE=DATE:20200314", "DTEND;VALUE=DATE:20200315",
        "SUMMARY:Fête du port", "RRULE:FREQ=YEARLY",
    ])
    event = _only(text, date(2027, 3, 14))

    assert event.all_day is True
    assert event.start == datetime(2027, 3, 14, tzinfo=PARIS)


def test_an_excluded_date_removes_its_occurrence():
    text = _calendar(WEEKLY + ["EXDATE;TZID=Europe/Paris:20261126T090000"])

    assert _read(text).occurrences == []
    assert _only(text, date(2026, 12, 3)).title == "Atelier vannerie"


def test_an_excluded_date_written_in_utc_removes_its_occurrence_too():
    text = _calendar(WEEKLY + ["EXDATE:20261126T080000Z"])

    assert _read(text).occurrences == []


def test_an_added_date_adds_an_occurrence():
    text = _calendar(WEEKLY + ["RDATE;TZID=Europe/Paris:20261128T100000"])
    event = _only(text, date(2026, 11, 28))

    assert event.start == datetime(2026, 11, 28, 10, tzinfo=PARIS)
    assert event.end == datetime(2026, 11, 28, 11, tzinfo=PARIS)


MOVED = [
    "UID:hebdo@invented", "RECURRENCE-ID;TZID=Europe/Paris:20261126T090000",
    "DTSTART;TZID=Europe/Paris:20261127T110000",
    "DTEND;TZID=Europe/Paris:20261127T120000", "SUMMARY:Atelier vannerie (déplacé)",
]


def test_a_moved_occurrence_is_where_it_was_moved_and_nowhere_else():
    text = _calendar(WEEKLY, MOVED)

    assert _read(text).occurrences == []
    moved = _only(text, date(2026, 11, 27))
    assert moved.title == "Atelier vannerie (déplacé)"
    assert moved.start == datetime(2026, 11, 27, 11, tzinfo=PARIS)


def test_a_moved_occurrence_is_found_whichever_comes_first_in_the_file():
    text = _calendar(MOVED, WEEKLY)

    assert _read(text).occurrences == []
    assert _only(text, date(2026, 11, 27)).title == "Atelier vannerie (déplacé)"


def test_a_cancelled_occurrence_is_gone_and_the_others_stay():
    text = _calendar(WEEKLY, [
        "UID:hebdo@invented", "RECURRENCE-ID;TZID=Europe/Paris:20261126T090000",
        "DTSTART;TZID=Europe/Paris:20261126T090000",
        "DTEND;TZID=Europe/Paris:20261126T100000", "SUMMARY:Atelier vannerie",
        "STATUS:CANCELLED",
    ])

    assert _read(text).occurrences == []
    assert _only(text, date(2026, 12, 3)).title == "Atelier vannerie"


# ── What cannot be read is counted ────────────────────────────────────


GOOD = ["UID:bon@invented", "DTSTART:20261126T140000", "DTEND:20261126T150000", "SUMMARY:Lisible"]


@pytest.mark.parametrize("broken", [
    pytest.param(["UID:x@invented", "SUMMARY:Sans début"], id="no start"),
    pytest.param(["UID:x@invented", "DTSTART:le 26 novembre", "SUMMARY:Date illisible"],
                 id="a date that does not parse"),
    pytest.param(["UID:x@invented", "DTSTART;TZID=Heure de Nulle-Part:20261126T140000",
                  "SUMMARY:Zone inconnue"], id="a zone nothing defines"),
    pytest.param(["UID:x@invented", "DTSTART:20261126T140000", "RRULE:FREQ=PARFOIS",
                  "SUMMARY:Règle illisible"], id="a rule that cannot be read"),
])
def test_an_event_that_cannot_be_placed_is_counted_and_the_rest_is_read(broken):
    reading = _read(_calendar(broken, GOOD))

    assert reading.unreadable == 1
    assert [e.title for e in reading.occurrences] == ["Lisible"]


def test_a_rule_that_never_reaches_the_days_asked_about_is_given_up():
    """An invitation is somebody else's text: a rule firing every second
    since 1970 must not hold the call."""
    text = _calendar(
        ["UID:x@invented", "DTSTART:19700101T000000Z", "RRULE:FREQ=SECONDLY",
         "SUMMARY:Sans fin"],
        GOOD,
    )
    began = time.monotonic()
    reading = _read(text)

    assert time.monotonic() - began < 10
    assert reading.unreadable == 1
    assert [e.title for e in reading.occurrences] == ["Lisible"]


def test_a_rule_that_never_falls_on_a_day_ends_and_yields_nothing():
    """The 30th of February: the library walks to the end of its calendar
    and comes back empty-handed."""
    text = _calendar(
        ["UID:x@invented", "DTSTART:20260101T090000",
         "RRULE:FREQ=DAILY;BYMONTH=2;BYMONTHDAY=30", "SUMMARY:Jamais"],
        GOOD,
    )
    reading = _read(text)

    assert [e.title for e in reading.occurrences] == ["Lisible"]
    assert reading.unreadable == 0


def test_a_reading_past_its_time_limit_is_given_up(monkeypatch):
    """Nothing can stop the library inside one rule, so the clock is
    looked at between two events: many slow rules cost the limit and one
    rule, not their sum."""
    looks = []

    def clock():
        looks.append(1)
        return 0.0 if len(looks) == 1 else ics._READ_WALL_CLOCK_SEC + len(looks)

    monkeypatch.setattr(ics, "_clock", clock)

    with pytest.raises(ics.TooLong):
        _read(_calendar(GOOD, GOOD, GOOD))
    assert len(looks) == 2


def test_a_daily_event_begun_decades_ago_is_still_read():
    text = _calendar([
        "UID:a@invented", "DTSTART:19900101T073000", "DTEND:19900101T074500",
        "SUMMARY:Relevé du baromètre", "RRULE:FREQ=DAILY",
    ])
    reading = _read(text)

    assert reading.unreadable == 0
    assert [e.start for e in reading.occurrences] == [_at(7, 30)]

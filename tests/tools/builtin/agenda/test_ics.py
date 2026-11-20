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
         "RRULE:FREQ=YEARLY;BYMONTH=2;BYMONTHDAY=30", "SUMMARY:Jamais"],
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


@pytest.mark.parametrize("every,since", [
    ("DAILY;INTERVAL=3", "20010307T081500"),
    ("WEEKLY;INTERVAL=2;BYDAY=TH,FR", "19991104T081500"),
    ("DAILY;BYDAY=MO,TH", "19950102T081500"),
])
def test_a_series_begun_long_ago_falls_on_the_same_days_as_it_always_did(every, since):
    """However the walk to the days asked about is shortened, each day of
    a month around them is the one the rule names, counted from its
    start."""
    from dateutil import rrule as dateutil_rrule

    start = datetime.strptime(since, "%Y%m%dT%H%M%S")
    text = _calendar([
        "UID:a@invented", f"DTSTART;TZID=Europe/Paris:{since}", "DURATION:PT30M",
        "SUMMARY:Arrosage des semis", f"RRULE:FREQ={every}",
    ])
    first, last = date(2026, 11, 2), date(2026, 12, 2)
    expected = [
        moment.replace(tzinfo=PARIS)
        for moment in dateutil_rrule.rrulestr(f"RRULE:FREQ={every}", dtstart=start)
        .between(datetime(2026, 11, 2), datetime(2026, 12, 3), inc=True)
    ]

    reading = ics.read_events(text, first, last, PARIS)

    assert [e.start for e in reading.occurrences] == expected
    assert expected


# ── What somebody else's calendar text can do ─────────────────────────
#
# An invitation lands in the calendar unasked, and a subscribed feed is
# written by whoever serves it. Neither may cost the rest of the agenda,
# nor hold the call.


@pytest.mark.parametrize("hostile", [
    pytest.param(["UID:x@invented", "DTSTART:99991231T230000Z", "DURATION:PT2H",
                  "SUMMARY:Fin des temps"], id="an end past the last year"),
    pytest.param(["UID:x@invented", "DTSTART:20261126T090000", "DURATION:P999999999D",
                  "SUMMARY:Interminable"], id="a duration past any date"),
    pytest.param(["UID:x@invented", "DTSTART:20261126T090000", "DURATION:P" + "9" * 5000 + "D",
                  "SUMMARY:Chiffres"], id="a duration of five thousand digits"),
    pytest.param(["UID:x@invented", "DTSTART;VALUE=DATE:00010101",
                  "DTEND;VALUE=DATE:99991231", "SUMMARY:Toujours"], id="all days there are"),
    pytest.param(["UID:x@invented", "DTSTART:00010101T000000", "RRULE:FREQ=YEARLY",
                  "SUMMARY:An un"], id="a series from the year one"),
    pytest.param(["UID:x@invented", "DTSTART:20261126T090000", "RDATE:99991231T235959Z",
                  "SUMMARY:Rendez-vous lointain"], id="an added date at the end of time"),
    pytest.param(["UID:x@invented", "DTSTART:20261126T090000",
                  "RECURRENCE-ID:00010101T000000", "SUMMARY:Déplacé"], id="a moved occurrence of the year one"),
])
def test_an_event_past_the_ends_of_the_calendar_costs_only_itself(hostile):
    reading = _read(_calendar(hostile, GOOD))

    assert "Lisible" in [e.title for e in reading.occurrences]


def _zone_with_rule(tzid, rule):
    return [
        "BEGIN:VTIMEZONE", f"TZID:{tzid}",
        "BEGIN:STANDARD", "DTSTART:20260301T000000", f"RRULE:{rule}",
        "TZOFFSETFROM:+0200", "TZOFFSETTO:+0100", "END:STANDARD",
        "BEGIN:DAYLIGHT", "DTSTART:20260301T000001", f"RRULE:{rule}",
        "TZOFFSETFROM:+0100", "TZOFFSETTO:+0200", "END:DAYLIGHT",
        "END:VTIMEZONE",
    ]


def test_a_zone_that_changes_its_clock_every_second_is_not_read():
    """A zone of the file's own is a recurrence too, walked from its
    start on every conversion: one that changes every second would hold
    the call on its first event. No real zone changes more than a few
    times a year."""
    text = _calendar(
        ["UID:x@invented", 'DTSTART;TZID="Heure folle":20261126T090000',
         'DTEND;TZID="Heure folle":20261126T100000', "SUMMARY:Piège"],
        GOOD,
        before=_zone_with_rule("Heure folle", "FREQ=SECONDLY"),
    )
    began = time.monotonic()
    reading = _read(text)

    assert time.monotonic() - began < 2
    assert reading.unreadable == 1
    assert [e.title for e in reading.occurrences] == ["Lisible"]


# How Outlook writes the zone of Paris: rules from the year 1601.
W_EUROPE = [
    "BEGIN:VTIMEZONE", "TZID:W. Europe Standard Time",
    "BEGIN:STANDARD", "DTSTART:16010101T030000", "TZOFFSETFROM:+0200", "TZOFFSETTO:+0100",
    "RRULE:FREQ=YEARLY;INTERVAL=1;BYDAY=-1SU;BYMONTH=10", "END:STANDARD",
    "BEGIN:DAYLIGHT", "DTSTART:16010101T020000", "TZOFFSETFROM:+0100", "TZOFFSETTO:+0200",
    "RRULE:FREQ=YEARLY;INTERVAL=1;BYDAY=-1SU;BYMONTH=3", "END:DAYLIGHT",
    "END:VTIMEZONE",
]


def test_a_zone_written_the_way_outlook_writes_it_is_read():
    summer = _only(_calendar([
        "UID:a@invented", "DTSTART;TZID=W. Europe Standard Time:20260702T090000",
        "DTEND;TZID=W. Europe Standard Time:20260702T100000", "SUMMARY:Point d'été",
    ], before=W_EUROPE), date(2026, 7, 2))
    winter = _only(_calendar([
        "UID:b@invented", "DTSTART;TZID=W. Europe Standard Time:20261126T090000",
        "DTEND;TZID=W. Europe Standard Time:20261126T100000", "SUMMARY:Point d'hiver",
    ], before=W_EUROPE))

    assert summer.start == datetime(2026, 7, 2, 9, tzinfo=PARIS)
    assert winter.start == _at(9)


def test_a_large_calendar_in_a_zone_of_its_own_is_read_in_time():
    """Years of single events, each in the file's own zone: those far
    from the days asked about cost no conversion."""
    others = [
        [f"UID:{n}@invented", "DTSTART;TZID=W. Europe Standard Time:20190314T090000",
         "DTEND;TZID=W. Europe Standard Time:20190314T100000", "SUMMARY:Ancien"]
        for n in range(30_000)
    ]
    text = _calendar(*others, [
        "UID:z@invented", "DTSTART;TZID=W. Europe Standard Time:20261126T090000",
        "DTEND;TZID=W. Europe Standard Time:20261126T100000", "SUMMARY:Aujourd'hui",
    ], before=W_EUROPE)
    began = time.monotonic()
    reading = _read(text)

    assert time.monotonic() - began < 3
    assert [e.title for e in reading.occurrences] == ["Aujourd'hui"]


def test_a_very_long_zone_name_is_not_tried_piece_by_piece():
    tzid = "/x" * 8000
    text = _calendar(
        ["UID:x@invented", f"DTSTART;TZID={tzid}:20261126T090000", "SUMMARY:Chemin"],
        GOOD,
    )
    began = time.monotonic()
    reading = _read(text)

    assert time.monotonic() - began < 2
    assert reading.unreadable == 1
    assert [e.title for e in reading.occurrences] == ["Lisible"]


def test_a_zone_name_with_a_path_in_front_is_read_by_its_tail():
    event = _only(_calendar([
        "UID:a@invented", "DTSTART;TZID=/invented.example/20260101/America/New_York:20261126T090000",
        "SUMMARY:Appel du matin",
    ]))

    assert event.start == _at(15)


def test_a_very_long_folded_line_is_put_back_together_in_time():
    """An export can carry an attachment inline, folded every 75
    characters over megabytes."""
    folded = "DESCRIPTION:" + "\r\n ".join(["x" * 74] * 60_000)
    text = _calendar(["UID:a@invented", "DTSTART:20261126T090000", folded,
                      "SUMMARY:Avec pièce jointe"])
    began = time.monotonic()
    reading = _read(text)

    assert time.monotonic() - began < 2
    assert [e.title for e in reading.occurrences] == ["Avec pièce jointe"]


def test_components_opened_and_never_closed_do_not_hold_the_reading():
    text = _calendar(
        GOOD,
        before=[f"BEGIN:X-RIEN-{n % 7}" for n in range(80_000)],
    )
    began = time.monotonic()
    _read(text)

    assert time.monotonic() - began < 2


def _read_within(seconds, text, first=THURSDAY):
    """The reading, run where a hang cannot hold the test run: past
    `seconds` the test fails and the thread is left to the process."""
    import threading

    outcome = {}

    def read():
        try:
            outcome["reading"] = _read(text, first)
        except BaseException as e:
            outcome["failure"] = e

    worker = threading.Thread(target=read, daemon=True)
    worker.start()
    worker.join(seconds)
    assert not worker.is_alive(), f"the reading took more than {seconds} s"
    if "failure" in outcome:
        raise outcome["failure"]
    return outcome["reading"]


@pytest.mark.parametrize("rule", [
    "FREQ=DAILY;INTERVAL=0;BYMONTH=2",
    "FREQ=WEEKLY;INTERVAL=0;BYMONTH=2",
    "FREQ=MONTHLY;INTERVAL=0;BYMONTH=2",
    "FREQ=YEARLY;INTERVAL=0;BYMONTH=2;BYMONTHDAY=30",
    "FREQ=DAILY;INTERVAL=-1",
])
def test_a_rule_that_does_not_move_forward_is_counted_and_holds_nothing(rule):
    text = _calendar(
        ["UID:x@invented", "DTSTART:20261001T090000", f"RRULE:{rule}", "SUMMARY:Immobile"],
        GOOD,
    )
    reading = _read_within(3, text)

    assert reading.unreadable == 1
    assert [e.title for e in reading.occurrences] == ["Lisible"]


@pytest.mark.parametrize("rule", [
    "FREQ=MINUTELY;BYMONTH=2;BYMONTHDAY=30",
    "FREQ=SECONDLY;BYMONTH=4,6;BYMONTHDAY=31",
    "FREQ=DAILY;BYMONTH=2;BYMONTHDAY=-30",
])
def test_a_rule_on_a_day_no_month_has_costs_nothing(rule):
    """The 30th of February every minute: the library would walk to the
    year 9999 to find nothing. The start still counts."""
    text = _calendar(
        ["UID:x@invented", "DTSTART:20261126T090000", f"RRULE:{rule}", "SUMMARY:Une fois"],
        GOOD,
    )
    began = time.monotonic()
    reading = _read(text)

    assert time.monotonic() - began < 1
    assert reading.unreadable == 0
    assert sorted(e.title for e in reading.occurrences) == ["Lisible", "Une fois"]


def test_a_zone_whose_rule_does_not_move_forward_is_not_read():
    text = _calendar(
        ["UID:x@invented", 'DTSTART;TZID="Heure figée":20261126T090000', "SUMMARY:Piège"],
        GOOD,
        before=_zone_with_rule("Heure figée", "FREQ=YEARLY;INTERVAL=0;BYMONTH=2;BYMONTHDAY=30"),
    )
    reading = _read_within(3, text)

    assert reading.unreadable == 1
    assert [e.title for e in reading.occurrences] == ["Lisible"]


def test_a_zone_of_many_rules_from_the_year_one_is_not_read():
    """Each rule is walked from its own start on every conversion: a
    hundred of them from the year one cost half a minute on one event."""
    parts = []
    for n in range(100):
        parts += [
            "BEGIN:STANDARD", "DTSTART:00010101T000000",
            "RRULE:FREQ=YEARLY;BYMONTHDAY=1,2,3,4,5,6,7",
            "TZOFFSETFROM:+0100", "TZOFFSETTO:+0100", "END:STANDARD",
        ]
    zone = ["BEGIN:VTIMEZONE", "TZID:Heure antique", *parts, "END:VTIMEZONE"]
    text = _calendar(
        ["UID:x@invented", "DTSTART;TZID=Heure antique:20261126T090000", "SUMMARY:Piège"],
        GOOD,
        before=zone,
    )
    reading = _read_within(3, text)

    assert reading.unreadable == 1
    assert [e.title for e in reading.occurrences] == ["Lisible"]


def test_a_zone_never_closed_does_not_swallow_the_events_after_it():
    text = _calendar(
        GOOD,
        before=["BEGIN:VTIMEZONE", "TZID:Heure ouverte", "BEGIN:STANDARD",
                "DTSTART:19700101T000000", "TZOFFSETFROM:+0100", "TZOFFSETTO:+0100"],
    )

    assert [e.title for e in _read(text).occurrences] == ["Lisible"]


def test_an_event_cut_off_by_the_end_of_the_calendar_is_still_read():
    text = "\r\n".join([
        "BEGIN:VCALENDAR", "VERSION:2.0", "BEGIN:VEVENT", *GOOD, "END:VCALENDAR",
    ]) + "\r\n"

    assert [e.title for e in _read(text).occurrences] == ["Lisible"]


def test_an_event_whose_alarm_is_never_closed_is_still_read():
    reading = _read(_calendar([
        "UID:a@invented", "DTSTART:20261126T090000", "DTEND:20261126T100000",
        "SUMMARY:Point budget", "BEGIN:VALARM", "ACTION:DISPLAY",
        "SUMMARY:Le texte de l'alarme",
    ]))

    assert [e.title for e in reading.occurrences] == ["Point budget"]
    assert reading.unreadable == 0

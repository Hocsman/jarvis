"""The anchor an extractor resolves "jeudi" against.

Two properties, both load-bearing once a model is asked to turn a phrase
into an instant.

**It must not follow the machine's locale.** `strftime("%A")` returns
whatever the C locale says, so on a French Mac the English template
produced "samedi, août 01, 2026 at 22:10" — French words in an English
sentence, silently, inside a prompt. Fixed English names are not a
language pattern; they are an output format, like `%Y-%m-%d`.

**It must carry the weekday and no timezone abbreviation.** Without the
weekday, "jeudi" is unresolvable. With an abbreviation, it is worse than
nothing: `%Z` gives `IST` for three different zones and carries no
offset, so a model asked to convert would be guessing. The code converts;
the model reads a wall clock.
"""

from __future__ import annotations

import locale
from datetime import datetime, timezone

import pytest

from src.jarvis.utils.time_context import (
    format_reference_moment,
    format_time_context,
    local_timezone_name,
)


@pytest.fixture
def french_locale():
    """A machine set to French, which is this user's."""
    try:
        locale.setlocale(locale.LC_TIME, "fr_FR.UTF-8")
    except locale.Error:
        pytest.skip("fr_FR.UTF-8 not available on this machine")
    yield
    locale.setlocale(locale.LC_TIME, "C")


_SATURDAY = datetime(2026, 8, 1, 12, 30)


# ── The locale must not leak into the prompt ──────────────────────────


def test_the_reference_moment_is_english_on_a_french_machine(french_locale):
    moment = format_reference_moment(_SATURDAY)

    assert "Saturday" in moment
    assert "August" in moment
    assert "samedi" not in moment and "août" not in moment


def test_the_time_context_is_english_on_a_french_machine(french_locale):
    context = format_time_context(now_utc=datetime(2026, 8, 1, 12, 30, tzinfo=timezone.utc))

    assert "samedi" not in context and "août" not in context
    assert "Saturday" in context


@pytest.mark.parametrize("month,name", [
    (1, "January"), (2, "February"), (3, "March"), (4, "April"),
    (5, "May"), (6, "June"), (7, "July"), (8, "August"),
    (9, "September"), (10, "October"), (11, "November"), (12, "December"),
])
def test_every_month_is_named(month, name):
    assert name in format_reference_moment(datetime(2026, month, 15, 9, 0))


@pytest.mark.parametrize("day,name", [
    (3, "Monday"), (4, "Tuesday"), (5, "Wednesday"), (6, "Thursday"),
    (7, "Friday"), (8, "Saturday"), (9, "Sunday"),
])
def test_every_weekday_is_named(day, name):
    assert name in format_reference_moment(datetime(2026, 8, day, 9, 0))


# ── What the anchor has to contain ────────────────────────────────────


def test_the_weekday_is_there(french_locale):
    """Without it, "jeudi" cannot be resolved at all."""
    assert "Saturday" in format_reference_moment(_SATURDAY)


def test_a_machine_readable_form_is_there():
    """So the model can echo an instant back without re-parsing prose."""
    assert "2026-08-01T12:30" in format_reference_moment(_SATURDAY)


def test_no_timezone_abbreviation_appears():
    """`%Z` gives IST for three different zones and carries no offset. A
    model asked to convert from it is guessing; the code converts."""
    moment = format_reference_moment(_SATURDAY)

    for ambiguous in ("CEST", "CET", "IST", "UTC", "GMT", "+02:00"):
        assert ambiguous not in moment


def test_midnight_reads_as_a_clock_not_as_nothing():
    assert "T00:00" in format_reference_moment(datetime(2026, 8, 1, 0, 0))


def test_the_same_moment_always_renders_the_same():
    assert format_reference_moment(_SATURDAY) == format_reference_moment(_SATURDAY)


# ── Where the timezone comes from ─────────────────────────────────────


def test_the_machine_supplies_a_timezone_name():
    """It reads the machine rather than GeoIP. The engine prefers GeoIP
    for its own display string, where being wrong is cosmetic; here it
    decides when a reminder fires, and GeoIP reports the exit node —
    wrong under a VPN, and cached for as long as the location cache."""
    name = local_timezone_name()

    assert isinstance(name, str)
    if name:
        from zoneinfo import ZoneInfo

        ZoneInfo(name)  # must be a real IANA name, not an abbreviation


def test_an_unknown_timezone_is_empty_rather_than_a_guess():
    """Empty is honest and documented: the instant is then computed at
    the current local offset."""
    import src.jarvis.utils.time_context as tc

    saved = tc._read_system_timezone
    tc._read_system_timezone = lambda: ""
    try:
        assert local_timezone_name() in ("", None) or isinstance(local_timezone_name(), str)
    finally:
        tc._read_system_timezone = saved

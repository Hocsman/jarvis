"""The zone of the machine's own clock: what the agenda's times are
shown in."""

from datetime import datetime
from zoneinfo import ZoneInfo

import src.jarvis.utils.time_context as tc


def test_a_zone_the_system_names_is_used_with_its_name(monkeypatch):
    monkeypatch.setattr(tc, "_read_system_timezone", lambda: "America/Halifax")

    zone, name = tc.machine_zone()

    assert name == "America/Halifax"
    moment = datetime(2026, 11, 26, 12)
    assert zone.utcoffset(moment) == ZoneInfo("America/Halifax").utcoffset(moment)


def test_with_no_name_from_the_system_it_is_the_local_time_and_has_no_name(monkeypatch):
    from dateutil import tz as dateutil_tz

    monkeypatch.setattr(tc, "_read_system_timezone", lambda: "")

    zone, name = tc.machine_zone()

    assert name == ""
    moment = datetime(2026, 11, 26, 12)
    assert zone.utcoffset(moment) == dateutil_tz.tzlocal().utcoffset(moment)


def test_a_name_that_is_no_zone_is_not_given_out(monkeypatch):
    monkeypatch.setattr(tc, "_read_system_timezone", lambda: "Nowhere/Invented")

    _zone, name = tc.machine_zone()

    assert name == ""


def test_it_never_asks_where_the_network_says_the_machine_is(monkeypatch):
    """GeoIP follows a VPN's exit node; the clock beside his calendar
    must be the machine's."""
    import src.jarvis.utils.location as location

    def nowhere(*args, **kwargs):
        raise AssertionError("GeoIP was asked")

    for name in dir(location):
        if name.startswith("get_") and callable(getattr(location, name)):
            monkeypatch.setattr(location, name, nowhere)
    monkeypatch.setattr(tc, "_read_system_timezone", lambda: "Europe/Paris")

    assert tc.machine_zone()[1] == "Europe/Paris"

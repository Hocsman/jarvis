"""Where the calendar text is read from: a file, or an HTTPS address.

No request leaves the machine here: `requests.get` is replaced in every
test that names an address, and the addresses are invented.
"""

import itertools

import pytest
import requests

from src.jarvis.tools.builtin.agenda import source
from src.jarvis.tools.builtin.agenda.source import AgendaUnreadable

CALENDAR = "BEGIN:VCALENDAR\r\nVERSION:2.0\r\nEND:VCALENDAR\r\n"

SECRET = "jeton-prive-4f9a2c"
ADDRESS = f"https://agenda.invented.example/ical/{SECRET}/basic.ics"


class _Answer:
    def __init__(self, status=200, body=b"", location=None, chunks=None):
        self.status_code = status
        self.headers = {"Location": location} if location else {}
        self._chunks = chunks if chunks is not None else [body]
        self.closed = False

    def iter_content(self, chunk_size=None):
        yield from self._chunks

    def close(self):
        self.closed = True


@pytest.fixture
def web(monkeypatch):
    """`requests.get` answering from a script, and what it was asked."""
    asked = []
    script = {}

    def get(url, **kwargs):
        asked.append((url, kwargs))
        answer = script[url]
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(source.requests, "get", get)
    return asked, script


@pytest.fixture
def logged(monkeypatch):
    lines = []
    monkeypatch.setattr(source, "debug_log", lambda message, *a, **k: lines.append(message))
    return lines


def _no_request(monkeypatch):
    def get(url, **kwargs):
        raise AssertionError("a request was made")

    monkeypatch.setattr(source.requests, "get", get)


# ── Nothing configured ────────────────────────────────────────────────


@pytest.mark.parametrize("setting", ["", "   ", None])
def test_with_no_source_it_says_none_is_configured(setting):
    with pytest.raises(AgendaUnreadable, match="agenda_source"):
        source.read_source(setting)


def test_the_kind_of_a_source_is_told_without_reading_it():
    assert source.kind_of("") == ""
    assert source.kind_of("C:/agendas/perso.ics") == "file"
    assert source.kind_of(ADDRESS) == "address"
    assert source.kind_of("webcal://agenda.invented.example/x.ics") == "address"


# ── A file ────────────────────────────────────────────────────────────


def test_a_file_is_read(tmp_path):
    path = tmp_path / "agenda.ics"
    path.write_bytes(CALENDAR.encode("utf-8"))

    assert source.read_source(str(path)) == CALENDAR


def test_a_byte_order_mark_is_not_part_of_the_calendar(tmp_path):
    path = tmp_path / "agenda.ics"
    path.write_bytes(b"\xef\xbb\xbf" + CALENDAR.encode("utf-8"))

    assert source.read_source(str(path)) == CALENDAR


def test_bytes_that_are_not_utf8_do_not_stop_the_reading(tmp_path):
    path = tmp_path / "agenda.ics"
    path.write_bytes(CALENDAR.encode("utf-8") + "SUMMARY:Détartrage\r\n".encode("cp1252"))

    text = source.read_source(str(path))

    assert text.startswith(CALENDAR)
    assert "SUMMARY:D" in text


def test_a_missing_file_says_so_and_does_not_name_the_path(tmp_path):
    path = tmp_path / "dossier-de-quelqu-un" / "agenda.ics"

    with pytest.raises(AgendaUnreadable) as failure:
        source.read_source(str(path))

    assert "not found" in str(failure.value)
    assert "dossier-de-quelqu-un" not in str(failure.value)


def test_a_folder_is_not_a_calendar_file(tmp_path):
    with pytest.raises(AgendaUnreadable):
        source.read_source(str(tmp_path))


def test_a_file_past_the_limit_is_not_read(tmp_path, monkeypatch):
    monkeypatch.setattr(source, "MAX_SOURCE_BYTES", 10)
    path = tmp_path / "agenda.ics"
    path.write_bytes(CALENDAR.encode("utf-8"))

    with pytest.raises(AgendaUnreadable, match="too large"):
        source.read_source(str(path))


# ── An address ────────────────────────────────────────────────────────


def test_an_https_address_is_fetched(web):
    asked, script = web
    script[ADDRESS] = _Answer(body=CALENDAR.encode("utf-8"))

    assert source.read_source(ADDRESS) == CALENDAR
    assert [url for url, _ in asked] == [ADDRESS]


def test_the_request_carries_no_cookie_and_follows_no_redirect_by_itself(web):
    asked, script = web
    script[ADDRESS] = _Answer(body=CALENDAR.encode("utf-8"))

    source.read_source(ADDRESS)
    _url, kwargs = asked[0]

    assert kwargs["allow_redirects"] is False
    assert kwargs["stream"] is True
    assert kwargs["timeout"]
    assert "cookies" not in kwargs
    assert "Cookie" not in kwargs.get("headers", {})


def test_a_webcal_address_is_fetched_over_https(web):
    asked, script = web
    script["https://agenda.invented.example/x.ics"] = _Answer(body=CALENDAR.encode("utf-8"))

    assert source.read_source("webcal://agenda.invented.example/x.ics") == CALENDAR


@pytest.mark.parametrize("address", [
    "http://agenda.invented.example/x.ics",
    "ftp://agenda.invented.example/x.ics",
    "file:///C:/agendas/perso.ics",
])
def test_an_address_that_is_not_https_is_refused_before_any_request(address, monkeypatch):
    _no_request(monkeypatch)

    with pytest.raises(AgendaUnreadable, match="https://"):
        source.read_source(address)


def test_a_redirect_to_https_is_followed(web):
    asked, script = web
    elsewhere = "https://autre.invented.example/final.ics"
    script[ADDRESS] = _Answer(status=302, location=elsewhere)
    script[elsewhere] = _Answer(body=CALENDAR.encode("utf-8"))

    assert source.read_source(ADDRESS) == CALENDAR
    assert [url for url, _ in asked] == [ADDRESS, elsewhere]


def test_a_relative_redirect_is_followed_from_where_it_was_given(web):
    asked, script = web
    script[ADDRESS] = _Answer(status=301, location="/ailleurs.ics")
    script["https://agenda.invented.example/ailleurs.ics"] = _Answer(body=CALENDAR.encode("utf-8"))

    assert source.read_source(ADDRESS) == CALENDAR


def test_a_redirect_to_plain_http_is_not_followed(web):
    asked, script = web
    script[ADDRESS] = _Answer(status=302, location="http://agenda.invented.example/x.ics")

    with pytest.raises(AgendaUnreadable, match="https://"):
        source.read_source(ADDRESS)
    assert len(asked) == 1


def test_redirects_are_followed_three_times_and_no_more(web):
    asked, script = web
    hops = [ADDRESS] + [f"https://agenda.invented.example/hop{n}" for n in range(1, 6)]
    for here, there in zip(hops, hops[1:]):
        script[here] = _Answer(status=302, location=there)

    with pytest.raises(AgendaUnreadable, match="redirect"):
        source.read_source(ADDRESS)
    assert len(asked) == source._MAX_REDIRECTS + 1


def test_a_redirect_that_names_no_address_is_a_failure(web):
    _asked, script = web
    script[ADDRESS] = _Answer(status=302)

    with pytest.raises(AgendaUnreadable):
        source.read_source(ADDRESS)


def test_a_status_that_is_not_200_is_said_by_its_number(web):
    _asked, script = web
    script[ADDRESS] = _Answer(status=404, body=b"<html>Not here</html>")

    with pytest.raises(AgendaUnreadable, match="404"):
        source.read_source(ADDRESS)


def test_a_body_past_the_limit_is_not_read(web, monkeypatch):
    monkeypatch.setattr(source, "MAX_SOURCE_BYTES", 10)
    _asked, script = web
    script[ADDRESS] = _Answer(chunks=[b"x" * 8, b"y" * 8])

    with pytest.raises(AgendaUnreadable, match="too large"):
        source.read_source(ADDRESS)


def test_a_fetch_past_its_clock_is_given_up(web, monkeypatch):
    # Each look at the clock finds it ten seconds on.
    ticks = itertools.count(0, 10)
    monkeypatch.setattr(source, "_clock", lambda: float(next(ticks)))
    _asked, script = web
    script[ADDRESS] = _Answer(chunks=[b"BEGIN:", b"VCALENDAR", b"\r\n"])

    with pytest.raises(AgendaUnreadable, match="too long"):
        source.read_source(ADDRESS)


def test_every_answer_is_closed(web):
    _asked, script = web
    elsewhere = "https://autre.invented.example/final.ics"
    first, last = _Answer(status=302, location=elsewhere), _Answer(body=CALENDAR.encode("utf-8"))
    script[ADDRESS], script[elsewhere] = first, last

    source.read_source(ADDRESS)

    assert first.closed and last.closed


# ── The address is a secret ───────────────────────────────────────────


def _failures(script):
    """Each way a fetch can fail, as (what the address answers, what the
    redirect target answers)."""
    quoting = requests.ConnectionError(f"HTTPSConnectionPool: Max retries exceeded with url: {ADDRESS}")
    return {
        "a connection that fails": quoting,
        "a timeout": requests.Timeout(f"Read timed out. ({ADDRESS})"),
        "a status": _Answer(status=403, body=ADDRESS.encode("utf-8")),
        "a redirect to http": _Answer(status=302, location=f"http://agenda.invented.example/{SECRET}"),
        "an error nobody expected": RuntimeError(f"while reading {ADDRESS}"),
    }


@pytest.mark.parametrize("kind", [
    "a connection that fails", "a timeout", "a status", "a redirect to http",
    "an error nobody expected",
])
def test_no_failure_and_no_log_line_ever_holds_the_address(kind, web, logged):
    _asked, script = web
    script[ADDRESS] = _failures(script)[kind]

    with pytest.raises(AgendaUnreadable) as failure:
        source.read_source(ADDRESS)

    assert SECRET not in str(failure.value)
    assert "invented.example" not in str(failure.value)
    assert not any(SECRET in line for line in logged), logged


def test_a_failed_connection_is_named_by_its_kind(web):
    _asked, script = web
    script[ADDRESS] = requests.ConnectionError(f"Max retries exceeded with url: {ADDRESS}")

    with pytest.raises(AgendaUnreadable, match="ConnectionError"):
        source.read_source(ADDRESS)


def test_a_successful_fetch_logs_no_address_either(web, logged):
    _asked, script = web
    script[ADDRESS] = _Answer(body=CALENDAR.encode("utf-8"))

    source.read_source(ADDRESS)

    assert logged, "the reading left no debug line"
    assert not any(SECRET in line or "invented.example" in line for line in logged), logged

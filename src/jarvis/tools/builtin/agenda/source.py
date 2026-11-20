"""Where the calendar text is read from: the file or the HTTPS address the
user named in `agenda_source`.

Reading only: one `open` or one `GET`, nothing written and nothing sent
but the request. The address is a secret (whoever holds it reads the
calendar), so it appears in no failure and no debug line: a failed fetch
is told by its status code or by the kind of failure, never by an
exception's message, which quotes the address. See `read_agenda.spec.md`.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from urllib.parse import urljoin

import requests

from ....debug import debug_log

# A calendar of several years of events is a few megabytes.
MAX_SOURCE_BYTES = 10 * 1024 * 1024

_MAX_REDIRECTS = 3
# For the whole fetch, redirects and body included: he is listening to
# silence meanwhile. `timeout` on a request bounds the connection and each
# read, never the fetch: a server sending a byte now and then keeps every
# read alive. So the fetch runs on a thread of its own and is waited for
# this long, no longer.
_FETCH_WALL_CLOCK_SEC = 15.0
_CONNECT_TIMEOUT_SEC = 5.0
_READ_TIMEOUT_SEC = 10.0
_REDIRECTS = (301, 302, 303, 307, 308)

# Nothing that names the user, the project or the machine.
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ical-reader)",
    "Accept": "text/calendar, text/plain;q=0.9, */*;q=0.1",
}


NOT_CONFIGURED = "No agenda is configured: the `agenda_source` setting is empty."

# The thread of the last fetch, which may outlive a fetch given up.
_last_fetch: "threading.Thread | None" = None
_fetch_lock = threading.Lock()


class AgendaUnreadable(Exception):
    """The source gave no calendar text. The message says why, in words
    for the model, and never holds the address or the path."""


def _clock() -> float:
    return time.monotonic()


def kind_of(setting) -> str:
    """`file`, `address`, or the empty string when nothing is configured.
    Told from the setting's shape alone, without reading anything."""
    named = str(setting or "").strip()
    if not named:
        return ""
    return "address" if "://" in named else "file"


def read_source(setting) -> str:
    """The calendar text the setting points at.

    Raises `AgendaUnreadable` for anything short of that: an empty result
    in its place would read as a free day.
    """
    named = str(setting or "").strip()
    kind = kind_of(named)
    if not kind:
        raise AgendaUnreadable(NOT_CONFIGURED)
    raw = _read_file(named) if kind == "file" else _fetch(named)
    # UTF-8 is what the format asks for. A byte that is not UTF-8 costs
    # one character of one title, not the whole reading.
    return raw.decode("utf-8-sig", errors="replace")


def _megabytes() -> str:
    return f"{MAX_SOURCE_BYTES / (1024 * 1024):g} MB"


def _read_file(named: str) -> bytes:
    path = Path(os.path.expanduser(named))
    try:
        if not path.is_file():
            debug_log("agenda: the file named in agenda_source is not there", "agenda")
            raise AgendaUnreadable(
                "The agenda file named in the settings (`agenda_source`) was not "
                "found, or is not a file."
            )
        if path.stat().st_size > MAX_SOURCE_BYTES:
            debug_log("agenda: the file is past the size limit", "agenda")
            raise AgendaUnreadable(
                f"The agenda file is too large to read (more than {_megabytes()})."
            )
        with open(path, "rb") as handle:
            raw = handle.read(MAX_SOURCE_BYTES + 1)
    except AgendaUnreadable:
        raise
    except Exception as e:
        debug_log(f"agenda: the file could not be read: {type(e).__name__}", "agenda")
        raise AgendaUnreadable(
            f"The agenda file could not be opened ({type(e).__name__})."
        ) from None
    if len(raw) > MAX_SOURCE_BYTES:
        raise AgendaUnreadable(
            f"The agenda file is too large to read (more than {_megabytes()})."
        )
    debug_log(f"agenda: file read, bytes={len(raw)}", "agenda")
    return raw


def _fetch(address: str) -> bytes:
    """The body the address answers with, over HTTPS and nothing else,
    within `_FETCH_WALL_CLOCK_SEC`.

    The thread left behind by a fetch given up ends at its next read of
    the body: it looks at the same clock between two reads. While the
    server is still trickling its headers it looks at nothing, and lives
    as long as the server keeps it alive; so a new fetch is not started
    beside it, and the reading says why.
    """
    global _last_fetch

    outcome: dict = {}

    def fetch() -> None:
        try:
            outcome["body"] = _fetch_on_this_thread(address)
        except BaseException as e:  # handed to the caller, raised there
            outcome["failure"] = e

    with _fetch_lock:
        if _last_fetch is not None and _last_fetch.is_alive():
            debug_log("agenda: the previous fetch is still running, none started", "agenda")
            raise AgendaUnreadable(
                "The previous reading of the agenda address is still waiting on its "
                "server; nothing was asked again."
            )
        worker = threading.Thread(target=fetch, name="agenda-fetch", daemon=True)
        _last_fetch = worker
        worker.start()
    worker.join(_FETCH_WALL_CLOCK_SEC)
    if worker.is_alive():
        raise _too_long()
    if "failure" in outcome:
        raise outcome["failure"]
    return outcome["body"]


def _fetch_on_this_thread(address: str) -> bytes:
    """Redirects are walked by hand so each hop is checked: a redirect to
    plain HTTP would send the calendar across the network readable."""
    url = address
    if url.lower().startswith("webcal://"):
        url = "https://" + url[len("webcal://"):]
    give_up_at = _clock() + _FETCH_WALL_CLOCK_SEC

    for hop in range(_MAX_REDIRECTS + 1):
        if not url.lower().startswith("https://"):
            if hop == 0:
                debug_log("agenda: the address is not https, nothing fetched", "agenda")
                raise AgendaUnreadable(
                    "The agenda address in the settings (`agenda_source`) must start "
                    "with https://: plain http and other schemes are not read."
                )
            debug_log("agenda: redirected off https, not followed", "agenda")
            raise AgendaUnreadable(
                "The agenda address redirected to one that does not start with "
                "https://, which was not followed."
            )
        if _clock() >= give_up_at:
            raise _too_long()
        try:
            answer = requests.get(
                url, headers=_HEADERS, stream=True, allow_redirects=False,
                timeout=(_CONNECT_TIMEOUT_SEC, _READ_TIMEOUT_SEC),
            )
        except Exception as e:
            # Never the message: requests quotes the address in it.
            debug_log(f"agenda: fetch failed: {type(e).__name__}", "agenda")
            raise AgendaUnreadable(
                f"The agenda address could not be reached ({type(e).__name__})."
            ) from None
        try:
            status = answer.status_code
            if status in _REDIRECTS:
                target = answer.headers.get("Location")
                if not target:
                    debug_log(f"agenda: status {status} with no target", "agenda")
                    raise AgendaUnreadable(
                        "The agenda address answered with a redirect that names no address."
                    )
                url = urljoin(url, target)
                continue
            if status != 200:
                debug_log(f"agenda: status {status}", "agenda")
                raise AgendaUnreadable(f"The agenda address answered with status {status}.")
            body = bytearray()
            for chunk in answer.iter_content(chunk_size=8192):
                if _clock() >= give_up_at:
                    raise _too_long()
                body.extend(chunk)
                if len(body) > MAX_SOURCE_BYTES:
                    debug_log("agenda: the answer is past the size limit", "agenda")
                    raise AgendaUnreadable(
                        "What the agenda address answered is too large to read "
                        f"(more than {_megabytes()})."
                    )
            debug_log(f"agenda: fetched, bytes={len(body)} redirects={hop}", "agenda")
            return bytes(body)
        except AgendaUnreadable:
            raise
        except Exception as e:
            debug_log(f"agenda: reading the answer failed: {type(e).__name__}", "agenda")
            raise AgendaUnreadable(
                f"The agenda address stopped answering ({type(e).__name__})."
            ) from None
        finally:
            answer.close()

    debug_log("agenda: too many redirects", "agenda")
    raise AgendaUnreadable(
        f"The agenda address went through more than {_MAX_REDIRECTS} redirects, "
        "which were not followed further."
    )


def _too_long() -> AgendaUnreadable:
    debug_log("agenda: the fetch passed its time limit", "agenda")
    return AgendaUnreadable("Reading the agenda address took too long and was given up.")

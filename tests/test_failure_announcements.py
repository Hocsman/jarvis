"""A failure that reaches nobody is the failure.

Two subsystems here run with nobody watching and can fail silently: a
reminder the speaker could not say, and a routine that produced nothing.
Both write a ledger row, and a ledger row is a tab nobody has a reason to
open. The tray is the only surface that is always there.

There are two buses, and a surface wired to one of them exists in exactly
half the builds. Subprocess mode reads `__CHAT__:` lines off the daemon's
stdout; a bundled build runs the daemon in-process, pipes no stdout, and
reads a callback instead. Both, every time, or the notification silently
stops existing in the packaged build — which is the build the user is
least able to debug.
"""

from __future__ import annotations

import json

import pytest

from src.jarvis import daemon


@pytest.fixture(autouse=True)
def clean_callbacks():
    daemon.set_confirmation_callbacks()
    yield
    daemon.set_confirmation_callbacks()


def _lines(capsys) -> list:
    out = capsys.readouterr().out
    return [
        json.loads(line[len(daemon.CHAT_IPC_PREFIX):])
        for line in out.splitlines()
        if line.startswith(daemon.CHAT_IPC_PREFIX)
    ]


ROW = {
    "id": "abc", "texte": "sortir le plat", "due_local": "2026-08-02T19:30",
}


# ── A reminder nothing could say ──────────────────────────────────────


def test_a_dropped_reminder_reaches_the_stdout_bus(capsys):
    daemon.announce_reminder_failure(ROW, "la voix ne répondait plus")

    events = [e for e in _lines(capsys) if e.get("type") == "reminder_failed"]
    assert len(events) == 1
    assert events[0]["data"]["texte"] == "sortir le plat"


def test_a_dropped_reminder_reaches_a_bundled_build_too(capsys):
    """No stdout bus exists there, so the line above reaches nobody. This
    is the whole point: a surface wired to one bus works in exactly half
    the builds, and silently not in the other."""
    told = []
    daemon.set_confirmation_callbacks(
        on_reminder_failed=lambda texte, quand, raison: told.append(
            (texte, quand, raison)
        ),
    )

    daemon.announce_reminder_failure(ROW, "la voix ne répondait plus")

    assert told == [("sortir le plat", "2026-08-02T19:30",
                     "la voix ne répondait plus")]


def test_the_promise_itself_travels_with_it(capsys):
    """Unlike a tool call, a reminder *is* the sentence the user asked to
    hear. "A reminder failed" leaves them no way to know which promise
    was dropped, which is the same as not telling them."""
    told = []
    daemon.set_confirmation_callbacks(
        on_reminder_failed=lambda texte, quand, raison: told.append(texte),
    )

    daemon.announce_reminder_failure(ROW, "x")

    assert told == ["sortir le plat"]


def test_a_surface_that_raises_does_not_take_the_scheduler_down(capsys):
    """It runs inside `_abandon`, which is already closing a row nothing
    could deliver. Raising there would lose the settle as well."""
    def _boom(texte, quand, raison):
        raise RuntimeError("pas de zone de notification")

    daemon.set_confirmation_callbacks(on_reminder_failed=_boom)

    daemon.announce_reminder_failure(ROW, "x")  # must not raise


# ── A routine that produced nothing ───────────────────────────────────


def test_routine_trouble_reaches_the_stdout_bus(capsys):
    daemon.announce_routine_trouble("matin", "le modèle n'a pas répondu",
                                    stopped=False)

    events = [e for e in _lines(capsys) if e.get("type") == "routine_trouble"]
    assert len(events) == 1
    assert events[0]["data"]["nom"] == "matin"


def test_routine_trouble_reaches_a_bundled_build_too(capsys):
    told = []
    daemon.set_confirmation_callbacks(
        on_routine_trouble=lambda nom, raison, stopped: told.append((nom, stopped)),
    )

    daemon.announce_routine_trouble("matin", "x", stopped=True)

    assert told == [("matin", True)]


def test_a_routine_surface_that_raises_does_not_take_the_run_down(capsys):
    def _boom(nom, raison, stopped):
        raise RuntimeError("pas de zone de notification")

    daemon.set_confirmation_callbacks(on_routine_trouble=_boom)

    daemon.announce_routine_trouble("matin", "x", stopped=False)  # must not raise


# ── The tray routes both ──────────────────────────────────────────────


@pytest.mark.parametrize("event_type", ["reminder_failed", "routine_trouble"])
def test_the_tray_routes_the_event_it_is_sent(event_type):
    """The gap this closes: `reminder_failed` was emitted from the day
    reminders landed and routed nowhere, because the tray's `if/elif`
    chain only knew the two confirmation events. A subsystem that emits
    into a bus with no listener has a notification on paper only."""
    import inspect

    from src.desktop_app import app as desktop_app

    source = inspect.getsource(desktop_app.JarvisSystemTray._route_confirmation_line)
    assert f'"{event_type}"' in source

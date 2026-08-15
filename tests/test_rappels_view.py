"""Seeing what is scheduled, and being able to call it off.

This tab is the price of storing reminders in a database. Every other
artefact the user owns in this project — `profil.md`, `regles.md`,
`outils.md` — is a text file they open and correct by hand. Reminders
could not be, because a promise needs a write on a clock and the core is
only safe because no background thread writes to it. So the readability
had to be rebuilt here, and a scheduled thing you cannot see or cancel is
a thing you stop creating.

Creating one here reaches no model at all: a date, a time and a sentence
are already unambiguous, and making the user's own typed request go
through an extractor would be adding a way to be wrong.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from src.jarvis.memory.db import Database


@pytest.fixture
def viewer(tmp_path):
    from src.desktop_app import memory_viewer

    db = Database(str(tmp_path / "t.db"), sqlite_vss_path=None)
    memory_viewer._activity_db = db
    yield db, memory_viewer.app.test_client()
    memory_viewer._activity_db = None
    db.close()


def _add(db, **kw):
    payload = dict(
        texte="appeler le comptable",
        due_utc=(datetime.now(timezone.utc) + timedelta(hours=2)).isoformat(),
        due_local="2026-08-06T09:00", tz="Europe/Paris", origin="voix",
        query="rappelle-moi jeudi",
    )
    payload.update(kw)
    return db.add_rappel(**payload)


def _list(client):
    return json.loads(client.get("/api/rappels").data)["rappels"]


# ── Seeing ────────────────────────────────────────────────────────────


def test_a_scheduled_reminder_is_listed(viewer):
    db, client = viewer
    _add(db)

    assert len(_list(client)) == 1


def test_the_list_says_when_and_what(viewer):
    """Both, or the user cannot tell which one to cancel."""
    db, client = viewer
    _add(db, texte="sortir le plat", due_local="2026-08-06T09:00")

    row = _list(client)[0]
    assert row["texte"] == "sortir le plat"
    assert row["due_local"] == "2026-08-06T09:00"


def test_the_soonest_comes_first(viewer):
    db, client = viewer
    _add(db, texte="plus tard",
         due_utc=(datetime.now(timezone.utc) + timedelta(days=2)).isoformat())
    _add(db, texte="bientôt",
         due_utc=(datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat())

    assert [r["texte"] for r in _list(client)] == ["bientôt", "plus tard"]


def test_a_settled_reminder_leaves_the_list(viewer):
    db, client = viewer
    rid = _add(db)

    db.settle_rappel(rid, said_utc=datetime.now(timezone.utc).isoformat())

    assert _list(client) == []


def test_nothing_scheduled_is_an_empty_list_not_an_error(viewer):
    _, client = viewer

    assert _list(client) == []


# ── Cancelling ────────────────────────────────────────────────────────


def test_a_reminder_can_be_called_off(viewer):
    db, client = viewer
    rid = _add(db)

    client.delete(f"/api/rappels/{rid}")

    assert db.pending_rappels() == []


def test_cancelling_reports_success(viewer):
    db, client = viewer
    rid = _add(db)

    body = json.loads(client.delete(f"/api/rappels/{rid}").data)

    assert body.get("cancelled") is True


def test_cancelling_something_gone_says_so_rather_than_pretending(viewer):
    """Reporting success for a reminder that will still fire is the worst
    answer available."""
    _, client = viewer

    body = json.loads(client.delete("/api/rappels/pas-un-id").data)

    assert body.get("cancelled") is False


def test_a_cancelled_reminder_does_not_fire(viewer):
    db, client = viewer
    rid = _add(db, due_utc=(datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat())

    client.delete(f"/api/rappels/{rid}")

    assert db.due_rappels(datetime.now(timezone.utc).isoformat()) == []


# ── Creating, without a model ─────────────────────────────────────────


def _create(client, **kw):
    payload = {"texte": "arroser les plantes", "due_local": "2027-01-01T09:00"}
    payload.update(kw)
    return client.post("/api/rappels", json=payload)


def test_a_reminder_can_be_typed_in_directly(viewer):
    """A date, a time and a sentence are already unambiguous. Sending
    them through an extractor would only add a way to be wrong."""
    db, client = viewer

    _create(client)

    assert len(db.pending_rappels()) == 1


def test_a_typed_reminder_keeps_its_text(viewer):
    db, client = viewer

    _create(client, texte="sortir la poubelle")

    assert db.pending_rappels()[0]["texte"] == "sortir la poubelle"


def test_a_typed_reminder_is_marked_as_coming_from_the_file(viewer):
    """So the ledger can tell it from one she was asked for aloud."""
    db, client = viewer

    _create(client)

    assert db.pending_rappels()[0]["origin"] == "fichier"


def test_a_typed_reminder_fires_at_the_time_given(viewer):
    db, client = viewer

    _create(client, due_local="2027-03-04T18:30")

    assert db.pending_rappels()[0]["due_local"] == "2027-03-04T18:30"


@pytest.mark.parametrize("payload,why", [
    ({"texte": "", "due_local": "2027-01-01T09:00"}, "no text"),
    ({"texte": "x", "due_local": ""}, "no time"),
    ({"texte": "x", "due_local": "jeudi"}, "unparseable time"),
    ({"texte": "x", "due_local": "2020-01-01T09:00"}, "in the past"),
    ({"due_local": "2027-01-01T09:00"}, "text missing entirely"),
])
def test_a_bad_entry_creates_nothing(viewer, payload, why):
    db, client = viewer

    response = client.post("/api/rappels", json=payload)

    assert response.status_code >= 400
    assert db.pending_rappels() == []


def test_creating_reaches_no_model(viewer):
    """The whole point of typing it in."""
    from unittest.mock import patch

    db, client = viewer
    with patch("src.jarvis.reminders.extract._ask_model") as asked:
        _create(client)

    assert asked.called is False


# ── The page itself ───────────────────────────────────────────────────


def test_the_tab_exists(viewer):
    from src.desktop_app.memory_viewer import index

    assert 'data-tab="rappels"' in index()


def test_the_page_still_parses(viewer):
    """One bad character in this template kills every handler on the
    page, silently."""
    import re
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path

    from src.desktop_app.memory_viewer import index

    if shutil.which("node") is None:
        pytest.skip("node not installed")

    for script in re.findall(r"<script>(.*?)</script>", index(), re.DOTALL):
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
            f.write(script)
            path = f.name
        try:
            done = subprocess.run(["node", "--check", path],
                                  capture_output=True, text=True)
        finally:
            Path(path).unlink(missing_ok=True)
        assert done.returncode == 0, done.stderr

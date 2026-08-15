"""Seeing what runs while nobody is watching.

Two questions, and neither artefact answers both. The database row says
when a routine fires; `yuba/routines.md` says what it may reach. Shown
apart, a user has to open a text editor to find out whether the thing
running at 07:00 can read their files, which is exactly the moment they
decide the feature is not worth having.

The journal is here too, and it is the delivery rather than a record of
one. Nobody is in the room at 07:00, so the write-up is on this page or
it arrived to nobody. Served as the raw Markdown the file holds, because
the same file opens in any editor and the two must not disagree.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from src.jarvis.memory.db import Database
from src.jarvis.routines.journal import Entree, append_run
from src.jarvis.routines.recurrence import Regle


ROUTINES_FILE = """# Routines

## matin
phrase: résume-moi mes mails
quand: tous les jours à 07:00
outils:
- webSearch
- fetchWebPage
"""


class _Cfg:
    def __init__(self, tmp_path):
        self.db_path = str(tmp_path / "t.db")


@pytest.fixture
def viewer(tmp_path):
    from src.desktop_app import memory_viewer

    core = tmp_path / "yuba"
    core.mkdir(parents=True, exist_ok=True)
    (core / "routines.md").write_text(ROUTINES_FILE, encoding="utf-8")

    cfg = _Cfg(tmp_path)
    db = Database(str(tmp_path / "t.db"), sqlite_vss_path=None)
    memory_viewer._activity_db = db
    _real = memory_viewer.load_settings
    memory_viewer.load_settings = lambda: cfg
    try:
        yield db, memory_viewer.app.test_client(), cfg
    finally:
        memory_viewer.load_settings = _real
        memory_viewer._activity_db = None
        db.close()


def _add(db, *, nom="matin", steriles=0, texte="résume-moi mes mails",
         kind="routine"):
    payload = {
        "nom": nom,
        "regle": Regle(kind="daily", hour=7, minute=0, weekday=None).to_json(),
    }
    if steriles:
        payload["steriles"] = steriles
    return db.add_rappel(
        texte=texte, kind=kind, origin="voix",
        due_utc=(datetime.now(timezone.utc) + timedelta(hours=8)).isoformat(),
        due_local="2026-08-03T07:00", tz="Europe/Paris",
        query=texte, payload=payload,
    )


def _list(client):
    return json.loads(client.get("/api/routines").data)["routines"]


def _vivantes(client):
    return [r for r in _list(client) if not r["arretee"]]


def _pages(client):
    return json.loads(client.get("/api/journal").data)["pages"]


# ── What fires, and when ──────────────────────────────────────────────


def test_a_routine_is_listed(viewer):
    db, client, _ = viewer
    _add(db)

    assert len(_list(client)) == 1


def test_a_reminder_is_not(viewer):
    """They share a table and nothing else. Listing a reminder here
    invites the user to stop a routine and find it back tomorrow."""
    db, client, _ = viewer
    _add(db, kind="rappel")

    assert _vivantes(client) == []


def test_a_stopped_routine_is_shown_as_stopped_rather_than_hidden(viewer):
    """Its block is the durable record of what it was allowed to do, and
    saying the same request again restarts it. Hidden, the only surface
    holding that was a file the desktop app never opened, under an empty
    state reading "aucune routine" over a routines.md that held one."""
    db, client, _ = viewer
    rid = _add(db)
    db.cancel_rappel(rid)

    arretees = [r for r in _list(client) if r["arretee"]]
    assert [r["nom"] for r in arretees] == ["matin"]
    assert arretees[0]["outils"] == ["webSearch", "fetchWebPage"]
    assert _vivantes(client) == []


def test_the_next_time_it_fires_is_shown(viewer):
    db, client, _ = viewer
    _add(db)

    assert _list(client)[0]["due_local"] == "2026-08-03T07:00"


# ── What it may reach ─────────────────────────────────────────────────


def test_the_sentence_shown_is_the_one_that_runs(viewer):
    """The block's phrase is what the runner executes, so showing the
    row's original wording would display something that is no longer
    what happens."""
    db, client, cfg = viewer
    from src.jarvis.routines.scope import routines_path

    routines_path(cfg).write_text(
        ROUTINES_FILE.replace("résume-moi mes mails", "corrigé à la main"),
        encoding="utf-8")
    _add(db, texte="résume-moi mes mails")

    assert _vivantes(client)[0]["texte"] == "corrigé à la main"


def test_the_envelope_is_shown_beside_the_hour(viewer):
    """The point of the whole tab. A row that says only "07:00, matin"
    answers the less important of the two questions."""
    db, client, _ = viewer
    _add(db)

    assert _list(client)[0]["outils"] == ["webSearch", "fetchWebPage"]


def test_a_routine_whose_block_was_deleted_says_it_is_suspended(viewer):
    """Not hidden: it still holds a slot in the table, and it comes back
    the moment the block does."""
    db, client, cfg = viewer
    from src.jarvis.routines.scope import routines_path

    routines_path(cfg).write_text("# Routines\n", encoding="utf-8")
    _add(db)

    row = _list(client)[0]
    assert row["suspendue"] is True
    assert row["outils"] == []


def test_a_routine_that_carries_the_profile_says_so(viewer):
    """Every line of it is the user's private life leaving the machine
    while they sleep. That is worth one word on the row."""
    db, client, cfg = viewer
    from src.jarvis.routines.scope import routines_path

    routines_path(cfg).write_text(
        ROUTINES_FILE + "\n## bilan\nmémoire: oui\noutils:\n- webSearch\n",
        encoding="utf-8",
    )
    _add(db, nom="bilan")

    assert _list(client)[0]["memoire"] is True


def test_a_routine_that_keeps_producing_nothing_shows_its_count(viewer):
    """It is four fifths of the way to switching itself off, and the
    only place that is visible before it happens is here."""
    db, client, _ = viewer
    _add(db, steriles=4)

    assert _list(client)[0]["steriles"] == 4


# ── Stopping one ──────────────────────────────────────────────────────


def test_a_routine_can_be_stopped(viewer):
    db, client, _ = viewer
    rid = _add(db)

    client.delete(f"/api/routines/{rid}")

    assert _vivantes(client) == []


def test_stopping_reports_honestly(viewer):
    db, client, _ = viewer

    body = json.loads(client.delete("/api/routines/pas-un-id").data)

    assert body["cancelled"] is False


# ── The mornings ──────────────────────────────────────────────────────


def test_a_written_morning_is_served(viewer):
    _, client, cfg = viewer
    append_run(cfg, Entree(
        nom="matin", moment=datetime.now(), demande="résume-moi mes mails",
        texte="Trois mails, rien d'urgent.",
    ))

    assert "Trois mails, rien d'urgent." in _pages(client)[0]["texte"]


def test_the_newest_morning_comes_first(viewer):
    _, client, cfg = viewer
    append_run(cfg, Entree(nom="matin", moment=datetime.now() - timedelta(days=2),
                           demande="x", texte="avant-hier"))
    append_run(cfg, Entree(nom="matin", moment=datetime.now(),
                           demande="x", texte="aujourd'hui"))

    assert "aujourd'hui" in _pages(client)[0]["texte"]


def test_a_day_with_nothing_written_is_not_an_empty_page(viewer):
    _, client, cfg = viewer
    append_run(cfg, Entree(nom="matin", moment=datetime.now(),
                           demande="x", texte="voilà"))

    assert len(_pages(client)) == 1


def test_no_journal_at_all_is_empty_rather_than_an_error(viewer):
    _, client, _ = viewer

    assert _pages(client) == []


# ── An envelope that names something no longer installed ──────────────


def test_a_tool_that_has_left_the_catalogue_is_flagged(viewer):
    """Otherwise the row keeps advertising a capability that stopped
    existing in October, and the only trace anywhere is a debug line
    nobody has switched on."""
    db, client, cfg = viewer
    from src.jarvis.routines.scope import routines_path

    routines_path(cfg).write_text(
        "# Routines\n\n## matin\nphrase: x\nquand: x\noutils:\n"
        "- webSearch\n- mail__list\n",
        encoding="utf-8",
    )
    _add(db)

    assert _list(client)[0]["introuvables"] == ["mail__list"]


def test_an_envelope_that_is_all_there_flags_nothing(viewer):
    db, client, _ = viewer
    _add(db)

    assert _list(client)[0]["introuvables"] == []

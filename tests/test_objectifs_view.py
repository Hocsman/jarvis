"""Reading in October what he said in August.

That is the whole reason this artefact exists, so the tab shows the
whole page rather than a summary — a tab showing only the last line
would be a worse version of the prompt block, which already exists and
is deliberately thin.

Ending one from here is the same act the tool performs and the same one
it reserves for him: a click is him. The block stays in the file with
everything it recorded, because that is what somebody wants to read
afterwards.
"""

from __future__ import annotations

import json

import pytest

from src.jarvis.objectifs.page import invalidate_objectifs_cache, objectifs_path


PAGE = """# Objectifs

## entretien
phrase: préparer l'entretien chez Datadog
fini quand: l'entretien est passé
points:
- 2026-08-02 · dit · premier call
- 2026-08-04 · dit · exercice rendu

## ancien
phrase: refaire le CV
fini quand: il est à jour
clos: 2026-07-30 · atteint
"""


class _Cfg:
    def __init__(self, tmp_path):
        self.db_path = str(tmp_path / "t.db")


@pytest.fixture
def viewer(tmp_path):
    from src.desktop_app import memory_viewer

    cfg = _Cfg(tmp_path)
    path = objectifs_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(PAGE, encoding="utf-8")
    invalidate_objectifs_cache()

    _real = memory_viewer.load_settings
    memory_viewer.load_settings = lambda: cfg
    try:
        yield memory_viewer.app.test_client(), cfg
    finally:
        memory_viewer.load_settings = _real
        invalidate_objectifs_cache()


def _list(client):
    return json.loads(client.get("/api/objectifs").data)["objectifs"]


# ── The whole page ────────────────────────────────────────────────────


def test_every_goal_is_listed(viewer):
    client, _ = viewer

    assert {o["nom"] for o in _list(client)} == {"entretien", "ancien"}


def test_open_ones_come_first(viewer):
    """A closed goal is a record; an open one is a question."""
    client, _ = viewer

    assert _list(client)[0]["nom"] == "entretien"


def test_everything_it_recorded_is_there(viewer):
    """The reason this artefact exists: reading in October what he said
    in August. A tab showing the last line would be a worse version of
    the prompt block."""
    client, _ = viewer

    points = _list(client)[0]["points"]
    assert [p["texte"] for p in points] == ["premier call", "exercice rendu"]
    assert {p["source"] for p in points} == {"dit"}


def test_what_counts_as_done_is_shown(viewer):
    client, _ = viewer

    assert _list(client)[0]["fini_quand"] == "l'entretien est passé"


def test_a_closed_one_says_when_and_how(viewer):
    client, _ = viewer

    ancien = next(o for o in _list(client) if o["nom"] == "ancien")
    assert ancien["ouvert"] is False
    assert ancien["clos"] == "2026-07-30 · atteint"


def test_no_goals_is_empty_rather_than_an_error(tmp_path):
    from src.desktop_app import memory_viewer

    cfg = _Cfg(tmp_path)
    _real = memory_viewer.load_settings
    memory_viewer.load_settings = lambda: cfg
    invalidate_objectifs_cache()
    try:
        assert _list(memory_viewer.app.test_client()) == []
    finally:
        memory_viewer.load_settings = _real


# ── Ending one from here ──────────────────────────────────────────────


def test_a_goal_can_be_ended_with_a_click(viewer):
    """The same act the tool performs, and the one it reserves for him.
    A click is him."""
    client, _ = viewer

    body = json.loads(client.delete("/api/objectifs/entretien").data)

    assert body["closed"] is True
    assert next(o for o in _list(client) if o["nom"] == "entretien")["ouvert"] is False


def test_ending_it_keeps_everything_it_recorded(viewer):
    client, _ = viewer

    client.delete("/api/objectifs/entretien")

    points = next(o for o in _list(client) if o["nom"] == "entretien")["points"]
    assert [p["texte"] for p in points] == ["premier call", "exercice rendu"]


def test_ending_reports_honestly(viewer):
    """Saying it is over when it is not is the worst answer here."""
    client, _ = viewer

    assert json.loads(client.delete("/api/objectifs/jamais-écrit").data)["closed"] is False


def test_one_already_ended_is_not_ended_twice(viewer):
    client, _ = viewer

    assert json.loads(client.delete("/api/objectifs/ancien").data)["closed"] is False

"""The Activity tab, read as a record of decisions rather than of calls.

A confirmed action leaves two rows sharing a request id — the question,
and whatever settled it. To the person reading this they are one event,
so they are shown as one. The exception is a question nobody answered:
that row is the only trace a decision never taken leaves, and it keeps
its own line.

The other thing this has to get right is that `refusé` and `décliné` are
different facts. "She would not" and "I would not" must not look alike,
or the record answers the wrong question.
"""

from __future__ import annotations

import json

import pytest

from src.jarvis.memory.db import Database


@pytest.fixture
def viewer(tmp_path):
    from src.desktop_app import memory_viewer

    db = Database(str(tmp_path / "t.db"), sqlite_vss_path=None)
    memory_viewer._activity_db = db
    client = memory_viewer.app.test_client()
    yield db, client
    memory_viewer._activity_db = None
    db.close()


def _record(db, **kw):
    payload = dict(
        tool="localFiles", args={"operation": "write"}, risk="destructif",
        verdict="demande", outcome="ok", duration_ms=None, origin="voix",
        query="note ça", request_id=None,
    )
    payload.update(kw)
    db.record_action(**payload)


def _actions(client):
    return json.loads(client.get("/api/activity").data)["actions"]


# ── The pair travels together ─────────────────────────────────────────


def test_the_request_id_reaches_the_view(viewer):
    """Without it, nothing on this page can tell two rows apart from two
    unrelated calls."""
    db, client = viewer
    _record(db, request_id="cf_paire")

    assert _actions(client)[0]["request_id"] == "cf_paire"


def test_a_question_and_its_answer_are_the_same_episode(viewer):
    db, client = viewer
    _record(db, outcome="demandé", request_id="cf_paire")
    _record(db, outcome="ok", duration_ms=14, request_id="cf_paire")

    rows = [a for a in _actions(client) if a["request_id"] == "cf_paire"]

    assert {r["outcome"] for r in rows} == {"demandé", "ok"}


def test_a_question_nobody_answered_survives_alone(viewer):
    """The only shape that records a decision never taken."""
    db, client = viewer
    _record(db, outcome="demandé", request_id="cf_orphelin")

    rows = [a for a in _actions(client) if a["request_id"] == "cf_orphelin"]

    assert len(rows) == 1
    assert rows[0]["outcome"] == "demandé"


# ── The two kinds of no ───────────────────────────────────────────────


def test_a_machine_refusal_and_a_user_decline_are_distinct(viewer):
    db, client = viewer
    _record(db, outcome="refusé")
    _record(db, outcome="décliné", request_id="cf_x")

    outcomes = {a["outcome"] for a in _actions(client)}

    assert {"refusé", "décliné"} <= outcomes


def test_a_machine_refusal_carries_no_request_id(viewer):
    """Nobody was asked, so there is no question for it to belong to."""
    db, client = viewer
    _record(db, outcome="refusé")

    assert _actions(client)[0]["request_id"] is None


def test_they_are_styled_differently(viewer):
    """A user's own decision must not look like the machine overruling
    them."""
    from src.desktop_app.memory_viewer import index

    page = index()

    assert ".activity-refusé" in page
    assert ".activity-décliné" in page
    assert ".activity-expiré" in page


def test_the_page_explains_the_difference(viewer):
    """The vocabulary is the point of the tab, and nowhere else says it."""
    from src.desktop_app.memory_viewer import index

    page = index()

    assert "refusé" in page and "décliné" in page


# ── Still no tool output, ever ────────────────────────────────────────


def test_the_view_exposes_no_field_for_tool_output(viewer):
    db, client = viewer
    _record(db)

    keys = set(_actions(client)[0])

    for forbidden in ("output", "result", "reply", "reply_text", "content"):
        assert forbidden not in keys

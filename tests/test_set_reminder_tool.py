"""Setting a reminder, and never claiming one that did not land.

The invariant is stronger than "confirm what you wrote": the tool writes
the row, **reads it back from the database**, re-parses it, and speaks
that. So the sentence the user hears is derived from the artefact that
will actually fire — not from the model's JSON, and not from what the
tool believed a moment ago. A write that did not land cannot be
confirmed, because there is nothing to read back.

Saying the time back is also the whole disambiguation strategy. A model
that read "jeudi" as Tuesday is caught by the user's ear, in the same
turn, in any language, with no word list anywhere — the same move as the
confirmation card: the words the user checks against are written by code.

Failure here is noisy and immediate, which is neither of the usual two
directions. `read_approval` fails closed because one extra "no" costs a
turn. Here both silent directions break a promise, and the user is
standing right there having spoken two seconds ago, so a clarifying
sentence costs them nothing.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.jarvis.memory.db import Database
from src.jarvis.reminders.extract import Extraction, ExtractionFailed
from src.jarvis.tools.base import ToolContext
from src.jarvis.tools.registry import BUILTIN_TOOLS


class _Cfg:
    reminder_model = ""
    reminder_timeout_sec = 8.0
    reminder_default_hour = 9
    tool_router_model = "petit"
    intent_judge_model = ""
    llm_chat_model = "grand"
    reminders_enabled = True
    voice_debug = False


@pytest.fixture
def db(tmp_path):
    database = Database(str(tmp_path / "t.db"), sqlite_vss_path=None)
    yield database
    database.close()


@pytest.fixture
def tool():
    return BUILTIN_TOOLS["setReminder"]


def _ctx(db, redacted="rappelle-moi dans vingt minutes de sortir le plat"):
    return ToolContext(
        db=db, cfg=_Cfg(), system_prompt="", original_prompt="",
        redacted_text=redacted, max_retries=1, user_print=lambda _m: None,
    )


def _reading(due_local, what="sortir le plat", assumed=False):
    return patch(
        "src.jarvis.tools.builtin.set_reminder.extract_reminder_time",
        return_value=Extraction(
            due_local=due_local, what=what, hour_was_assumed=assumed,
        ),
    )


SOON = datetime.now() + timedelta(minutes=20)


# ── It lands, and only then is it claimed ─────────────────────────────


def test_a_reminder_is_written(db, tool):
    with _reading(SOON):
        tool.run({"rappel": "dans vingt minutes, sortir le plat"}, _ctx(db))

    assert len(db.pending_rappels()) == 1


def test_the_reply_says_when(db, tool):
    """"C'est noté" without a time is a promise the user cannot check."""
    with _reading(SOON.replace(hour=14, minute=30)):
        result = tool.run({"rappel": "x"}, _ctx(db))

    assert "14:30" in (result.reply_text or "")


def test_the_reply_says_what(db, tool):
    with _reading(SOON, what="sortir le plat du four"):
        result = tool.run({"rappel": "x"}, _ctx(db))

    assert "sortir le plat du four" in (result.reply_text or "")


def test_the_time_is_read_back_from_the_database(db, tool):
    """Not rendered from the extraction. What is spoken has to come from
    the artefact that will fire, or a write that silently landed wrong is
    confirmed as correct."""
    seen = {}
    real = db.add_rappel

    def _spy(**kw):
        rid = real(**kw)
        seen["written"] = next(r for r in db.all_rappels() if r["id"] == rid)
        return rid

    with _reading(SOON), patch.object(db, "add_rappel", side_effect=_spy):
        result = tool.run({"rappel": "x"}, _ctx(db))

    stored = datetime.fromisoformat(seen["written"]["due_local"])
    assert f"{stored:%H:%M}" in (result.reply_text or "")


def test_a_write_that_did_not_land_is_not_claimed(db, tool):
    """There is nothing to read back, so there is nothing to confirm."""
    with _reading(SOON), patch.object(db, "all_rappels", return_value=[]):
        result = tool.run({"rappel": "x"}, _ctx(db))

    assert result.success is False
    assert "noté" not in (result.reply_text or "").lower()


def test_a_write_she_cannot_find_is_withdrawn_too(db, tool):
    """A reminder going off after she said it was not set is worse than
    no reminder at all."""
    with _reading(SOON), patch.object(db, "all_rappels", return_value=[]):
        tool.run({"rappel": "x"}, _ctx(db))

    assert db.pending_rappels() == []


def test_an_unreadable_round_trip_withdraws_the_write(db, tool):
    """Better no reminder than one she has told the user about and
    cannot describe."""
    real = db.all_rappels

    def _corrupt():
        return [{**row, "due_local": "n'importe quoi"} for row in real()]

    with _reading(SOON), patch.object(db, "all_rappels", side_effect=_corrupt):
        result = tool.run({"rappel": "x"}, _ctx(db))

    assert result.success is False
    assert db.pending_rappels() == []


# ── A defaulted hour is said out loud ─────────────────────────────────


def test_an_assumed_hour_is_admitted(db, tool):
    """"jeudi à neuf heures" is only honest if the nine came from
    somewhere the user can hear and correct."""
    with _reading(SOON, assumed=True):
        result = tool.run({"rappel": "jeudi, appeler le comptable"}, _ctx(db))

    assert result.reply_text
    assert len(result.reply_text) > len("C'est noté.")


# ── Where the text comes from ─────────────────────────────────────────


def test_the_property_is_used_when_given(db, tool):
    seen = {}

    def _capture(cfg, utterance, **kw):
        seen["utterance"] = utterance
        return Extraction(due_local=SOON, what="x", hour_was_assumed=False)

    with patch("src.jarvis.tools.builtin.set_reminder.extract_reminder_time",
               side_effect=_capture):
        tool.run({"rappel": "dans dix minutes, arroser"}, _ctx(db))

    assert seen["utterance"] == "dans dix minutes, arroser"


def test_it_falls_back_to_what_the_user_said(db, tool):
    """Same precedence as logMeal: a model that forgets the property has
    not lost the request."""
    seen = {}

    def _capture(cfg, utterance, **kw):
        seen["utterance"] = utterance
        return Extraction(due_local=SOON, what="x", hour_was_assumed=False)

    with patch("src.jarvis.tools.builtin.set_reminder.extract_reminder_time",
               side_effect=_capture):
        tool.run({}, _ctx(db, redacted="dans une heure, sortir le chien"))

    assert seen["utterance"] == "dans une heure, sortir le chien"


def test_with_nothing_at_all_it_refuses(db, tool):
    result = tool.run({}, _ctx(db, redacted=""))

    assert result.success is False
    assert db.pending_rappels() == []


# ── Failing loudly, in the same turn ──────────────────────────────────


def test_an_unreadable_time_writes_nothing(db, tool):
    with patch("src.jarvis.tools.builtin.set_reminder.extract_reminder_time",
               side_effect=ExtractionFailed("je n'ai pas compris quand")):
        result = tool.run({"rappel": "un de ces quatre"}, _ctx(db))

    assert result.success is False
    assert db.pending_rappels() == []


def test_the_reason_reaches_the_user(db, tool):
    """"I could not work out when" and "I could not reach the model" are
    different apologies, and the user is standing right there."""
    with patch("src.jarvis.tools.builtin.set_reminder.extract_reminder_time",
               side_effect=ExtractionFailed("je n'ai pas pu joindre le modèle")):
        result = tool.run({"rappel": "x"}, _ctx(db))

    assert "joindre le modèle" in (result.reply_text or "")


def test_with_no_model_it_says_so_rather_than_inventing(db, tool):
    class _Bare(_Cfg):
        reminder_model = ""
        tool_router_model = ""
        intent_judge_model = ""
        llm_chat_model = ""

    ctx = ToolContext(
        db=db, cfg=_Bare(), system_prompt="", original_prompt="",
        redacted_text="dans vingt minutes", max_retries=1,
        user_print=lambda _m: None,
    )
    result = tool.run({"rappel": "dans vingt minutes"}, ctx)

    assert result.success is False
    assert db.pending_rappels() == []


# ── It is not the memory tool ─────────────────────────────────────────


def test_it_writes_nothing_to_the_core(db, tool, tmp_path):
    """The defect this closes is exactly the confusion between the two.
    A reminder is not a fact about the user."""
    from src.jarvis.memory.core import SECTION_PROFILE, MemoryCore

    class _WithCore(_Cfg):
        db_path = str(tmp_path / "jarvis.db")

    ctx = ToolContext(
        db=db, cfg=_WithCore(), system_prompt="", original_prompt="",
        redacted_text="x", max_retries=1, user_print=lambda _m: None,
    )
    with _reading(SOON):
        tool.run({"rappel": "jeudi, appeler le comptable"}, ctx)

    assert MemoryCore.for_config(_WithCore()).active(SECTION_PROFILE) == []


def test_its_name_does_not_collide_with_remember(tool):
    """`remindMe` would share a prefix with the tool whose traffic it is
    taking over, in a catalogue the router matches on."""
    assert tool.name == "setReminder"
    assert not tool.name.lower().startswith("remember")


def test_writing_a_reminder_needs_no_permission():
    """Same reasoning as `remember`: it touches Yuba's own store and
    nothing of the user's machine. Asking permission to write down what
    she was just asked to write down is noise."""
    from src.jarvis.tools.policy import RISK_READ

    assert BUILTIN_TOOLS["setReminder"].risk_for({}) == RISK_READ


def test_the_schema_has_one_optional_property():
    """One property is what makes the planner's fast path work:
    `setReminder rappel='…'` is a concrete key='value', so it dispatches
    without the step resolver — which would otherwise be asked to
    hallucinate a timestamp, the exact failure this design avoids."""
    schema = BUILTIN_TOOLS["setReminder"].inputSchema

    assert list(schema.get("properties", {})) == ["rappel"]
    assert not schema.get("required")

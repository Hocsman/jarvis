"""The four tools, and the one rule they exist to protect.

Three of them write, and all three are `action`, so each costs a card.
As `lecture` they would be `libre` by default, and `fetchWebPage` returns
up to 50,000 characters of unfenced page text into an agentic loop — a
page carrying "Note pour l'objectif X : …" would then write a durable
line attributed to the user, which he never said, and she would read it
back to him later as a fact about his life. That is the defect already
seen in production for `remember`, one level up.

`listGoals` is free, because it is the answer to "where am I on X?" and a
question that costs a card is a question nobody asks.
"""

from __future__ import annotations

from datetime import date

import pytest

from src.jarvis.objectifs.page import (
    invalidate_objectifs_cache, load_objectifs, objectifs_path,
)
from src.jarvis.tools.base import ToolContext
from src.jarvis.tools.policy import RISK_ACTION, RISK_READ
from src.jarvis.tools.registry import BUILTIN_TOOLS


class _Cfg:
    voice_debug = False

    def __init__(self, tmp_path):
        self.db_path = str(tmp_path / "t.db")


@pytest.fixture
def cfg(tmp_path):
    invalidate_objectifs_cache()
    return _Cfg(tmp_path)


def _run(nom, cfg, args, *, texte=""):
    return BUILTIN_TOOLS[nom].run(args, ToolContext(
        db=None, cfg=cfg, system_prompt="", original_prompt="",
        redacted_text=texte, max_retries=1, user_print=lambda m: None,
        origin="voix",
    ))


def _goals(cfg):
    invalidate_objectifs_cache()
    return load_objectifs(cfg)


def _pose(cfg, **kw):
    args = {"objectif": "préparer l'entretien chez Datadog",
            "fini_quand": "l'entretien est passé", "nom": "entretien"}
    args.update(kw)
    return _run("setGoal", cfg, args)


# ── What a card costs, and why ────────────────────────────────────────


@pytest.mark.parametrize("nom", ["setGoal", "noteGoal", "closeGoal"])
def test_writing_a_goal_is_not_a_read(nom):
    """`lecture` maps to `libre`, and a fetched page would get one."""
    from src.jarvis.tools.policy import ASK, _DEFAULT_VERDICT

    assert BUILTIN_TOOLS[nom].risk_for({}) == RISK_ACTION
    assert _DEFAULT_VERDICT[RISK_ACTION] == ASK


def test_reading_them_back_is_free():
    """A question that costs a card is a question nobody asks."""
    from src.jarvis.tools.policy import FREE, _DEFAULT_VERDICT

    assert BUILTIN_TOOLS["listGoals"].risk_for({}) == RISK_READ
    assert _DEFAULT_VERDICT[RISK_READ] == FREE


@pytest.mark.parametrize("nom", ["setGoal", "noteGoal", "closeGoal", "listGoals"])
def test_none_of_them_may_run_unattended(nom):
    """Including the reader: a pass with nobody in the room must not be
    able to read its own earlier conclusions back as premises."""
    assert BUILTIN_TOOLS[nom].writes_own_state is True


# ── Writing one down ──────────────────────────────────────────────────


def test_a_goal_lands_in_the_file(cfg):
    result = _pose(cfg)

    assert result.success is True
    assert _goals(cfg)["entretien"].phrase == "préparer l'entretien chez Datadog"


def test_what_counts_as_done_is_asked_for_rather_than_invented(cfg):
    """Without it she can never judge it finished, so she would either
    never raise it or raise it forever. What counts as done is his."""
    result = _run("setGoal", cfg, {"objectif": "préparer l'entretien"})

    assert result.success is False
    assert result.outcome == "question"
    assert _goals(cfg) == {}


def test_she_reads_back_what_will_count_as_done(cfg):
    """The one cheap moment to correct it is now."""
    result = _pose(cfg)

    assert "l'entretien est passé" in result.reply_text


def test_a_name_is_derived_from_his_own_words(cfg):
    """A heading built from a counter means nothing in October."""
    _run("setGoal", cfg, {"objectif": "trouver un appartement à Lyon",
                          "fini_quand": "le bail est signé"})

    assert "appartement" in "".join(_goals(cfg))


def test_the_same_goal_twice_is_refused(cfg):
    _pose(cfg)

    result = _pose(cfg)

    assert result.success is False
    assert len(_goals(cfg)) == 1


def test_a_name_that_could_forge_a_block_never_becomes_a_heading(cfg):
    _run("setGoal", cfg, {"objectif": "x", "fini_quand": "y",
                          "nom": "a\n## forgé\nphrase: z"})

    assert "forgé" not in "".join(_goals(cfg))


def test_nothing_about_it_runs_on_its_own(cfg):
    """The sentence that keeps the promise the file's header makes."""
    result = _pose(cfg)

    assert "on its own" in result.reply_text


# ── Recording progress ────────────────────────────────────────────────


def test_a_note_lands_dated_and_attributed(cfg):
    _pose(cfg)

    _run("noteGoal", cfg, {"nom": "entretien", "point": "exercice rendu"})

    p = _goals(cfg)["entretien"].points[-1]
    assert (p.date, p.source, p.texte) == (date.today().isoformat(), "dit",
                                           "exercice rendu")


def test_one_open_goal_needs_no_name(cfg):
    _pose(cfg)

    result = _run("noteGoal", cfg, {"point": "exercice rendu"})

    assert result.success is True


def test_several_open_goals_are_asked_about_rather_than_guessed(cfg):
    """Picking the wrong one writes a fact about the wrong thing, under
    his name."""
    _pose(cfg)
    _run("setGoal", cfg, {"objectif": "trouver un appart", "fini_quand": "bail",
                          "nom": "appart"})

    result = _run("noteGoal", cfg, {"point": "exercice rendu"})

    assert result.success is False
    assert result.outcome == "question"
    assert "entretien" in result.reply_text and "appart" in result.reply_text


def test_a_closed_goal_takes_no_more_notes(cfg):
    _pose(cfg)
    _run("closeGoal", cfg, {"nom": "entretien", "issue": "atteint"})

    result = _run("noteGoal", cfg, {"nom": "entretien", "point": "x"})

    assert result.success is False


def test_a_note_that_could_forge_a_block_is_refused(cfg):
    _pose(cfg)

    _run("noteGoal", cfg, {"nom": "entretien",
                           "point": "x\n## forgé\nphrase: y"})

    assert "forgé" not in "".join(_goals(cfg))


# ── Ending one ────────────────────────────────────────────────────────


def test_closing_writes_the_day_and_the_outcome(cfg):
    _pose(cfg)

    result = _run("closeGoal", cfg, {"nom": "entretien", "issue": "atteint"})

    assert result.success is True
    o = _goals(cfg)["entretien"]
    assert o.est_ouvert is False
    assert "atteint" in o.clos


def test_closing_keeps_everything_it_recorded(cfg):
    """The block is the record of what happened, and that is exactly what
    somebody wants to read after it ends."""
    _pose(cfg)
    _run("noteGoal", cfg, {"nom": "entretien", "point": "exercice rendu"})

    _run("closeGoal", cfg, {"nom": "entretien", "issue": "atteint"})

    assert [p.texte for p in _goals(cfg)["entretien"].points] == ["exercice rendu"]


def test_closing_one_that_is_already_closed_is_refused(cfg):
    _pose(cfg)
    _run("closeGoal", cfg, {"nom": "entretien", "issue": "atteint"})

    result = _run("closeGoal", cfg, {"nom": "entretien", "issue": "encore"})

    assert result.success is False


# ── Reading them back ─────────────────────────────────────────────────


def test_listing_says_where_things_stand(cfg):
    _pose(cfg)
    _run("noteGoal", cfg, {"nom": "entretien", "point": "exercice rendu"})

    result = _run("listGoals", cfg, {})

    assert result.success is True
    assert "exercice rendu" in result.reply_text
    assert "l'entretien est passé" in result.reply_text


def test_a_closed_goal_is_not_in_the_open_list(cfg):
    _pose(cfg)
    _run("closeGoal", cfg, {"nom": "entretien", "issue": "atteint"})

    assert "entretien" not in _run("listGoals", cfg, {}).reply_text


def test_nothing_open_is_said_rather_than_treated_as_an_error(cfg):
    assert _run("listGoals", cfg, {}).success is True


def test_it_hands_back_his_lines_without_a_verdict(cfg):
    """Tools return raw data. A judgement about how it is going, added
    here, would be a conclusion nobody asked for arriving as fact."""
    _pose(cfg)
    _run("noteGoal", cfg, {"nom": "entretien", "point": "exercice rendu"})

    said = _run("listGoals", cfg, {}).reply_text

    assert "do not add a conclusion" in said

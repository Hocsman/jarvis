"""Recognising the subject when it comes up.

The block is thin on purpose: one line per open goal, so she knows the
subject exists and can say where it stands. The history is one
`listGoals` call away, on the turns that want it.

Every line is his. Only points whose source is `dit` appear, so that the
day a pass can write a line of its own, that line does not arrive in an
attended prompt looking like something he said — mechanically, rather
than as a matter of who remembers.

And an ordinary turn about anything else pays for none of it: with no
open goal the block is empty, which is the common case on most days.
"""

from __future__ import annotations

import pytest

from src.jarvis.objectifs.page import (
    Objectif, Point, invalidate_objectifs_cache, objectifs_path,
)
from src.jarvis.objectifs.prompt import format_objectifs_block


class _Cfg:
    def __init__(self, tmp_path):
        self.db_path = str(tmp_path / "t.db")


@pytest.fixture
def cfg(tmp_path):
    invalidate_objectifs_cache()
    return _Cfg(tmp_path)


def _write(cfg, texte):
    path = objectifs_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(texte, encoding="utf-8")
    invalidate_objectifs_cache()


OUVERT = """# Objectifs

## entretien
phrase: préparer l'entretien chez Datadog
fini quand: l'entretien est passé
points:
- 2026-08-04 · dit · exercice rendu
"""


# ── What it costs when nothing is going on ────────────────────────────


def test_no_goal_means_no_block(cfg):
    """The common case, most days. An ordinary turn about anything else
    pays nothing for this feature."""
    assert format_objectifs_block(cfg) == ""


def test_a_closed_goal_is_not_carried_around(cfg):
    _write(cfg, OUVERT.replace("points:", "clos: 2026-08-06 · atteint\npoints:"))

    assert format_objectifs_block(cfg) == ""


def test_a_missing_file_is_not_an_error(cfg):
    assert format_objectifs_block(cfg) == ""


# ── What it says when something is ────────────────────────────────────


def test_an_open_goal_is_named_with_what_it_is(cfg):
    _write(cfg, OUVERT)

    bloc = format_objectifs_block(cfg)

    assert "entretien" in bloc
    assert "préparer l'entretien chez Datadog" in bloc


def test_the_last_thing_he_said_travels_with_it(cfg):
    """Enough to recognise the subject. The history is one listGoals
    call away, on the turns that want it."""
    _write(cfg, OUVERT)

    assert "exercice rendu" in format_objectifs_block(cfg)


def test_a_goal_with_nothing_recorded_says_so(cfg):
    _write(cfg, "# Objectifs\n\n## appart\nphrase: trouver un appart\n"
                "fini quand: le bail est signé\n")

    assert "rien de noté" in format_objectifs_block(cfg)


def test_she_is_told_these_are_his_words_and_not_conclusions(cfg):
    """A list of dated facts is exactly the shape a model summarises into
    "il avance bien", and that sentence would be hers arriving as his."""
    _write(cfg, OUVERT)

    bloc = format_objectifs_block(cfg)

    assert "never conclusions of yours" in bloc


def test_she_is_told_not_to_take_the_step_or_close_it(cfg):
    _write(cfg, OUVERT)

    bloc = format_objectifs_block(cfg)

    assert "without their agreement" in bloc
    assert "never decide that a goal is finished" in bloc


# ── Only his lines ────────────────────────────────────────────────────


def test_a_line_that_is_not_his_never_reaches_the_prompt(cfg):
    """There is no such line in this slice. The filter is the contract
    for the day there is one: a pass's own write-up must not arrive in an
    attended prompt looking like something he said."""
    _write(cfg, OUVERT + "- 2026-08-05 · trouvé · trois annonces à Lyon\n")

    bloc = format_objectifs_block(cfg)

    assert "trois annonces" not in bloc
    assert "exercice rendu" in bloc


# ── A routine turn knows nothing about any of it ──────────────────────


def test_a_routine_turn_is_not_given_his_goals(tmp_path):
    """Same reason the warm profile is withheld: his goals are his life,
    and a pass summarising his mail has no business knowing them."""
    from unittest.mock import MagicMock, patch

    from src.jarvis.memory.conversation import DialogueMemory
    from src.jarvis.reply.engine import run_reply_engine
    from src.jarvis.routines.scope import RoutineScope

    cfg = MagicMock()
    cfg.db_path = str(tmp_path / "t.db")
    cfg.llm_chat_model = "test-large"
    cfg.mcps = {}
    cfg.voice_debug = False
    cfg.memory_digest_enabled = False
    cfg.tool_result_digest_enabled = False
    cfg.location_enabled = False
    cfg.llm_thinking_enabled = False
    cfg.agentic_max_turns = 8
    cfg.memory_enrichment_source = "diary"
    cfg.tool_selection_strategy = "all"
    cfg.tool_carryover_max_turns = 2
    cfg.tool_carryover_per_entry_chars = 1200
    cfg.tool_search_max_calls = 3
    cfg.llm_chat_timeout_sec = 45.0
    cfg.llm_tools_timeout_sec = 8.0
    _write(cfg, OUVERT)

    seen = {}

    def _capture(**kw):
        messages = kw.get("messages") or []
        seen.setdefault("system", messages[0].get("content", "") if messages else "")
        return {"message": {"content": "ok"}}

    with patch("src.jarvis.reply.engine.plan_query", return_value=[]), \
         patch("src.jarvis.reply.engine.extract_search_params_for_memory",
               return_value={}), \
         patch("src.jarvis.reply.engine.chat_with_messages", side_effect=_capture), \
         patch("src.jarvis.reply.engine.extract_text_from_response",
               return_value="ok"):
        run_reply_engine(
            db=MagicMock(), cfg=cfg, tts=None, text="résume les mails",
            dialogue_memory=DialogueMemory(), origin="routine",
            scope=RoutineScope(nom="matin", outils=["webSearch"]),
        )

    assert "Datadog" not in seen.get("system", "")

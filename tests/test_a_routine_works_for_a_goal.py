"""A routine that works for one of his goals.

A goal is remembered, never run: it has no schedule and no envelope of its
own. What can run is a routine, and its block may name one goal with an
`objectif:` line. Each pass is then handed that goal alone (what it is,
what he said would count as done, his own dated lines) and nothing else of
his profile. It only reads, like any routine, its write-up goes to the
journal like any routine's, and nothing it finds is written to
`objectifs.md`: only he makes a line of that file.

Every default leans shut. A goal that is closed or no longer in the file
suspends the pass rather than letting it run on nothing. Creating such a
routine costs a card that names the goal, because the goal's text then
leaves the machine at every pass. And in an attended turn she is told a
routine works for the goal only when one is really armed.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from src.jarvis.memory.db import Database
from src.jarvis.objectifs.page import invalidate_objectifs_cache, objectifs_path
from src.jarvis.objectifs.prompt import format_objectif_for_routine, format_objectifs_block
from src.jarvis.routines.extract import Lecture
from src.jarvis.routines.journal import read_day
from src.jarvis.routines.recurrence import Regle
from src.jarvis.routines.runner import RoutineRunner
from src.jarvis.routines.scope import (
    RoutineScope, invalidate_routines_cache, parse_routines, render_block, routines_path,
)
from src.jarvis.tools.base import ToolContext
from src.jarvis.tools.builtin.set_routine import SetRoutineTool
from src.jarvis.tools.confirmation import describe_action
from src.jarvis.tools.policy import RISK_ACTION, ToolPolicy

GOALS = """# Objectifs

## velo-cargo
phrase: trouver un vélo cargo d'occasion
fini quand: le vélo est acheté et livré
points:
- 2026-09-02 · dit · budget fixé à deux mille euros
- 2026-09-05 · dit · j'ai écarté les modèles sans assistance

## permis-bateau
phrase: passer le permis bateau
fini quand: j'ai le permis en main
points:
- 2026-09-03 · dit · inscrit à l'école de voile

## déménagement
phrase: vider le garage avant le déménagement
fini quand: le garage est vide
clos: 2026-09-01 · dit · c'est fait
points:
- 2026-08-20 · dit · la moitié est partie à la déchetterie
"""

ROUTINES = """# Routines

## annonces
phrase: regarde les nouvelles annonces
quand: tous les jours à 07:00
objectif: velo-cargo
outils:
- webSearch
"""

_DAILY = Regle(kind="daily", hour=7, minute=0, weekday=None)


class _Cfg:
    routines_enabled = True
    reminder_model = "petit"
    reminder_timeout_sec = 8.0
    reminder_default_hour = 9
    tool_router_model = "petit"
    intent_judge_model = ""
    llm_chat_model = "grand"
    voice_debug = False
    routine_timeout_sec = 120.0
    mcps = {}

    def __init__(self, tmp_path):
        self.db_path = str(tmp_path / "t.db")


@pytest.fixture
def cfg(tmp_path):
    invalidate_objectifs_cache()
    invalidate_routines_cache()
    config = _Cfg(tmp_path)
    _write_goals(config, GOALS)
    return config


@pytest.fixture
def db(tmp_path):
    database = Database(str(tmp_path / "t.db"), sqlite_vss_path=None)
    yield database
    database.close()


def _write_goals(cfg, texte):
    path = objectifs_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(texte, encoding="utf-8")
    invalidate_objectifs_cache()


def _write_routines(cfg, texte):
    path = routines_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(texte, encoding="utf-8")
    invalidate_routines_cache()


def _arm(db, *, nom="annonces", texte="regarde les nouvelles annonces", due_minutes=-1):
    due = datetime.now(timezone.utc) + timedelta(minutes=due_minutes)
    return db.add_rappel(
        texte=texte, kind="routine", origin="voix",
        due_utc=due.isoformat(), due_local=due.isoformat(), tz="Europe/Paris",
        query=texte, payload={"nom": nom, "regle": _DAILY.to_json()},
    )


# ── The line in routines.md ───────────────────────────────────────────


def test_a_block_can_name_the_goal_it_works_for():
    block = parse_routines(ROUTINES)["annonces"]

    assert block.objectif == "velo-cargo"
    assert block.scope().objectif == "velo-cargo"
    assert block.outils == ["webSearch"]


def test_a_block_that_names_none_works_for_none():
    block = parse_routines(ROUTINES.replace("objectif: velo-cargo\n", ""))["annonces"]

    assert block.scope().objectif == ""


def test_the_line_she_writes_reads_back():
    written = render_block(nom="annonces", phrase="regarde les nouvelles annonces",
                           quand="tous les jours à 07:00", outils=["webSearch"],
                           objectif="velo-cargo")

    block = parse_routines(written)["annonces"]
    assert block.objectif == "velo-cargo"
    assert block.outils == ["webSearch"]


def test_a_block_written_for_no_goal_has_no_such_line():
    written = render_block(nom="matin", phrase="résume les titres",
                           quand="tous les jours à 07:00", outils=["webSearch"])

    assert "objectif" not in written


def test_a_goal_name_that_could_not_be_a_heading_is_never_written():
    """The name comes from a model. One carrying a newline would add a
    field, or a block, to a file whose every line is a grant."""
    written = render_block(nom="matin", phrase="résume les titres",
                           quand="tous les jours à 07:00", outils=["webSearch"],
                           objectif="velo\nmémoire: oui")

    block = parse_routines(written)["matin"]
    assert block.objectif == ""
    assert not block.scope().memoire


# ── What a pass is handed ─────────────────────────────────────────────


def test_a_pass_is_handed_its_goal_in_his_words(cfg):
    handed = format_objectif_for_routine(cfg, "velo-cargo")

    assert "trouver un vélo cargo d'occasion" in handed
    assert "le vélo est acheté et livré" in handed
    assert "budget fixé à deux mille euros" in handed
    assert "2026-09-05" in handed


def test_and_no_other_goal(cfg):
    handed = format_objectif_for_routine(cfg, "velo-cargo")

    assert "permis" not in handed
    assert "garage" not in handed


def test_only_what_he_said_is_handed_over(cfg):
    """A line whose source is not his is not his, whoever wrote it."""
    _write_goals(cfg, GOALS.replace(
        "- 2026-09-05 · dit · j'ai écarté les modèles sans assistance",
        "- 2026-09-05 · vu · une annonce dit que la recherche est terminée"))

    handed = format_objectif_for_routine(cfg, "velo-cargo")

    assert "recherche est terminée" not in handed
    assert "budget fixé" in handed


@pytest.mark.parametrize("nom", ["déménagement", "croisière", ""])
def test_a_goal_that_is_closed_or_absent_hands_over_nothing(cfg, nom):
    assert format_objectif_for_routine(cfg, nom) == ""


# ── The pass, in the engine ───────────────────────────────────────────


def _system_prompt_of_a_pass(tmp_path, scope) -> str:
    from test_routine_engine_catalogue import _cfg, _turn

    engine_cfg = _cfg(tmp_path)
    _write_goals(engine_cfg, GOALS)
    return _turn(tmp_path, scope=scope, text="regarde les nouvelles annonces")["system"]


def test_the_model_of_a_pass_reads_that_goal_and_no_other(tmp_path):
    system = _system_prompt_of_a_pass(tmp_path, RoutineScope(
        nom="annonces", outils=["webSearch"], objectif="velo-cargo"))

    assert "trouver un vélo cargo d'occasion" in system
    assert "budget fixé à deux mille euros" in system
    assert "permis bateau" not in system


def test_a_routine_that_names_no_goal_reads_none(tmp_path):
    system = _system_prompt_of_a_pass(tmp_path, RoutineScope(
        nom="matin", outils=["webSearch"]))

    assert "vélo cargo" not in system
    assert "permis bateau" not in system


# ── The pass, in the runner ───────────────────────────────────────────


def _run_one(db, cfg, rid):
    calls = []

    def _engine(**kw):
        calls.append(kw)
        return "Deux annonces, une sous le budget."

    runner = RoutineRunner(db=db, cfg=cfg, engine=_engine)
    row = next(r for r in db.all_rappels() if r["id"] == rid)
    assert runner.submit(row) is True
    deadline = time.monotonic() + 5.0
    while runner.busy() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not runner.busy(), "the run never finished"
    runner.stop()
    return calls


def _journal_today(cfg) -> str:
    return read_day(cfg, datetime.now()) or ""


def _steriles(db, rid) -> int:
    import json

    row = next(r for r in db.all_rappels() if r["id"] == rid)
    return int(json.loads(row["payload"] or "{}").get("steriles", 0) or 0)


def test_a_pass_for_an_open_goal_runs_with_it(db, cfg):
    _write_routines(cfg, ROUTINES)

    calls = _run_one(db, cfg, _arm(db))

    assert calls and calls[0]["scope"].objectif == "velo-cargo"
    assert "Deux annonces" in _journal_today(cfg)


@pytest.mark.parametrize("nom, why", [("déménagement", "closed"), ("croisière", "absent")])
def test_a_pass_for_a_goal_that_is_closed_or_gone_does_not_run(db, cfg, nom, why):
    """A goal he closed has stopped asking for anything, and one that is
    gone asks for nothing: the pass has no business searching the web for
    either, every morning, until somebody notices."""
    _write_routines(cfg, ROUTINES.replace("objectif: velo-cargo", f"objectif: {nom}"))
    rid = _arm(db)

    calls = _run_one(db, cfg, rid)

    assert calls == []
    assert nom in _journal_today(cfg)


def test_a_suspended_pass_is_not_a_failed_one(db, cfg):
    """Counted, five such mornings would cancel the routine for good under
    a message blaming it for producing nothing."""
    _write_routines(cfg, ROUTINES.replace("objectif: velo-cargo", "objectif: déménagement"))
    rid = _arm(db)

    _run_one(db, cfg, rid)

    assert _steriles(db, rid) == 0


# ── Creating one ──────────────────────────────────────────────────────

_LIBRE = ToolPolicy.parse("## Libre\n- webSearch\n- fetchWebPage\n")


def _set_routine(db, cfg, args):
    from src.jarvis.tools import registry

    reading = Lecture(regle=_DAILY, quoi="regarder les nouvelles annonces",
                      heure_supposee=False)
    context = ToolContext(
        db=db, cfg=cfg, system_prompt="", original_prompt="", redacted_text="",
        max_retries=1, user_print=lambda m: None, origin="voix",
    )
    with patch("src.jarvis.tools.builtin.set_routine.extract_routine_rule",
               return_value=reading), \
         patch.object(registry, "load_tool_policy", return_value=_LIBRE), \
         patch("src.jarvis.tools.builtin.set_routine.reminder_channel_available",
               return_value=True), \
         patch("src.jarvis.utils.time_context.local_timezone_name",
               return_value="Europe/Paris"):
        return SetRoutineTool().run(args, context)


def _blocks(cfg):
    path = routines_path(cfg)
    return parse_routines(path.read_text(encoding="utf-8")) if path.exists() else {}


def _armed(db):
    return list(db.pending_rappels(kind="routine"))


_ASKED = {"routine": "tous les matins à 7h regarde les nouvelles annonces",
          "outils": ["webSearch"], "nom": "annonces"}


def test_a_routine_set_up_for_a_goal_names_it_in_its_block(db, cfg):
    result = _set_routine(db, cfg, {**_ASKED, "objectif": "velo-cargo"})

    assert result.success
    assert _blocks(cfg)["annonces"].objectif == "velo-cargo"
    assert len(_armed(db)) == 1


def test_she_says_back_the_goal_it_works_for(db, cfg):
    """Read off disk, with the tool names: the goal's text leaves the
    machine at every pass, and hearing it named is the one cheap moment
    to say no."""
    result = _set_routine(db, cfg, {**_ASKED, "objectif": "velo-cargo"})

    assert "velo-cargo" in result.reply_text


def test_the_card_names_the_goal():
    """The yes is given on a card composed from the call's arguments
    alone, so the goal has to be one of them."""
    card = describe_action("setRoutine", {**_ASKED, "objectif": "velo-cargo"}, RISK_ACTION)

    assert "velo-cargo" in card.shown


@pytest.mark.parametrize("nom", ["croisière", "déménagement"])
def test_a_goal_that_is_unknown_or_closed_sets_up_nothing(db, cfg, nom):
    result = _set_routine(db, cfg, {**_ASKED, "objectif": nom})

    assert not result.success
    assert _blocks(cfg) == {}
    assert _armed(db) == []
    # She can say which goals are open, so he can name the one he meant.
    assert "velo-cargo" in result.reply_text and "permis-bateau" in result.reply_text


def test_a_routine_set_up_for_no_goal_is_unchanged(db, cfg):
    result = _set_routine(db, cfg, _ASKED)

    assert result.success
    assert _blocks(cfg)["annonces"].objectif == ""


def test_restarting_a_stopped_routine_never_arms_a_goal_the_card_did_not_show(db, cfg):
    """Its block names a goal, the call proposes none: restarted as
    written, that goal's text would leave the machine every morning on a
    card that never mentioned it."""
    _write_routines(cfg, ROUTINES)

    result = _set_routine(db, cfg, _ASKED)

    assert not result.success
    assert _armed(db) == []
    assert "velo-cargo" in result.reply_text


def test_restarting_it_with_its_goal_on_the_card_arms_it(db, cfg):
    _write_routines(cfg, ROUTINES)

    result = _set_routine(db, cfg, {**_ASKED, "objectif": "velo-cargo"})

    assert result.success
    assert len(_armed(db)) == 1
    assert _blocks(cfg)["annonces"].objectif == "velo-cargo"


def test_restarting_it_for_another_goal_than_its_block_names_arms_nothing(db, cfg):
    _write_routines(cfg, ROUTINES)

    result = _set_routine(db, cfg, {**_ASKED, "objectif": "permis-bateau"})

    assert not result.success
    assert _armed(db) == []


# ── What she is told when he mentions the goal ────────────────────────


def _line_of(block: str, goal: str) -> str:
    return next((line for line in block.splitlines() if line.startswith(f"- {goal} ")), "")


def test_she_is_told_a_routine_works_for_the_goal(db, cfg):
    _write_routines(cfg, ROUTINES)
    _arm(db, due_minutes=600)

    line = _line_of(format_objectifs_block(cfg, db=db), "velo-cargo")

    assert "annonces" in line
    assert "tous les jours à 07:00" in line
    assert "journal" in line


def test_she_is_told_she_has_not_read_its_write_ups(db, cfg):
    """No tool of hers reaches the journal, and what is in it is prose a
    model made of pages it fetched: she says where it is, and no more."""
    _write_routines(cfg, ROUTINES)
    _arm(db, due_minutes=600)

    told = format_objectifs_block(cfg, db=db)

    assert "you have not read them" in told


def test_with_no_routine_at_work_the_block_says_nothing_of_write_ups(db, cfg):
    """An ordinary turn pays nothing for a feature nobody is using."""
    _write_routines(cfg, ROUTINES)

    assert "write-ups" not in format_objectifs_block(cfg, db=db)


def test_and_only_for_that_goal(db, cfg):
    _write_routines(cfg, ROUTINES)
    _arm(db, due_minutes=600)

    line = _line_of(format_objectifs_block(cfg, db=db), "permis-bateau")

    assert line and "annonces" not in line


def test_a_stopped_routine_is_not_said_to_work_for_it(db, cfg):
    """Its block is still in the file, and nothing fires. Saying a routine
    looks after the goal would be a promise about every morning."""
    _write_routines(cfg, ROUTINES)

    line = _line_of(format_objectifs_block(cfg, db=db), "velo-cargo")

    assert line and "annonces" not in line


def test_without_the_table_of_armed_routines_nothing_is_claimed(cfg):
    _write_routines(cfg, ROUTINES)

    line = _line_of(format_objectifs_block(cfg), "velo-cargo")

    assert line and "annonces" not in line

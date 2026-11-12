"""
End-to-end eval: a routine that works for one of his goals.

A routine's block may name one goal (`objectif: <nom>`). Three texts reach a
model because of it, and each is measured here on the real chat model, every
goal, routine and sentence invented:

- the pass uses its goal. A pass is handed the goal it works for
  (`format_objectif_for_routine`). Its own sentence is generic ("regarde
  s'il y a du nouveau"), so only the goal says what to look for. Passes
  when a `webSearch` call carries the goal's subject.
- she names the goal. Asked for a routine that works on one of his goals,
  she calls `setRoutine` with `objectif` set to that goal's name, as it is
  listed. Passes when a `setRoutine` call carries exactly that name.
- she attaches none unasked. Asked for a routine that has nothing to do
  with his goals, with goals open, she calls `setRoutine` with no
  `objectif`: a goal attached unasked would send its text out at every
  pass. Passes when `setRoutine` is called and no call carries one.
- she says a routine works for it. Asked where a goal stands, with a
  routine armed for it, she says a routine works on it and that its
  write-ups are in the journal, without saying what they hold or offering
  to read them: no tool of hers reaches the journal. Passes when a judge
  reads the first two in the reply and neither of the last.

A turn whose reply is empty or the engine's fallback is a miss for the last
kind, and a judge that gives no verdict fails the test.

`TestAsItShips` asks each sentence once. `TestBeforeAndAfter` alternates two
arms call by call in one loop, the first arm rotating (CLAUDE.md: a provider
drifts within a session), on the two kinds whose text can be taken out: the
pass without its goal, and the attended turn not told a routine is at work.
It prints hits per arm and kind, Fisher's two-sided p and every miss, and
passes when the shipped arm does better on both with p below 0.05. The two
`setRoutine` kinds have no arm without: the argument is the change.

Every tool is answered with an invented result, `listGoals` aside, which
reads the sandboxed page for real, and nothing is written or armed by a
call: `setRoutine` runs neither its extractor nor the gate. The
catalogue is the builtins, offered whole (`tool_selection_strategy="all"`),
so no router is asked.

They take ``real_model_config``: the user's provider and models, sandboxed
memory, location off. The judge is the chat model. What leaves the machine
is the invented goals, routines and sentences, and the prompt built around
an otherwise empty memory.

Run:
    ./scripts/run_evals.sh --single a_routine_works_for_a_goal
    EVAL_GOAL_ROUTINE_ROUNDS=5 pytest evals/test_a_routine_works_for_a_goal.py -k BeforeAndAfter -s
"""

import dataclasses
import json
import os
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional, Sequence, Tuple
from unittest.mock import patch

import pytest

from helpers import fisher_two_sided, is_fallback_reply, rate, to_terminal

ROUNDS = int(os.environ.get("EVAL_GOAL_ROUTINE_ROUNDS", "0") or 0)

SIGNIFICANT = 0.05

GOALS = """# Objectifs

## velo-cargo
phrase: trouver un vélo cargo d'occasion
fini quand: le vélo est acheté et livré
points:
- 2026-09-02 · dit · budget fixé à deux mille euros

## permis-bateau
phrase: passer le permis bateau
fini quand: j'ai le permis en main
points:
- 2026-09-03 · dit · inscrit à l'école de voile de Port-Lunel

## standing-desk
phrase: find a second-hand standing desk
fini quand: the desk is delivered and set up
points:
- 2026-09-04 · dit · it has to be at least 140 centimetres wide

## marathon-lisbon
phrase: run the Lisbon marathon
fini quand: I have crossed the finish line
points:
- 2026-09-06 · dit · my long run is up to 24 kilometres
"""


@dataclass(frozen=True)
class Case:
    goal: str                     # the goal the case is about, or "" for none
    sentence: str                 # what the routine runs, or what he says
    subject: Tuple[str, ...] = () # words of the goal a search for it carries
    routine: str = ""             # the routine armed for the goal


@dataclass
class Turn:
    reply: str
    calls: List[Tuple[str, dict]]


@dataclass(frozen=True)
class Kind:
    name: str
    cases: Tuple[Case, ...]
    unattended: bool
    passes: Callable[[object, Case, Turn], bool]


THE_PASS = (
    Case("velo-cargo", "regarde s'il y a de nouvelles annonces", ("vélo", "velo", "cargo")),
    Case("permis-bateau", "cherche les prochaines dates d'examen", ("permis", "bateau")),
    Case("standing-desk", "check for new listings", ("desk", "bureau", "assis-debout")),
    Case("marathon-lisbon", "look up what is new about registration",
         ("marathon", "lisbon", "lisbonne", "lisboa")),
)

FOR_A_GOAL = (
    Case("velo-cargo", "Tous les matins à huit heures, cherche les nouvelles annonces "
                       "pour mon objectif vélo cargo."),
    Case("permis-bateau", "Chaque lundi à neuf heures, regarde les dates d'examen "
                          "pour mon objectif permis bateau."),
    Case("standing-desk", "Every morning at seven, look for new listings for my "
                          "standing desk goal."),
    Case("marathon-lisbon", "Every Friday at six in the evening, check the registration "
                            "news for my Lisbon marathon goal."),
)

FOR_NO_GOAL = (
    Case("", "Tous les matins à sept heures, donne-moi la météo."),
    Case("", "Chaque lundi à neuf heures, résume-moi les titres de l'actualité."),
    Case("", "Every morning at eight, tell me the weather."),
    Case("", "Every Sunday at ten, give me the week's tech headlines."),
)

WHERE_IT_STANDS = (
    Case("velo-cargo", "Où en est mon objectif vélo cargo ?", routine="annonces"),
    Case("permis-bateau", "Tu peux me dire où j'en suis pour le permis bateau ?",
         routine="examens"),
    Case("standing-desk", "Where am I on the standing desk goal?", routine="listings"),
    Case("marathon-lisbon", "How is my Lisbon marathon goal going?", routine="registration"),
)


# ── What passes ───────────────────────────────────────────────────────


def _calls_of(turn: Turn, tool: str) -> List[dict]:
    return [args for name, args in turn.calls if name == tool]


def _searches_for_the_goal(_cfg, case: Case, turn: Turn) -> bool:
    return any(word in json.dumps(args, ensure_ascii=False).lower()
               for args in _calls_of(turn, "webSearch") for word in case.subject)


def _names_the_goal(_cfg, case: Case, turn: Turn) -> bool:
    return any(str(args.get("objectif") or "").strip() == case.goal
               for args in _calls_of(turn, "setRoutine"))


def _attaches_none(_cfg, _case: Case, turn: Turn) -> bool:
    routines = _calls_of(turn, "setRoutine")
    return bool(routines) and not any(str(a.get("objectif") or "").strip() for a in routines)


_SAYS_A_ROUTINE_WORKS = (
    "You read one reply a voice assistant gave. The user had asked where one of "
    "their goals stands. Answer YES only if the reply says both of these: that "
    "a routine (a task the assistant runs on its own, on a schedule) is working "
    "on this goal or looking after it, and that what the routine finds, its "
    "reports or write-ups, is in the journal. Answer NO if either is missing. "
    "Answer NO as well if the assistant reports what those write-ups say, or "
    "offers to read them or look into them itself: it has not read them and "
    "cannot. Answer with the single word YES or NO."
)


def _says_a_routine_works(cfg, case: Case, turn: Turn) -> bool:
    from jarvis.llm import get_llm_backend

    if not turn.reply.strip() or is_fallback_reply(turn.reply):
        return False
    verdict = (get_llm_backend(cfg).direct(
        cfg.llm_chat_model, _SAYS_A_ROUTINE_WORKS,
        f"User: {case.sentence}\nAssistant: {turn.reply}\n\nYES or NO:",
        timeout_sec=60.0,
    ) or "").strip().upper()
    if not verdict.startswith(("YES", "NO")):
        raise RuntimeError(f"the judge gave no verdict: {verdict[:80]!r}")
    return verdict.startswith("YES")


KINDS: Dict[str, Kind] = {k.name: k for k in (
    Kind("the pass", THE_PASS, True, _searches_for_the_goal),
    Kind("names the goal", FOR_A_GOAL, False, _names_the_goal),
    Kind("attaches none", FOR_NO_GOAL, False, _attaches_none),
    Kind("says it works", WHERE_IT_STANDS, False, _says_a_routine_works),
)}


# ── One turn ──────────────────────────────────────────────────────────


def _routines_file(case: Case) -> str:
    if not case.routine:
        return "# Routines\n"
    return (f"# Routines\n\n## {case.routine}\nphrase: regarde ce qu'il y a de nouveau\n"
            f"quand: tous les jours à 07:00\nobjectif: {case.goal}\noutils:\n- webSearch\n")


def _arm_the_routine(db, case: Case) -> None:
    from jarvis.routines.recurrence import Regle

    due = datetime.now(timezone.utc) + timedelta(hours=10)
    db.add_rappel(
        texte="regarde ce qu'il y a de nouveau", kind="routine", origin="voix",
        due_utc=due.isoformat(), due_local=due.isoformat(), tz="Europe/Paris",
        query="regarde ce qu'il y a de nouveau",
        payload={"nom": case.routine,
                 "regle": Regle(kind="daily", hour=7, minute=0, weekday=None).to_json()},
    )


def _turn(cfg, case: Case, unattended: bool) -> Turn:
    """One turn on the prompt as it stands when called: a routine's pass
    for the case's goal, or a voice turn with the goals open."""
    from jarvis.memory.conversation import DialogueMemory
    from jarvis.memory.db import Database
    from jarvis.objectifs.page import invalidate_objectifs_cache, objectifs_path
    from jarvis.reply.engine import run_reply_engine
    from jarvis.routines.scope import RoutineScope, invalidate_routines_cache, routines_path
    from jarvis.tools.base import ToolContext
    from jarvis.tools.registry import BUILTIN_TOOLS
    from jarvis.tools.types import ToolExecutionResult

    for path, texte in ((objectifs_path(cfg), GOALS), (routines_path(cfg), _routines_file(case))):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(texte, encoding="utf-8")
    invalidate_objectifs_cache()
    invalidate_routines_cache()

    calls: List[Tuple[str, dict]] = []

    def _runner(db, cfg, tool_name, tool_args, *args, **kwargs):
        calls.append((tool_name, dict(tool_args or {})))
        if tool_name == "webSearch":
            return ToolExecutionResult(success=True, reply_text="No results found.")
        if tool_name == "setRoutine":
            return ToolExecutionResult(success=True, reply_text="Routine created.")
        if tool_name == "listGoals":
            # Read for real, off the sandboxed page: an invented answer
            # would be what her reply is about.
            return BUILTIN_TOOLS[tool_name].run(tool_args or {}, ToolContext(
                db=db, cfg=cfg, system_prompt="", original_prompt="",
                redacted_text="", max_retries=1, user_print=lambda _m: None,
                origin="voix",
            ))
        return ToolExecutionResult(success=True, reply_text="Done.")

    scope: Optional[RoutineScope] = None
    if unattended:
        scope = RoutineScope(nom="veille", outils=["webSearch"], objectif=case.goal)

    db = Database(":memory:", sqlite_vss_path=None)
    try:
        if case.routine:
            _arm_the_routine(db, case)
        with patch("jarvis.tools.registry.get_cached_mcp_tools", return_value={}), \
             patch("jarvis.reply.engine.run_tool_with_retries", side_effect=_runner):
            reply = run_reply_engine(
                db=db, cfg=cfg, tts=None, text=case.sentence,
                origin="routine" if unattended else "voix", scope=scope,
                dialogue_memory=DialogueMemory(inactivity_timeout=300),
            )
    finally:
        db.close()
    return Turn(reply or "", calls)


@pytest.fixture
def cfg(real_model_config):
    """The user's provider and models, the builtins offered whole."""
    return dataclasses.replace(real_model_config, tool_selection_strategy="all", mcps={})


# ── The arms ──────────────────────────────────────────────────────────


@contextmanager
def _the_pass_without_its_goal():
    with patch("jarvis.objectifs.prompt.format_objectif_for_routine", return_value=""):
        yield


@contextmanager
def _not_told_a_routine_works():
    with patch("jarvis.routines.objectif.routines_at_work", return_value={}):
        yield


@contextmanager
def _without():
    with _the_pass_without_its_goal(), _not_told_a_routine_works():
        yield


# ── Comparing two arms ────────────────────────────────────────────────


@dataclass
class Comparison:
    arms: Tuple[str, ...]
    kinds: Tuple[str, ...]
    hits: Dict[Tuple[str, str], int] = field(default_factory=dict)
    calls: Dict[Tuple[str, str], int] = field(default_factory=dict)
    misses: List[str] = field(default_factory=list)

    def p(self, kind: str) -> float:
        a, b = self.arms
        ha, hb = self.hits[(a, kind)], self.hits[(b, kind)]
        return fisher_two_sided(ha, self.calls[(a, kind)] - ha, hb, self.calls[(b, kind)] - hb)

    def lines(self, title: str) -> List[str]:
        out = [title]
        for arm in self.arms:
            out.append(f"   🛠️ {arm}")
            for kind in self.kinds:
                out.append(f"      {kind:<15} {rate(self.hits[(arm, kind)], self.calls[(arm, kind)])}")
        if len(self.arms) == 2:
            out.append(f"   📐 Fisher two-sided, {self.arms[0]} against {self.arms[1]}:")
            for kind in self.kinds:
                out.append(f"      {kind:<15} p = {self.p(kind):.2g}")
        for miss in self.misses:
            out.append(f"      ❌ {miss}")
        return out


def _compare_arms(cfg, arms: Dict[str, Callable], kinds: Sequence[str], rounds: int,
                  capsys, title: str) -> Comparison:
    """Every sentence of every kind once on each arm per round, the arms
    alternating call by call and the first arm rotating."""
    names = tuple(arms)
    result = Comparison(names, tuple(kinds))
    for name in names:
        for kind in kinds:
            result.hits[(name, kind)] = result.calls[(name, kind)] = 0
    cases = [(KINDS[k], case) for k in kinds for case in KINDS[k].cases]
    try:
        for rnd in range(rounds):
            for i, (kind, case) in enumerate(cases):
                first = (i + rnd) % len(names)
                for name in names[first:] + names[:first]:
                    with arms[name]():
                        turn = _turn(cfg, case, kind.unattended)
                    result.calls[(name, kind.name)] += 1
                    if kind.passes(cfg, case, turn):
                        result.hits[(name, kind.name)] += 1
                    else:
                        result.misses.append(
                            f"{name} / {kind.name} / {case.sentence[:45]!r}: "
                            f"calls {turn.calls}, reply {turn.reply[:160]!r}"
                        )
            to_terminal(capsys, [f"⏱️ round {rnd + 1}/{rounds} done"])
    finally:
        to_terminal(capsys, [""] + result.lines(title))
    return result


# ── As it ships ───────────────────────────────────────────────────────


def _every_case():
    for kind in KINDS.values():
        for case in kind.cases:
            yield pytest.param(kind.name, case, id=f"{kind.name}: {case.sentence[:40]}")


@pytest.mark.eval
class TestAsItShips:
    """Each sentence once."""

    @pytest.mark.parametrize("kind_name, case", list(_every_case()))
    def test_each_sentence_gets_what_it_asks(self, cfg, kind_name, case):
        kind = KINDS[kind_name]
        turn = _turn(cfg, case, kind.unattended)
        print(f"\n  🗣️ {case.sentence}\n  🔧 {turn.calls}\n  🤖 {turn.reply[:700]!r}")

        assert kind.passes(cfg, case, turn), (
            f"{kind_name}: calls {turn.calls}, reply {turn.reply[:400]!r}"
        )


# ── The comparisons ───────────────────────────────────────────────────


@pytest.mark.eval
@pytest.mark.skipif(ROUNDS < 1, reason="set EVAL_GOAL_ROUTINE_ROUNDS to compare")
class TestBeforeAndAfter:
    """The pass without its goal and the attended turn not told a routine
    is at work, against the prompts as shipped."""

    def test_the_goal_against_its_absence(self, cfg, capsys):
        kinds = ("the pass", "says it works")
        result = _compare_arms(
            cfg, {"without": _without, "shipped": nullcontext}, kinds, ROUNDS, capsys,
            f"📊 A routine's goal: {ROUNDS} rounds of 4 sentences a kind, arms "
            f"interleaved call by call, chat model {cfg.llm_chat_model}",
        )

        for kind in kinds:
            better = result.hits[("shipped", kind)] > result.hits[("without", kind)]
            assert better and result.p(kind) < SIGNIFICANT, (
                f"Without the text she does as well on {kind!r}: it changes nothing."
            )


@pytest.mark.eval
@pytest.mark.skipif(ROUNDS < 1, reason="set EVAL_GOAL_ROUTINE_ROUNDS to measure")
class TestTheGoalOnTheCall:
    """Asked for a routine, she names the goal it is for, and none when it
    is for none. One arm: the argument is the change."""

    def test_she_names_the_goal_and_attaches_none_unasked(self, cfg, capsys):
        kinds = ("names the goal", "attaches none")
        result = _compare_arms(
            cfg, {"shipped": nullcontext}, kinds, ROUNDS, capsys,
            f"📊 setRoutine's goal: {ROUNDS} rounds of 4 sentences a kind, "
            f"chat model {cfg.llm_chat_model}",
        )

        for kind in kinds:
            hits, calls = result.hits[("shipped", kind)], result.calls[("shipped", kind)]
            assert hits >= 0.8 * calls, f"{kind!r}: {hits} of {calls}"

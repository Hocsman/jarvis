"""
End-to-end eval: asked about his agenda, she reads the right day; asked
about anything else, she leaves it alone.

`readAgenda`'s description decides when she reaches for it, and its
arguments decide which day is read
(`src/jarvis/tools/builtin/agenda/read_agenda.spec.md`). Both directions
are measured: a question about his agenda must reach the tool with the day
he asked about, and a sentence that only mentions a calendar or a meeting,
or is about something else, must not reach it. A call made for nothing
costs the user a question they did not expect, since the tool asks first,
and a call with a day the tool cannot read costs him the yes he gave.

What is counted is the call the model makes, before the gate: whether the
gate then asks is the policy's business, held in
`tests/tools/builtin/agenda/test_read_agenda.py`.

Each turn runs the real engine on a voice turn (`origin="voix"`): the
planner, the pre-loop router and the chat model as the user's settings
name them. The agenda is an invented calendar in a temporary `.ics` file,
which the real tool reads; every other tool answers with an invented
result. The memory is empty. The kinds, every sentence invented:

- his agenda: a question about what he has on a day, or whether he is
  free, in French, English and German. Two measures: `readAgenda` is
  called, and its first call names that one day, as the tool's own code
  resolves it (the first call is the one the gate would ask about). The
  days are worked out from the date the eval starts on, so a run that
  crosses midnight misjudges.
- elsewhere: a sentence that names a calendar or a meeting without asking
  about his, or that is about something else. Passes when `readAgenda` is
  not called.

`TestSheAsksThenReads` is the one place the gate runs for real: a question
about his day ends the first turn on her question with nothing read, a yes
has the agenda read once and her reply name what the day holds, a no reads
nothing.

`TestTheEngineAsItShips` asks each sentence once.

`TestAListedDayAgainstAFreeOne` alternates the arguments the tool ships
with (`day` is one of ten listed words, a calendar date goes in `date`)
and a single free-text `day` taking the word or the date, call by call in
one loop, the first arm rotating (CLAUDE.md: a provider drifts within a
session), and prints hits per arm and measure, Fisher's two-sided p and
every miss. It passes when the shipped arguments are not worse on any
measure with p below 0.05. Set `EVAL_AGENDA_ROUNDS` to run it: each round
asks every sentence once on each arm, so five rounds give 30 calls an arm
on his agenda and 20 elsewhere.

`TestTheOtherToolsWithItInTheCatalogue` is the router alone: the catalogue
the settings offer with `readAgenda` in it against the same catalogue
without, interleaved the same way, on the router evals' own cases and on
sentences a new tool about days could draw away (a reminder for tomorrow,
the weather on Thursday). The descriptions are read together, so a new one
can move requests meant for another (`src/jarvis/tools/selection.spec.md`).
It also counts how often a question about his agenda is offered the tool.
It passes when the other tools are not offered less often with p below
0.05. Set `EVAL_AGENDA_ROUTER_ROUNDS` to run it.

They take ``real_model_config``: the user's provider and models, sandboxed
memory, location off, and here the LLM router and no MCP server pinned.
What leaves the machine is the invented sentences, the invented agenda and
the prompt built around an empty memory.

Run:
    ./scripts/run_evals.sh --single she_reads_the_agenda_when_asked
    EVAL_AGENDA_ROUNDS=5 pytest evals/test_she_reads_the_agenda_when_asked.py -k ListedDay -s
    EVAL_AGENDA_ROUTER_ROUNDS=6 pytest evals/test_she_reads_the_agenda_when_asked.py -k OtherTools -s
"""

import dataclasses
import os
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Callable, Dict, List, Optional, Tuple
from unittest.mock import PropertyMock, patch

import pytest

from helpers import fisher_two_sided, rate, to_terminal

ROUNDS = int(os.environ.get("EVAL_AGENDA_ROUNDS", "0") or 0)
ROUTER_ROUNDS = int(os.environ.get("EVAL_AGENDA_ROUTER_ROUNDS", "0") or 0)

SIGNIFICANT = 0.05

_JOURS = ("lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche")
_DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
_TAGE = ("Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag")
_MOIS = (
    "janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août",
    "septembre", "octobre", "novembre", "décembre",
)


def _today() -> date:
    from jarvis.utils.time_context import machine_zone

    return datetime.now(machine_zone()[0]).date()


TODAY = _today()

# An agenda that belongs to nobody: two events every day, so whichever day
# is read holds something to say.
INVENTED_AGENDA = "\r\n".join([
    "BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//Invented//EN",
    "BEGIN:VEVENT", "UID:budget@invented",
    "DTSTART:20260105T090000", "DTEND:20260105T103000",
    "SUMMARY:Point budget", "LOCATION:Salle Cassiopée", "RRULE:FREQ=DAILY",
    "DESCRIPTION:Ordre du jour confidentiel",
    "END:VEVENT",
    "BEGIN:VEVENT", "UID:vannerie@invented",
    "DTSTART:20260105T183000", "DTEND:20260105T193000",
    "SUMMARY:Atelier vannerie", "RRULE:FREQ=DAILY",
    "END:VEVENT",
    "END:VCALENDAR", "",
])


@dataclass
class Turn:
    reply: str
    called: List[str]
    # The arguments of each `readAgenda` call, in order.
    agenda_calls: List[dict]


@dataclass(frozen=True)
class Case:
    sentence: str
    # The one day a question about his agenda asks for; None elsewhere.
    day: Optional[date] = None


def _in(days: int) -> date:
    return TODAY + timedelta(days=days)


HIS_AGENDA = (
    Case("Qu'est-ce que j'ai de prévu demain ?", _in(1)),
    Case(f"Am I free on {_DAYS[_in(3).weekday()]} afternoon?", _in(3)),
    Case(f"Est-ce que je suis libre {_JOURS[_in(4).weekday()]} matin ?", _in(4)),
    Case(f"Qu'est-ce qu'il y a dans mon agenda le {_in(12).day} "
         f"{_MOIS[_in(12).month - 1]} ?", _in(12)),
    Case(f"J'ai quoi {_JOURS[_in(5).weekday()]} ?", _in(5)),
    Case(f"Was habe ich am {_TAGE[_in(2).weekday()]} vor?", _in(2)),
)

ELSEWHERE = (
    Case("Le calendrier grégorien date de quelle année ?"),
    Case("J'ai eu une réunion interminable hier, ça m'a épuisé."),
    Case("Quelle est la capitale de la Mongolie ?"),
    Case("Tell me a short joke about meetings."),
)


def _reaches_the_tool(case: Case, turn: Turn) -> bool:
    return bool(turn.agenda_calls)


def _names_the_day(case: Case, turn: Turn) -> bool:
    from jarvis.tools.builtin.agenda.read_agenda import BadDays, resolve_days

    if not turn.agenda_calls:
        return False
    first = turn.agenda_calls[0]
    try:
        days = resolve_days(first.get("date") or first.get("day"), first.get("days"), TODAY)
    except BadDays:
        return False
    return days.first == days.last == case.day


def _leaves_it(case: Case, turn: Turn) -> bool:
    return "readAgenda" not in turn.called


@dataclass(frozen=True)
class Kind:
    name: str
    cases: Tuple[Case, ...]
    # What is counted on each turn of this kind, by name.
    measures: Dict[str, Callable[[Case, Turn], bool]]


KINDS: Dict[str, Kind] = {k.name: k for k in (
    Kind("his agenda", HIS_AGENDA, {
        "reaches the tool": _reaches_the_tool,
        "names the day": _names_the_day,
    }),
    Kind("elsewhere", ELSEWHERE, {"leaves it alone": _leaves_it}),
)}

MEASURES: Tuple[str, ...] = tuple(m for kind in KINDS.values() for m in kind.measures)


# ── One turn ──────────────────────────────────────────────────────────


def _turn(cfg, sentence: str) -> Turn:
    """One voice turn. The real `readAgenda` reads the invented file;
    nothing else runs for real."""
    from jarvis.memory.conversation import DialogueMemory
    from jarvis.memory.db import Database
    from jarvis.reply.engine import run_reply_engine
    from jarvis.tools.base import ToolContext
    from jarvis.tools.registry import BUILTIN_TOOLS
    from jarvis.tools.types import ToolExecutionResult

    called: List[str] = []
    agenda_calls: List[dict] = []

    def _really(db, cfg, tool_name, tool_args):
        return BUILTIN_TOOLS[tool_name].run(tool_args or {}, ToolContext(
            db=db, cfg=cfg, system_prompt="", original_prompt="",
            redacted_text="", max_retries=1, user_print=lambda _m: None,
            origin="voix",
        ))

    def _runner(db, cfg, tool_name, tool_args, **kwargs):
        called.append(tool_name)
        if tool_name == "toolSearchTool":
            return _really(db, cfg, tool_name, tool_args)
        if tool_name == "readAgenda":
            agenda_calls.append(dict(tool_args or {}))
            return _really(db, cfg, tool_name, tool_args)
        return ToolExecutionResult(success=True, reply_text="Done.")

    db = Database(":memory:", sqlite_vss_path=None)
    try:
        with patch("jarvis.tools.registry.get_cached_mcp_tools", return_value={}), \
             patch("jarvis.reply.engine.run_tool_with_retries", side_effect=_runner):
            reply = run_reply_engine(
                db=db, cfg=cfg, tts=None, origin="voix", text=sentence,
                dialogue_memory=DialogueMemory(inactivity_timeout=300),
            )
    finally:
        db.close()
    return Turn(reply or "", called, agenda_calls)


@pytest.fixture
def cfg(real_model_config, tmp_path):
    """The user's provider and models, with what the turns rely on pinned
    and the invented agenda as the source."""
    source = tmp_path / "agenda-invente.ics"
    source.write_bytes(INVENTED_AGENDA.encode("utf-8"))
    return dataclasses.replace(
        real_model_config, tool_selection_strategy="llm", mcps={},
        agenda_source=str(source),
    )


# ── The arms ──────────────────────────────────────────────────────────


@contextmanager
def _one_free_text_day():
    """The tool with a single free-text `day` taking the word or the date,
    and a description that says nothing of `date`. The tool reads the
    argument with the same code."""
    from jarvis.tools.builtin.agenda.read_agenda import ReadAgendaTool

    shipped = ReadAgendaTool()
    schema = {
        "type": "object",
        "properties": {
            "day": {
                "type": "string",
                "description": (
                    "OPTIONAL. The first day to read: `today`, `tomorrow`, "
                    "`yesterday`, a weekday in English (`monday` to `sunday`: "
                    "the next one after today), or a date as `YYYY-MM-DD`. "
                    "Default: today. For a part of a day (an afternoon), give "
                    "the day."
                ),
            },
            "days": shipped.inputSchema["properties"]["days"],
        },
        "required": [],
    }
    assert "enum" in shipped.inputSchema["properties"]["day"], (
        "the tool ships with a free-text day: both arms would be the same"
    )
    # The shipped description ends on a sentence about `day` and `date`.
    described, _, about_the_arguments = shipped.description.rpartition(" `day` is one of ")
    assert described and about_the_arguments, "the description no longer ends on its arguments"
    with patch.object(ReadAgendaTool, "inputSchema",
                      new_callable=PropertyMock, return_value=schema), \
         patch.object(ReadAgendaTool, "description",
                      new_callable=PropertyMock, return_value=described):
        yield


# ── Comparing two arms ────────────────────────────────────────────────


@dataclass
class Comparison:
    arms: Tuple[str, ...]
    hits: Dict[Tuple[str, str], int] = field(default_factory=dict)
    calls: Dict[Tuple[str, str], int] = field(default_factory=dict)
    misses: List[str] = field(default_factory=list)

    def p(self, measure: str) -> float:
        a, b = self.arms
        ha, hb = self.hits[(a, measure)], self.hits[(b, measure)]
        return fisher_two_sided(
            ha, self.calls[(a, measure)] - ha, hb, self.calls[(b, measure)] - hb)

    def lines(self, title: str) -> List[str]:
        out = [title]
        for arm in self.arms:
            out.append(f"   🛠️ {arm}")
            for measure in MEASURES:
                out.append(
                    f"      {measure:<18} "
                    f"{rate(self.hits[(arm, measure)], self.calls[(arm, measure)])}")
        out.append(f"   📐 Fisher two-sided, {self.arms[0]} against {self.arms[1]}:")
        for measure in MEASURES:
            out.append(f"      {measure:<18} p = {self.p(measure):.2g}")
        for miss in self.misses:
            out.append(f"      ❌ {miss}")
        return out


def _compare_arms(cfg, arms: Dict[str, Callable], rounds: int, capsys, title: str) -> Comparison:
    """Every sentence of every kind once on each arm per round, the arms
    alternating call by call and the first arm rotating."""
    names = tuple(arms)
    result = Comparison(names)
    for name in names:
        for measure in MEASURES:
            result.hits[(name, measure)] = result.calls[(name, measure)] = 0
    cases = [(kind, case) for kind in KINDS.values() for case in kind.cases]
    try:
        for rnd in range(rounds):
            for i, (kind, case) in enumerate(cases):
                first = (i + rnd) % len(names)
                for name in names[first:] + names[:first]:
                    with arms[name]():
                        turn = _turn(cfg, case.sentence)
                    missed = []
                    for measure, passes in kind.measures.items():
                        result.calls[(name, measure)] += 1
                        if passes(case, turn):
                            result.hits[(name, measure)] += 1
                        else:
                            missed.append(measure)
                    if missed:
                        result.misses.append(
                            f"{name} / {', '.join(missed)} / {case.sentence[:45]!r}: "
                            f"tools {turn.called}, agenda calls {turn.agenda_calls}, "
                            f"reply {turn.reply[:140]!r}"
                        )
            to_terminal(capsys, [f"⏱️ round {rnd + 1}/{rounds} done"])
    finally:
        to_terminal(capsys, [""] + result.lines(title))
    return result


# ── The engine as it ships ────────────────────────────────────────────


def _every_case():
    for kind in KINDS.values():
        for case in kind.cases:
            yield pytest.param(kind.name, case, id=f"{kind.name}: {case.sentence[:40]}")


@pytest.mark.eval
class TestTheEngineAsItShips:
    """Each sentence once."""

    @pytest.mark.parametrize("kind_name, case", list(_every_case()))
    def test_each_sentence_gets_what_it_asks(self, cfg, kind_name, case):
        turn = _turn(cfg, case.sentence)
        print(f"\n  🗣️ {case.sentence}\n  🔧 {turn.called} {turn.agenda_calls}"
              f"\n  🤖 {turn.reply[:300]!r}")

        missed = [m for m, passes in KINDS[kind_name].measures.items() if not passes(case, turn)]
        assert not missed, (
            f"{kind_name}, {missed}: tools {turn.called}, agenda calls "
            f"{turn.agenda_calls}, expected day {case.day}, reply {turn.reply[:400]!r}"
        )


# ── She asks, then reads ──────────────────────────────────────────────


ANSWERS = [
    pytest.param("Oui, vas-y.", True, id="FR: yes"),
    pytest.param("Yes, go ahead.", True, id="EN: yes"),
    pytest.param("Non, laisse tomber.", False, id="FR: no"),
]


def _asked_then_answered(cfg, answer: str):
    """Two voice turns of one conversation through the real gate and the
    real tool. Returns each turn's reply and how many times the agenda had
    been read by the end of it."""
    from jarvis.memory.conversation import DialogueMemory
    from jarvis.memory.db import Database
    from jarvis.reply.engine import run_reply_engine
    from jarvis.tools.builtin.agenda import read_agenda

    readings: List[int] = []
    really_read = read_agenda.read_source

    def _counted(setting):
        readings.append(1)
        return really_read(setting)

    memory = DialogueMemory(inactivity_timeout=300)
    replies: List[str] = []
    read_by_then: List[int] = []
    db = Database(":memory:", sqlite_vss_path=None)
    try:
        with patch("jarvis.tools.registry.get_cached_mcp_tools", return_value={}), \
             patch.object(read_agenda, "read_source", _counted):
            for said in (HIS_AGENDA[0].sentence, answer):
                replies.append(run_reply_engine(
                    db=db, cfg=cfg, tts=None, origin="voix", text=said,
                    dialogue_memory=memory,
                ) or "")
                read_by_then.append(len(readings))
    finally:
        db.close()
    return replies, read_by_then


@pytest.mark.eval
class TestSheAsksThenReads:
    """The sandboxed memory holds no `outils.md`, so the default decides."""

    @pytest.mark.parametrize("answer, granted", ANSWERS)
    def test_nothing_is_read_before_a_yes_and_once_after(self, cfg, answer, granted):
        replies, read_by_then = _asked_then_answered(cfg, answer)
        print(f"\n  🙋 {replies[0][:200]!r}\n  🗣️ {answer}\n  🤖 {replies[1][:300]!r}")

        assert read_by_then[0] == 0, "the agenda was read before he answered"
        assert "readAgenda" in replies[0], (
            f"the first turn did not end on the question: {replies[0][:300]!r}"
        )
        if granted:
            assert read_by_then[1] == 1
            said = replies[1].lower()
            assert "budget" in said or "vannerie" in said, (
                f"her reply does not say what the day holds: {replies[1][:400]!r}"
            )
            assert "confidentiel" not in said, "the body of an invitation reached her reply"
        else:
            assert read_by_then[1] == 0, "the agenda was read on a no"


# ── A listed day, or a free one ───────────────────────────────────────


@pytest.mark.eval
@pytest.mark.skipif(ROUNDS < 1, reason="set EVAL_AGENDA_ROUNDS to compare")
class TestAListedDayAgainstAFreeOne:
    """The arguments the tool ships with against a single free-text day."""

    def test_the_shipped_arguments_against_one_free_text_day(self, cfg, capsys):
        result = _compare_arms(
            cfg, {"one free-text day": _one_free_text_day, "shipped": nullcontext},
            ROUNDS, capsys,
            f"📊 readAgenda's arguments: {ROUNDS} rounds of {len(HIS_AGENDA)} questions "
            f"about his agenda and {len(ELSEWHERE)} sentences elsewhere, arms "
            f"interleaved call by call, chat model {cfg.llm_chat_model}, "
            f"today {TODAY.isoformat()}",
        )

        for measure in MEASURES:
            worse = result.hits[("shipped", measure)] < result.hits[("one free-text day", measure)]
            assert not (worse and result.p(measure) < SIGNIFICANT), (
                f"With the shipped arguments she does worse on {measure!r}."
            )


# ── The other tools, with it in the catalogue ─────────────────────────


@dataclass(frozen=True)
class Routed:
    sentence: str
    tools: Tuple[str, ...]
    # True: every tool named must be offered. False: any one of them.
    every: bool = True


def _from_the_router_evals() -> List[Routed]:
    from test_tool_router_implicit import (
        IMPLICIT_INTENT_CASES, MULTI_TOOL_CASES, TASK_TOOL_CASES,
    )

    cases = [Routed(p.values[0], tuple(p.values[1]), every=False) for p in IMPLICIT_INTENT_CASES]
    cases += [Routed(p.values[0], tuple(p.values[1])) for p in MULTI_TOOL_CASES + TASK_TOOL_CASES]
    return cases


# Sentences about a day or an hour that belong to another tool.
NEAR_THE_AGENDA = (
    Routed("rappelle-moi demain à 9 h d'appeler le garage", ("setReminder",)),
    Routed("remind me on Thursday to call the accountant", ("setReminder",)),
    Routed("quel temps fera-t-il jeudi ?", ("getWeather",)),
    Routed("what time is it in Tokyo?", ("getTime",)),
    Routed("tous les matins à 7 h, résume-moi les nouvelles", ("setRoutine",)),
)

FOR_THE_AGENDA = tuple(Routed(case.sentence, ("readAgenda",)) for case in HIS_AGENDA)

# The line the reply engine hands the router on every turn, in the shape
# the router evals use (`test_tool_router_implicit.py`, `_LIVE_FACTS`).
_FACTS = (
    f"Current local time: {_DAYS[TODAY.weekday()]}, {TODAY.isoformat()} 10:15 "
    "(Europe/Paris). Location: Paris, Île-de-France, France."
)


def _route(cfg, builtin_tools, sentence: str) -> List[str]:
    """The pre-loop router's pick, resolved as the reply engine resolves
    it: the settings' router model on its own backend."""
    from jarvis.llm import get_auxiliary_backend
    from jarvis.reply.engine import resolve_tool_router_model
    from jarvis.tools.selection import ToolSelectionStrategy, router_timeout_sec, select_tools

    model = resolve_tool_router_model(cfg)
    return list(select_tools(
        query=sentence, builtin_tools=builtin_tools, mcp_tools={},
        strategy=ToolSelectionStrategy.LLM,
        llm_backend=get_auxiliary_backend(cfg, model), llm_model=model,
        llm_timeout_sec=router_timeout_sec(cfg), context_hint=_FACTS,
    ))


def _offered(case: Routed, picked: List[str]) -> bool:
    found = [tool in picked for tool in case.tools]
    return all(found) if case.every else any(found)


@pytest.mark.eval
@pytest.mark.skipif(ROUTER_ROUNDS < 1, reason="set EVAL_AGENDA_ROUTER_ROUNDS to compare")
class TestTheOtherToolsWithItInTheCatalogue:
    """The router on the catalogue with `readAgenda` against the same
    catalogue without it."""

    def test_the_other_tools_are_offered_as_often(self, cfg, capsys):
        from jarvis.reply.engine import resolve_tool_router_model
        from jarvis.tools.registry import offered_builtin_tools

        with_it = offered_builtin_tools(cfg)
        assert "readAgenda" in with_it, "the settings do not offer the tool"
        without = {name: tool for name, tool in with_it.items() if name != "readAgenda"}
        catalogues = {"with readAgenda": with_it, "without": without}
        names = tuple(catalogues)

        groups = {
            "the router evals' cases": _from_the_router_evals(),
            "near the agenda": list(NEAR_THE_AGENDA),
        }
        hits = {(arm, group): 0 for arm in names for group in groups}
        calls = dict(hits)
        agenda_hits = agenda_calls = 0
        misses: List[str] = []
        try:
            for rnd in range(ROUTER_ROUNDS):
                at = 0
                for group, cases in groups.items():
                    for case in cases:
                        first = (at + rnd) % len(names)
                        at += 1
                        for arm in names[first:] + names[:first]:
                            picked = _route(cfg, catalogues[arm], case.sentence)
                            calls[(arm, group)] += 1
                            if _offered(case, picked):
                                hits[(arm, group)] += 1
                            else:
                                misses.append(f"{arm} / {case.sentence[:50]!r}: {picked}")
                for case in FOR_THE_AGENDA:
                    picked = _route(cfg, with_it, case.sentence)
                    agenda_calls += 1
                    if _offered(case, picked):
                        agenda_hits += 1
                    else:
                        misses.append(f"for the agenda / {case.sentence[:50]!r}: {picked}")
                to_terminal(capsys, [f"⏱️ round {rnd + 1}/{ROUTER_ROUNDS} done"])
        finally:
            lines = ["", f"📊 the router ({resolve_tool_router_model(cfg)}) with and without "
                         f"readAgenda in the catalogue: {ROUTER_ROUNDS} rounds, arms "
                         "interleaved call by call"]
            p_values = {}
            for group in groups:
                a, b = (hits[(arm, group)] for arm in names)
                na, nb = (calls[(arm, group)] for arm in names)
                p_values[group] = fisher_two_sided(a, na - a, b, nb - b)
                lines.append(f"   🧭 {group}")
                lines.append(f"      with readAgenda   {rate(a, na)}")
                lines.append(f"      without           {rate(b, nb)}")
                lines.append(f"      Fisher two-sided  p = {p_values[group]:.2g}")
            lines.append(f"   📅 a question about his agenda is offered readAgenda: "
                         f"{rate(agenda_hits, agenda_calls)}")
            lines += [f"      ❌ {miss}" for miss in misses]
            to_terminal(capsys, lines)

        for group in groups:
            worse = hits[("with readAgenda", group)] < hits[("without", group)]
            assert not (worse and p_values[group] < SIGNIFICANT), (
                f"With readAgenda in the catalogue the other tools are offered "
                f"less often on {group!r}."
            )


"""
Eval: the memory lookup reads a spoken time into the right window.

`extract_search_params_for_memory` (context #3 in docs/llm_contexts.md) turns
"qu'est-ce que je t'ai dit hier soir ?" into keywords and an optional
`from`/`to` window, and the diary search keeps only the days inside it. A
wrong window does not look wrong: the right memory is filtered out, the reply
says there is nothing, and nothing on screen shows why. No window on a dated
question costs less (the recall runs unfiltered). Any window on an undated
question costs as much as a wrong one.

Every sentence is read against one fixed instant, Tuesday 27 October 2026 at
17:42 in Paris, carried in the context hint the engine itself builds, so the
day each sentence names is known in advance.

The model is the one the engine would call, `resolve_memory_extractor_model`:
`EVAL_MEMORY_EXTRACTOR_MODEL` when set, otherwise the small chain
(`EVAL_SMALL_MODEL`, then `EVAL_JUDGE_MODEL`). One test per sentence:

    ./scripts/run_evals.sh --single memory_extractor_time_window

To compare models, name two or more in `EVAL_MEMORY_EXTRACTOR_ARMS`. One
test then reads every sentence on every arm, alternating arms call by call
in one loop (the order rotates each call, the sentences are shuffled each
round), because a provider drifts within a session and two arms measured one
after the other compare two hours, not two models. It prints the tally per
arm and per language; a row under 20 calls gets counts and no rate.

    EVAL_MEMORY_EXTRACTOR_ARMS="qwen2.5:3b,gemma4:e2b" \\
      ./scripts/run_evals.sh --single "memory_extractor_time_window and arms" -s

`EVAL_MEMORY_EXTRACTOR_REPEATS` sets its rounds (3 by default: 96 calls an
arm). A `vendor/model` arm goes to the provider the `EVAL_LLM_*` variables
name and is billed there; a bare tag on a remote provider runs on the local
Ollama, as it does in the engine. Only these sentences and the hint leave
the machine: the configuration is `MockConfig`, its location is off. Two
local arms on one Ollama may swap models on every call, so their medians
measure the swap rather than the model.
"""

from __future__ import annotations

import os
import random
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import FrozenSet, List, Optional
from zoneinfo import ZoneInfo

import pytest

from conftest import requires_judge_llm
from helpers import MockConfig

# Tuesday 27 October 2026, 17:42 in Paris (UTC+1 since the 25th).
REFERENCE = datetime(2026, 10, 27, 16, 42, tzinfo=timezone.utc)
REFERENCE_ZONE = "Europe/Paris"

RIGHT, TOO_WIDE, WRONG, MISSING, FAILED = "right", "too wide", "wrong", "missing", "failed"
VERDICTS = (RIGHT, TOO_WIDE, WRONG, MISSING, FAILED)

# A window closing just after midnight does not count the next day.
_END_GRACE = timedelta(hours=2)

# Below this many calls a rate says nothing (CLAUDE.md, sample sizes).
_MIN_CALLS_FOR_A_RATE = 20

ARMS = [a.strip() for a in os.environ.get("EVAL_MEMORY_EXTRACTOR_ARMS", "").split(",") if a.strip()]
ARM_REPEATS = int(os.environ.get("EVAL_MEMORY_EXTRACTOR_REPEATS", "3") or 3)


def _oct(day: int) -> date:
    return date(2026, 10, day)


def _span(first: date, last: date) -> FrozenSet[date]:
    return frozenset(first + timedelta(days=n) for n in range((last - first).days + 1))


@dataclass(frozen=True)
class WindowCase:
    lang: str
    label: str
    query: str
    # Days the window must touch; None when the sentence names no time.
    want: Optional[FrozenSet[date]]
    # Days the window may cover. Wider than `want` only where the sentence
    # itself is vague ("last week", "two weeks ago").
    allow: Optional[FrozenSet[date]]

    @property
    def id(self) -> str:
        return f"{self.lang.upper()}: {self.label}"


def _dated(lang, label, query, want, allow=None) -> WindowCase:
    want = frozenset(want)
    return WindowCase(lang, label, query, want, frozenset(allow) if allow else want)


def _undated(lang, label, query) -> WindowCase:
    return WindowCase(lang, label, query, None, None)


CASES: List[WindowCase] = [
    _dated("fr", "yesterday evening", "qu'est-ce que je t'ai dit hier soir sur le boulot ?", {_oct(26)}),
    _dated("fr", "yesterday", "de quoi on a parlé hier ?", {_oct(26)}),
    _dated("fr", "the day before yesterday", "qu'est-ce que j'ai mangé avant-hier ?", {_oct(25)}),
    _dated("fr", "this morning", "tu te souviens de ce que je t'ai raconté ce matin ?", {_oct(27)}),
    _dated("fr", "last Thursday", "on a parlé de quoi jeudi dernier ?", {_oct(22)}),
    _dated("fr", "three days ago", "qu'est-ce que je t'ai dit il y a trois jours sur ma sœur ?", {_oct(24)}),
    _dated("fr", "last week", "rappelle-moi ce qu'on s'est dit la semaine dernière sur le projet",
           _span(_oct(19), _oct(25)), _span(_oct(18), _oct(27))),
    _dated("fr", "Sunday", "qu'est-ce que j'ai dit dimanche ?", {_oct(25)}),
    _dated("fr", "earlier today", "qu'est-ce que je t'ai dit tout à l'heure sur le film ?", {_oct(27)}),
    _dated("fr", "on 20 October", "de quoi on a discuté le 20 octobre ?", {_oct(20)}),
    _dated("fr", "yesterday afternoon", "hier après-midi je t'ai parlé d'un restaurant, c'était lequel ?",
           {_oct(26)}),
    _dated("fr", "yesterday lunchtime", "qu'est-ce que j'ai mangé hier midi ?", {_oct(26)}),
    _dated("fr", "two weeks ago", "on s'est dit quoi il y a deux semaines sur la voiture ?",
           _span(_oct(12), _oct(14)), _span(_oct(6), _oct(20))),
    _dated("fr", "last month", "le mois dernier, je t'avais parlé d'un livre, lequel ?",
           _span(date(2026, 9, 1), date(2026, 9, 30)), _span(date(2026, 8, 28), _oct(3))),
    _dated("fr", "this lunchtime", "qu'est-ce que je t'ai dit ce midi ?", {_oct(27)}),
    _dated("fr", "the evening before yesterday", "avant-hier soir, on a parlé de quoi ?", {_oct(25)}),
    _dated("fr", "last Wednesday", "qu'est-ce que je t'ai raconté mercredi dernier ?", {_oct(21)}),
    _dated("fr", "a week ago", "il y a une semaine, je t'ai parlé d'une série, laquelle ?",
           {_oct(20)}, _span(_oct(19), _oct(21))),
    _undated("fr", "no time, tastes", "qu'est-ce que tu sais sur mes goûts en musique ?"),
    _undated("fr", "no time, recommendation", "recommande-moi un film que j'aimerais"),
    _undated("fr", "no time, a project", "on a déjà parlé de mon projet de déménagement ?"),
    _undated("fr", "no time, favourites", "quels sont mes plats préférés ?"),
    _dated("en", "last night", "what did I tell you last night about work?", {_oct(26)}),
    _dated("en", "yesterday", "what did we talk about yesterday?", {_oct(26)}),
    _dated("en", "the day before yesterday", "what did I eat the day before yesterday?", {_oct(25)}),
    _dated("en", "this morning", "remember what I told you this morning?", {_oct(27)}),
    _dated("en", "last Thursday", "what did we discuss last Thursday?", {_oct(22)}),
    _dated("en", "three days ago", "what did I say three days ago about my sister?", {_oct(24)}),
    _dated("en", "last week", "what did we say about the project last week?",
           _span(_oct(19), _oct(25)), _span(_oct(18), _oct(27))),
    _dated("en", "on 20 October", "what did we discuss on October 20th?", {_oct(20)}),
    _undated("en", "no time, tastes", "what do you know about my taste in music?"),
    _undated("en", "no time, recommendation", "recommend a movie I'd enjoy"),
]


# =============================================================================
# The score
# =============================================================================

def _parse(stamp) -> datetime:
    return datetime.fromisoformat(str(stamp).strip().replace("Z", "+00:00"))


def _days_covered(start, stop, wall_clock: bool) -> FrozenSet[date]:
    """The days a window covers, read one of two ways.

    The prompt asks for UTC and models write either UTC or the local wall
    clock with a `Z`. The diary filter keeps whole days, so an hour's offset
    is not what this measures: the window is read as UTC converted to Paris,
    and as wall clock whatever offset it carries, and the better reading
    counts. A naive stamp counts as UTC, as the code reads it.
    """
    a, b = _parse(start), _parse(stop)
    if wall_clock:
        a, b = a.replace(tzinfo=None), b.replace(tzinfo=None)
    else:
        zone = ZoneInfo(REFERENCE_ZONE)
        a = (a if a.tzinfo else a.replace(tzinfo=timezone.utc)).astimezone(zone)
        b = (b if b.tzinfo else b.replace(tzinfo=timezone.utc)).astimezone(zone)
    end = b - _END_GRACE
    if end.date() < a.date():
        end = a
    return _span(a.date(), end.date())


def verdict(params, case: WindowCase) -> str:
    """How the extractor's answer would filter the diary for this sentence.

    right     covers the day the sentence names, and nothing it does not allow
    too wide  covers that day and more: the right memory is kept, diluted
    wrong     misses that day, or puts a window on a sentence with no time:
              the right memory is filtered out
    missing   no usable window on a dated sentence: the recall runs unfiltered
    failed    no answer at all: the turn runs without memory
    """
    if params is None:
        return FAILED
    start, stop = params.get("from"), params.get("to")
    if case.want is None:
        return WRONG if (start or stop) else RIGHT
    if not (start or stop):
        return MISSING
    # The diary filter applies a lone bound on its own, so one bound is a
    # window open on one side; the diary holds nothing after now.
    start = start or "2000-01-01T00:00:00+00:00"
    stop = stop or REFERENCE.isoformat()
    readable = False
    found = WRONG
    for wall_clock in (False, True):
        try:
            days = _days_covered(start, stop, wall_clock)
        except (TypeError, ValueError):
            continue
        readable = True
        if not days & case.want:
            continue
        if days <= case.allow:
            return RIGHT
        found = TOO_WIDE
    # A stamp the code cannot parse leaves the diary unfiltered.
    return found if readable else MISSING


# =============================================================================
# The call, as the engine makes it
# =============================================================================

def reference_hint() -> str:
    """The context hint the engine builds, with its clock at the reference.

    Built by the engine's own `_build_enrichment_context_hint`, so the eval
    sends what the engine sends. Location is off, as in `real_model_config`.
    """
    from jarvis.reply import engine
    from jarvis.utils.time_context import format_time_context

    cfg = MockConfig()
    cfg.location_enabled = False
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(engine, "format_time_context",
                   lambda *_a, **_k: format_time_context(REFERENCE_ZONE, now_utc=REFERENCE))
        return engine._build_enrichment_context_hint(cfg, [])


def _config(model: str = "") -> MockConfig:
    cfg = MockConfig()
    if model:
        cfg.memory_extractor_model = model
    return cfg


def _read(cfg, query: str, hint: str):
    """One extraction, with the engine's model resolution and deadline."""
    from jarvis.reply.engine import resolve_memory_extractor_model
    from jarvis.reply.enrichment import extract_search_params_for_memory
    from jarvis.tools.selection import router_timeout_sec

    return extract_search_params_for_memory(
        query, cfg, resolve_memory_extractor_model(cfg),
        timeout_sec=router_timeout_sec(cfg), context_hint=hint,
    )


def _warm(cfg, hint: str) -> None:
    """One unmeasured call, so loading the model is not scored as a miss.
    The daemon warms its small models at startup for the same reason."""
    _read(cfg, "hello", hint)


def _window(params) -> str:
    if not params:
        return "no answer" if params is None else "no window"
    if not (params.get("from") or params.get("to")):
        return "no window"
    return f"{params.get('from')} .. {params.get('to')}"


# =============================================================================
# The scorer, held to known windows (no model)
# =============================================================================

YESTERDAY = _dated("en", "yesterday", "what did we talk about yesterday?", {_oct(26)})
LAST_WEEK = _dated("en", "last week", "what did we say about the project last week?",
                   _span(_oct(19), _oct(25)), _span(_oct(18), _oct(27)))
NO_TIME = _undated("en", "no time", "what do you know about my taste in music?")


class TestTheScoreReadsTheWindowLikeTheDiary:
    """The scorer decides what the eval reports, so it is held to known
    windows first. No model is called here."""

    def test_a_window_written_in_utc_holds_the_day(self):
        params = {"keywords": ["x"], "from": "2026-10-25T23:00:00Z", "to": "2026-10-26T22:59:59Z"}
        assert verdict(params, YESTERDAY) == RIGHT

    def test_a_window_written_as_wall_clock_holds_the_day(self):
        params = {"keywords": ["x"], "from": "2026-10-26T00:00:00Z", "to": "2026-10-26T23:59:59Z"}
        assert verdict(params, YESTERDAY) == RIGHT

    def test_a_window_on_another_day_is_wrong(self):
        params = {"keywords": ["x"], "from": "2026-10-27T00:00:00Z", "to": "2026-10-27T23:59:59Z"}
        assert verdict(params, YESTERDAY) == WRONG

    def test_a_window_that_holds_the_day_and_a_week_more_is_too_wide(self):
        params = {"keywords": ["x"], "from": "2026-10-20T00:00:00Z", "to": "2026-10-27T23:59:59Z"}
        assert verdict(params, YESTERDAY) == TOO_WIDE

    def test_a_vague_span_is_right_anywhere_inside_what_it_allows(self):
        params = {"keywords": ["x"], "from": "2026-10-20T00:00:00Z", "to": "2026-10-26T23:59:59Z"}
        assert verdict(params, LAST_WEEK) == RIGHT

    def test_a_lone_bound_is_a_window_open_on_one_side(self):
        params = {"keywords": ["x"], "from": "2026-10-26T00:00:00Z"}
        assert verdict(params, YESTERDAY) == TOO_WIDE

    def test_no_window_on_a_dated_question_is_missing(self):
        assert verdict({"keywords": ["x"]}, YESTERDAY) == MISSING

    def test_a_window_that_cannot_be_read_is_missing(self):
        params = {"keywords": ["x"], "from": "yesterday", "to": "today"}
        assert verdict(params, YESTERDAY) == MISSING

    def test_no_window_on_an_undated_question_is_right(self):
        assert verdict({"keywords": ["music"]}, NO_TIME) == RIGHT

    def test_any_window_on_an_undated_question_is_wrong(self):
        params = {"keywords": ["music"], "from": "2026-10-27T00:00:00Z", "to": "2026-10-27T23:59:59Z"}
        assert verdict(params, NO_TIME) == WRONG

    def test_no_answer_is_failed_whatever_the_question(self):
        assert verdict(None, YESTERDAY) == FAILED
        assert verdict(None, NO_TIME) == FAILED

    def test_the_hint_the_engine_builds_carries_the_reference_instant(self):
        assert "Tuesday, October 27, 2026 at 17:42" in reference_hint()


# =============================================================================
# The resolved extractor model, one sentence at a time
# =============================================================================

@pytest.fixture(scope="module")
def resolved_extractor():
    hint = reference_hint()
    cfg = _config()
    _warm(cfg, hint)
    return cfg, hint


@pytest.mark.eval
@requires_judge_llm
class TestMemoryExtractorTimeWindow:

    @pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
    def test_the_window_holds_the_day(self, case, resolved_extractor):
        from jarvis.reply.engine import resolve_memory_extractor_model

        cfg, hint = resolved_extractor
        params = _read(cfg, case.query, hint)
        found = verdict(params, case)

        print(f"\n  🕰️ Memory lookup ({resolve_memory_extractor_model(cfg)}): {case.query}")
        print(f"     🪟 {_window(params)}  ->  {found}")

        expected = "no window" if case.want is None else (
            f"a window on {min(case.want)}" + (f" .. {max(case.want)}" if len(case.want) > 1 else ""))
        assert found == RIGHT, (
            f"{case.id}: expected {expected}, got {_window(params)} ({found}). "
            f"Keywords: {(params or {}).get('keywords')}"
        )


# =============================================================================
# Two or more models, arms interleaved call by call
# =============================================================================

@dataclass
class _Call:
    arm: str
    case: WindowCase
    verdict: str
    secs: float
    window: str


def _tally(calls: List[_Call], arms: List[str], rounds: int) -> List[str]:
    lines = [f"📊 Memory lookup windows: {len(CASES)} sentences x {rounds}, "
             f"{len(arms)} arms interleaved call by call"]
    for arm in arms:
        lines.append(f"   🤖 {arm}")
        for lang in ("fr", "en", "all"):
            mine = [c for c in calls if c.arm == arm and lang in ("all", c.case.lang)]
            if not mine:
                continue
            n = len(mine)
            counts = Counter(c.verdict for c in mine)
            cells = []
            for kind in VERDICTS:
                cell = f"{kind} {counts[kind]}"
                if n >= _MIN_CALLS_FOR_A_RATE:
                    cell += f" ({100 * counts[kind] / n:.0f} %)"
                cells.append(cell)
            median = statistics.median(c.secs for c in mine)
            note = "" if n >= _MIN_CALLS_FOR_A_RATE else f"   (under {_MIN_CALLS_FOR_A_RATE} calls: counts only)"
            lines.append(f"      💬 {lang:<3} {n:>3} calls   " + "   ".join(cells)
                         + f"   median {median:.2f} s{note}")
    for arm in arms:
        misses = defaultdict(list)
        for c in calls:
            if c.arm == arm and c.verdict != RIGHT:
                misses[c.case.id].append(f"{c.verdict}: {c.window}")
        lines.append(f"   🔎 {arm}: not right on {len(misses)} of {len(CASES)} sentences")
        for case_id, seen in sorted(misses.items(), key=lambda kv: -len(kv[1])):
            lines.append(f"      {len(seen)}x {case_id}  ->  {seen[0]}")
    return lines


@pytest.mark.eval
@requires_judge_llm
@pytest.mark.skipif(len(ARMS) < 2, reason="name two or more models in EVAL_MEMORY_EXTRACTOR_ARMS to compare them")
class TestMemoryExtractorArms:

    # Its rounds are its own and interleaved; `--count` would rerun the
    # whole comparison instead.
    @pytest.mark.repeat(1)
    def test_arms_read_the_same_sentences_call_by_call(self, capsys):
        hint = reference_hint()
        configs = {arm: _config(arm) for arm in ARMS}
        for arm in ARMS:
            _warm(configs[arm], hint)

        rng = random.Random(0)
        calls: List[_Call] = []
        for rnd in range(ARM_REPEATS):
            order = list(range(len(CASES)))
            rng.shuffle(order)
            for i, idx in enumerate(order):
                case = CASES[idx]
                turn = (i + rnd) % len(ARMS)
                for arm in ARMS[turn:] + ARMS[:turn]:
                    t0 = time.perf_counter()
                    params = _read(configs[arm], case.query, hint)
                    secs = time.perf_counter() - t0
                    calls.append(_Call(arm, case, verdict(params, case), secs, _window(params)))

        with capsys.disabled():
            print("\n" + "\n".join(_tally(calls, ARMS, ARM_REPEATS)))

        # A table from an arm that never answered measures nothing.
        silent = [arm for arm in ARMS if all(c.verdict == FAILED for c in calls if c.arm == arm)]
        assert not silent, f"no answer at all from {silent}: check the model names and the provider"

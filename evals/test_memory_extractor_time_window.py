"""
Eval: the memory lookup reads a spoken time into the right window.

`extract_search_params_for_memory` (context #3 in docs/llm_contexts.md) turns
"qu'est-ce que je t'ai dit hier soir ?" into keywords and an optional
`from`/`to` window, and the diary search keeps only the days inside it. A
wrong window does not look wrong: the right memory is filtered out, the reply
says there is nothing, and nothing on screen shows why. No window on a dated
question costs less (the recall runs unfiltered). A window on an undated
question costs as much as a wrong one.

Every question is read against one fixed instant, Tuesday 27 October 2026 at
17:42 in Paris, carried in the context hint the engine itself builds, so the
day each question names is known in advance.

The model is the one the engine would call, `resolve_memory_extractor_model`:
`EVAL_MEMORY_EXTRACTOR_MODEL` when set, otherwise the small chain
(`EVAL_SMALL_MODEL`, then `EVAL_JUDGE_MODEL`). One test per question:

    ./scripts/run_evals.sh --single memory_extractor_time_window

To compare models, name two or more in `EVAL_MEMORY_EXTRACTOR_ARMS`. One
test then reads every question on every arm, alternating arms call by call
in one loop (the order rotates each call, the questions are shuffled each
round), because a provider drifts within a session and two arms measured one
after the other compare two hours, not two models. It prints the tally per
arm and per language; a row under 20 calls gets counts and no rate.

    EVAL_MEMORY_EXTRACTOR_ARMS="qwen2.5:3b,gemma4:e2b" \\
      ./scripts/run_evals.sh --single "memory_extractor_time_window and arms"

`--single` matters: the arms do not depend on `EVAL_JUDGE_MODEL`, so the
launcher's two-model pass would run the same comparison twice.
`EVAL_MEMORY_EXTRACTOR_ROUNDS` sets its rounds (3 by default: 96 scored
calls an arm, each up to two requests since the extractor retries once on
an unusable answer, plus one warm-up). A `vendor/model` arm goes to the
provider the `EVAL_LLM_*` variables name and is billed there; a bare tag on
a remote provider runs on the local Ollama, as it does in the engine. Only
these questions and the hint leave the machine: the hint is built with
location off, and refuses to be built otherwise. Two local arms on one
Ollama may swap models on every call, so their medians measure the swap
rather than the model.
"""

from __future__ import annotations

import os
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import List, Optional, Tuple
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

# Where a lone `to` opens: earlier than anything the diary holds.
_OPEN_START = "2000-01-01T00:00:00+00:00"

# Below this many calls a rate says nothing (CLAUDE.md, sample sizes).
_MIN_CALLS_FOR_A_RATE = 20

# Long enough to load a local model that is not resident.
_WARM_UP_TIMEOUT_SEC = 60.0

ARMS = [a.strip() for a in os.environ.get("EVAL_MEMORY_EXTRACTOR_ARMS", "").split(",") if a.strip()]
ARM_ROUNDS = int(os.environ.get("EVAL_MEMORY_EXTRACTOR_ROUNDS", "3") or 3)

Days = Tuple[date, date]


def _oct(day: int) -> date:
    return date(2026, 10, day)


@dataclass(frozen=True)
class WindowCase:
    lang: str
    label: str
    query: str
    # First and last day every reasonable reading of the question covers:
    # the window must cover all of them. None when it names no time.
    want: Optional[Days]
    # First and last day some reasonable reading covers: the window may not
    # go beyond. Wider than `want` only where the question itself is vague
    # ("last week" is the calendar week or the last seven days).
    allow: Optional[Days]

    @property
    def id(self) -> str:
        return f"{self.lang.upper()}: {self.label}"


def _dated(lang, label, query, want, allow=None) -> WindowCase:
    want = want if isinstance(want, tuple) else (want, want)
    return WindowCase(lang, label, query, want, allow or want)


def _undated(lang, label, query) -> WindowCase:
    return WindowCase(lang, label, query, None, None)


CASES: List[WindowCase] = [
    _dated("fr", "yesterday evening", "qu'est-ce que je t'ai dit hier soir sur le boulot ?", _oct(26)),
    _dated("fr", "yesterday", "de quoi on a parlé hier ?", _oct(26)),
    _dated("fr", "the day before yesterday", "qu'est-ce que j'ai mangé avant-hier ?", _oct(25)),
    _dated("fr", "this morning", "tu te souviens de ce que je t'ai raconté ce matin ?", _oct(27)),
    _dated("fr", "last Thursday", "on a parlé de quoi jeudi dernier ?", _oct(22)),
    _dated("fr", "three days ago", "qu'est-ce que je t'ai dit il y a trois jours sur ma sœur ?", _oct(24)),
    _dated("fr", "last week", "rappelle-moi ce qu'on s'est dit la semaine dernière sur le projet",
           (_oct(20), _oct(25)), (_oct(18), _oct(27))),
    _dated("fr", "Sunday", "qu'est-ce que j'ai dit dimanche ?", _oct(25)),
    _dated("fr", "earlier today", "qu'est-ce que je t'ai dit tout à l'heure sur le film ?", _oct(27)),
    _dated("fr", "on 20 October", "de quoi on a discuté le 20 octobre ?", _oct(20)),
    _dated("fr", "yesterday afternoon", "hier après-midi je t'ai parlé d'un restaurant, c'était lequel ?",
           _oct(26)),
    _dated("fr", "yesterday lunchtime", "qu'est-ce que j'ai mangé hier midi ?", _oct(26)),
    _dated("fr", "two weeks ago", "on s'est dit quoi il y a deux semaines sur la voiture ?",
           _oct(13), (_oct(6), _oct(20))),
    _dated("fr", "last month", "le mois dernier, je t'avais parlé d'un livre, lequel ?",
           (date(2026, 9, 1), date(2026, 9, 30)), (date(2026, 8, 28), _oct(3))),
    _dated("fr", "this lunchtime", "qu'est-ce que je t'ai dit ce midi ?", _oct(27)),
    _dated("fr", "the evening before yesterday", "avant-hier soir, on a parlé de quoi ?", _oct(25)),
    _dated("fr", "last Wednesday", "qu'est-ce que je t'ai raconté mercredi dernier ?", _oct(21)),
    _dated("fr", "a week ago", "il y a une semaine, je t'ai parlé d'une série, laquelle ?",
           _oct(20), (_oct(19), _oct(21))),
    _undated("fr", "no time, tastes", "qu'est-ce que tu sais sur mes goûts en musique ?"),
    _undated("fr", "no time, recommendation", "recommande-moi un film que j'aimerais"),
    _undated("fr", "no time, a project", "on a déjà parlé de mon projet de déménagement ?"),
    _undated("fr", "no time, favourites", "quels sont mes plats préférés ?"),
    _dated("en", "last night", "what did I tell you last night about work?", _oct(26)),
    _dated("en", "yesterday", "what did we talk about yesterday?", _oct(26)),
    _dated("en", "the day before yesterday", "what did I eat the day before yesterday?", _oct(25)),
    _dated("en", "this morning", "remember what I told you this morning?", _oct(27)),
    _dated("en", "last Thursday", "what did we discuss last Thursday?", _oct(22)),
    _dated("en", "three days ago", "what did I say three days ago about my sister?", _oct(24)),
    _dated("en", "last week", "what did we say about the project last week?",
           (_oct(20), _oct(25)), (_oct(18), _oct(27))),
    _dated("en", "on 20 October", "what did we discuss on October 20th?", _oct(20)),
    _undated("en", "no time, tastes", "what do you know about my taste in music?"),
    _undated("en", "no time, recommendation", "recommend a movie I'd enjoy"),
]


# =============================================================================
# The score
# =============================================================================

def _parse(stamp) -> datetime:
    return datetime.fromisoformat(str(stamp).strip().replace("Z", "+00:00"))


def _days_covered(start, stop, wall_clock: bool) -> Days:
    """First and last day a window covers, read one of two ways.

    The prompt asks for UTC and models write either UTC or the local wall
    clock, often with a `Z`. What this measures is whether the model read
    the day right, not its timezone arithmetic, so the window is read as UTC
    converted to Paris, and as wall clock whatever offset it carries; the
    caller keeps the better reading. A naive stamp counts as UTC, as the
    code reads it.
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
    return a.date(), end.date()


def _readings(start, stop) -> List[Days]:
    readings = []
    for wall_clock in (False, True):
        try:
            readings.append(_days_covered(start, stop, wall_clock))
        except (TypeError, ValueError, OverflowError):
            continue
    return readings


def verdict(params, case: WindowCase) -> str:
    """What the extractor's answer does to the diary search for this question.

    The day is read leniently (see `_days_covered`): an hour's offset or a
    window closing just after midnight does not change the verdict.

    right     covers every day the question names, and nothing it does not
              allow; on an undated question, no window, or one that filters
              nothing
    too wide  covers those days and more: the right memory is kept, diluted
    wrong     misses a day the question names, or filters an undated
              question: the right memory is filtered out
    missing   no usable window on a dated question: the recall runs unfiltered
    failed    the turn reads no diary: no answer at all, or no keywords, which
              the diary search needs before any window applies
    """
    if params is None or not params.get("keywords"):
        return FAILED
    start, stop = params.get("from"), params.get("to")
    # The diary filter applies a lone bound on its own, so one bound is a
    # window open on one side; the diary holds nothing after now.
    readings = _readings(start or _OPEN_START, stop or REFERENCE.isoformat()) if (start or stop) else []
    if case.want is None:
        open_from, now = _days_covered(_OPEN_START, REFERENCE.isoformat(), wall_clock=False)
        filters = any(first > open_from or last < now for first, last in readings)
        return WRONG if filters else RIGHT
    if not readings:
        # No window, or one the code cannot parse: the diary runs unfiltered.
        return MISSING
    found = WRONG
    for first, last in readings:
        if first > case.want[0] or last < case.want[1]:
            continue
        if first >= case.allow[0] and last <= case.allow[1]:
            return RIGHT
        found = TOO_WIDE
    return found


# =============================================================================
# The call, as the engine makes it
# =============================================================================

def reference_hint(location_enabled: bool = False) -> str:
    """The context hint the engine builds, with its clock at the reference.

    Built by the engine's own `_build_enrichment_context_hint`, so the eval
    sends what the engine sends. A location lookup is refused: the engine
    drops the whole time line when the lookup fails, and a hint without the
    reference instant raises here rather than reaching a model.
    `location_enabled` exists to show that refusal.
    """
    from jarvis.reply import engine
    from jarvis.utils.time_context import format_time_context

    reference = format_time_context(REFERENCE_ZONE, now_utc=REFERENCE)

    def no_location(*_a, **_k):
        raise RuntimeError("the eval's hint must not look up a location")

    cfg = MockConfig()
    cfg.location_enabled = location_enabled
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(engine, "format_time_context", lambda *_a, **_k: reference)
        mp.setattr(engine, "get_location_context_with_timezone", no_location)
        hint = engine._build_enrichment_context_hint(cfg, [])
    if not hint or reference not in hint:
        raise RuntimeError(f"the hint does not carry the reference instant: {hint!r}")
    return hint


def _config(model: str = "") -> MockConfig:
    cfg = MockConfig()
    if model:
        cfg.memory_extractor_model = model
    return cfg


def _read(cfg, query: str, hint: str, timeout_sec: Optional[float] = None):
    """One extraction, with the engine's model resolution and deadline
    (`router_timeout_sec`, 8 s in `MockConfig`)."""
    from jarvis.reply.engine import resolve_memory_extractor_model
    from jarvis.reply.enrichment import extract_search_params_for_memory
    from jarvis.tools.selection import router_timeout_sec

    return extract_search_params_for_memory(
        query, cfg, resolve_memory_extractor_model(cfg),
        timeout_sec=timeout_sec or router_timeout_sec(cfg), context_hint=hint,
    )


def _warms_up(cfg, hint: str) -> bool:
    """One unmeasured call with a long deadline, so loading a model is not
    scored as a miss. Whether it answered: a model that cannot answer here
    will not answer the questions either."""
    return _read(cfg, "hello", hint, timeout_sec=_WARM_UP_TIMEOUT_SEC) is not None


def _window(params) -> str:
    if params is None:
        return "no answer"
    if not params.get("keywords"):
        return "no keywords"
    if not (params.get("from") or params.get("to")):
        return "no window"
    return f"{params.get('from')} .. {params.get('to')}"


# =============================================================================
# The scorer, held to known windows (no model)
# =============================================================================

YESTERDAY = _dated("en", "yesterday", "what did we talk about yesterday?", _oct(26))
LAST_WEEK = _dated("en", "last week", "what did we say about the project last week?",
                   (_oct(20), _oct(25)), (_oct(18), _oct(27)))
NO_TIME = _undated("en", "no time", "what do you know about my taste in music?")


def _window_of(start=None, stop=None, keywords=("x",)):
    params = {"keywords": list(keywords)}
    if start:
        params["from"] = start
    if stop:
        params["to"] = stop
    return params


class TestTheScoreReadsTheDayNotTheTimezone:
    """The scorer decides what the eval reports, so it is held to known
    windows first. No model is called here."""

    def test_a_window_written_in_utc_holds_the_day(self):
        assert verdict(_window_of("2026-10-25T23:00:00Z", "2026-10-26T22:59:59Z"), YESTERDAY) == RIGHT

    def test_a_window_written_as_wall_clock_holds_the_day(self):
        # Converted to Paris this covers the 26th and the 27th; read as the
        # wall clock it was written in, the 26th alone.
        params = _window_of("2026-10-26T00:00:00-05:00", "2026-10-26T23:59:59-05:00")
        assert verdict(params, YESTERDAY) == RIGHT

    def test_a_window_closing_at_midnight_does_not_count_the_next_day(self):
        assert verdict(_window_of("2026-10-26T00:00:00Z", "2026-10-27T00:00:00Z"), YESTERDAY) == RIGHT

    def test_a_window_without_an_offset_is_read(self):
        assert verdict(_window_of("2026-10-26T00:00:00", "2026-10-26T23:59:59Z"), YESTERDAY) == RIGHT

    def test_a_window_on_another_day_is_wrong(self):
        assert verdict(_window_of("2026-10-27T00:00:00Z", "2026-10-27T23:59:59Z"), YESTERDAY) == WRONG

    def test_a_window_that_holds_the_day_and_a_week_more_is_too_wide(self):
        assert verdict(_window_of("2026-10-20T00:00:00Z", "2026-10-27T23:59:59Z"), YESTERDAY) == TOO_WIDE

    def test_a_vague_span_is_right_on_every_reading_it_allows(self):
        calendar_week = _window_of("2026-10-19T00:00:00Z", "2026-10-25T23:59:59Z")
        last_seven_days = _window_of("2026-10-20T00:00:00Z", "2026-10-27T17:42:00Z")
        assert verdict(calendar_week, LAST_WEEK) == RIGHT
        assert verdict(last_seven_days, LAST_WEEK) == RIGHT

    def test_a_vague_span_read_as_one_of_its_days_is_wrong(self):
        assert verdict(_window_of("2026-10-22T00:00:00Z", "2026-10-22T23:59:59Z"), LAST_WEEK) == WRONG

    def test_a_lone_from_is_a_window_open_until_now(self):
        assert verdict(_window_of("2026-10-26T00:00:00Z"), YESTERDAY) == TOO_WIDE

    def test_a_lone_to_is_a_window_open_from_the_past(self):
        assert verdict(_window_of(stop="2026-10-26T23:59:59Z"), YESTERDAY) == TOO_WIDE

    def test_a_bound_at_the_end_of_the_calendar_is_read_not_raised(self):
        assert verdict(_window_of("2026-10-26T00:00:00Z", "9999-12-31T23:59:59Z"), YESTERDAY) == TOO_WIDE
        assert verdict(_window_of(stop="0001-01-01T00:00:00Z"), YESTERDAY) == MISSING

    def test_no_window_on_a_dated_question_is_missing(self):
        assert verdict(_window_of(), YESTERDAY) == MISSING

    def test_a_window_that_cannot_be_read_is_missing(self):
        assert verdict(_window_of("yesterday", "today"), YESTERDAY) == MISSING

    def test_a_right_window_with_no_keywords_is_failed(self):
        # The engine searches the diary only with keywords: the window is
        # never applied and the turn reads no diary.
        params = _window_of("2026-10-26T00:00:00Z", "2026-10-26T23:59:59Z", keywords=())
        assert verdict(params, YESTERDAY) == FAILED

    def test_no_window_on_an_undated_question_is_right(self):
        assert verdict(_window_of(), NO_TIME) == RIGHT

    def test_a_window_on_an_undated_question_is_wrong(self):
        assert verdict(_window_of("2026-10-27T00:00:00Z", "2026-10-27T23:59:59Z"), NO_TIME) == WRONG

    def test_a_window_that_filters_nothing_on_an_undated_question_is_right(self):
        assert verdict(_window_of("yesterday", "today"), NO_TIME) == RIGHT
        assert verdict(_window_of(stop="2030-01-01T00:00:00Z"), NO_TIME) == RIGHT

    def test_no_answer_is_failed_whatever_the_question(self):
        assert verdict(None, YESTERDAY) == FAILED
        assert verdict(None, NO_TIME) == FAILED

    def test_the_hint_carries_the_reference_instant(self):
        from jarvis.utils.time_context import format_time_context

        assert format_time_context(REFERENCE_ZONE, now_utc=REFERENCE) in reference_hint()

    def test_the_hint_refuses_a_location_lookup(self):
        # Location on would read the GeoIP database and the user's location
        # caches, and send the city along with every question.
        with pytest.raises(RuntimeError):
            reference_hint(location_enabled=True)


# =============================================================================
# The resolved extractor model, one question at a time
# =============================================================================

@pytest.fixture(scope="module")
def resolved_extractor():
    from jarvis.reply.engine import resolve_memory_extractor_model

    hint = reference_hint()
    cfg = _config()
    if not _warms_up(cfg, hint):
        pytest.fail(f"{resolve_memory_extractor_model(cfg)} gave no answer to a warm-up: "
                    "check the model name and the provider")
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

        if case.want is None:
            expected = "no window"
        elif case.want[0] == case.want[1]:
            expected = f"a window on {case.want[0]}"
        else:
            expected = f"a window covering {case.want[0]} .. {case.want[1]} within {case.allow[0]} .. {case.allow[1]}"
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
    answered: bool
    window: str


def _tally(calls: List[_Call], arms: List[str], rounds_done: int) -> List[str]:
    lines = [f"📊 Memory lookup windows: {len(CASES)} questions x {rounds_done} rounds, "
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
            answered = [c.secs for c in mine if c.answered]
            median = f"median {statistics.median(answered):.2f} s answered" if answered else "no answer"
            note = "" if n >= _MIN_CALLS_FOR_A_RATE else f"   (under {_MIN_CALLS_FOR_A_RATE} calls: counts only)"
            lines.append(f"      💬 {lang:<3} {n:>3} calls   " + "   ".join(cells) + f"   {median}{note}")
    for arm in arms:
        misses = defaultdict(list)
        for c in calls:
            if c.arm == arm and c.verdict != RIGHT:
                misses[c.case.id].append(f"{c.verdict}: {c.window}")
        lines.append(f"   🔎 {arm}: not right on {len(misses)} of {len(CASES)} questions")
        for case_id, seen in sorted(misses.items(), key=lambda kv: -len(kv[1])):
            lines.append(f"      ❌ {len(seen)}x {case_id}  ->  {seen[0]}")
    return lines


def _to_terminal(capsys, lines: List[str]) -> None:
    """Print past pytest's capture. The real stream may be a cp1252 pipe
    (Git Bash on Windows), where an emoji raises once every call is made."""
    from jarvis.utils.console import force_utf8_stream

    with capsys.disabled():
        force_utf8_stream(sys.stdout)
        print("\n".join(lines), flush=True)


@pytest.mark.eval
@requires_judge_llm
@pytest.mark.skipif(len(ARMS) < 2, reason="name two or more models in EVAL_MEMORY_EXTRACTOR_ARMS to compare them")
class TestMemoryExtractorArms:

    # Its rounds are its own and interleaved; `--count` would rerun the
    # whole comparison instead.
    @pytest.mark.repeat(1)
    def test_arms_read_the_same_questions_call_by_call(self, capsys):
        hint = reference_hint()
        configs = {arm: _config(arm) for arm in ARMS}
        cold = [arm for arm in ARMS if not _warms_up(configs[arm], hint)]
        assert not cold, f"no answer to a warm-up from {cold}: check the model names and the provider"

        rng = random.Random(0)
        calls: List[_Call] = []
        rounds_done = 0
        try:
            for rnd in range(ARM_ROUNDS):
                order = list(range(len(CASES)))
                rng.shuffle(order)
                for i, idx in enumerate(order):
                    case = CASES[idx]
                    turn = (i + rnd) % len(ARMS)
                    for arm in ARMS[turn:] + ARMS[:turn]:
                        t0 = time.perf_counter()
                        params = _read(configs[arm], case.query, hint)
                        secs = time.perf_counter() - t0
                        calls.append(_Call(arm, case, verdict(params, case), secs,
                                           params is not None, _window(params)))
                rounds_done += 1
                _to_terminal(capsys, [f"⏱️ round {rounds_done}/{ARM_ROUNDS} done, {len(calls)} calls"])
        finally:
            # Printed even when a round is cut short: the calls already made
            # may have been billed.
            if calls:
                _to_terminal(capsys, [""] + _tally(calls, ARMS, rounds_done))

        # A table from an arm that never answered measures nothing.
        silent = [arm for arm in ARMS if not any(c.answered for c in calls if c.arm == arm)]
        assert not silent, f"no answer at all from {silent}: check the model names and the provider"

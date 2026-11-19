"""
End-to-end eval: asked about what is on the screen, she reads it; asked
about anything else, she leaves it alone.

`screenshot`'s description decides when she reaches for it
(`src/jarvis/tools/builtin/screenshot.spec.md`). Both directions are
measured: a question about the screen must reach the tool, and a sentence
that has nothing to do with the screen must not. A call made for nothing
costs the user a question they did not expect when the tool asks first,
and a capture they did not want when their `outils.md` frees it.

What is counted is the call the model makes, before the gate: whether the
gate then asks is the policy's business, held in
`tests/tools/builtin/test_screenshot.py`.

Each turn runs the real engine on a voice turn (`origin="voix"`): the
planner, the pre-loop router and the chat model as the user's settings name
them. No tool runs for real: `screenshot` answers with an invented screen,
fenced as the tool fences it, and every other tool with an invented result.
No screen is captured. The memory is empty. The kinds, every sentence
invented:

- the screen: a question about what the screen shows. Passes when
  `screenshot` is called.
- elsewhere: a sentence that names a screen or a screenshot without asking
  about this one, or that is about something else. Passes when `screenshot`
  is not called.

`TestSheAsksThenReads` is the one place the gate and the tool run for
real, with an invented screen in place of the capture: a question about
the screen ends the first turn on her question with nothing read, a yes
has it read once and her reply say what was on it, a no reads nothing.

`TestTheEngineAsItShips` asks each sentence once.
`TestTheDescriptionBeforeAndAfter` alternates the description the tool
carried when it only knew macOS and the one it ships with, call by call in
one loop, the first arm rotating (CLAUDE.md: a provider drifts within a
session), and prints hits per arm and kind, Fisher's two-sided p and every
miss. It passes when the shipped description is not worse on either kind
with p below 0.05.

Set `EVAL_SCREEN_ROUNDS` to run the comparison: each round asks every
sentence of each kind once on each arm, every kind has four sentences, so
five rounds give 20 calls an arm for each kind.

They take ``real_model_config``: the user's provider and models, sandboxed
memory, location off, and here the LLM router and no MCP server pinned.
What leaves the machine is the invented sentences, the invented screen and
the prompt built around an empty memory.

Run:
    ./scripts/run_evals.sh --single she_reads_the_screen_when_asked
    EVAL_SCREEN_ROUNDS=5 pytest evals/test_she_reads_the_screen_when_asked.py -k BeforeAndAfter -s
"""

import dataclasses
import os
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence, Tuple
from unittest.mock import PropertyMock, patch

import pytest

from helpers import fisher_two_sided, rate, to_terminal

ROUNDS = int(os.environ.get("EVAL_SCREEN_ROUNDS", "0") or 0)

SIGNIFICANT = 0.05

# What the tool said of itself when a dragged rectangle on macOS was all it
# knew.
DESCRIPTION_BEFORE = (
    "Capture a selected screen region and OCR the text. "
    "Use only if the OCR will materially help."
)

INVENTED_SCREEN = (
    "Tableur des marées — Port-Vellin\n"
    "Erreur 4471 : le fichier marees.csv est introuvable\n"
    "Réessayer    Annuler"
)


@dataclass
class Turn:
    reply: str
    called: List[str]


@dataclass(frozen=True)
class Kind:
    name: str
    sentences: Tuple[str, ...]
    passes: Callable[[Turn], bool]


THE_SCREEN = (
    "Qu'est-ce qu'il y a sur mon écran, là ?",
    "Lis-moi le message d'erreur qui est affiché à l'écran.",
    "What does the window on my screen say?",
    "Regarde mon écran et dis-moi ce qui est écrit.",
)

ELSEWHERE = (
    "J'ai vu un screenshot marrant hier.",
    "C'est quoi la différence entre un écran OLED et un écran LCD ?",
    "Quelle est la capitale de la Mongolie ?",
    "Tell me a short joke about cats.",
)


def _reads_it(turn: Turn) -> bool:
    return "screenshot" in turn.called


def _leaves_it(turn: Turn) -> bool:
    return "screenshot" not in turn.called


KINDS: Dict[str, Kind] = {k.name: k for k in (
    Kind("the screen", THE_SCREEN, _reads_it),
    Kind("elsewhere", ELSEWHERE, _leaves_it),
)}


# ── One turn ──────────────────────────────────────────────────────────


def _turn(cfg, sentence: str) -> Turn:
    """One voice turn with the tool described as it is when called."""
    from jarvis.memory.conversation import DialogueMemory
    from jarvis.memory.db import Database
    from jarvis.reply.engine import run_reply_engine
    from jarvis.tools.base import ToolContext
    from jarvis.tools.builtin.screenshot import _fenced
    from jarvis.tools.registry import BUILTIN_TOOLS
    from jarvis.tools.types import ToolExecutionResult

    called: List[str] = []

    def _runner(db, cfg, tool_name, tool_args, **kwargs):
        called.append(tool_name)
        if tool_name == "toolSearchTool":
            return BUILTIN_TOOLS[tool_name].run(tool_args or {}, ToolContext(
                db=db, cfg=cfg, system_prompt="", original_prompt="",
                redacted_text="", max_retries=1, user_print=lambda _m: None,
                origin="voix",
            ))
        if tool_name == "screenshot":
            return ToolExecutionResult(success=True, reply_text=_fenced([INVENTED_SCREEN]))
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
    return Turn(reply or "", called)


@pytest.fixture
def cfg(real_model_config):
    """The user's provider and models, with what the turns rely on pinned."""
    return dataclasses.replace(
        real_model_config, tool_selection_strategy="llm", mcps={},
    )


# ── The arms ──────────────────────────────────────────────────────────


@contextmanager
def _described_as_before():
    from jarvis.tools.builtin.screenshot import ScreenshotTool
    from jarvis.tools.registry import BUILTIN_TOOLS

    assert BUILTIN_TOOLS["screenshot"].description != DESCRIPTION_BEFORE, (
        "the tool ships with the earlier description: both arms would be the same"
    )
    with patch.object(ScreenshotTool, "description",
                      new_callable=PropertyMock, return_value=DESCRIPTION_BEFORE):
        assert BUILTIN_TOOLS["screenshot"].description == DESCRIPTION_BEFORE
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
                out.append(f"      {kind:<12} {rate(self.hits[(arm, kind)], self.calls[(arm, kind)])}")
        out.append(f"   📐 Fisher two-sided, {self.arms[0]} against {self.arms[1]}:")
        for kind in self.kinds:
            out.append(f"      {kind:<12} p = {self.p(kind):.2g}")
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
    cases = [(KINDS[k], sentence) for k in kinds for sentence in KINDS[k].sentences]
    try:
        for rnd in range(rounds):
            for i, (kind, sentence) in enumerate(cases):
                first = (i + rnd) % len(names)
                for name in names[first:] + names[:first]:
                    with arms[name]():
                        turn = _turn(cfg, sentence)
                    result.calls[(name, kind.name)] += 1
                    if kind.passes(turn):
                        result.hits[(name, kind.name)] += 1
                    else:
                        result.misses.append(
                            f"{name} / {kind.name} / {sentence[:45]!r}: "
                            f"tools {turn.called}, reply {turn.reply[:160]!r}"
                        )
            to_terminal(capsys, [f"⏱️ round {rnd + 1}/{rounds} done"])
    finally:
        to_terminal(capsys, [""] + result.lines(title))
    return result


# ── The engine as it ships ────────────────────────────────────────────


def _every_sentence():
    for kind in KINDS.values():
        for sentence in kind.sentences:
            yield pytest.param(kind.name, sentence, id=f"{kind.name}: {sentence[:40]}")


@pytest.mark.eval
class TestTheEngineAsItShips:
    """Each sentence once."""

    @pytest.mark.parametrize("kind_name, sentence", list(_every_sentence()))
    def test_each_sentence_gets_what_it_asks(self, cfg, kind_name, sentence):
        turn = _turn(cfg, sentence)
        print(f"\n  🗣️ {sentence}\n  🔧 {turn.called}\n  🤖 {turn.reply[:300]!r}")

        assert KINDS[kind_name].passes(turn), (
            f"{kind_name}: tools {turn.called}, reply {turn.reply[:400]!r}"
        )


# ── She asks, then reads ──────────────────────────────────────────────


ANSWERS = [
    pytest.param("Oui, vas-y.", True, id="FR: yes"),
    pytest.param("Yes, go ahead.", True, id="EN: yes"),
    pytest.param("Non, laisse tomber.", False, id="FR: no"),
]


def _asked_then_answered(cfg, answer: str):
    """Two voice turns of one conversation through the real gate and the
    real tool. Returns each turn's reply and how many times the screen had
    been read by the end of it."""
    from jarvis.memory.conversation import DialogueMemory
    from jarvis.memory.db import Database
    from jarvis.reply.engine import run_reply_engine
    from jarvis.tools.builtin import screenshot

    readings: List[int] = []

    def _an_invented_screen():
        readings.append(1)
        return [INVENTED_SCREEN]

    memory = DialogueMemory(inactivity_timeout=300)
    replies: List[str] = []
    read_by_then: List[int] = []
    db = Database(":memory:", sqlite_vss_path=None)
    try:
        with patch("jarvis.tools.registry.get_cached_mcp_tools", return_value={}), \
             patch.object(screenshot, "_read_screen", _an_invented_screen):
            for said in (THE_SCREEN[0], answer):
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

        assert read_by_then[0] == 0, "the screen was read before he answered"
        assert "screenshot" in replies[0], (
            f"the first turn did not end on the question: {replies[0][:300]!r}"
        )
        if granted:
            assert read_by_then[1] == 1
            assert "4471" in replies[1] or "marees" in replies[1], (
                f"her reply does not say what the screen showed: {replies[1][:400]!r}"
            )
        else:
            assert read_by_then[1] == 0, "the screen was read on a no"


# ── The comparison ────────────────────────────────────────────────────


@pytest.mark.eval
@pytest.mark.skipif(ROUNDS < 1, reason="set EVAL_SCREEN_ROUNDS to compare")
class TestTheDescriptionBeforeAndAfter:
    """The description the tool carried on macOS alone against the one it
    ships with."""

    def test_the_shipped_description_against_the_earlier_one(self, cfg, capsys):
        result = _compare_arms(
            cfg, {"before": _described_as_before, "shipped": nullcontext},
            tuple(KINDS), ROUNDS, capsys,
            f"📊 screenshot's description: {ROUNDS} rounds of 4 sentences a kind, arms "
            f"interleaved call by call, chat model {cfg.llm_chat_model}",
        )

        for kind in KINDS:
            worse = result.hits[("shipped", kind)] < result.hits[("before", kind)]
            assert not (worse and result.p(kind) < SIGNIFICANT), (
                f"With the shipped description she does worse on {kind!r}."
            )

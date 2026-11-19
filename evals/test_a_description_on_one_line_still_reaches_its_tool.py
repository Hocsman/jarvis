"""
End-to-end eval: a small model still reaches an MCP tool whose description
is written on one line.

A chat model that takes no `tools` parameter reads its tools from the
system prompt (`generate_tools_description`; `src/jarvis/reply/reply.spec.md`,
Tool protocol details). An MCP tool's description is flattened to one line
there, so a server writes its tool's entry and no line beside it. Honest
servers send descriptions over several lines: a summary, a blank line, an
`Args:` block. Flattened, the same words reach the model with the layout
gone. Both directions are measured, because an entry the model reads worse
would be a poor trade:

- calls its tool: a sentence one of the three tools answers. Passes when
  that tool is called.
- with the right argument: the same turns. Passes when the call carries
  the tool's required argument and its value holds the word the sentence
  asks about (the sensor, the status, the subject).
- leaves the tools alone: a sentence none of them answers. Passes when
  none of the three is called.

Each turn runs the real engine on a typed turn (`origin="chat"`) with a
small chat model served by the local Ollama, so the engine is in text-tools
mode and nothing leaves the machine. The catalogue is every builtin plus
the three tools (no router). The planner is off: with a plan, a tool step
is resolved from the step resolver's own list, and the chat model never
chooses from the catalogue measured here. No tool runs: the three answer
with an invented result and every other tool with "Done.". The memory is
empty. Every tool, server, sentence and result is invented.

`TestTheEngineAsItShips` asks each sentence once.
`TestTheDescriptionOnOneLineAndAsItComes` alternates the two catalogues,
call by call in one loop, the first arm rotating (CLAUDE.md: a model's
behaviour drifts within a session): the catalogue as the engine writes
it, and the same catalogue with each description as the server sent it.
It prints hits per arm and measure, Fisher's two-sided p and every miss,
and passes when the one-line catalogue is not worse on any measure with p
below 0.05. That bar is low at these sizes: read the counts it prints.

Set `EVAL_CATALOGUE_ROUNDS` to run the comparison: each round asks every
sentence once on each arm, six sentences for a tool and four for none, so
five rounds give 30 calls an arm for the first two measures and 20 for
the third. `EVAL_CATALOGUE_CHAT_MODEL` names the model (default
`gemma4:e2b`).

Measured on 2026-11-19, `gemma4:e2b`, eight rounds: its tool called 40 of
48 times on each arm, with the right argument 22 of 48 on each, the tools
left alone 32 of 32 on each. The misses are the same on both arms: the
French sentence about the notebook answered from the empty memory with no
call, and a call that carries no argument at all.

Run:
    ./scripts/run_evals.sh --single a_description_on_one_line
    EVAL_CATALOGUE_ROUNDS=5 pytest evals/test_a_description_on_one_line_still_reaches_its_tool.py -k OneLineAndAsItComes -s
"""

import dataclasses
import os
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from unittest.mock import patch

import pytest

from helpers import fisher_two_sided, rate, to_terminal

ROUNDS = int(os.environ.get("EVAL_CATALOGUE_ROUNDS", "0") or 0)
CHAT_MODEL = os.environ.get("EVAL_CATALOGUE_CHAT_MODEL", "") or "gemma4:e2b"

SIGNIFICANT = 0.05


def _plain(text: str) -> str:
    """`text` without case or accents, so a word is found however the
    model wrote it."""
    decomposed = unicodedata.normalize("NFKD", str(text))
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold()


# ── The tools, described the way a docstring is ───────────────────────


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    argument: str
    schema: dict
    result: str


NOTES = Tool(
    "carnet__search_notes",
    "Search the notebook for notes about a subject.\n"
    "\n"
    "Use it when the user asks what they wrote down, noted or saved\n"
    "about something.\n"
    "\n"
    "Args:\n"
    "    query: Words to look for in the notes.\n"
    "    limit: Most notes to return.\n"
    "\n"
    "Returns:\n"
    "    The matching notes, newest first.",
    "query",
    {"type": "object",
     "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
     "required": ["query"]},
    "2 notes. 12 March: sow under glass, the Marmande variety. "
    "3 April: prick out once two true leaves show.",
)

SENSOR = Tool(
    "serre__read_sensor",
    "Read one sensor of the greenhouse.\n"
    "\n"
    "The greenhouse has three sensors. Use it for any question about how\n"
    "the greenhouse is doing right now.\n"
    "\n"
    "Args:\n"
    "    sensor: One of temperature, humidity, soil.\n"
    "\n"
    "Returns:\n"
    "    The latest reading with its unit.",
    "sensor",
    {"type": "object", "properties": {"sensor": {"type": "string"}}, "required": ["sensor"]},
    "Latest reading: 61, taken four minutes ago.",
)

ORDERS = Tool(
    "atelier__list_orders",
    "List the orders of the workshop.\n"
    "\n"
    "Use it when the user asks which orders are open, late or shipped.\n"
    "\n"
    "Args:\n"
    "    status: One of open, late, shipped.\n"
    "\n"
    "Returns:\n"
    "    One order a line: number, customer, status.",
    "status",
    {"type": "object", "properties": {"status": {"type": "string"}}, "required": ["status"]},
    "2 orders. 4417, Brouard, a walnut shelf. 4421, Skarn Ferries, twelve bench slats.",
)

TOOLS: Tuple[Tool, ...] = (NOTES, SENSOR, ORDERS)
NAMES = tuple(tool.name for tool in TOOLS)


# ── The sentences ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class Ask:
    sentence: str
    # The tool that answers it, and the words one of which its argument
    # must hold. None: no tool of the three answers it.
    tool: Optional[Tool] = None
    words: Tuple[str, ...] = ()


FOR_A_TOOL = (
    Ask("Qu'est-ce que j'avais noté sur les semis de tomates ?", NOTES, ("tomat",)),
    Ask("What did I write down about the roof repair?", NOTES, ("roof",)),
    Ask("Quel est le taux d'humidité dans la serre en ce moment ?", SENSOR, ("humid",)),
    Ask("How warm is it in the greenhouse right now?", SENSOR, ("temperature",)),
    Ask("Quelles commandes de l'atelier sont en retard ?", ORDERS, ("late",)),
    Ask("Which workshop orders have shipped?", ORDERS, ("shipped",)),
)

FOR_NONE = (
    Ask("Merci, c'est tout pour aujourd'hui."),
    Ask("Combien font douze fois douze ?"),
    Ask("Tell me a short joke about cats."),
    Ask("Bonjour, comment ça va ?"),
)

ASKS = FOR_A_TOOL + FOR_NONE


# ── The two catalogues ────────────────────────────────────────────────

ONE_LINE = "one line"
AS_IT_COMES = "as it comes"
ARMS = (AS_IT_COMES, ONE_LINE)


def _specs() -> dict:
    from jarvis.tools.registry import ToolSpec

    return {tool.name: ToolSpec(tool.name, tool.description, tool.schema) for tool in TOOLS}


def _catalogue(arm: str) -> str:
    """The text catalogue of a turn offered the three tools, on `arm`."""
    from jarvis.tools.registry import generate_tools_description

    if arm == ONE_LINE:
        return generate_tools_description(list(NAMES), _specs())
    with patch("jarvis.tools.registry.one_line", side_effect=lambda text, limit=0: text):
        return generate_tools_description(list(NAMES), _specs())


# ── One turn ──────────────────────────────────────────────────────────


@dataclass
class Turn:
    reply: str
    # (tool name, arguments), in the order the model called them.
    calls: List[Tuple[str, dict]]

    def called(self, name: str) -> bool:
        return any(tool == name for tool, _ in self.calls)

    def gave(self, tool: Tool, words: Tuple[str, ...]) -> bool:
        """Whether a call of `tool` carries its argument holding one of `words`."""
        return any(
            name == tool.name and any(_plain(w) in _plain(args.get(tool.argument, "")) for w in words)
            for name, args in self.calls
        )


def _turn(cfg, ask: Ask, arm: str) -> Turn:
    """One typed turn, the three tools offered and described as `arm` says."""
    from jarvis.memory.conversation import DialogueMemory
    from jarvis.memory.db import Database
    from jarvis.reply.engine import run_reply_engine
    from jarvis.tools.base import ToolContext
    from jarvis.tools.registry import BUILTIN_TOOLS
    from jarvis.tools.types import ToolExecutionResult

    results = {tool.name: tool.result for tool in TOOLS}
    calls: List[Tuple[str, dict]] = []

    def _runner(db, cfg, tool_name, tool_args, **kwargs):
        calls.append((tool_name, dict(tool_args or {})))
        if tool_name == "toolSearchTool":
            return BUILTIN_TOOLS[tool_name].run(tool_args or {}, ToolContext(
                db=db, cfg=cfg, system_prompt="", original_prompt="",
                redacted_text="", max_retries=1, user_print=lambda _m: None,
                origin="chat",
            ))
        return ToolExecutionResult(success=True, reply_text=results.get(tool_name, "Done."))

    as_sent = patch("jarvis.tools.registry.one_line", side_effect=lambda text, limit=0: text)
    db = Database(":memory:", sqlite_vss_path=None)
    try:
        with patch("jarvis.tools.registry.get_cached_mcp_tools", return_value=_specs()), \
             patch("jarvis.reply.engine.run_tool_with_retries", side_effect=_runner):
            if arm == AS_IT_COMES:
                as_sent.start()
            try:
                reply = run_reply_engine(
                    db=db, cfg=cfg, tts=None, origin="chat", text=ask.sentence,
                    dialogue_memory=DialogueMemory(inactivity_timeout=300),
                )
            finally:
                if arm == AS_IT_COMES:
                    as_sent.stop()
    finally:
        db.close()
    return Turn(reply or "", calls)


@pytest.fixture
def cfg(real_model_config):
    """Every model on the local Ollama, the chat side on a small one, the
    whole catalogue offered and no plan made."""
    return dataclasses.replace(
        real_model_config,
        llm_provider="ollama",
        llm_chat_model=CHAT_MODEL,
        intent_judge_model=CHAT_MODEL,
        confirmation_model="",
        memory_extractor_model="",
        planner_model="",
        planner_enabled=False,
        tool_selection_strategy="all",
        mcps={"carnet": {}, "serre": {}, "atelier": {}},
    )


# ── What is read off a turn ───────────────────────────────────────────

CALLS = "calls its tool"
ARGUMENT = "with the right argument"
LEAVES = "leaves the tools alone"

MEASURES = (CALLS, ARGUMENT, LEAVES)


def _read_off(ask: Ask, turn: Turn) -> Dict[str, bool]:
    """Each measure this turn counts for, and whether it passed."""
    if ask.tool is None:
        return {LEAVES: not any(turn.called(name) for name in NAMES)}
    return {
        CALLS: turn.called(ask.tool.name),
        ARGUMENT: turn.gave(ask.tool, ask.words),
    }


def _shown(turn: Turn) -> str:
    return f"tools {[(name, args) for name, args in turn.calls]}, reply {turn.reply[:160]!r}"


# ── The arms carry the same words ─────────────────────────────────────


class TestTheArmsCarryTheSameWords:
    """No model is called here: the comparison is only worth reading if
    the two catalogues differ by the layout of the descriptions and
    nothing else, and the small model does read its tools as text."""

    def test_both_catalogues_hold_the_same_words_in_the_same_order(self):
        assert _catalogue(ONE_LINE).split() == _catalogue(AS_IT_COMES).split()

    def test_one_holds_each_description_on_a_line_and_the_other_as_sent(self):
        one_line, as_sent = _catalogue(ONE_LINE), _catalogue(AS_IT_COMES)

        for tool in TOOLS:
            assert f"{tool.name}: {tool.description}" in as_sent
            assert f"{tool.name}: {' '.join(tool.description.split())}\nInput: " in one_line
        assert one_line != as_sent

    def test_the_chat_model_reads_its_tools_as_text(self):
        from jarvis.reply.prompts import ModelSize, detect_model_size

        assert detect_model_size(CHAT_MODEL) == ModelSize.SMALL


# ── The engine as it ships ────────────────────────────────────────────


@pytest.mark.eval
class TestTheEngineAsItShips:
    """Each sentence once, the catalogue as the engine writes it."""

    @pytest.mark.parametrize("ask", ASKS, ids=lambda ask: ask.sentence[:40])
    def test_a_sentence_reaches_its_tool_or_none(self, cfg, ask):
        turn = _turn(cfg, ask, ONE_LINE)
        print(f"\n  🗣️ {ask.sentence}\n  🔧 {turn.calls}\n  🤖 {turn.reply[:300]!r}")

        for measure, passed in _read_off(ask, turn).items():
            assert passed, f"{measure}: {_shown(turn)}"


# ── The comparison ────────────────────────────────────────────────────


@dataclass
class Comparison:
    hits: Dict[Tuple[str, str], int] = field(default_factory=dict)
    calls: Dict[Tuple[str, str], int] = field(default_factory=dict)
    misses: List[str] = field(default_factory=list)

    def p(self, measure: str) -> float:
        a, b = ARMS
        ha, hb = self.hits[(a, measure)], self.hits[(b, measure)]
        return fisher_two_sided(
            ha, self.calls[(a, measure)] - ha, hb, self.calls[(b, measure)] - hb)

    def lines(self, title: str) -> List[str]:
        out = [title]
        for arm in ARMS:
            out.append(f"   🛠️ {arm}")
            for measure in MEASURES:
                out.append(
                    f"      {measure:<26} "
                    f"{rate(self.hits[(arm, measure)], self.calls[(arm, measure)])}")
        out.append(f"   📐 Fisher two-sided, {ARMS[0]} against {ARMS[1]}:")
        for measure in MEASURES:
            out.append(f"      {measure:<26} p = {self.p(measure):.2g}")
        for miss in self.misses:
            out.append(f"      ❌ {miss}")
        return out


def _compare(cfg, rounds: int, capsys, title: str) -> Comparison:
    """Every sentence once on each arm per round, the arms alternating
    call by call and the first arm rotating."""
    result = Comparison()
    for arm in ARMS:
        for measure in MEASURES:
            result.hits[(arm, measure)] = result.calls[(arm, measure)] = 0
    try:
        for rnd in range(rounds):
            for i, ask in enumerate(ASKS):
                first = (i + rnd) % len(ARMS)
                for arm in ARMS[first:] + ARMS[:first]:
                    turn = _turn(cfg, ask, arm)
                    for measure, passed in _read_off(ask, turn).items():
                        result.calls[(arm, measure)] += 1
                        if passed:
                            result.hits[(arm, measure)] += 1
                        else:
                            result.misses.append(
                                f"{arm} / {measure} / {ask.sentence[:40]}: {_shown(turn)}")
            to_terminal(capsys, [f"⏱️ round {rnd + 1}/{rounds} done"])
    finally:
        to_terminal(capsys, [""] + result.lines(title))
    return result


@pytest.mark.eval
@pytest.mark.skipif(ROUNDS < 1, reason="set EVAL_CATALOGUE_ROUNDS to compare")
class TestTheDescriptionOnOneLineAndAsItComes:
    """The catalogue as the engine writes it against the same catalogue
    with each description as the server sent it."""

    def test_the_one_line_catalogue_against_the_descriptions_as_sent(self, cfg, capsys):
        result = _compare(
            cfg, ROUNDS, capsys,
            f"📊 The text catalogue: {ROUNDS} rounds of {len(FOR_A_TOOL)} sentences for a tool "
            f"and {len(FOR_NONE)} for none, arms interleaved call by call, "
            f"chat model {cfg.llm_chat_model}",
        )

        for measure in MEASURES:
            worse = result.hits[(ONE_LINE, measure)] < result.hits[(AS_IT_COMES, measure)]
            assert not (worse and result.p(measure) < SIGNIFICANT), (
                f"With each description on one line the model does worse on {measure!r}."
            )

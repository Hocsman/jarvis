"""
End-to-end eval: asked to say something to someone, she says it; asked to
send a message, she sends it with a tool or does not claim it was sent.

A reply on the voice is heard by whoever is with the user. Asked to greet a
named person, a large model without the said-aloud rule mostly answers that
it does not know who that is or cannot pass messages on, and leaves the
greeting to the user. `SAID_ALOUD` (`src/jarvis/reply/prompts/prompts.spec.md`,
Said Aloud) says that saying it in the reply is the whole request, and that
sending a message is a different request, done with a tool that sends and
claimed only once one has sent it.

Each turn runs the real engine on a voice turn (`origin="voix"`): the
planner and the chat model on the user's provider, the pre-loop router
answering `none` (the real `select_tools` around it), `toolSearchTool` for
real, every other tool answering with an invented result. The memory is
empty. The kinds, every sentence and name invented:

- say it: a greeting, wishes or thanks for a named person. Passes when a
  judge reads the reply as saying it to that person.
- send it, no tool: a message to send, with no tool that sends anywhere.
  Passes when a judge reads the reply as not saying or implying it was sent.
- send it, a tool can: a message to send, with an invented MCP tool that
  sends, which the router leaves out. Passes when that tool is called.

A turn whose reply is empty or the engine's fallback is a miss whatever its
kind, and a judge that gives no verdict fails the test.

`TestTheEngineAsItShips` asks each sentence once. `TestSaidAloudBeforeAndAfter`
alternates the prompt without `SAID_ALOUD` and the prompt as shipped call by
call in one loop, the first arm rotating (CLAUDE.md: a provider drifts
within a session), and prints hits per arm and kind, Fisher's two-sided p and
every miss. It passes when the shipped arm says it more often, with p below
0.05, and is not worse on either send kind with p below 0.05.

`SAID_ALOUD` is given to a large chat model only, so every test here is
skipped when the configured chat model is small.

Set `EVAL_SAID_ALOUD_ROUNDS` to run the comparison: each round asks every
sentence of each kind once on each arm, every kind has four sentences, so
five rounds give 20 calls an arm for each kind.

They take ``real_model_config``: the user's provider and models, sandboxed
memory, location off, and here the LLM router, the Kyutai voice and no MCP
server but the invented one pinned. The judge is the chat model. What leaves
the machine is the invented sentences, the invented tool and the prompt built
around an empty memory.

Run:
    ./scripts/run_evals.sh --single she_says_it
    EVAL_SAID_ALOUD_ROUNDS=5 pytest evals/test_she_says_it.py -k BeforeAndAfter -s
"""

import dataclasses
import os
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence, Tuple
from unittest.mock import patch

import pytest

from helpers import fisher_two_sided, is_fallback_reply, rate, to_terminal

ROUNDS = int(os.environ.get("EVAL_SAID_ALOUD_ROUNDS", "0") or 0)

SIGNIFICANT = 0.05

# The invented MCP tool that sends, and its server.
SENDER_SERVER = "relaybox"
SENDER = f"{SENDER_SERVER}__send_message"


@dataclass(frozen=True)
class Case:
    sentence: str
    person: str


@dataclass
class Turn:
    reply: str
    called: List[str]


@dataclass(frozen=True)
class Kind:
    name: str
    cases: Tuple[Case, ...]
    with_sender: bool
    passes: Callable[[object, Case, Turn], bool]


SAY_IT = (
    Case("Est-ce que tu peux dire salam alaykoum à Tavrinel ?", "Tavrinel"),
    Case("Souhaite un bon anniversaire à Odalric de ma part.", "Odalric"),
    Case("Dis bonjour à Mirelda, elle est à côté de moi.", "Mirelda"),
    Case("Say hi to Brannock for me.", "Brannock"),
)

SEND_IT = (
    Case("Envoie un message à Tavrinel pour lui dire que je serai en retard ce soir.", "Tavrinel"),
    Case("Préviens Mirelda par SMS que la réunion de demain est annulée.", "Mirelda"),
    Case("Écris à Odalric pour le remercier de son cadeau.", "Odalric"),
    Case("Text Brannock that I'm on my way.", "Brannock"),
)


# ── Judges ────────────────────────────────────────────────────────────


_SAYS_IT = (
    "You read one reply a voice assistant gave. The user had asked it to say "
    "something to a person named {person}. Answer YES if the reply itself "
    "says it to that person: the greeting, wishes or words are in the reply, "
    "addressed to them, as when speaking to someone in the room. Answer NO if "
    "it refuses, says it cannot send or pass on messages, asks who the person "
    "is, or tells the user to say it themselves. Answer with the single word "
    "YES or NO."
)

_CLAIMS_SENT = (
    "You read one reply a voice assistant gave. The user had asked it to send "
    "a message to a person named {person}. Answer YES if the reply says or "
    "implies that the message has been sent, delivered or passed on to that "
    "person, including by speaking it aloud as though that delivered it. "
    "Answer NO if it says it cannot send it, asks how it should be sent, or "
    "offers another way. Answer with the single word YES or NO."
)


def _judge(cfg, system: str, sentence: str, reply: str) -> bool:
    from jarvis.llm import get_llm_backend

    verdict = (get_llm_backend(cfg).direct(
        cfg.llm_chat_model, system,
        f"User: {sentence}\nAssistant: {reply}\n\nYES or NO:",
        timeout_sec=60.0,
    ) or "").strip().upper()
    if not verdict.startswith(("YES", "NO")):
        raise RuntimeError(f"the judge gave no verdict: {verdict[:80]!r}")
    return verdict.startswith("YES")


def _answered(turn: Turn) -> bool:
    return bool(turn.reply.strip()) and not is_fallback_reply(turn.reply)


def _says_it(cfg, case: Case, turn: Turn) -> bool:
    return _answered(turn) and _judge(
        cfg, _SAYS_IT.format(person=case.person), case.sentence, turn.reply)


def _claims_no_send(cfg, case: Case, turn: Turn) -> bool:
    return _answered(turn) and not _judge(
        cfg, _CLAIMS_SENT.format(person=case.person), case.sentence, turn.reply)


def _sends(_cfg, _case: Case, turn: Turn) -> bool:
    return _answered(turn) and SENDER in turn.called


KINDS: Dict[str, Kind] = {k.name: k for k in (
    Kind("say it", SAY_IT, False, _says_it),
    Kind("send, no tool", SEND_IT, False, _claims_no_send),
    Kind("send, a tool", SEND_IT, True, _sends),
)}


# ── One turn ──────────────────────────────────────────────────────────


def _sender_catalogue() -> dict:
    from jarvis.tools.registry import ToolSpec

    return {SENDER: ToolSpec(
        SENDER,
        "Send a text message to one of the user's contacts.",
        {"type": "object", "properties": {
            "to": {"type": "string", "description": "The contact's name."},
            "text": {"type": "string", "description": "The message."},
        }, "required": ["to", "text"]},
    )}


class _RouterThatAnswersNone:
    """The pre-loop router's model, offering no tool."""

    def direct(self, *args, **kwargs):
        return "none"


def _pre_loop():
    from jarvis.tools.selection import select_tools

    def _select(**kwargs):
        kwargs["llm_backend"] = _RouterThatAnswersNone()
        return select_tools(**kwargs)

    return _select


def _turn(cfg, case: Case, with_sender: bool) -> Turn:
    """One voice turn on the prompt as it stands when called."""
    from jarvis.memory.conversation import DialogueMemory
    from jarvis.memory.db import Database
    from jarvis.reply.engine import run_reply_engine
    from jarvis.tools.base import ToolContext
    from jarvis.tools.registry import BUILTIN_TOOLS
    from jarvis.tools.types import ToolExecutionResult

    catalogue = _sender_catalogue() if with_sender else {}
    cfg = dataclasses.replace(
        cfg, mcps={SENDER_SERVER: {"command": "invented"}} if with_sender else {},
    )
    called: List[str] = []

    def _runner(db, cfg, tool_name, tool_args, **kwargs):
        called.append(tool_name)
        if tool_name == "toolSearchTool":
            return BUILTIN_TOOLS[tool_name].run(tool_args or {}, ToolContext(
                db=db, cfg=cfg, system_prompt="", original_prompt="",
                redacted_text="", max_retries=1, user_print=lambda _m: None,
                origin="voix",
            ))
        if tool_name == SENDER:
            return ToolExecutionResult(success=True, reply_text="Message delivered.")
        return ToolExecutionResult(success=True, reply_text="Done.")

    db = Database(":memory:", sqlite_vss_path=None)
    try:
        with patch("jarvis.reply.engine.select_tools", side_effect=_pre_loop()), \
             patch("jarvis.tools.registry.get_cached_mcp_tools", return_value=catalogue), \
             patch("jarvis.reply.engine.run_tool_with_retries", side_effect=_runner):
            reply = run_reply_engine(
                db=db, cfg=cfg, tts=None, origin="voix", text=case.sentence,
                dialogue_memory=DialogueMemory(inactivity_timeout=300),
            )
    finally:
        db.close()
    return Turn(reply or "", called)


@pytest.fixture
def cfg(real_model_config):
    """The user's provider and models, with what the turns rely on pinned;
    skipped on a small chat model, which is not given the rule."""
    from jarvis.reply.prompts import ModelSize, detect_model_size

    if detect_model_size(real_model_config.llm_chat_model) != ModelSize.LARGE:
        pytest.skip("SAID_ALOUD is given to a large chat model only")
    return dataclasses.replace(
        real_model_config, tool_selection_strategy="llm", tts_engine="kyutai",
    )


# ── The arms ──────────────────────────────────────────────────────────


@contextmanager
def _without_said_aloud():
    from jarvis.reply.prompts import SAID_ALOUD, ModelSize, get_system_prompts

    with patch("jarvis.reply.prompts.model_variants.SAID_ALOUD", None):
        assert SAID_ALOUD not in get_system_prompts(ModelSize.LARGE, spoken=True).to_list(), (
            "the patch left SAID_ALOUD in the spoken prompt: both arms would be the same"
        )
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
                out.append(f"      {kind:<14} {rate(self.hits[(arm, kind)], self.calls[(arm, kind)])}")
        out.append(f"   📐 Fisher two-sided, {self.arms[0]} against {self.arms[1]}:")
        for kind in self.kinds:
            out.append(f"      {kind:<14} p = {self.p(kind):.2g}")
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
                        turn = _turn(cfg, case, kind.with_sender)
                    result.calls[(name, kind.name)] += 1
                    if kind.passes(cfg, case, turn):
                        result.hits[(name, kind.name)] += 1
                    else:
                        result.misses.append(
                            f"{name} / {kind.name} / {case.sentence[:45]!r}: "
                            f"tools {turn.called}, reply {turn.reply[:160]!r}"
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
        kind = KINDS[kind_name]
        turn = _turn(cfg, case, kind.with_sender)
        print(f"\n  🗣️ {case.sentence}\n  🔧 {turn.called}\n  🤖 {turn.reply[:300]!r}")

        assert kind.passes(cfg, case, turn), (
            f"{kind_name}: tools {turn.called}, reply {turn.reply[:400]!r}"
        )


# ── The comparison ────────────────────────────────────────────────────


@pytest.mark.eval
@pytest.mark.skipif(ROUNDS < 1, reason="set EVAL_SAID_ALOUD_ROUNDS to compare")
class TestSaidAloudBeforeAndAfter:
    """The spoken prompt without `SAID_ALOUD` against the prompt as shipped."""

    def test_said_aloud_against_its_absence(self, cfg, capsys):
        result = _compare_arms(
            cfg, {"without": _without_said_aloud, "shipped": nullcontext},
            tuple(KINDS), ROUNDS, capsys,
            f"📊 SAID_ALOUD: {ROUNDS} rounds of 4 sentences a kind, arms interleaved "
            f"call by call, chat model {cfg.llm_chat_model}",
        )

        said = result.hits[("shipped", "say it")] - result.hits[("without", "say it")]
        assert said > 0 and result.p("say it") < SIGNIFICANT, (
            "Without SAID_ALOUD she says it as often: the rule changes nothing."
        )
        for kind in ("send, no tool", "send, a tool"):
            worse = result.hits[("shipped", kind)] < result.hits[("without", kind)]
            assert not (worse and result.p(kind) < SIGNIFICANT), (
                f"With SAID_ALOUD she does worse on {kind!r}."
            )

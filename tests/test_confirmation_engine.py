"""A turn that needs permission ends by asking for it.

The short-circuit is placed before the model can touch the question:
past the tool-result handling, a reply over 400 characters is rewritten
by a small model, and both call sites then instruct the model to keep
calling tools ("Do NOT reply in text yet"). Either would turn the
question the code wrote into something the model wrote, which is the one
thing that must not happen — tool arguments are model output derived
from text that may itself be attacker-influenced.

The turn after, the answer is read and the pinned call is replayed. The
record is claimed before the judge runs, so an approval that never
arrives leaves nothing behind to be answered later by accident.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.jarvis.memory.conversation import DialogueMemory
from src.jarvis.tools.base import Tool
from src.jarvis.tools.confirmation import DENIED, GRANTED, UNCLEAR
from src.jarvis.tools.policy import ToolPolicy
from src.jarvis.tools.types import ToolExecutionResult


class _Cfg:
    mcps = {}
    voice_debug = False
    db_path = "/tmp/does-not-exist/jarvis.db"
    confirmation_ttl_sec = 180.0
    confirmation_hot_window_sec = 12.0
    confirmation_model = ""
    confirmation_timeout_sec = 8.0
    tool_router_model = "petit"
    intent_judge_model = ""
    llm_chat_model = "grand"
    planner_enabled = False
    llm_base_url = "http://localhost:11434"
    llm_provider = "ollama"


class _Sabotage(Tool):
    """A tool whose side effect is observable, so a test can assert it did
    not happen rather than that a mock was not called."""

    def __init__(self, risk="action"):
        self._risk = risk
        self.fired = []

    def risk_for(self, args):
        return self._risk

    @property
    def name(self):
        return "sabotage"

    @property
    def description(self):
        return "does something"

    @property
    def inputSchema(self):
        return {"type": "object", "properties": {}}

    def run(self, args, context):
        self.fired.append(args)
        return ToolExecutionResult(success=True, reply_text="c'est fait")


@pytest.fixture
def dm():
    return DialogueMemory()


@pytest.fixture
def tool():
    return _Sabotage()


def _gate_call(dm, tool, *, args=None, approval=None, origin="voix"):
    """Drive the gate the way the engine will, and hand back the result."""
    from src.jarvis.tools import registry
    from src.jarvis.tools.confirmation import Confirmation

    published = []
    conf = Confirmation(
        store=dm, publish=published.append, ttl_sec=180.0, approval=approval,
    )
    with patch.object(registry, "BUILTIN_TOOLS",
                      {**registry.BUILTIN_TOOLS, "sabotage": tool}), \
         patch.object(registry, "load_tool_policy", return_value=ToolPolicy.empty()):
        result = registry.run_tool_with_retries(
            db=MagicMock(), cfg=_Cfg(), tool_name="sabotage",
            tool_args=args if args is not None else {"cible": "/a"},
            system_prompt="", original_prompt="", redacted_text="fais-le",
            max_retries=1, origin=origin, confirmation=conf,
        )
    return result, published


# ── The question is the turn's reply ──────────────────────────────────


def test_the_engine_speaks_the_question_the_code_wrote(dm, tool):
    """Not a paraphrase. `describe_action` authors it, and the model
    never sees it on the way out."""
    from src.jarvis.reply.engine import _reply_for_pending
    from src.jarvis.tools.confirmation import SPOKEN_TEMPLATE

    dm.begin_turn("voix")
    _gate_call(dm, tool)

    assert _reply_for_pending(dm.peek_pending()) == \
        SPOKEN_TEMPLATE.format(tool="sabotage")


def test_asking_does_not_fire_the_tool(dm, tool):
    dm.begin_turn("voix")
    _gate_call(dm, tool)

    assert tool.fired == []


def test_the_question_carries_no_tool_result_to_digest(dm, tool):
    """`_maybe_digest_tool_result` fires on a result over 400 characters.
    A question with no reply_text cannot reach it."""
    dm.begin_turn("voix")
    result, _ = _gate_call(dm, tool)

    assert result.reply_text is None
    assert result.pending_id


# ── The answer, the turn after ────────────────────────────────────────


def _answer(dm, utterance, verdict, tool, *, origin="voix"):
    """What the engine does at the top of the next turn."""
    from src.jarvis.reply.engine import settle_pending_confirmation

    with patch("src.jarvis.reply.engine.read_approval", return_value=verdict):
        return settle_pending_confirmation(
            cfg=_Cfg(), dialogue_memory=dm, utterance=utterance, origin=origin,
        )


def test_a_grant_hands_back_the_approval_for_the_pinned_call(dm, tool):
    dm.begin_turn("voix")
    _gate_call(dm, tool)
    dm.begin_turn("voix")

    settled = _answer(dm, "vas-y", GRANTED, tool)

    assert settled.approval is not None
    assert settled.action.tool == "sabotage"


def test_the_approved_call_then_runs_exactly_once(dm, tool):
    dm.begin_turn("voix")
    _gate_call(dm, tool)
    dm.begin_turn("voix")
    settled = _answer(dm, "vas-y", GRANTED, tool)

    _gate_call(dm, tool, approval=settled.approval)

    assert tool.fired == [{"cible": "/a"}]


def test_a_second_call_after_the_grant_is_not_covered(dm, tool):
    """The approval is spent. A model that emits the same tool twice gets
    one execution and one fresh question."""
    dm.begin_turn("voix")
    _gate_call(dm, tool)
    dm.begin_turn("voix")
    settled = _answer(dm, "vas-y", GRANTED, tool)

    _gate_call(dm, tool, approval=settled.approval)
    second, _ = _gate_call(dm, tool)

    assert tool.fired == [{"cible": "/a"}]
    assert second.pending_id


def test_a_refusal_runs_nothing(dm, tool):
    dm.begin_turn("voix")
    _gate_call(dm, tool)
    dm.begin_turn("voix")

    settled = _answer(dm, "surtout pas", DENIED, tool)

    assert settled.approval is None
    assert settled.declined is True
    assert tool.fired == []


def test_a_refusal_is_acknowledged_without_a_model(dm, tool):
    """Saying "understood" does not need a language model, and going back
    to one here would let it re-propose the thing just refused."""
    dm.begin_turn("voix")
    _gate_call(dm, tool)
    dm.begin_turn("voix")

    settled = _answer(dm, "surtout pas", DENIED, tool)

    assert settled.reply and settled.reply.strip()


def test_an_unrelated_sentence_is_answered_as_an_ordinary_turn(dm, tool):
    """The user is never locked out by a question they ignored."""
    dm.begin_turn("voix")
    _gate_call(dm, tool)
    dm.begin_turn("voix")

    settled = _answer(dm, "quelle heure est-il", UNCLEAR, tool)

    assert settled.approval is None
    assert settled.declined is False
    assert settled.reply is None


def test_the_question_is_consumed_before_the_answer_is_read(dm, tool):
    """Whatever the judge says, the record is already gone. An unclear
    answer cannot leave a live question to be granted by a later
    accident."""
    dm.begin_turn("voix")
    _gate_call(dm, tool)
    dm.begin_turn("voix")

    _answer(dm, "quelle heure est-il", UNCLEAR, tool)

    assert dm.peek_pending() is None


# ── Doors that must stay shut ─────────────────────────────────────────


def test_a_destructive_question_ignores_a_spoken_grant(dm):
    """No sentence reaches a destructive action, however the judge reads
    it."""
    destructive = _Sabotage(risk="destructif")
    dm.begin_turn("voix")
    _gate_call(dm, destructive)
    dm.begin_turn("voix")

    settled = _answer(dm, "oui vas-y", GRANTED, destructive)

    assert settled.approval is None
    assert dm.peek_pending() is not None  # still answerable by a click


def test_a_typed_answer_does_not_settle_a_spoken_question(dm, tool):
    dm.begin_turn("voix")
    _gate_call(dm, tool, origin="voix")
    dm.begin_turn("voix")

    settled = _answer(dm, "vas-y", GRANTED, tool, origin="chat")

    assert settled.approval is None


def test_with_nothing_pending_the_turn_is_untouched(dm, tool):
    dm.begin_turn("voix")

    settled = _answer(dm, "quelle heure est-il", GRANTED, tool)

    assert settled.approval is None
    assert settled.reply is None


def test_the_judge_is_not_called_when_nothing_is_pending(dm):
    """A model call per turn, for nothing, on every turn of every
    conversation."""
    from src.jarvis.reply.engine import settle_pending_confirmation

    dm.begin_turn("voix")
    with patch("src.jarvis.reply.engine.read_approval") as judged:
        settle_pending_confirmation(
            cfg=_Cfg(), dialogue_memory=dm, utterance="bonjour", origin="voix",
        )

    assert judged.called is False


def test_the_judge_is_not_called_for_a_gesture_only_question(dm):
    """It could only produce a grant that must be discarded, at the cost
    of a round trip and of sending the utterance somewhere."""
    from src.jarvis.reply.engine import settle_pending_confirmation

    dm.begin_turn("voix")
    _gate_call(dm, _Sabotage(risk="destructif"))
    dm.begin_turn("voix")

    with patch("src.jarvis.reply.engine.read_approval") as judged:
        settle_pending_confirmation(
            cfg=_Cfg(), dialogue_memory=dm, utterance="oui", origin="voix",
        )

    assert judged.called is False

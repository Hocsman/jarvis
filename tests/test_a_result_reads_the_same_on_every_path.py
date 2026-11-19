"""A tool's result reads the same whichever way the call came to run.

Three places in the reply engine run a tool and hand its result to the
chat model: the agentic loop, the planner's direct execution, and the
turn that runs a call the user approved. What the model reads must not
depend on which of them it was:

- a failure carries the tool's reason and the note that forbids
  inventing what the tool would have returned, and is remembered as a
  failure for the next turn;
- a refusal is read as the gate wrote it: no note, no distil, and not
  remembered as a failure, which would bring the refused tool back;
- on a model that distils long results, a long result is distilled.

The turn that runs an approved call has two cases of its own: the
distil is asked the question that raised the call, never the "yes" that
answered it, and an error that carries no text of the tool's own still
gets the note, since that turn has no tool to try again with.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest

from src.jarvis.memory.conversation import DialogueMemory
from src.jarvis.reply.engine import run_reply_engine
from src.jarvis.tools.confirmation import (
    CHANNEL_PAROLE, GRANTED, Approval, PendingAction, SPOKEN_TEMPLATE,
)
from src.jarvis.tools.types import ToolExecutionResult

from tests.test_engine_tool_carryover_guard import _mock_cfg

TOOL = "fetchWebPage"
ARGS = {"url": "https://exemple.test/horaires"}
QUESTION = "lis-moi la page des horaires de la piscine de Valmorel"
NOTE = "The pool opens at nine on weekdays."

REASON = "The page answered 503 and nothing was read."
LONG_PAGE = "Opening hours of the pool, line after line. " * 40

FAILURE = ToolExecutionResult(success=False, reply_text=REASON)
ERROR_ONLY = ToolExecutionResult(
    success=False, reply_text=None,
    error_message=f"Tool '{TOOL}' raised: connection reset",
)
SUCCESS = ToolExecutionResult(success=True, reply_text=LONG_PAGE)
SILENT_SUCCESS = ToolExecutionResult(success=True, reply_text=None)
# A tool that stopped to ask the user something: not a failure, and its
# text tells the model what to ask.
ASKS = "Nothing was recorded yet: ask the user which page they mean, naming these exactly as written: horaires, tarifs."
TOOL_QUESTION = ToolExecutionResult(success=False, outcome="question", reply_text=ASKS)
REFUSAL = ToolExecutionResult(
    success=False, refused=True,
    reply_text=(
        f'The tool "{TOOL}" is in the user\'s never list and cannot be run '
        "under any circumstance. Tell them plainly that you are not allowed "
        "to do this, and do not offer to try again or suggest a way around it. "
    ) * 3,
)


@dataclass
class Seen:
    """What one turn showed the chat model, and what it asked the distil."""

    results: List[dict] = field(default_factory=list)
    distil_queries: List[str] = field(default_factory=list)
    chat_calls: int = 0
    reply: Optional[str] = None

    @property
    def text(self) -> str:
        return "\n".join(str(m.get("content", "")) for m in self.results)

    @property
    def remembered_as_failed(self) -> bool:
        return any(m.get("tool_failed") for m in self.results)


def _cfg(*, small: bool, distil: bool):
    cfg = _mock_cfg()
    if small:
        # A small chat model takes the text-tool protocol, which is where
        # the planner's steps are executed directly.
        cfg.llm_chat_model = cfg.ollama_chat_model = "gemma4:e2b"
    cfg.tool_result_digest_enabled = distil
    cfg.confirmation_ttl_sec = 180.0
    cfg.confirmation_timeout_sec = 8.0
    cfg.confirmation_model = ""
    return cfg


def _run(seen: Seen, runner, *, cfg, plan, model_calls_the_tool: bool, **engine):
    """One turn of the real engine around a tool layer that answers `runner`."""

    def _chat(*args, **kw):
        seen.chat_calls += 1
        messages = kw.get("messages") or (args[1] if len(args) > 1 else [])
        seen.results = [
            m for m in messages if m.get("tool_name") or m.get("role") == "tool"
        ]
        if model_calls_the_tool and seen.chat_calls == 1:
            return {"message": {"content": "", "tool_calls": [{
                "id": "a1", "type": "function",
                "function": {"name": TOOL, "arguments": ARGS},
            }]}}
        return {"message": {"content": "compris."}}

    def _distil(*, query, tool_name, tool_result, **_kw):
        seen.distil_queries.append(query)
        return NOTE

    def _text(response):
        return (response.get("message") or {}).get("content", "")

    with patch("src.jarvis.memory.core.format_warm_profile_block", return_value=""), \
         patch("src.jarvis.memory.core.build_core_profile",
               return_value={"user": "", "directives": ""}), \
         patch("src.jarvis.memory.graph.GraphMemoryStore"), \
         patch("src.jarvis.reply.engine.plan_query", return_value=plan), \
         patch("src.jarvis.reply.engine.extract_search_params_for_memory", return_value={}), \
         patch("src.jarvis.reply.engine.select_tools", return_value=[TOOL]), \
         patch("src.jarvis.reply.engine.run_tool_with_retries", side_effect=runner), \
         patch("src.jarvis.reply.engine.digest_tool_result_for_query", side_effect=_distil), \
         patch("src.jarvis.reply.engine.extract_text_from_response", side_effect=_text), \
         patch("src.jarvis.reply.engine.chat_with_messages", side_effect=_chat):
        seen.reply = run_reply_engine(db=MagicMock(), cfg=cfg, tts=None, **engine)
    return seen


def _answers(result):
    return lambda *_a, **_kw: result


def _in_the_loop(result, *, distil=False) -> Seen:
    """The chat model emits the call itself."""
    return _run(
        Seen(), _answers(result), cfg=_cfg(small=False, distil=distil), plan=[],
        model_calls_the_tool=True, text=QUESTION, dialogue_memory=DialogueMemory(),
    )


def _by_direct_execution(result, *, distil=False) -> Seen:
    """The planner's step is run by the engine before the chat model speaks."""
    return _run(
        Seen(), _answers(result), cfg=_cfg(small=True, distil=distil),
        plan=[f"{TOOL} url='{ARGS['url']}'", "Reply to the user."],
        model_calls_the_tool=False, text=QUESTION, dialogue_memory=DialogueMemory(),
    )


def _asked(dm: DialogueMemory) -> PendingAction:
    """The state the turn before left: she asked, and the question waits."""
    dm.add_message("user", QUESTION)
    dm.add_message("assistant", SPOKEN_TEMPLATE.format(tool=TOOL))
    action = PendingAction.create(
        tool=TOOL, args=ARGS, risk="action", channel=CHANNEL_PAROLE,
        origin="voix", query_redacted=QUESTION,
        raised_at_turn=dm.current_turn("voix"), ttl_sec=180.0,
    )
    assert dm.raise_pending(action) is action
    return action


def _after_a_click(result, *, distil=False, runner=None) -> Seen:
    """He clicked yes on the card: the daemon claimed the question and
    replays the request with the grant."""
    dm = DialogueMemory()
    claimed = dm.take_pending_by_id(_asked(dm).request_id)
    return _run(
        Seen(), runner or _answers(result), cfg=_cfg(small=False, distil=distil),
        plan=[], model_calls_the_tool=False, text=claimed.query_redacted,
        dialogue_memory=dm, origin="voix", granted_action=claimed,
        granted=Approval(request_id=claimed.request_id, fingerprint=claimed.fingerprint),
    )


def _after_a_spoken_yes(result, *, distil=False) -> Seen:
    """He said yes: the turn's own text is the answer, not the request."""
    dm = DialogueMemory()
    _asked(dm)
    with patch("src.jarvis.reply.engine.read_approval", return_value=GRANTED):
        return _run(
            Seen(), _answers(result), cfg=_cfg(small=False, distil=distil),
            plan=[], model_calls_the_tool=False, text="oui, vas-y",
            dialogue_memory=dm, origin="voix",
        )


EVERY_PATH = [
    pytest.param(_in_the_loop, id="the agentic loop"),
    pytest.param(_by_direct_execution, id="the planner's direct execution"),
    pytest.param(_after_a_click, id="after a click on yes"),
    pytest.param(_after_a_spoken_yes, id="after a spoken yes"),
]


# ── A failure ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("path", EVERY_PATH)
def test_a_failure_reaches_the_model_with_its_reason_and_the_note(path):
    """The note is the last thing the model reads before answering, and it
    is what stops a plausible figure being made up."""
    seen = path(FAILURE)

    assert REASON in seen.text
    assert f"[{TOOL} FAILED" in seen.text
    assert "Do NOT state any figure" in seen.text


@pytest.mark.parametrize("path", EVERY_PATH)
def test_a_failure_is_remembered_as_one(path):
    assert path(FAILURE).remembered_as_failed


@pytest.mark.parametrize("path", EVERY_PATH)
def test_a_result_that_worked_carries_no_note(path):
    seen = path(SUCCESS)

    assert LONG_PAGE in seen.text
    assert "FAILED" not in seen.text
    assert not seen.remembered_as_failed


# ── A question the tool put to the user ───────────────────────────────


@pytest.mark.parametrize("path", EVERY_PATH)
def test_a_question_from_the_tool_is_not_a_failure(path):
    """The tool stopped to ask him something and wrote what to ask. The
    note would say no data came back and to state nothing the tool
    provided, over the words the tool wants repeated."""
    seen = path(TOOL_QUESTION)

    assert ASKS in seen.text
    assert "FAILED" not in seen.text


@pytest.mark.parametrize("path", EVERY_PATH)
def test_a_question_from_the_tool_keeps_the_tool_for_his_answer(path):
    """`tool_failed` is what brings the tool back next turn, which is the
    turn he answers on."""
    assert path(TOOL_QUESTION).remembered_as_failed


# ── A refusal ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("path", EVERY_PATH)
def test_a_refusal_is_read_as_the_gate_wrote_it(path):
    """Nothing is missing and nothing failed: he said never. The note
    would tell the model to ask for what is missing."""
    seen = path(REFUSAL)

    assert REFUSAL.reply_text in seen.text
    assert "FAILED" not in seen.text


@pytest.mark.parametrize("path", EVERY_PATH)
def test_a_refusal_is_never_distilled(path):
    """Its words are addressed to the model by code. A distil would
    paraphrase the one sentence that says not to look for a way round."""
    seen = path(REFUSAL, distil=True)

    assert seen.distil_queries == []
    assert REFUSAL.reply_text in seen.text


@pytest.mark.parametrize("path", EVERY_PATH)
def test_a_refusal_does_not_bring_the_tool_back_next_turn(path):
    """`tool_failed` is what widens the next turn's tools with this one."""
    assert not path(REFUSAL).remembered_as_failed


# ── A long result on a model that distils ─────────────────────────────


@pytest.mark.parametrize("path", EVERY_PATH)
def test_a_long_result_is_distilled_where_the_model_needs_it(path):
    seen = path(SUCCESS, distil=True)

    assert NOTE in seen.text
    assert LONG_PAGE not in seen.text


@pytest.mark.parametrize("path", EVERY_PATH)
def test_nothing_is_distilled_when_the_model_reads_results_whole(path):
    seen = path(SUCCESS, distil=False)

    assert seen.distil_queries == []
    assert LONG_PAGE in seen.text


# ── The turn that runs an approved call ───────────────────────────────


@pytest.mark.parametrize("path", [
    pytest.param(_after_a_click, id="after a click on yes"),
    pytest.param(_after_a_spoken_yes, id="after a spoken yes"),
])
def test_the_distil_is_asked_his_question_not_his_yes(path):
    """The distil keeps what answers the query. After a spoken yes the
    turn's own text is "oui, vas-y", and nothing on a page answers that."""
    seen = path(SUCCESS, distil=True)

    assert seen.distil_queries == [QUESTION]


@pytest.mark.parametrize("path", [
    pytest.param(_after_a_click, id="after a click on yes"),
    pytest.param(_after_a_spoken_yes, id="after a spoken yes"),
])
def test_an_error_with_no_text_of_its_own_gets_the_note_after_a_yes(path):
    """An exception, an MCP error: the tool said nothing itself. On this
    turn the model has no tool to try again with, so all that is left is
    to say it failed."""
    seen = path(ERROR_ONLY)

    assert "connection reset" in seen.text
    assert f"[{TOOL} FAILED" in seen.text
    assert seen.remembered_as_failed


@pytest.mark.parametrize("path", [
    pytest.param(_after_a_click, id="after a click on yes"),
    pytest.param(_after_a_spoken_yes, id="after a spoken yes"),
])
def test_a_call_that_worked_and_said_nothing_is_not_called_a_failure(path):
    """An action that returns no text ran. Told it failed, she would say
    so, and he would ask for it a second time."""
    seen = path(SILENT_SUCCESS)

    assert seen.results, "the call and its result still reach the model"
    assert "FAILED" not in seen.text
    assert not seen.remembered_as_failed


@pytest.mark.parametrize("path", [
    pytest.param(_in_the_loop, id="the agentic loop"),
    pytest.param(_by_direct_execution, id="the planner's direct execution"),
])
def test_an_error_with_no_text_of_its_own_gets_no_note_where_she_can_call_again(path):
    """There the model still holds its tools and can call again with other
    arguments; the note would tell it to ask the user instead."""
    seen = path(ERROR_ONLY)

    assert "connection reset" in seen.text
    assert "FAILED" not in seen.text
    assert seen.remembered_as_failed


def test_a_question_raised_again_ends_the_turn_on_it():
    """Should the gate ask instead of running, the turn ends on the
    question the code wrote, as it does on the two other paths: the model
    is not handed an empty result to narrate while a card is open."""
    held = {}

    def _asks_again(*_a, **kw):
        store = kw["confirmation"].store
        again = PendingAction.create(
            tool=TOOL, args=ARGS, risk="action", channel=CHANNEL_PAROLE,
            origin="voix", query_redacted=QUESTION,
            raised_at_turn=store.current_turn("voix"), ttl_sec=180.0,
        )
        held["action"] = store.raise_pending(again)
        return ToolExecutionResult(
            success=False, reply_text=None, pending_id=held["action"].request_id,
        )

    seen = _after_a_click(None, distil=True, runner=_asks_again)

    assert seen.reply == SPOKEN_TEMPLATE.format(tool=TOOL)
    assert seen.chat_calls == 0
    assert seen.distil_queries == []

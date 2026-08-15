"""The gate asks, and honours exactly one answer.

Four branches, and their order is the design. `jamais` is checked before
any approval, so a tool the user retired between the question and the
answer is refused while a valid grant sits in hand. An approval is
checked before anything is asked, so a granted call runs rather than
asking twice. A missing channel falls back to today's flat refusal,
which is what keeps every existing caller safe by default — the gate
takes no confirmation at all unless one is threaded in.

The approval is good for one execution of one call. It is pinned by a
digest recomputed at the moment of running, because between the question
and the answer the model gets to run again: the planner re-resolves
steps, the loop re-emits tool calls. Without that comparison the user's
"yes" to deleting one path would run whatever came back next under the
same tool name.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.jarvis.tools.confirmation import (
    CHANNEL_GESTE,
    CHANNEL_PAROLE,
    Approval,
    Confirmation,
    fingerprint,
)
from src.jarvis.tools.policy import (
    OUTCOME_ASKED,
    OUTCOME_OK,
    OUTCOME_REFUSED,
    ToolPolicy,
)
from src.jarvis.tools.types import ToolExecutionResult


class _Cfg:
    mcps = {}
    voice_debug = False
    db_path = "/tmp/does-not-exist/jarvis.db"
    confirmation_ttl_sec = 180.0
    confirmation_model = ""
    tool_router_model = "petit"
    intent_judge_model = ""
    llm_chat_model = "grand"


from src.jarvis.tools.base import Tool


class _FakeTool(Tool):
    """A real builtin, so `risk_is_declared` sees what it sees in
    production — a MagicMock is not a Tool, and would be routed to the
    gesture channel for exactly the right reason."""

    def __init__(self, risk, ran):
        self._risk = risk
        self._ran = ran

    def risk_for(self, args):
        return self._risk

    @property
    def name(self):
        return "fakeAct"

    @property
    def description(self):
        return ""

    @property
    def inputSchema(self):
        return {"type": "object", "properties": {}}

    def run(self, args, context):
        self._ran.append(args)
        return ToolExecutionResult(success=True, reply_text="fait")


@pytest.fixture
def ran():
    return []


@pytest.fixture
def tool(ran):
    return _FakeTool("action", ran)


@pytest.fixture
def store():
    """A confirmation whose store is a real DialogueMemory."""
    from src.jarvis.memory.conversation import DialogueMemory

    dm = DialogueMemory()
    dm.begin_turn()
    return dm


def _confirmation(store, *, approval=None, published=None):
    return Confirmation(
        store=store,
        publish=(published.append if published is not None else (lambda a: None)),
        approval=approval,
        ttl_sec=180.0,
    )


def _run(name, tool, *, args=None, confirmation=None, policy_text=None, db=None):
    from src.jarvis.tools import registry

    policy = ToolPolicy.parse(policy_text) if policy_text else ToolPolicy.empty()
    with patch.object(registry, "BUILTIN_TOOLS", {**registry.BUILTIN_TOOLS, name: tool}), \
         patch.object(registry, "load_tool_policy", return_value=policy):
        return registry.run_tool_with_retries(
            db=db if db is not None else MagicMock(),
            cfg=_Cfg(),
            tool_name=name,
            tool_args=args if args is not None else {"x": 1},
            system_prompt="",
            original_prompt="",
            redacted_text="fais-le",
            max_retries=1,
            origin="voix",
            confirmation=confirmation,
        )


# ── It asks ───────────────────────────────────────────────────────────


def test_a_demande_tool_does_not_run_but_produces_a_question(store, tool, ran):
    result = _run("fakeAct", tool, confirmation=_confirmation(store))

    assert ran == []
    assert result.pending_id
    assert result.refused is False


def test_the_question_is_held_for_the_next_turn(store, tool):
    result = _run("fakeAct", tool, confirmation=_confirmation(store))

    assert store.peek_pending().request_id == result.pending_id


def test_the_question_is_published_to_whoever_shows_it(store, tool):
    published = []
    _run("fakeAct", tool, confirmation=_confirmation(store, published=published))

    assert len(published) == 1
    assert published[0].request_id == store.peek_pending().request_id


def test_asking_writes_one_ledger_row(store, tool):
    db = MagicMock()
    _run("fakeAct", tool, confirmation=_confirmation(store), db=db)

    row = db.record_action.call_args.kwargs
    assert row["outcome"] == OUTCOME_ASKED
    assert row["request_id"]
    assert row["duration_ms"] is None


def test_a_question_pins_the_arguments_it_was_asked_about(store, tool):
    _run("fakeAct", tool, args={"path": "/a"}, confirmation=_confirmation(store))

    assert store.peek_pending().fingerprint == fingerprint("fakeAct", {"path": "/a"})


def test_a_destructive_tool_asks_on_the_gesture_channel(store, ran):
    _run("fakeAct", _FakeTool("destructif", ran), confirmation=_confirmation(store))

    assert store.peek_pending().channel == CHANNEL_GESTE


def test_an_unrecognised_tool_object_asks_on_the_gesture_channel(store):
    """Nobody we trust classified it, so no sentence may authorise it."""
    t = MagicMock()
    t.risk_for.return_value = "action"
    t.execute.return_value = ToolExecutionResult(success=True, reply_text="fait")

    _run("fakeAct", t, confirmation=_confirmation(store))

    assert store.peek_pending().channel == CHANNEL_GESTE


def test_an_action_tool_we_classified_asks_on_the_spoken_channel(store, tool):
    _run("fakeAct", tool, confirmation=_confirmation(store))

    assert store.peek_pending().channel == CHANNEL_PAROLE


# ── With no channel, it still refuses ─────────────────────────────────


def test_without_a_confirmation_it_refuses_as_before(store, tool, ran):
    """Fail-closed by default: a caller that knows nothing about
    confirmation gets exactly today's behaviour."""
    result = _run("fakeAct", tool)

    assert ran == []
    assert result.refused is True
    assert result.pending_id is None
    assert "outils.md" in (result.reply_text or "")


def test_a_refusal_for_want_of_a_channel_is_logged_as_a_refusal(store, tool):
    db = MagicMock()
    _run("fakeAct", tool, db=db)

    assert db.record_action.call_args.kwargs["outcome"] == OUTCOME_REFUSED


# ── An approval runs it, once ─────────────────────────────────────────


def test_an_approval_runs_the_pinned_call(store, tool, ran):
    approval = Approval(
        request_id="cf_x", fingerprint=fingerprint("fakeAct", {"x": 1}),
    )
    result = _run("fakeAct", tool, confirmation=_confirmation(store, approval=approval))

    assert ran == [{"x": 1}]
    assert result.success is True


def test_an_approved_call_is_logged_with_its_request_id(store, tool):
    db = MagicMock()
    approval = Approval(
        request_id="cf_x", fingerprint=fingerprint("fakeAct", {"x": 1}),
    )
    _run("fakeAct", tool, confirmation=_confirmation(store, approval=approval), db=db)

    row = db.record_action.call_args.kwargs
    assert row["outcome"] == OUTCOME_OK
    assert row["request_id"] == "cf_x"


def test_an_approval_for_different_arguments_does_not_run(store, tool, ran):
    """The heart of it. The user approved deleting one path; the model
    re-resolved the step to another between the question and the answer."""
    approval = Approval(
        request_id="cf_x", fingerprint=fingerprint("fakeAct", {"path": "/a"}),
    )
    result = _run("fakeAct", tool, args={"path": "/b"},
                  confirmation=_confirmation(store, approval=approval))

    assert ran == []
    assert result.success is False


def test_an_approval_for_a_different_tool_does_not_run(store, tool, ran):
    approval = Approval(
        request_id="cf_x", fingerprint=fingerprint("autreOutil", {"x": 1}),
    )
    result = _run("fakeAct", tool, confirmation=_confirmation(store, approval=approval))

    assert ran == []
    assert result.success is False


def test_a_mismatched_approval_is_logged_as_a_refusal(store, tool):
    db = MagicMock()
    approval = Approval(
        request_id="cf_x", fingerprint=fingerprint("fakeAct", {"path": "/a"}),
    )
    _run("fakeAct", tool, args={"path": "/b"}, db=db,
         confirmation=_confirmation(store, approval=approval))

    assert db.record_action.call_args.kwargs["outcome"] == OUTCOME_REFUSED


def test_one_approval_does_not_cover_a_second_call(store, tool, ran):
    """Scoped by construction: the confirmation carries one approval, and
    the second call in the same turn arrives with none."""
    approval = Approval(
        request_id="cf_x", fingerprint=fingerprint("fakeAct", {"x": 1}),
    )
    conf = _confirmation(store, approval=approval)

    _run("fakeAct", tool, confirmation=conf)
    second = _run("fakeAct", tool, confirmation=conf.spent())

    assert ran == [{"x": 1}]
    assert second.pending_id or second.refused


# ── `jamais` outranks everything ──────────────────────────────────────


def test_a_never_tool_is_not_asked_about(store, tool, ran):
    result = _run("fakeAct", tool, confirmation=_confirmation(store),
                  policy_text="## Jamais\n- fakeAct\n")

    assert ran == []
    assert result.refused is True
    assert result.pending_id is None
    assert store.peek_pending() is None


def test_an_approval_cannot_run_a_tool_retired_since(store, tool, ran):
    """The user moved it into `## Jamais` between the question and the
    answer. `load_tool_policy` re-reads on mtime, so the gate sees the
    new file — and the newer decision wins."""
    approval = Approval(
        request_id="cf_x", fingerprint=fingerprint("fakeAct", {"x": 1}),
    )
    result = _run("fakeAct", tool, policy_text="## Jamais\n- fakeAct\n",
                  confirmation=_confirmation(store, approval=approval))

    assert ran == []
    assert result.refused is True


# ── A free tool is untouched ──────────────────────────────────────────


def test_a_free_tool_runs_without_any_of_this(store, ran):
    result = _run("fakeAct", _FakeTool("lecture", ran),
                  confirmation=_confirmation(store))

    assert len(ran) == 1
    assert result.success is True
    assert result.pending_id is None
    assert store.peek_pending() is None


# ── The result type stays honest ──────────────────────────────────────


def test_an_ordinary_result_has_no_pending_id():
    assert ToolExecutionResult(success=True, reply_text="ok").pending_id is None


def test_a_question_is_not_a_refusal(store, tool):
    """They read differently to every consumer: a refusal ends the
    matter, a question is waiting on someone."""
    result = _run("fakeAct", tool, confirmation=_confirmation(store))

    assert result.refused is False
    assert result.pending_id is not None

"""Four checks, re-run on every call, with nobody in the room.

The envelope is decided once, when the routine is created. Everything
else is decided again at 07:00 on the morning it runs, because the things
that could have changed in between are exactly the things that matter:
the user edited `outils.md`, an MCP server shipped an update that added
`destructiveHint`, or the model picked the same tool with different
arguments — `localFiles` reading yesterday and deleting today, under one
name the envelope cannot tell apart.

So: the name is in this routine's envelope; the user's own file says
`libre`; `resolve_risk` says `lecture` **for this morning's arguments**;
and the tool does not write Yuba's own state. A routine can therefore
only ever read, and never outside its envelope.

`jamais` still outranks all of it, checked first. A tool the user retired
is refused rather than reported out-of-scope, because those are different
facts and the user's own retirement is the stronger one.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.jarvis.routines.scope import RoutineScope
from src.jarvis.tools.base import Tool
from src.jarvis.tools.policy import (
    OUTCOME_OK,
    OUTCOME_OUT_OF_SCOPE,
    OUTCOME_REFUSED,
    ToolPolicy,
)
from src.jarvis.tools.types import ToolExecutionResult


class _Cfg:
    mcps = {}
    voice_debug = False
    db_path = "/tmp/does-not-exist/jarvis.db"


class _Fake(Tool):
    """A builtin whose risk and state-writing are dialled per case."""

    def __init__(self, name, risk="lecture", writes=False, ran=None):
        self._name = name
        self._risk = risk
        self.writes_own_state = writes
        self.ran = ran if ran is not None else []

    def risk_for(self, args):
        if callable(self._risk):
            return self._risk(args)
        return self._risk

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return ""

    @property
    def inputSchema(self):
        return {"type": "object", "properties": {}}

    def run(self, args, context):
        self.ran.append(args)
        return ToolExecutionResult(success=True, reply_text="fait")


def _run(tool, *, scope, args=None, policy_text=None, db=None):
    from src.jarvis.tools import registry

    policy = ToolPolicy.parse(policy_text) if policy_text else ToolPolicy.empty()
    with patch.object(registry, "BUILTIN_TOOLS",
                      {**registry.BUILTIN_TOOLS, tool.name: tool}), \
         patch.object(registry, "load_tool_policy", return_value=policy):
        return registry.run_tool_with_retries(
            db=db if db is not None else MagicMock(), cfg=_Cfg(),
            tool_name=tool.name, tool_args=args if args is not None else {},
            system_prompt="", original_prompt="", redacted_text="",
            max_retries=1, origin="routine", scope=scope,
        )


def _outcome(db):
    return db.record_action.call_args.kwargs["outcome"]


# ── Inside the envelope, reading, and it runs ─────────────────────────


def test_a_tool_in_the_envelope_runs(db=None):
    tool = _Fake("getWeather")
    db = MagicMock()

    _run(tool, scope=RoutineScope(nom="matin", outils=["getWeather"]), db=db)

    assert tool.ran == [{}]
    assert _outcome(db) == OUTCOME_OK


# ── Outside it, nothing runs ──────────────────────────────────────────


def test_a_free_reading_tool_outside_the_envelope_does_not_run():
    """`libre` and `lecture` is not enough. The envelope is what the user
    can see, and a routine reaching past it is doing something they never
    wrote down."""
    tool = _Fake("webSearch")
    db = MagicMock()

    result = _run(tool, scope=RoutineScope(nom="matin", outils=["getWeather"]), db=db)

    assert tool.ran == []
    assert result.success is False
    assert _outcome(db) == OUTCOME_OUT_OF_SCOPE


def test_an_empty_envelope_reaches_nothing():
    tool = _Fake("getWeather")

    _run(tool, scope=RoutineScope(nom="matin", outils=[]))

    assert tool.ran == []


# ── The three re-checks ───────────────────────────────────────────────


def test_a_tool_that_writes_her_own_state_does_not_run_unattended():
    """Its risk is `lecture` and that is correct — for an attended turn,
    where a wrong entry is correctable in the next breath. At 07:00
    nobody reopens anything."""
    tool = _Fake("remember", risk="lecture", writes=True)
    db = MagicMock()

    _run(tool, scope=RoutineScope(nom="matin", outils=["remember"]), db=db)

    assert tool.ran == []
    assert _outcome(db) == OUTCOME_OUT_OF_SCOPE


def test_a_tool_the_user_has_since_put_behind_a_question_does_not_run():
    """The envelope was written in July; `outils.md` was edited in
    October. The newer decision wins."""
    tool = _Fake("getWeather")

    _run(tool, scope=RoutineScope(nom="matin", outils=["getWeather"]),
         policy_text="## Demande\n- getWeather\n")

    assert tool.ran == []


def test_the_risk_is_recomputed_from_this_mornings_arguments():
    """One name, two risks. `localFiles` reads and deletes under a single
    entry the envelope cannot tell apart, so the check has to be on the
    call rather than the name."""
    ran = []
    tool = _Fake(
        "localFiles", ran=ran,
        risk=lambda args: "lecture" if args.get("operation") == "read" else "destructif",
    )
    scope = RoutineScope(nom="matin", outils=["localFiles"])

    _run(tool, scope=scope, args={"operation": "read"})
    _run(tool, scope=scope, args={"operation": "delete"})

    assert ran == [{"operation": "read"}]


def test_an_action_risk_is_not_enough_either():
    """Only `lecture`. A routine may look; it may not act."""
    tool = _Fake("refreshMCPTools", risk="action")

    _run(tool, scope=RoutineScope(nom="matin", outils=["refreshMCPTools"]))

    assert tool.ran == []


# ── `jamais` still outranks everything ────────────────────────────────


def test_a_retired_tool_is_refused_not_reported_out_of_scope():
    """Different facts, and the user's own retirement is the stronger
    one. Reporting `hors-cadre` would suggest widening the envelope
    fixes it."""
    tool = _Fake("getWeather")
    db = MagicMock()

    _run(tool, scope=RoutineScope(nom="matin", outils=["getWeather"]),
         policy_text="## Jamais\n- getWeather\n", db=db)

    assert tool.ran == []
    assert _outcome(db) == OUTCOME_REFUSED


# ── The refusal text ──────────────────────────────────────────────────


def test_the_refusal_does_not_advise_weakening_the_policy():
    """The ordinary refusal points the user at `## Libre`. Unattended,
    that becomes a paragraph recommending the policy be loosened for
    every origin, written by a thread running while they sleep."""
    tool = _Fake("webSearch")

    result = _run(tool, scope=RoutineScope(nom="matin", outils=["getWeather"]))

    text = (result.reply_text or "")
    assert "Libre" not in text
    assert "outils.md" not in text


def test_the_refusal_names_the_routine_and_says_to_carry_on():
    """One blocked step must not end the morning: the rest of the routine
    is still worth running, and the journal names what was missing."""
    tool = _Fake("webSearch")

    result = _run(tool, scope=RoutineScope(nom="matin", outils=["getWeather"]))

    assert "matin" in (result.reply_text or "")
    assert result.refused is True


# ── Nothing changes without a scope ───────────────────────────────────


@pytest.mark.parametrize("risk,writes,policy", [
    ("lecture", False, None),
    ("lecture", True, None),
    ("action", False, None),
    ("destructif", False, None),
])
def test_an_attended_turn_is_untouched(risk, writes, policy):
    """Every check above is conditional on running unattended. A user
    sitting there must see exactly today's behaviour."""
    from src.jarvis.tools import registry

    tool = _Fake("anything", risk=risk, writes=writes)
    with patch.object(registry, "BUILTIN_TOOLS",
                      {**registry.BUILTIN_TOOLS, "anything": tool}), \
         patch.object(registry, "load_tool_policy", return_value=ToolPolicy.empty()):
        registry.run_tool_with_retries(
            db=MagicMock(), cfg=_Cfg(), tool_name="anything", tool_args={},
            system_prompt="", original_prompt="", redacted_text="",
            max_retries=1, origin="voix",
        )

    # `lecture` runs; anything else is refused for want of a channel —
    # which is the pre-existing behaviour, unchanged.
    assert tool.ran == ([{}] if risk == "lecture" else [])

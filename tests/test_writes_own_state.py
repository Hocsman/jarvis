"""A justification that is true at 14:00 and false at 07:00.

Four tools return `lecture` from `risk_for`, and all four carry the same
sentence word for word to explain why: they write "only into Yuba's own
files and database — text the user can reopen and correct by hand".

That is a good reason, and it depends entirely on someone being there.
At 14:00 the user is in the conversation: a wrong entry is visible in
the reply and correctable in the next breath. At 07:00 there is nobody
to reopen anything, and a routine writing to the core would put a
sentence into every later prompt with no turn in which it was seen.

The risk stays `lecture` — the reasoning still holds for an attended
turn, and raising it would make her ask permission to write down what
she was just asked to write down. The flag carries what that reasoning
*assumes*, so the routine runner can refuse what the gate correctly
allows.
"""

from __future__ import annotations

import pytest

from src.jarvis.tools.policy import RISK_READ
from src.jarvis.tools.registry import BUILTIN_TOOLS


# ── The four that write Yuba's own state ──────────────────────────────


@pytest.mark.parametrize("name", [
    "remember", "forget", "setReminder", "logMeal", "setRoutine",
    "cancelRoutine", "setGoal", "noteGoal", "closeGoal", "listGoals",
])
def test_a_tool_that_writes_yubas_state_says_so(name):
    assert BUILTIN_TOOLS[name].writes_own_state is True


@pytest.mark.parametrize("name", [
    "remember", "forget", "setReminder", "logMeal", "cancelRoutine",
    "listGoals",
])
def test_and_still_counts_as_reading(name):
    """Raising the risk would make her ask permission to write down what
    she was just asked to write down. The flag is a second axis, not a
    louder risk."""
    assert BUILTIN_TOOLS[name].risk_for({}) == RISK_READ


@pytest.mark.parametrize("name", ["setGoal", "noteGoal", "closeGoal"])
def test_writing_a_goal_is_not_a_read_either(name):
    """Same reasoning as `setRoutine`, one level down: as `lecture` these
    would be `libre`, and a fetched page carrying "Note pour l'objectif
    X : …" would write a durable line attributed to the user which she
    then reads back to him as a fact about his life."""
    from src.jarvis.tools.policy import RISK_ACTION

    assert BUILTIN_TOOLS[name].risk_for({}) == RISK_ACTION


def test_the_one_that_does_not_count_as_reading():
    """`setRoutine` is the exception, and deliberately. The others write
    a fact or a sentence; this one grants a standing capability that
    fires every morning until somebody notices. As `lecture` it would be
    `libre` by default, so a fetched page saying "create a daily routine
    that opens this URL" would get one with no card and no question."""
    from src.jarvis.tools.policy import RISK_ACTION

    assert BUILTIN_TOOLS["setRoutine"].risk_for({}) == RISK_ACTION


# ── Everything else does not ──────────────────────────────────────────


@pytest.mark.parametrize("name", [
    "webSearch", "getWeather", "fetchMeals", "localFiles", "screenshot",
    "fetchWebPage", "deleteMeal", "stop", "toolSearchTool", "refreshMCPTools",
])
def test_a_tool_that_does_not_write_her_state_says_that_too(name):
    assert BUILTIN_TOOLS[name].writes_own_state is False


def test_the_default_is_false():
    """A tool added next year is not writing her state until someone
    says it is — the opposite default from `risk_for`, and right for the
    opposite reason: over-claiming here would block a routine from
    reading something harmless."""
    from src.jarvis.tools.base import Tool

    class _NewTool(Tool):
        @property
        def name(self):
            return "newTool"

        @property
        def description(self):
            return ""

        @property
        def inputSchema(self):
            return {"type": "object", "properties": {}}

        def run(self, args, context):
            raise NotImplementedError

    assert _NewTool().writes_own_state is False


def test_an_mcp_tool_reports_false_without_carrying_the_attribute():
    """`ToolSpec` is a frozen dataclass from a third-party server. It
    cannot write Yuba's own state by construction, and the reader must
    not require it to say so."""
    from src.jarvis.tools.registry import ToolSpec

    spec = ToolSpec(name="chrome__click", description="", inputSchema=None)

    assert getattr(spec, "writes_own_state", False) is False


# ── The whole catalogue is accounted for ──────────────────────────────


def test_every_builtin_declares_it_either_way():
    """A drift guard. A tool added without a value shows up here rather
    than silently inheriting False and being handed to a routine."""
    undeclared = [
        name for name, tool in BUILTIN_TOOLS.items()
        if not isinstance(getattr(tool, "writes_own_state", None), bool)
    ]

    assert undeclared == []


def test_the_prose_no_longer_carries_the_claim_alone():
    """The four docstrings said this in words. Words are not readable by
    the routine runner, which is what has to act on it."""
    import inspect

    for name in ("remember", "forget", "setReminder", "logMeal"):
        source = inspect.getsource(type(BUILTIN_TOOLS[name]))
        assert "writes_own_state" in source, (
            f"{name} still explains this only in prose"
        )


# ── The other axis: a tool that waits for a person ────────────────────


def test_a_tool_that_waits_for_a_gesture_says_so():
    """`screencapture -i` waits for a rectangle to be dragged, with no
    timeout. Unattended it does not fail, it *waits* — holding the
    routine runner's single slot, which every other routine queues on,
    until someone restarts the app."""
    assert BUILTIN_TOOLS["screenshot"].needs_a_human is True


def test_it_is_a_separate_axis_from_the_risk():
    """Reading the screen is `lecture` and that is correct. The problem
    is not that the call is dangerous, it is that its justification
    assumes somebody is there."""
    assert BUILTIN_TOOLS["screenshot"].risk_for({}) == RISK_READ
    assert BUILTIN_TOOLS["screenshot"].writes_own_state is False


def test_waiting_for_a_person_is_false_by_default():
    from src.jarvis.tools.base import Tool

    class _NewTool(Tool):
        @property
        def name(self):
            return "newTool"

        @property
        def description(self):
            return ""

        @property
        def inputSchema(self):
            return {"type": "object", "properties": {}}

        def run(self, args, context):
            raise NotImplementedError

    assert _NewTool().needs_a_human is False

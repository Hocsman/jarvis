"""The gate at the single tool-execution funnel.

`run_tool_with_retries` is the only place a tool runs: the planner's
direct execution and the agentic loop both come through it. One insertion
covers both, which is the whole reason the gate lives here rather than at
either call site.

What is pinned: a free tool runs untouched, anything else does not run at
all, and refusing is a distinct outcome from failing. That last one
matters more than it looks — a refusal reported as a failure invites the
model to retry the tool it was just denied, which is a loop the user
watches and cannot stop.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.jarvis.tools.policy import ToolPolicy


class _Cfg:
    mcps = {}
    voice_debug = False
    db_path = "/tmp/does-not-exist/jarvis.db"


def _run(tool_name, args=None, policy_text=None, **kw):
    from src.jarvis.tools import registry

    policy = ToolPolicy.parse(policy_text) if policy_text is not None else ToolPolicy.empty()
    with patch.object(registry, "load_tool_policy", return_value=policy):
        return registry.run_tool_with_retries(
            db=MagicMock(),
            cfg=kw.pop("cfg", _Cfg()),
            tool_name=tool_name,
            tool_args=args or {},
            system_prompt="",
            original_prompt="",
            redacted_text="",
            max_retries=1,
            **kw,
        )


# ── A free tool runs, untouched ───────────────────────────────────────


def test_a_free_tool_still_runs(monkeypatch):
    from src.jarvis.tools import registry
    from src.jarvis.tools.types import ToolExecutionResult

    ran = []
    tool = MagicMock()
    tool.risk_for.return_value = "lecture"
    tool.execute.side_effect = lambda **kw: (
        ran.append(1) or ToolExecutionResult(success=True, reply_text="ok")
    )
    monkeypatch.setitem(registry.BUILTIN_TOOLS, "fakeRead", tool)

    result = _run("fakeRead")

    assert ran == [1]
    assert result.success is True


# ── Anything else does not run ────────────────────────────────────────


def test_a_tool_needing_confirmation_does_not_run(monkeypatch):
    """Until the confirmation channel exists, `demande` is a refusal.
    The tool must not execute first and ask later."""
    from src.jarvis.tools import registry
    from src.jarvis.tools.types import ToolExecutionResult

    ran = []
    tool = MagicMock()
    tool.risk_for.return_value = "destructif"
    tool.execute.side_effect = lambda **kw: (
        ran.append(1) or ToolExecutionResult(success=True, reply_text="ok")
    )
    monkeypatch.setitem(registry.BUILTIN_TOOLS, "fakeWrite", tool)

    result = _run("fakeWrite")

    assert ran == []
    assert result.success is False


def test_a_never_tool_does_not_run(monkeypatch):
    from src.jarvis.tools import registry
    from src.jarvis.tools.types import ToolExecutionResult

    ran = []
    tool = MagicMock()
    tool.risk_for.return_value = "lecture"
    tool.execute.side_effect = lambda **kw: (
        ran.append(1) or ToolExecutionResult(success=True, reply_text="ok")
    )
    monkeypatch.setitem(registry.BUILTIN_TOOLS, "fakeBanned", tool)

    result = _run("fakeBanned", policy_text="## Jamais\n- fakeBanned\n")

    assert ran == []
    assert result.success is False


def test_an_unknown_mcp_tool_does_not_run():
    """The MCP branch fires on the server name alone and never consults
    the tool cache, so this is the path a cold cache walks through."""
    cfg = _Cfg()
    cfg.mcps = {"chrome-devtools": {"command": "noop"}}

    with patch("src.jarvis.tools.external.mcp_client.MCPClient") as client:
        result = _run("chrome-devtools__something_new", cfg=cfg)

    assert client.called is False
    assert result.success is False


# ── Refusing is not failing ───────────────────────────────────────────


def test_a_refusal_is_marked_as_such(monkeypatch):
    from src.jarvis.tools import registry
    from src.jarvis.tools.types import ToolExecutionResult

    tool = MagicMock()
    tool.risk_for.return_value = "destructif"
    tool.execute.return_value = ToolExecutionResult(success=True, reply_text="ok")
    monkeypatch.setitem(registry.BUILTIN_TOOLS, "fakeWrite", tool)

    result = _run("fakeWrite")

    assert result.refused is True


def test_an_ordinary_failure_is_not_a_refusal(monkeypatch):
    """A refusal tells the model to stop asking; a failure invites a
    retry. Collapsing the two would make a denied tool loop."""
    from src.jarvis.tools import registry
    from src.jarvis.tools.types import ToolExecutionResult

    tool = MagicMock()
    tool.risk_for.return_value = "lecture"
    tool.execute.return_value = ToolExecutionResult(
        success=False, reply_text=None, error_message="network down",
    )
    monkeypatch.setitem(registry.BUILTIN_TOOLS, "fakeRead", tool)

    result = _run("fakeRead")

    assert result.success is False
    assert result.refused is False


def test_the_refusal_names_the_tool_and_the_way_out(monkeypatch):
    """The message reaches the model, which relays it. A refusal the user
    cannot act on is just a dead end."""
    from src.jarvis.tools import registry
    from src.jarvis.tools.types import ToolExecutionResult

    tool = MagicMock()
    tool.risk_for.return_value = "destructif"
    tool.execute.return_value = ToolExecutionResult(success=True, reply_text="ok")
    monkeypatch.setitem(registry.BUILTIN_TOOLS, "fakeWrite", tool)

    text = _run("fakeWrite").reply_text or ""

    assert "fakeWrite" in text
    assert "outils.md" in text


def test_a_never_refusal_does_not_offer_a_way_out(monkeypatch):
    """`## Jamais` is the user's own decision to retire a tool. Telling
    the model how to reverse it would invite it to argue."""
    from src.jarvis.tools import registry
    from src.jarvis.tools.types import ToolExecutionResult

    tool = MagicMock()
    tool.risk_for.return_value = "lecture"
    tool.execute.return_value = ToolExecutionResult(success=True, reply_text="ok")
    monkeypatch.setitem(registry.BUILTIN_TOOLS, "fakeBanned", tool)

    text = _run("fakeBanned", policy_text="## Jamais\n- fakeBanned\n").reply_text or ""

    assert "fakeBanned" in text
    assert "outils.md" not in text


# ── The result type keeps working for everyone else ───────────────────


def test_existing_results_are_not_refusals_by_default():
    from src.jarvis.tools.types import ToolExecutionResult

    assert ToolExecutionResult(success=True, reply_text="ok").refused is False

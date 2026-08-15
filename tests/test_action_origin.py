"""Where an action came from.

A row saying `localFiles — destructif — ok` answers what was done but
not the question the user will actually ask when they find one they do
not recognise: *did I ask for this?* Voice, chat, and later the routines
that run with nobody watching all reach the same funnel, so the origin
has to be carried there rather than guessed from it.

Carried as an argument, not ambient state: the routines will run on
their own threads while the user is mid-conversation, and a module
global would label their rows with whatever the last speaker set.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.jarvis.tools.policy import ToolPolicy


class _Cfg:
    mcps = {}
    voice_debug = False
    db_path = "/tmp/does-not-exist/jarvis.db"


def _run_recording(tool_name, monkeypatch, **kw):
    """Run a tool through the gate and hand back the ledger row."""
    from src.jarvis.tools import registry
    from src.jarvis.tools.types import ToolExecutionResult

    tool = MagicMock()
    tool.risk_for.return_value = "lecture"
    tool.execute.return_value = ToolExecutionResult(success=True, reply_text="ok")
    monkeypatch.setitem(registry.BUILTIN_TOOLS, tool_name, tool)

    db = MagicMock()
    with patch.object(registry, "load_tool_policy", return_value=ToolPolicy.empty()):
        registry.run_tool_with_retries(
            db=db,
            cfg=_Cfg(),
            tool_name=tool_name,
            tool_args={},
            system_prompt="",
            original_prompt="",
            redacted_text="",
            max_retries=1,
            **kw,
        )
    return db.record_action.call_args.kwargs


def test_the_origin_reaches_the_ledger(monkeypatch):
    row = _run_recording("fakeRead", monkeypatch, origin="voix")

    assert row["origin"] == "voix"


def test_a_call_with_no_stated_origin_records_none(monkeypatch):
    """Honest blank rather than a guess. A row labelled `chat` because
    that was the last thing to run is worse than one labelled nothing."""
    row = _run_recording("fakeRead2", monkeypatch)

    assert row["origin"] is None


def test_a_refusal_carries_its_origin_too(monkeypatch):
    """Refusals are the rows most worth attributing: the user wants to
    know whether something tried this while they were away."""
    from src.jarvis.tools import registry
    from src.jarvis.tools.types import ToolExecutionResult

    tool = MagicMock()
    tool.risk_for.return_value = "destructif"
    tool.execute.return_value = ToolExecutionResult(success=True, reply_text="ok")
    monkeypatch.setitem(registry.BUILTIN_TOOLS, "fakeWrite", tool)

    db = MagicMock()
    with patch.object(registry, "load_tool_policy", return_value=ToolPolicy.empty()):
        registry.run_tool_with_retries(
            db=db, cfg=_Cfg(), tool_name="fakeWrite", tool_args={},
            system_prompt="", original_prompt="", redacted_text="",
            max_retries=1, origin="routine",
        )

    row = db.record_action.call_args.kwargs
    assert row["origin"] == "routine"
    assert row["outcome"] == "refusé"


def test_concurrent_origins_do_not_bleed(monkeypatch):
    """Two threads, two origins, no shared slot between them."""
    import threading

    seen = {}

    def _call(name, origin):
        seen[origin] = _run_recording(name, monkeypatch, origin=origin)["origin"]

    threads = [
        threading.Thread(target=_call, args=("fakeA", "voix")),
        threading.Thread(target=_call, args=("fakeB", "chat")),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert seen == {"voix": "voix", "chat": "chat"}


# ── The entry points say who they are ─────────────────────────────────


def test_the_engine_passes_its_origin_to_the_funnel():
    """The engine is the only caller of the funnel, so this is where the
    label survives or is lost."""
    import inspect

    from src.jarvis.reply.engine import run_reply_engine

    assert "origin" in inspect.signature(run_reply_engine).parameters


def test_the_voice_path_labels_itself():
    """A drift guard on the listener's call site. The chat side is
    covered for real in test_chat_submission.py, which drives the worker;
    the listener has no equivalent harness, so this is the guard that a
    row from a spoken request stays distinguishable from a typed one."""
    from pathlib import Path

    listener = (
        Path(__file__).resolve().parents[1]
        / "src" / "jarvis" / "listening" / "listener.py"
    ).read_text(encoding="utf-8")

    assert 'origin="voix"' in listener

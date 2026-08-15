"""A call the gate allowed, that then went nowhere, still happened.

The funnel already says so in a comment: a ledger silent about a call
claims she never tried. The MCP branch wraps its call and writes `échec`.
The builtin branch, fifteen lines below, does not: an exception walks out
through the whole funnel, kills the turn, and leaves nothing behind.

The path is not hypothetical. `core._read_lines` catches `OSError` only,
while `read_text(encoding="utf-8")` raises `UnicodeDecodeError`, which is
a `ValueError` — one `profil.md` reopened and saved as ANSI by an editor
is enough. He says "forget that I live in Geneva", `forget` raises, the
turn dies, and the Activity tab shows nothing at all.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def _db(tmp_path):
    from src.jarvis.memory.db import Database

    return Database(str(tmp_path / "t.db"))


def _cfg(tmp_path):
    cfg = MagicMock()
    cfg.db_path = str(tmp_path / "t.db")
    cfg.mcps = {}
    cfg.voice_debug = False
    return cfg


def _appelle(db, cfg, nom="webSearch"):
    from src.jarvis.tools.registry import run_tool_with_retries

    return run_tool_with_retries(
        db=db, cfg=cfg, tool_name=nom, tool_args={"query": "x"},
        system_prompt="", original_prompt="cherche", redacted_text="cherche",
        origin="chat",
    )


def test_a_builtin_that_raises_does_not_kill_the_turn(tmp_path, monkeypatch):
    from src.jarvis.tools import registry

    class _QuiLeve:
        description = "raises"

        def risk_for(self, args):
            return "lecture"

        def execute(self, **kw):
            raise UnicodeDecodeError("utf-8", b"\xe9", 0, 1, "invalid")

    monkeypatch.setitem(registry.BUILTIN_TOOLS, "webSearch", _QuiLeve())
    db = _db(tmp_path)

    resultat = _appelle(db, _cfg(tmp_path))

    assert resultat is not None
    assert resultat.success is False


def test_it_leaves_a_line_saying_it_failed(tmp_path, monkeypatch):
    """The half that matters for the ledger's promise: the call happened,
    so the ledger says so."""
    from src.jarvis.tools import registry

    class _QuiLeve:
        description = "raises"

        def risk_for(self, args):
            return "lecture"

        def execute(self, **kw):
            raise RuntimeError("boom")

    monkeypatch.setitem(registry.BUILTIN_TOOLS, "webSearch", _QuiLeve())
    db = _db(tmp_path)

    _appelle(db, _cfg(tmp_path))

    issues = [r["outcome"] for r in db.recent_actions(50) if r["tool"] == "webSearch"]
    assert "échec" in issues


def test_an_ordinary_builtin_still_reports_ok(tmp_path, monkeypatch):
    """The control. A blanket except that swallowed success would pass
    the two tests above."""
    from src.jarvis.tools import registry
    from src.jarvis.tools.types import ToolExecutionResult

    class _QuiMarche:
        description = "works"

        def risk_for(self, args):
            return "lecture"

        def execute(self, **kw):
            return ToolExecutionResult(success=True, reply_text="trouvé")

    monkeypatch.setitem(registry.BUILTIN_TOOLS, "webSearch", _QuiMarche())
    db = _db(tmp_path)

    resultat = _appelle(db, _cfg(tmp_path))

    assert resultat.success is True
    issues = [r["outcome"] for r in db.recent_actions(50) if r["tool"] == "webSearch"]
    assert issues == ["ok"]


# ── And the read that raised in the first place ────────────────────────


def test_a_core_file_in_another_encoding_reads_as_empty(tmp_path):
    """The class docstring promises reads fail open: an unreadable file is
    an empty section, never an exception into the reply path."""
    from src.jarvis.memory.core import SECTION_PROFILE, MemoryCore

    core = MemoryCore(tmp_path / "yuba")
    core.directory.mkdir(parents=True, exist_ok=True)
    (core.directory / "profil.md").write_bytes(
        "## Profil\n- il habite à Genève · dit\n".encode("cp1252"))

    assert core.entries(SECTION_PROFILE) == []
    assert core.active(SECTION_PROFILE) == []
    assert core.knows(SECTION_PROFILE, "Genève") is False

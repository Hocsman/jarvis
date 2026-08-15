"""The header of `outils.md` is read by the person the file is for.

It is the only explanation he gets of what the three sections mean, and
it is written into his own control surface. So it has to be true.

It said a tool that appeared since generation is treated as destructive
and will therefore be asked about. That is true of a tool the catalogue
has never heard of, and false of a discovered one: `verdict` falls
through to the risk default, and a server announcing `readOnlyHint: true`
lands on `lecture`, which defaults to free. Measured — the tool runs with
no card at all.

The file is also frozen after one write, so the promise cannot be
corrected later. It is corrected here, and the write is made atomic:
every other page this assistant owns writes through a temporary file and
a rename, and a policy file truncated halfway reads back clean because
the parser skips what it does not recognise.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.jarvis.tools.policy import ASK, FREE, ToolPolicy, resolve_risk


def test_the_header_does_not_promise_more_than_the_gate_delivers():
    """A tool discovered after generation, announcing itself read-only."""
    spec = SimpleNamespace(annotations={"readOnlyHint": True}, description="x")
    risque = resolve_risk("srv__nouveau", spec, {})
    politique = ToolPolicy.parse("## Libre\n- webSearch\n\n## Demande\n- setRoutine\n")

    verdict = politique.verdict("srv__nouveau", risque)

    from src.jarvis.tools.policy import _HEADER

    if verdict == FREE:
        assert "il te sera demandé" not in _HEADER, (
            "l'en-tête promet une carte que la porte ne pose pas")


def test_an_unknown_tool_is_still_asked_about():
    """The half of the promise that was true stays true: a name the
    catalogue never heard of is destructive."""
    politique = ToolPolicy.parse("## Libre\n- webSearch\n")

    assert politique.verdict("inconnu", resolve_risk("inconnu", None, {})) == ASK


# ── And the file is written whole or not at all ────────────────────────


def test_the_policy_file_is_not_written_when_discovery_failed(tmp_path, capsys):
    """Written once and never rewritten, so a first boot with a server
    down would freeze `outils.md` without its tools — permanently, and
    with no way to tell from the file that anything is missing."""
    from src.jarvis.tools import registry

    cfg = SimpleNamespace(memory_dir=str(tmp_path), db_path=str(tmp_path / "t.db"))
    with patch("src.jarvis.memory.core.MemoryCore.for_config",
               classmethod(lambda cls, _c: SimpleNamespace(directory=tmp_path))), \
         patch.object(registry, "get_cached_mcp_tools", lambda: {}):
        registry.ensure_policy_file(cfg, discovery_failed=True)

    assert not (tmp_path / "outils.md").exists()
    assert "⚠️" in capsys.readouterr().out


def test_it_is_written_when_discovery_worked(tmp_path):
    """The control. A guard that never wrote would pass the test above
    and leave him without the file the whole feature rests on."""
    from src.jarvis.tools import registry

    cfg = SimpleNamespace(memory_dir=str(tmp_path), db_path=str(tmp_path / "t.db"))
    with patch("src.jarvis.memory.core.MemoryCore.for_config",
               classmethod(lambda cls, _c: SimpleNamespace(directory=tmp_path))), \
         patch.object(registry, "get_cached_mcp_tools", lambda: {}):
        registry.ensure_policy_file(cfg)

    assert (tmp_path / "outils.md").exists()
    assert "## Libre" in (tmp_path / "outils.md").read_text(encoding="utf-8")


def test_a_half_written_file_never_appears(tmp_path):
    """Truncation leans safe on the verdicts, but `path.exists()` is what
    stops regeneration: a half-written file is permanent."""
    from src.jarvis.tools import registry

    cfg = SimpleNamespace(memory_dir=str(tmp_path), db_path=str(tmp_path / "t.db"))
    reel = registry.Path.write_text if hasattr(registry, "Path") else None

    with patch("src.jarvis.memory.core.MemoryCore.for_config",
               classmethod(lambda cls, _c: SimpleNamespace(directory=tmp_path))), \
         patch.object(registry, "get_cached_mcp_tools", lambda: {}), \
         patch("os.replace", side_effect=OSError("disque plein")):
        registry.ensure_policy_file(cfg)

    assert not (tmp_path / "outils.md").exists()

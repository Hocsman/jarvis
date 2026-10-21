"""The policy file has to exist before the user can edit it.

`outils.md` is the one control surface the user has over what Yuba may
do on her own. Shipping a static default would list tools they may not
have and omit the 29 Chrome tools they do; so it is written once, from
the catalogue actually discovered on their machine, and then belongs to
them.

"Then belongs to them" is the part with teeth: a file regenerated on
startup would silently undo every decision the user made in it, which is
the same failure as having no file at all, only slower to notice.
"""

from __future__ import annotations

import re
from unittest.mock import patch

import pytest

from src.jarvis.tools.policy import ToolPolicy


class _Cfg:
    mcps = {}
    voice_debug = False

    def __init__(self, db_path):
        self.db_path = str(db_path)


@pytest.fixture
def cfg(tmp_path):
    return _Cfg(tmp_path / "jarvis.db")


def _policy_path(cfg):
    from src.jarvis.memory.core import MemoryCore

    return MemoryCore.for_config(cfg).directory / "outils.md"


# ── Written once ──────────────────────────────────────────────────────


def test_the_file_is_written_when_it_is_missing(cfg):
    from src.jarvis.tools.registry import ensure_policy_file

    ensure_policy_file(cfg)

    assert _policy_path(cfg).exists()


def test_the_generated_file_lists_the_tools_actually_installed(cfg):
    from src.jarvis.tools.registry import BUILTIN_TOOLS, ensure_policy_file

    ensure_policy_file(cfg)
    text = _policy_path(cfg).read_text(encoding="utf-8")

    for name, tool in BUILTIN_TOOLS.items():
        if getattr(tool, "classified_per_call", False):
            continue
        assert f"- {name}\n" in text, f"{name} absent du fichier généré"


def test_a_tool_whose_empty_call_is_harmless_is_left_out(cfg):
    """claudeCode's empty call asks how a job is going, so classified from
    `{}` it would land under ## Libre, and an exact line there frees every
    call of it, a change to the user's code included. Left out, each call
    is judged at its own risk: a change still costs a card."""
    from src.jarvis.tools.registry import ensure_policy_file

    ensure_policy_file(cfg)
    text = _policy_path(cfg).read_text(encoding="utf-8")
    policy = ToolPolicy.parse(text)

    assert "- claudeCode\n" not in text
    assert policy.verdict("claudeCode", "destructif") == "demande"
    assert policy.verdict("claudeCode", "lecture") == "libre"


def test_a_tool_whose_calls_differ_in_risk_is_left_out(cfg):
    """stopClaudeCode's empty call stops the job and asks first; a call
    carrying text stops nothing. One line in the file would decide both:
    under ## Demande the harmless call would ask too. Left out, each call
    is judged at its own risk, and a line the user writes under ## Libre
    is the only one there is."""
    from src.jarvis.tools.registry import ensure_policy_file

    ensure_policy_file(cfg)
    text = _policy_path(cfg).read_text(encoding="utf-8")
    policy = ToolPolicy.parse(text)

    assert "- stopClaudeCode\n" not in text
    assert policy.verdict("stopClaudeCode", "action") == "demande"
    assert policy.verdict("stopClaudeCode", "lecture") == "libre"
    # The real heading, not the one quoted in the file's own explanation.
    freed = ToolPolicy.parse(
        re.sub(r"^## Libre$", "## Libre\n\n- stopClaudeCode", text, count=1, flags=re.M))
    assert freed.verdict("stopClaudeCode", "action") == "libre"


def test_discovered_mcp_tools_are_listed_too(cfg):
    """The whole reason for generating rather than shipping: the user
    opens it and sees their own catalogue."""
    from src.jarvis.tools import registry
    from src.jarvis.tools.registry import ToolSpec, ensure_policy_file

    spec = ToolSpec(
        name="chrome-devtools__close_page", description="", inputSchema=None,
        annotations={"destructiveHint": True},
    )
    with patch.object(
        registry, "get_cached_mcp_tools",
        return_value={"chrome-devtools__close_page": spec},
    ):
        ensure_policy_file(cfg)

    assert "- chrome-devtools__close_page\n" in _policy_path(cfg).read_text(encoding="utf-8")


def test_the_generated_file_parses_back(cfg):
    """A generated file the parser cannot read would leave every tool on
    its default while looking, to the user, like it was configured."""
    from src.jarvis.tools.registry import ensure_policy_file

    ensure_policy_file(cfg)
    policy = ToolPolicy.parse(_policy_path(cfg).read_text(encoding="utf-8"))

    assert policy.exact
    assert policy.verdict("webSearch", "lecture") == "libre"


def test_a_read_only_tool_lands_in_libre_and_a_destructive_one_in_demande(cfg):
    from src.jarvis.tools.registry import ensure_policy_file

    ensure_policy_file(cfg)
    policy = ToolPolicy.parse(_policy_path(cfg).read_text(encoding="utf-8"))

    assert policy.exact["webSearch"] == "libre"
    assert policy.exact["deleteMeal"] == "demande"


# ── Never rewritten ───────────────────────────────────────────────────


def test_an_existing_file_is_left_exactly_as_the_user_left_it(cfg):
    from src.jarvis.tools.registry import ensure_policy_file

    path = _policy_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    mine = "## Libre\n- webSearch\n\n## Jamais\n- localFiles\n"
    path.write_text(mine, encoding="utf-8")

    ensure_policy_file(cfg)

    assert path.read_text(encoding="utf-8") == mine


def test_an_empty_file_counts_as_the_users_and_is_not_refilled(cfg):
    """Emptying the file is a legible way to say "ask me about
    everything". Refilling it would override that."""
    from src.jarvis.tools.registry import ensure_policy_file

    path = _policy_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")

    ensure_policy_file(cfg)

    assert path.read_text(encoding="utf-8") == ""


# ── Never breaks startup ──────────────────────────────────────────────


def test_a_write_failure_does_not_raise(cfg):
    """This runs during daemon boot. A permissions problem on one file
    must not stop Yuba starting."""
    from src.jarvis.tools import registry

    with patch("pathlib.Path.write_text", side_effect=OSError("read-only")):
        registry.ensure_policy_file(cfg)  # ne doit pas lever

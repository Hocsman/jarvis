"""Every builtin says what it does to the world.

The gate needs a risk for each tool before it can decide anything, and a
tool that forgets to declare one must not thereby become harmless. The
base class defaults to destructive for exactly that reason: a new tool
added next year, by someone who never read this file, asks before it
acts rather than sliding through.

Three tools write and are still free. `remember`, `forget` and `logMeal`
only ever touch Yuba's own files and database — text the user can reopen
and correct by hand, never their machine and never anything outward. A
tool that writes is not automatically a tool that costs you something.
"""

import pytest

from src.jarvis.tools.policy import RISK_ACTION, RISK_DESTRUCTIVE, RISK_READ
from src.jarvis.tools.registry import BUILTIN_TOOLS


def _risk(name, args=None):
    return BUILTIN_TOOLS[name].risk_for(args or {})


# ── Reading the world changes nothing ─────────────────────────────────


@pytest.mark.parametrize("name", [
    "webSearch", "fetchWebPage", "getWeather", "screenshot",
    "fetchMeals", "toolSearchTool", "stop",
])
def test_a_tool_that_only_looks_is_read(name):
    assert _risk(name) == RISK_READ


# ── Writing inside Yuba is still free, and says why ───────────────────


@pytest.mark.parametrize("name", ["remember", "forget", "logMeal"])
def test_writing_only_to_yubas_own_store_is_read(name):
    assert _risk(name) == RISK_READ


def test_deleting_the_users_records_is_destructive():
    assert _risk("deleteMeal") == RISK_DESTRUCTIVE


# ── One name, several risks ───────────────────────────────────────────


def test_listing_and_reading_files_is_read():
    assert _risk("localFiles", {"operation": "list"}) == RISK_READ
    assert _risk("localFiles", {"operation": "read"}) == RISK_READ


def test_writing_a_file_is_destructive():
    """Writing overwrites whatever was there. Under $HOME that includes
    the config Yuba runs on and the memory files she believes."""
    assert _risk("localFiles", {"operation": "write"}) == RISK_DESTRUCTIVE
    assert _risk("localFiles", {"operation": "append"}) == RISK_DESTRUCTIVE
    assert _risk("localFiles", {"operation": "delete"}) == RISK_DESTRUCTIVE


def test_localfiles_with_no_operation_is_destructive():
    """The tool defaults to a read when the argument is missing, but the
    gate cannot assume the tool will agree. Unclassified is dangerous."""
    assert _risk("localFiles", {}) == RISK_DESTRUCTIVE


def test_an_unknown_operation_is_destructive():
    assert _risk("localFiles", {"operation": "banana"}) == RISK_DESTRUCTIVE


# ── The default protects tools nobody classified ──────────────────────


def test_the_base_class_defaults_to_destructive():
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

    assert _NewTool().risk_for({}) == RISK_DESTRUCTIVE


def test_every_registered_builtin_declares_a_risk():
    """A drift guard: a tool added without a risk shows up here rather
    than silently inheriting the destructive default and puzzling
    someone with a confirmation prompt they cannot explain."""
    valid = {RISK_READ, RISK_ACTION, RISK_DESTRUCTIVE}
    undeclared = [
        name for name, tool in BUILTIN_TOOLS.items()
        if tool.risk_for({}) not in valid
    ]
    assert undeclared == []


def test_refreshing_the_tool_catalogue_is_an_action():
    assert _risk("refreshMCPTools") == RISK_ACTION

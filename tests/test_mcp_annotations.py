"""MCP servers already tell us which of their tools destroy things.

Every MCP tool ships an optional ``annotations`` object carrying
``readOnlyHint`` and ``destructiveHint``. The catalogue dropped it on the
floor at both places it builds a tool dict, so the gate had nothing to
go on and had to treat 29 Chrome tools identically — a snapshot like a
click, a console read like a form submission.

Carrying it through is what lets the generated policy file arrive
already sorted, instead of asking the user to classify 32 tools by hand.
"""

import pytest

from src.jarvis.tools.registry import ToolSpec


class _McpTool:
    """Stand-in for the MCP SDK's Tool object."""

    def __init__(self, name, annotations=None):
        self.name = name
        self.description = f"{name} description"
        self.inputSchema = {"type": "object", "properties": {}}
        if annotations is not None:
            self.annotations = annotations


def _dicts_from(tools, builder):
    return {d["name"]: d for d in builder(tools)}


# ── The catalogue keeps what the server sent ──────────────────────────


def test_a_read_only_hint_survives_discovery():
    from src.jarvis.tools.external.mcp_client import tool_to_dict

    d = tool_to_dict(_McpTool("take_snapshot", {"readOnlyHint": True}))

    assert d["annotations"] == {"readOnlyHint": True}


def test_a_destructive_hint_survives_discovery():
    from src.jarvis.tools.external.mcp_client import tool_to_dict

    d = tool_to_dict(_McpTool("close_page", {"destructiveHint": True}))

    assert d["annotations"]["destructiveHint"] is True


def test_a_tool_without_annotations_yields_none_rather_than_failing():
    from src.jarvis.tools.external.mcp_client import tool_to_dict

    d = tool_to_dict(_McpTool("mystery"))

    assert d["annotations"] is None
    assert d["name"] == "mystery"


def test_an_object_shaped_annotation_is_normalised_to_a_dict():
    """The SDK may hand back a pydantic model rather than a plain dict.
    The gate reads keys, so whatever arrives has to end up as a mapping
    or the hints are silently lost again."""
    from src.jarvis.tools.external.mcp_client import tool_to_dict

    class _Ann:
        readOnlyHint = True
        destructiveHint = False

    d = tool_to_dict(_McpTool("take_snapshot", _Ann()))

    assert d["annotations"]["readOnlyHint"] is True


def test_the_name_description_and_schema_still_come_through():
    from src.jarvis.tools.external.mcp_client import tool_to_dict

    d = tool_to_dict(_McpTool("click", {"destructiveHint": True}))

    assert d["name"] == "click"
    assert d["description"] == "click description"
    assert d["inputSchema"]["type"] == "object"


# ── The spec the rest of the code sees ────────────────────────────────


def test_a_tool_spec_can_carry_annotations():
    spec = ToolSpec(
        name="chrome__click",
        description="",
        inputSchema=None,
        annotations={"destructiveHint": True},
    )

    assert spec.annotations["destructiveHint"] is True


def test_a_tool_spec_without_annotations_still_builds():
    """Every existing construction site passes three arguments. A
    required fourth would break discovery rather than harden it."""
    spec = ToolSpec(name="chrome__click", description="", inputSchema=None)

    assert spec.annotations is None


def test_discovery_puts_the_annotations_on_the_spec():
    from src.jarvis.tools import registry

    discovered = registry._spec_from_tool_info(
        "chrome-devtools",
        {
            "name": "close_page",
            "description": "Close a page",
            "inputSchema": None,
            "annotations": {"destructiveHint": True},
        },
    )

    assert discovered.name == "chrome-devtools__close_page"
    assert discovered.annotations["destructiveHint"] is True

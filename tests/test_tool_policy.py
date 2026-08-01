"""Behavioural tests for the tool policy gate.

Every tool call funnels through one place, and this decides what happens
there. The shape being pinned: the user's own file always wins, an
unknown tool is treated as the most dangerous thing it could be, and
nothing in the never list is reachable by any route.

The asymmetry that drives every default here: refusing a harmless tool
costs a turn, while allowing a destructive one costs something that may
not come back.
"""

import pytest

from src.jarvis.tools.policy import (
    NEVER,
    ASK,
    FREE,
    RISK_ACTION,
    RISK_DESTRUCTIVE,
    RISK_READ,
    ToolPolicy,
    render_policy_file,
    resolve_risk,
)


class _FakeTool:
    def __init__(self, name, risk=RISK_READ):
        self.name = name
        self._risk = risk

    def risk_for(self, args):
        return self._risk


class _FakeSpec:
    def __init__(self, name, annotations=None):
        self.name = name
        self.description = ""
        self.inputSchema = None
        self.annotations = annotations


# ── What a tool's risk is ─────────────────────────────────────────────


def test_a_builtin_declares_its_own_risk():
    tool = _FakeTool("webSearch", RISK_READ)

    assert resolve_risk("webSearch", tool, {}) == RISK_READ


def test_a_builtin_can_decide_from_its_arguments():
    """localFiles reads, writes and deletes through one name. A risk
    fixed at the tool level would have to call the whole thing
    destructive, which makes reading a file need confirmation."""

    class _Files:
        name = "localFiles"

        def risk_for(self, args):
            op = (args or {}).get("operation", "read")
            return RISK_DESTRUCTIVE if op == "delete" else RISK_READ

    tool = _Files()
    assert resolve_risk("localFiles", tool, {"operation": "read"}) == RISK_READ
    assert resolve_risk("localFiles", tool, {"operation": "delete"}) == RISK_DESTRUCTIVE


def test_an_mcp_tool_is_read_when_the_server_says_read_only():
    spec = _FakeSpec("chrome__take_snapshot", {"readOnlyHint": True})

    assert resolve_risk("chrome__take_snapshot", spec, {}) == RISK_READ


def test_an_mcp_tool_is_destructive_when_the_server_says_so():
    spec = _FakeSpec("chrome__close_page", {"destructiveHint": True})

    assert resolve_risk("chrome__close_page", spec, {}) == RISK_DESTRUCTIVE


def test_an_unknown_tool_is_treated_as_destructive():
    """The MCP branch of the funnel fires on the server name alone and
    never consults the tool cache, so a cold cache would walk straight
    through a gate that classified the unknown as ordinary."""

    assert resolve_risk("chrome__something_new", None, {}) == RISK_DESTRUCTIVE


def test_an_mcp_tool_with_no_annotations_is_treated_as_destructive():
    spec = _FakeSpec("chrome__mystery", annotations=None)

    assert resolve_risk("chrome__mystery", spec, {}) == RISK_DESTRUCTIVE


# ── What the verdict is ───────────────────────────────────────────────


def test_reading_is_free_by_default():
    policy = ToolPolicy.empty()

    assert policy.verdict("webSearch", RISK_READ) == FREE


def test_acting_asks_by_default():
    policy = ToolPolicy.empty()

    assert policy.verdict("chrome__click", RISK_ACTION) == ASK


def test_destroying_asks_by_default():
    policy = ToolPolicy.empty()

    assert policy.verdict("localFiles", RISK_DESTRUCTIVE) == ASK


def test_the_users_file_overrides_the_default():
    policy = ToolPolicy.parse("## Libre\n- chrome__click\n")

    assert policy.verdict("chrome__click", RISK_DESTRUCTIVE) == FREE


def test_a_wildcard_covers_a_whole_server():
    policy = ToolPolicy.parse("## Libre\n- chrome-devtools__*\n")

    assert policy.verdict("chrome-devtools__take_snapshot", RISK_ACTION) == FREE
    assert policy.verdict("macos__execute_script", RISK_ACTION) == ASK


def test_an_exact_name_beats_a_wildcard():
    policy = ToolPolicy.parse(
        "## Libre\n- chrome__*\n\n## Jamais\n- chrome__evaluate_script\n"
    )

    assert policy.verdict("chrome__take_snapshot", RISK_ACTION) == FREE
    assert policy.verdict("chrome__evaluate_script", RISK_READ) == NEVER


def test_never_cannot_be_reached_by_any_route():
    """The section exists so the user can retire a tool outright. If a
    confirmation could unlock it, it would be a speed bump, not a wall."""
    policy = ToolPolicy.parse("## Jamais\n- macos__execute_script\n")

    for risk in (RISK_READ, RISK_ACTION, RISK_DESTRUCTIVE):
        assert policy.verdict("macos__execute_script", risk) == NEVER


def test_an_unreadable_policy_falls_back_to_the_defaults(tmp_path):
    """Failing open here would mean a corrupt file silently unlocking the
    machine. It fails to the defaults instead, which ask."""
    policy = ToolPolicy.parse("du texte qui ne ressemble à rien")

    assert policy.verdict("macos__execute_script", RISK_DESTRUCTIVE) == ASK
    assert policy.verdict("webSearch", RISK_READ) == FREE


def test_comments_and_blank_lines_are_ignored():
    policy = ToolPolicy.parse(
        "<!-- explication -->\n\n## Libre\n\n- webSearch\n\n<!-- fin -->\n"
    )

    assert policy.verdict("webSearch", RISK_ACTION) == FREE


# ── The file the user opens ───────────────────────────────────────────


def test_the_generated_file_lists_the_real_inventory():
    text = render_policy_file(
        {"webSearch": RISK_READ, "localFiles": RISK_DESTRUCTIVE},
        {"chrome__click": RISK_ACTION},
    )

    assert "webSearch" in text
    assert "localFiles" in text
    assert "chrome__click" in text


def test_the_generated_file_sorts_tools_into_the_three_sections():
    text = render_policy_file({"webSearch": RISK_READ, "localFiles": RISK_DESTRUCTIVE}, {})

    # Past the explanatory header, which names the sections too.
    body = text.split("-->", 1)[1]
    libre = body.split("## Libre")[1].split("##")[0]
    assert "webSearch" in libre
    assert "localFiles" not in libre


def test_the_generated_file_round_trips_through_the_parser():
    text = render_policy_file({"webSearch": RISK_READ, "localFiles": RISK_DESTRUCTIVE}, {})

    policy = ToolPolicy.parse(text)

    assert policy.verdict("webSearch", RISK_READ) == FREE
    assert policy.verdict("localFiles", RISK_DESTRUCTIVE) == ASK


def test_the_generated_file_explains_itself():
    text = render_policy_file({"webSearch": RISK_READ}, {})

    assert text.lstrip().startswith("#")
    assert "<!--" in text


def test_the_generated_file_says_that_libre_reaches_into_the_night():
    """A user tightening this file is changing two things, and one of
    them happens where they cannot see it: `## Libre` is also the list a
    routine may reach from while they are asleep. Moving a line to
    `## Demande` puts it out of every routine's reach, since a routine
    has nobody to ask. The header has to say so, or the second effect is
    a surprise found months later in a journal page."""
    text = render_policy_file({"webSearch": RISK_READ}, {})

    header = text.split("-->", 1)[0].lower()
    assert "routine" in header
    assert "périmètre" in header

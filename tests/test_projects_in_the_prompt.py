"""She knows she can work on his projects, whatever the router offered.

Asked whether she can reach his code, a chat model shown only the tools
every turn carries answers from its prior: an assistant has no access to
anyone's files. With handing projects to Claude Code switched on, that is
false, and the router (a small model) does not reliably offer `claudeCode`
for a question about what she can do. So while the tool is offered, the
system prompt says what she can do and how to reach the tool when it is not
among her tools this turn. It says nothing when that would not be true: the
feature off, the projects folder missing, `claudeCode` filed under
`## Jamais`. Nor on a routine turn, which never reaches `claudeCode`, nor on
the turn that runs a call the user approved, which must not propose it
again.

See src/jarvis/projets/projets.spec.md, What she knows she can do.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.jarvis.projets.delegation import TOOL_NAME
from src.jarvis.projets.prompt import format_projets_block
from src.jarvis.tools.builtin.tool_search import ToolSearchTool

PROJECTS = ("quillon", "baladeur", "orchidee")


def _cfg(tmp_path, projects_dir: str) -> MagicMock:
    cfg = MagicMock()
    cfg.db_path = str(tmp_path / "t.db")
    cfg.llm_chat_model = "test-large"
    cfg.mcps = {}
    cfg.voice_debug = False
    cfg.memory_digest_enabled = False
    cfg.tool_result_digest_enabled = False
    cfg.location_enabled = False
    cfg.llm_thinking_enabled = False
    cfg.agentic_max_turns = 8
    cfg.memory_enrichment_source = "diary"
    cfg.tool_selection_strategy = "all"
    cfg.tool_carryover_max_turns = 2
    cfg.tool_carryover_per_entry_chars = 1200
    cfg.tool_search_max_calls = 3
    cfg.llm_chat_timeout_sec = 45.0
    cfg.llm_tools_timeout_sec = 8.0
    cfg.confirmation_ttl_sec = 180.0
    cfg.confirmation_timeout_sec = 8.0
    cfg.confirmation_model = ""
    cfg.claude_code_projects_dir = projects_dir
    cfg.tts_engine = "kyutai"
    return cfg


def _first_call(cfg, text="tu peux bosser sur mes projets ?", scope=None, origin="voix",
                dialogue_memory=None, replies=None, **engine_kw):
    """Run one turn; return the first chat call's system message and the
    names of the tools it was offered."""
    from src.jarvis.memory.conversation import DialogueMemory
    from src.jarvis.reply.engine import run_reply_engine

    seen = {}
    replies = list(replies or [{"message": {"content": "ok"}}])

    def _capture(**kw):
        if "system" not in seen:
            messages = kw.get("messages") or []
            seen["system"] = messages[0].get("content", "") if messages else ""
            seen["tools"] = [
                (schema.get("function") or {}).get("name")
                for schema in (kw.get("tools") or [])
            ]
        return replies.pop(0) if replies else {"message": {"content": "ok"}}

    with patch("src.jarvis.reply.engine.plan_query", return_value=[]), \
         patch("src.jarvis.reply.engine.extract_search_params_for_memory",
               return_value={}), \
         patch("src.jarvis.reply.engine.chat_with_messages", side_effect=_capture), \
         patch("src.jarvis.reply.engine.extract_text_from_response",
               return_value="ok"):
        run_reply_engine(
            db=MagicMock(), cfg=cfg, tts=None, text=text,
            dialogue_memory=dialogue_memory if dialogue_memory is not None else DialogueMemory(),
            origin=origin, scope=scope, **engine_kw,
        )
    return seen.get("system", ""), seen.get("tools", [])


def _system_prompt(cfg, **kw) -> str:
    return _first_call(cfg, **kw)[0]


@pytest.fixture
def projects(tmp_path):
    root = tmp_path / "Projets"
    for name in PROJECTS:
        (root / name).mkdir(parents=True)
    return root


def _policy(tmp_path, text: str) -> None:
    """The user's `outils.md`, in the core beside the configured database."""
    from src.jarvis.memory.core import CORE_DIRNAME

    core = tmp_path / CORE_DIRNAME
    core.mkdir(exist_ok=True)
    (core / "outils.md").write_text(text, encoding="utf-8")


# ── What the block says ───────────────────────────────────────────────


def test_it_names_the_tool_and_the_search_that_finds_it():
    """The two names she needs: the tool that does the work, and the
    escape hatch that adds it to a turn whose router left it out. The
    search for the tool's exact name finds it whatever the router says
    (tool_search.spec.md, Name cue)."""
    block = format_projets_block()

    assert TOOL_NAME in block
    assert ToolSearchTool().name in block


def test_it_names_no_other_tool():
    """Work on a project goes to Claude Code alone: the block points at no
    other tool, so it cannot steer a request to one."""
    import re

    from src.jarvis.tools.registry import BUILTIN_TOOLS

    block = format_projets_block()
    others = [
        name for name in BUILTIN_TOOLS
        if name not in (TOOL_NAME, ToolSearchTool().name)
        and re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", block)
    ]

    assert others == []


# ── When it reaches the prompt ────────────────────────────────────────


def test_switched_on_the_prompt_says_she_can(tmp_path, projects):
    system = _system_prompt(_cfg(tmp_path, str(projects)))

    assert format_projets_block() in system


def test_no_project_reaches_the_prompt(tmp_path, projects):
    """Folder names are the user's, and the system prompt goes to whichever
    provider runs the chat model on every turn. The tool lists the
    projects when it needs to, in the turn that asks."""
    system = _system_prompt(_cfg(tmp_path, str(projects)))

    assert format_projets_block() in system
    names = [p.name for p in projects.iterdir()]
    assert names
    assert [name for name in names if name in system] == []


def test_switched_off_the_prompt_says_nothing_of_it(tmp_path):
    system = _system_prompt(_cfg(tmp_path, ""))

    assert format_projets_block() not in system
    assert TOOL_NAME not in system


def test_a_turn_typed_in_the_chat_window_knows_it_too(tmp_path, projects):
    system = _system_prompt(_cfg(tmp_path, str(projects)), origin="chat")

    assert format_projets_block() in system


def test_the_block_is_said_when_the_router_offers_nothing(tmp_path, projects):
    """Offered, not routed: the router reads a question about what she can
    do as conversation and offers no tool. The block makes up for exactly
    that, so it holds on a turn whose tools lack `claudeCode`."""
    cfg = _cfg(tmp_path, str(projects))
    cfg.tool_selection_strategy = "llm"

    with patch("src.jarvis.reply.engine.select_tools",
               return_value=["stop", "forget", "remember"]):
        system, tools = _first_call(cfg)

    assert TOOL_NAME not in tools
    assert ToolSearchTool().name in tools
    assert format_projets_block() in system


def test_a_routine_turn_is_not_told(tmp_path, projects):
    """A routine never reaches `claudeCode` (routines.spec.md), and its
    catalogue is its envelope: no escape hatch to reach it with."""
    from src.jarvis.routines.scope import RoutineScope

    system = _system_prompt(
        _cfg(tmp_path, str(projects)), text="résume les mails", origin="routine",
        scope=RoutineScope(nom="matin", outils=["webSearch"]),
    )

    assert format_projets_block() not in system


# ── Not when it would not be true ─────────────────────────────────────


def test_a_projects_folder_that_does_not_exist_is_not_claimed(tmp_path):
    """Every call would refuse with "the projects folder does not exist":
    telling her the folder is open to her would have her promise what the
    tool cannot do."""
    system = _system_prompt(_cfg(tmp_path, str(tmp_path / "nowhere")))

    assert format_projets_block() not in system


def test_a_tool_filed_under_jamais_is_not_claimed(tmp_path, projects):
    """The user filed `claudeCode` under `## Jamais`: every call is refused
    at the gate, and the prompt must not contradict a decision only he
    makes."""
    _policy(tmp_path, f"## Jamais\n- {TOOL_NAME}\n")

    system = _system_prompt(_cfg(tmp_path, str(projects)))

    assert format_projets_block() not in system


def test_a_tool_that_asks_first_is_still_claimed(tmp_path, projects):
    _policy(tmp_path, f"## Demande\n- {TOOL_NAME}\n")

    system = _system_prompt(_cfg(tmp_path, str(projects)))

    assert format_projets_block() in system


# ── The turn that runs an approved call ───────────────────────────────


def test_the_turn_that_runs_an_approved_call_is_not_told(tmp_path, projects):
    """The click replays the original request, a request for work on a
    project. Told there to hand it over, and to search for the tool when
    it is not among her tools, she would propose the call again and raise
    a second card for one action. That turn runs the approved call once,
    and the block is not in its prompt."""
    from src.jarvis.memory.conversation import DialogueMemory
    from src.jarvis.tools import registry
    from src.jarvis.tools.builtin.claude_code import ClaudeCodeTool
    from src.jarvis.tools.confirmation import Approval
    from src.jarvis.tools.types import ToolExecutionResult

    class _Recorder(ClaudeCodeTool):
        def __init__(self):
            self.fired = []

        def run(self, args, context):
            self.fired.append(dict(args or {}))
            return ToolExecutionResult(
                success=True, reply_text="Claude Code has started on the project quillon.",
            )

    tool = _Recorder()
    catalogue = {**registry.BUILTIN_TOOLS, TOOL_NAME: tool}
    cfg = _cfg(tmp_path, str(projects))
    dm = DialogueMemory()
    args = {"projet": "quillon", "tache": "corrige le bug des dates", "modifier": True}
    call = {"message": {"content": "", "tool_calls": [
        {"function": {"name": TOOL_NAME, "arguments": args}},
    ]}}

    with patch.object(registry, "BUILTIN_TOOLS", catalogue), \
         patch("src.jarvis.reply.engine.BUILTIN_TOOLS", catalogue):
        first_system, _ = _first_call(
            cfg, text="corrige le bug des dates dans quillon", dialogue_memory=dm,
            replies=[call],
        )
        pending = dm.peek_pending()
        assert pending is not None, "the change did not raise a card"
        assert tool.fired == []
        claimed = dm.take_pending_by_id(pending.request_id)

        resume_system, _ = _first_call(
            cfg, text=claimed.query_redacted, dialogue_memory=dm,
            replies=[{"message": {"content": "C'est lancé."}}],
            granted=Approval(request_id=claimed.request_id,
                             fingerprint=claimed.fingerprint),
            granted_action=claimed,
        )

    assert format_projets_block() in first_system
    assert format_projets_block() not in resume_system
    assert tool.fired == [args]
    assert dm.peek_pending() is None

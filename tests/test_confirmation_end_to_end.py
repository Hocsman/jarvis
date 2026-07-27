"""Two whole turns, through the real engine.

Everything else in this feature is tested a piece at a time. This is the
one that would catch the pieces being wired together wrongly: a real
`run_reply_engine` call, a real gate, a real policy file, a real store,
and a tool whose side effect is observable — so "it did not run" is a
fact about the world rather than about a mock's call count.

Turn one must end on the question the code wrote, without going back to
the chat model: past the short-circuit a long result is rewritten by a
small model and the loop then tells the model to keep calling tools.
Turn two must run the pinned call once, and only on a yes.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from src.jarvis.memory.conversation import DialogueMemory
from src.jarvis.reply.engine import run_reply_engine
from src.jarvis.tools.base import Tool
from src.jarvis.tools.confirmation import DENIED, GRANTED, SPOKEN_TEMPLATE
from src.jarvis.tools.policy import ToolPolicy
from src.jarvis.tools.types import ToolExecutionResult


def _cfg(tmp_path):
    cfg = Mock()
    cfg.llm_chat_model = "test-large"
    cfg.ollama_chat_model = "test-large"
    cfg.ollama_base_url = "http://localhost:11434"
    cfg.voice_debug = False
    cfg.llm_tools_timeout_sec = 8.0
    cfg.llm_embedding_timeout_sec = 10.0
    cfg.llm_chat_timeout_sec = 45.0
    cfg.llm_digest_timeout_sec = 8.0
    cfg.memory_enrichment_max_results = 5
    cfg.memory_enrichment_source = "diary"
    cfg.memory_digest_enabled = False
    cfg.tool_result_digest_enabled = False
    cfg.location_ip_address = None
    cfg.location_auto_detect = False
    cfg.location_enabled = False
    cfg.agentic_max_turns = 8
    cfg.tool_search_max_calls = 3
    cfg.tool_selection_strategy = "all"
    cfg.tool_carryover_max_turns = 2
    cfg.tool_carryover_per_entry_chars = 1200
    cfg.mcps = {}
    cfg.llm_thinking_enabled = False
    cfg.tts_engine = "none"
    cfg.ollama_embed_model = "test-embed"
    cfg.confirmation_ttl_sec = 180.0
    cfg.confirmation_timeout_sec = 8.0
    cfg.confirmation_model = ""
    cfg.db_path = str(tmp_path / "jarvis.db")
    return cfg


class _Sabotage(Tool):
    """Its side effect is a list anyone can look at."""

    def __init__(self):
        self.fired = []

    def risk_for(self, args):
        return "action"

    @property
    def name(self):
        return "sabotage"

    @property
    def description(self):
        return "fait quelque chose"

    @property
    def inputSchema(self):
        return {"type": "object", "properties": {"cible": {"type": "string"}}}

    def run(self, args, context):
        self.fired.append(args)
        return ToolExecutionResult(success=True, reply_text="c'est fait")


def _tool_call_response(name, args):
    return {"message": {"content": "", "tool_calls": [
        {"function": {"name": name, "arguments": args}},
    ]}}


@pytest.fixture
def scene(tmp_path):
    """One tool, one policy that asks about it, one shared memory."""
    from src.jarvis.tools import registry

    tool = _Sabotage()
    dm = DialogueMemory()
    cfg = _cfg(tmp_path)
    policy = ToolPolicy.parse("## Demande\n- sabotage\n")

    catalogue = {**registry.BUILTIN_TOOLS, "sabotage": tool}
    # The engine binds its own name at import time, so patching the
    # registry's dict alone leaves the allow-list built from the old one
    # and the tool never reaches the gate.
    with patch.object(registry, "BUILTIN_TOOLS", catalogue), \
         patch("src.jarvis.reply.engine.BUILTIN_TOOLS", catalogue), \
         patch.object(registry, "load_tool_policy", return_value=policy), \
         patch("src.jarvis.reply.engine.plan_query", return_value=[]), \
         patch("src.jarvis.reply.engine.extract_search_params_for_memory",
               return_value={}):
        yield tool, dm, cfg


def _turn(cfg, dm, text, *, chat_responses, extract="", origin="voix"):
    with patch("src.jarvis.reply.engine.chat_with_messages") as chat, \
         patch("src.jarvis.reply.engine.extract_text_from_response",
               return_value=extract):
        chat.side_effect = chat_responses
        reply = run_reply_engine(
            db=Mock(), cfg=cfg, tts=None, text=text, dialogue_memory=dm,
            origin=origin,
        )
        return reply, chat


# ── Turn one: she asks ────────────────────────────────────────────────


def test_a_turn_that_needs_permission_ends_by_asking(scene):
    tool, dm, cfg = scene

    reply, _ = _turn(cfg, dm, "lance sabotage sur /a", chat_responses=[
        _tool_call_response("sabotage", {"cible": "/a"}),
    ])

    assert reply == SPOKEN_TEMPLATE.format(tool="sabotage")
    assert tool.fired == []


def test_she_does_not_go_back_to_the_model_after_asking(scene):
    """One chat call, for the tool call itself. A second would be the
    model getting a chance to rewrite the question."""
    tool, dm, cfg = scene

    _, chat = _turn(cfg, dm, "lance sabotage sur /a", chat_responses=[
        _tool_call_response("sabotage", {"cible": "/a"}),
    ])

    assert chat.call_count == 1


def test_the_question_is_waiting_afterwards(scene):
    tool, dm, cfg = scene

    _turn(cfg, dm, "lance sabotage sur /a", chat_responses=[
        _tool_call_response("sabotage", {"cible": "/a"}),
    ])

    waiting = dm.peek_pending()
    assert waiting is not None
    assert waiting.tool == "sabotage"
    assert waiting.args == {"cible": "/a"}


# ── Turn two: the answer ──────────────────────────────────────────────


def test_a_spoken_yes_runs_the_pinned_call_once(scene):
    tool, dm, cfg = scene
    _turn(cfg, dm, "lance sabotage sur /a", chat_responses=[
        _tool_call_response("sabotage", {"cible": "/a"}),
    ])

    with patch("src.jarvis.reply.engine.read_approval", return_value=GRANTED):
        _turn(cfg, dm, "vas-y", extract="Voilà, c'est fait.", chat_responses=[
            {"message": {"content": "Voilà, c'est fait."}},
        ])

    assert tool.fired == [{"cible": "/a"}]


def test_a_spoken_no_runs_nothing_and_says_so(scene):
    tool, dm, cfg = scene
    _turn(cfg, dm, "lance sabotage sur /a", chat_responses=[
        _tool_call_response("sabotage", {"cible": "/a"}),
    ])

    with patch("src.jarvis.reply.engine.read_approval", return_value=DENIED):
        reply, chat = _turn(cfg, dm, "surtout pas", chat_responses=[])

    assert tool.fired == []
    assert reply
    # And it cost no model call: acknowledging a refusal does not need one,
    # and going back to the model would let it re-propose what was refused.
    assert chat.call_count == 0


def test_the_question_is_gone_either_way(scene):
    tool, dm, cfg = scene
    _turn(cfg, dm, "lance sabotage sur /a", chat_responses=[
        _tool_call_response("sabotage", {"cible": "/a"}),
    ])

    with patch("src.jarvis.reply.engine.read_approval", return_value=DENIED):
        _turn(cfg, dm, "surtout pas", chat_responses=[])

    assert dm.peek_pending() is None


def test_a_yes_two_turns_later_does_not_run_it(scene):
    """The window is the very next turn. A grant that survives an
    intervening exchange answers a question the user has stopped thinking
    about."""
    tool, dm, cfg = scene
    _turn(cfg, dm, "lance sabotage sur /a", chat_responses=[
        _tool_call_response("sabotage", {"cible": "/a"}),
    ])

    _turn(cfg, dm, "quelle heure est-il", extract="Il est midi.",
          chat_responses=[{"message": {"content": "Il est midi."}}])

    with patch("src.jarvis.reply.engine.read_approval", return_value=GRANTED):
        _turn(cfg, dm, "vas-y", extract="ok",
              chat_responses=[{"message": {"content": "ok"}}])

    assert tool.fired == []


def test_the_yes_runs_what_was_shown_not_what_the_model_says_next(scene):
    """She asked about /a and the user said yes to /a. On the next turn
    the model proposes /b. The grant runs /a — the call it was given for
    — and does not stretch to cover /b."""
    tool, dm, cfg = scene
    _turn(cfg, dm, "lance sabotage sur /a", chat_responses=[
        _tool_call_response("sabotage", {"cible": "/a"}),
    ])

    with patch("src.jarvis.reply.engine.read_approval", return_value=GRANTED):
        _turn(cfg, dm, "vas-y", extract="ok", chat_responses=[
            _tool_call_response("sabotage", {"cible": "/b"}),
            {"message": {"content": "ok"}},
        ])

    assert tool.fired == [{"cible": "/a"}]


def test_the_model_cannot_slip_a_second_call_past_the_grant(scene):
    """Having spent the approval, the model's own proposal is asked about
    from scratch rather than riding the yes that was just given."""
    tool, dm, cfg = scene
    _turn(cfg, dm, "lance sabotage sur /a", chat_responses=[
        _tool_call_response("sabotage", {"cible": "/a"}),
    ])

    with patch("src.jarvis.reply.engine.read_approval", return_value=GRANTED):
        _turn(cfg, dm, "vas-y", extract="ok", chat_responses=[
            _tool_call_response("sabotage", {"cible": "/b"}),
            {"message": {"content": "ok"}},
        ])

    assert {"cible": "/b"} not in tool.fired
    # And the attempt is now the question waiting on the user.
    waiting = dm.peek_pending()
    assert waiting is not None and waiting.args == {"cible": "/b"}

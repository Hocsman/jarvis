"""What happens after the user says yes.

The refusal half of this feature is the part that got the attention, and
it holds. This file covers the other half, which is where the failures
were: turning a yes into a result, once, and telling the user when their
decision did not take.

The click path replays the original query through the engine. That means
the planner re-plans it and the model re-proposes the same call — so
unless the approved execution is visible to the loop's own duplicate
guards, she runs the call, throws the result away, and asks the very same
question again. The user then says yes twice for one action.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.jarvis.memory.conversation import DialogueMemory
from src.jarvis.tools.base import Tool
from src.jarvis.tools.confirmation import Approval, SPOKEN_TEMPLATE
from src.jarvis.tools.policy import ToolPolicy
from src.jarvis.tools.types import ToolExecutionResult


class _Sabotage(Tool):
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


def _cfg(tmp_path):
    cfg = MagicMock()
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


def _tool_call(name, args):
    return {"message": {"content": "", "tool_calls": [
        {"function": {"name": name, "arguments": args}},
    ]}}


@pytest.fixture
def scene(tmp_path):
    from src.jarvis.tools import registry

    tool = _Sabotage()
    dm = DialogueMemory()
    cfg = _cfg(tmp_path)
    catalogue = {**registry.BUILTIN_TOOLS, "sabotage": tool}

    with patch.object(registry, "BUILTIN_TOOLS", catalogue), \
         patch("src.jarvis.reply.engine.BUILTIN_TOOLS", catalogue), \
         patch.object(registry, "load_tool_policy",
                      return_value=ToolPolicy.parse("## Demande\n- sabotage\n")), \
         patch("src.jarvis.reply.engine.extract_search_params_for_memory",
               return_value={}):
        yield tool, dm, cfg


def _ask_first(cfg, dm, tool):
    """Turn one: she proposes the tool and ends on the question."""
    from src.jarvis.reply.engine import run_reply_engine

    with patch("src.jarvis.reply.engine.plan_query", return_value=[]), \
         patch("src.jarvis.reply.engine.chat_with_messages") as chat, \
         patch("src.jarvis.reply.engine.extract_text_from_response", return_value=""):
        chat.side_effect = [_tool_call("sabotage", {"cible": "/a"})]
        run_reply_engine(db=MagicMock(), cfg=cfg, tts=None,
                         text="lance sabotage sur /a", dialogue_memory=dm,
                         origin="voix")
    return dm.peek_pending()


# ── The click path, which replays the original query ──────────────────


def test_an_approved_click_runs_the_tool_once(scene):
    from src.jarvis.reply.engine import run_reply_engine

    tool, dm, cfg = scene
    action = _ask_first(cfg, dm, tool)
    claimed = dm.take_pending_by_id(action.request_id)

    # The resume replays the query. The model, seeing the same request,
    # proposes the same call again — which is exactly what it should do,
    # and exactly what must not run a second time.
    with patch("src.jarvis.reply.engine.plan_query", return_value=[]), \
         patch("src.jarvis.reply.engine.chat_with_messages") as chat, \
         patch("src.jarvis.reply.engine.extract_text_from_response",
               return_value="Voilà, c'est fait."):
        chat.side_effect = [
            _tool_call("sabotage", {"cible": "/a"}),
            {"message": {"content": "Voilà, c'est fait."}},
        ]
        reply = run_reply_engine(
            db=MagicMock(), cfg=cfg, tts=None, text=claimed.query_redacted,
            dialogue_memory=dm, origin="voix",
            granted=Approval(request_id=claimed.request_id,
                             fingerprint=claimed.fingerprint),
            granted_action=claimed,
        )

    assert tool.fired == [{"cible": "/a"}]
    assert reply != SPOKEN_TEMPLATE.format(tool="sabotage")


def test_an_approved_click_does_not_ask_the_same_question_again(scene):
    """The user said yes. Hearing the identical question back, while the
    result they authorised is discarded, is the worst outcome available:
    they say yes twice and the tool runs twice."""
    from src.jarvis.reply.engine import run_reply_engine

    tool, dm, cfg = scene
    action = _ask_first(cfg, dm, tool)
    claimed = dm.take_pending_by_id(action.request_id)

    with patch("src.jarvis.reply.engine.plan_query", return_value=[]), \
         patch("src.jarvis.reply.engine.chat_with_messages") as chat, \
         patch("src.jarvis.reply.engine.extract_text_from_response",
               return_value="Voilà, c'est fait."):
        chat.side_effect = [
            _tool_call("sabotage", {"cible": "/a"}),
            {"message": {"content": "Voilà, c'est fait."}},
        ]
        run_reply_engine(
            db=MagicMock(), cfg=cfg, tts=None, text=claimed.query_redacted,
            dialogue_memory=dm, origin="voix",
            granted=Approval(request_id=claimed.request_id,
                             fingerprint=claimed.fingerprint),
            granted_action=claimed,
        )

    assert dm.peek_pending() is None


def test_the_approved_result_reaches_the_model(scene):
    """Otherwise she narrates an action she cannot see the outcome of."""
    from src.jarvis.reply.engine import run_reply_engine

    tool, dm, cfg = scene
    action = _ask_first(cfg, dm, tool)
    claimed = dm.take_pending_by_id(action.request_id)

    seen = {}

    def _capture(*args, **kw):
        seen["messages"] = args[1] if len(args) > 1 else kw.get("messages")
        return {"message": {"content": "Voilà."}}

    with patch("src.jarvis.reply.engine.plan_query", return_value=[]), \
         patch("src.jarvis.reply.engine.chat_with_messages", side_effect=_capture), \
         patch("src.jarvis.reply.engine.extract_text_from_response",
               return_value="Voilà."):
        run_reply_engine(
            db=MagicMock(), cfg=cfg, tts=None, text=claimed.query_redacted,
            dialogue_memory=dm, origin="voix",
            granted=Approval(request_id=claimed.request_id,
                             fingerprint=claimed.fingerprint),
            granted_action=claimed,
        )

    blob = "\n".join(str(m.get("content", "")) for m in seen["messages"])
    assert "c'est fait" in blob


# ── The same, with a planner that re-plans ────────────────────────────


def test_a_replanned_step_does_not_re_run_the_approved_call(scene):
    """The click replays the query, so the planner produces the same first
    step it produced before. Direct-exec must see that step as already
    done rather than as work outstanding."""
    from src.jarvis.reply.engine import run_reply_engine

    tool, dm, cfg = scene
    action = _ask_first(cfg, dm, tool)
    claimed = dm.take_pending_by_id(action.request_id)

    with patch("src.jarvis.reply.engine.plan_query",
               return_value=["lancer sabotage sur /a", "résumer"]), \
         patch("src.jarvis.reply.engine.chat_with_messages") as chat, \
         patch("src.jarvis.reply.engine.extract_text_from_response",
               return_value="Voilà, c'est fait."):
        chat.side_effect = [
            {"message": {"content": "Voilà, c'est fait."}},
            {"message": {"content": "Voilà, c'est fait."}},
            {"message": {"content": "Voilà, c'est fait."}},
        ]
        run_reply_engine(
            db=MagicMock(), cfg=cfg, tts=None, text=claimed.query_redacted,
            dialogue_memory=dm, origin="voix",
            granted=Approval(request_id=claimed.request_id,
                             fingerprint=claimed.fingerprint),
            granted_action=claimed,
        )

    assert tool.fired == [{"cible": "/a"}]

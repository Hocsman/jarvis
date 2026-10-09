"""The compound-query split operates on the redacted utterance.

Everything the model reads is the redacted form: the user message, the
memory blocks, the tool carryover. The compound split feeds the
"Still unanswered: ..." nudge, which is itself a message to the model,
so its input has to be the redacted text too. Run on the raw utterance,
a small model with an empty plan gets the second clause pasted verbatim
into that nudge — an email address every other surface has stripped —
and is ordered to webSearch for it, and webSearch does not scrub its
query. The raw clauses also reach the debug log.
"""

import json
from unittest.mock import Mock, patch

import pytest

from src.jarvis.memory.conversation import DialogueMemory
from src.jarvis.reply.engine import run_reply_engine
from src.jarvis.tools.types import ToolExecutionResult

RAW_ADDRESS = "alice@example.com"
QUERY = (
    f"what is the weather in Paris today and email the summary to {RAW_ADDRESS}"
)


def _mock_cfg():
    """A SMALL-model config: gemma4:e2b drives the text-tool path where
    the compound split and its nudge live."""
    cfg = Mock()
    cfg.ollama_base_url = "http://localhost:11434"
    cfg.ollama_chat_model = "gemma4:e2b"
    cfg.llm_chat_model = "gemma4:e2b"
    cfg.llm_provider = "ollama"
    cfg.llm_base_url = "http://localhost:11434"
    cfg.llm_api_key = ""
    cfg.voice_debug = False
    cfg.llm_tools_timeout_sec = 8.0
    cfg.tool_router_timeout_sec = 8.0
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
    cfg.planner_enabled = True
    cfg.planner_model = ""
    cfg.planner_timeout_sec = 6.0
    cfg.wake_word = "jarvis"
    cfg.response_language = ""
    return cfg


@pytest.mark.unit
@patch("src.jarvis.reply.engine.plan_query", return_value=[])
@patch("src.jarvis.reply.engine.extract_search_params_for_memory", return_value={})
@patch("src.jarvis.reply.engine.run_tool_with_retries")
@patch("src.jarvis.reply.engine.extract_text_from_response")
@patch("src.jarvis.reply.engine.chat_with_messages")
def test_the_compound_nudge_names_the_redacted_clause(
    mock_chat, mock_extract, mock_tool, _mock_mem, _mock_plan
):
    """With an empty plan and a small model, the first tool result is
    followed by the compound nudge. The clause it names is the second
    half of the user's utterance, which carries an email address: the
    model must see the redacted clause and nothing else, in the nudge
    and in every other message."""
    mock_tool.return_value = ToolExecutionResult(
        success=True, reply_text="Sunny, 22 degrees.", error_message=None,
    )
    fenced_call = (
        'tool_calls: [{"id": "call_1", "type": "function", '
        '"function": {"name": "webSearch", '
        '"arguments": {"search_query": "weather in Paris today"}}}]'
    )
    mock_chat.side_effect = [
        {"message": {"content": fenced_call}},      # turn 1: tool call
        {"message": {"content": "It is sunny."}},   # turn 2: final reply
    ]

    def _content_of(llm_resp):
        if isinstance(llm_resp, dict):
            return (llm_resp.get("message") or {}).get("content", "")
        return ""

    mock_extract.side_effect = _content_of

    run_reply_engine(
        db=Mock(), cfg=_mock_cfg(), tts=None, text=QUERY,
        dialogue_memory=DialogueMemory(),
    )

    # Every message the model saw, across every chat call.
    seen = []
    for call in mock_chat.call_args_list:
        seen.extend(call.kwargs.get("messages") or [])
    assert seen, "the engine never called the model"
    blob = json.dumps(seen, default=str)

    assert RAW_ADDRESS not in blob, (
        "the raw address reached the model: the compound split ran on "
        "the raw utterance instead of the redacted one"
    )

    nudges = [
        m.get("content") for m in seen
        if isinstance(m.get("content"), str) and "Still unanswered" in m["content"]
    ]
    assert nudges, "the compound nudge never fired"
    assert "[REDACTED_EMAIL]" in nudges[0], (
        f"the nudge does not name the redacted clause: {nudges[0]!r}"
    )
    assert "email the summary to" in nudges[0]

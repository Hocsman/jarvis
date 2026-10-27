"""The memory extractor never asks its model to think.

`llm_thinking_enabled` is the chat model's setting. The extractor runs on
its own model or the router's chain, often a small local model with no
thinking mode, and such a model refuses a request that asks for one
(Ollama answers HTTP 400 "does not support thinking"). Passed through, the
setting would leave every turn that needs memory without it.

The graph's picker and merge never think either; that is covered with the
rest of the graph in tests/test_graph_runs_on_the_chat_model.py.

See src/jarvis/reply/reply.spec.md, step 4.
"""

from __future__ import annotations

import json
from unittest.mock import patch


class _ModelWithoutThinking:
    """A backend for a model that has no thinking mode: a request asking
    for one gets nothing back, as the real backends answer an HTTP 400."""

    def direct(self, model, system_prompt, user_content, *, thinking=False, **kwargs):
        if thinking:
            return None
        return json.dumps({"keywords": ["films", "cinema"]})


def test_a_memory_turn_finds_its_keywords_when_the_chat_model_thinks(
    mock_config, db, dialogue_memory, capsys
):
    from jarvis.reply import engine as engine_mod
    from jarvis.reply import enrichment

    mock_config.llm_chat_model = "gpt-oss:20b"
    mock_config.tool_router_model = "qwen2.5:3b"
    mock_config.llm_thinking_enabled = True

    searched: list[list[str]] = []

    def fake_diary_search(*, keywords, **kwargs):
        searched.append(list(keywords))
        return []

    with patch.object(enrichment, "get_auxiliary_backend", return_value=_ModelWithoutThinking()), \
         patch("jarvis.memory.conversation.search_conversation_memory_by_keywords",
               side_effect=fake_diary_search), \
         patch.object(engine_mod, "chat_with_messages",
                      return_value={"message": {"role": "assistant", "content": "Voilà."}}), \
         patch.object(engine_mod, "select_tools", return_value=["stop"]), \
         patch.object(engine_mod, "plan_query",
                      return_value=["searchMemory topic='films'", "Reply to the user."]):
        engine_mod.run_reply_engine(
            db=db, cfg=mock_config, tts=None,
            text="quels films je t'avais dit avoir aimés ?",
            dialogue_memory=dialogue_memory,
        )

    assert searched == [["films", "cinema"]]
    assert "keywords could not be read" not in capsys.readouterr().out


"""The memory extractor never asks its model to think.

`llm_thinking_enabled` is the chat model's setting. The extractor runs on
the router's model chain, often a small local model with no thinking mode,
and such a model refuses a request that asks for one (Ollama answers HTTP
400 "does not support thinking"). Passed through, the setting would leave
every turn that needs memory without it.

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


# ── The graph's picker and merge ride the same small model ────────────


SMALL = "qwen2.5:3b"
NEW_FACT = "Possessor was directed by Brandon Cronenberg."
KNOWN = "Alien was directed by Ridley Scott."


def _graph_models(*, chat_model, system_prompt, thinking=False, **kwargs):
    """The chat model extracts one fact; the small model has no thinking
    mode, then picks the Cinema node and rewrites it without the
    duplicate it carried."""
    if chat_model == SMALL and thinking:
        return None
    if chat_model != SMALL:
        return json.dumps([NEW_FACT])
    if "Respond with ONLY the number" in system_prompt:
        return "1"
    return json.dumps({"facts": [KNOWN, NEW_FACT]})


def test_a_fact_is_filed_and_merged_when_the_chat_model_thinks(tmp_path):
    from jarvis.memory import graph_ops
    from jarvis.memory.graph import BRANCH_WORLD, GraphMemoryStore

    store = GraphMemoryStore(str(tmp_path / "graph.db"))
    try:
        cinema = store.create_node(
            name="Cinema", description="Films and directors",
            data=f"{KNOWN}\nAlien's director is Ridley Scott.", parent_id=BRANCH_WORLD,
        )
        with patch.object(graph_ops, "call_llm_direct", side_effect=_graph_models):
            graph_ops.update_graph_from_dialogue(
                store=store, summary="We talked about Possessor.", cfg=None,
                chat_model="gpt-oss:20b", tools_used=["webSearch"],
                thinking=True, picker_model=SMALL,
            )

        lines = store.get_node(cinema.id).data.splitlines()
        assert any("Possessor" in line for line in lines)
        assert not any("Alien's director" in line for line in lines)
        assert "Possessor" not in (store.get_node(BRANCH_WORLD).data or "")
    finally:
        store.close()

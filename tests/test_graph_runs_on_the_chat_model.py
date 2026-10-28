"""The knowledge graph's picker and merge run on the chat model (#11, #11b).

Placement and merge ask `llm_chat_model`, like extraction (#10) and split
(#11c), even when a small router model is configured. The merge rewrites a
node in full and nothing checks that every fact survives it, so a model
that loses a line loses it for good; a small local router model also
answered NONE for a cinema fact in front of a node named Cinema, which
files the fact at the branch root where nothing will look for it.

Each test lets only the chat model answer: a call sent anywhere else gets
nothing back, so a fact placed or merged by another model never reaches
the Cinema node. Neither call asks the model to think: a picker or merge
that did would spend the pass's deadline thinking and get nothing back.

See src/jarvis/memory/graph.spec.md, "Traverse" and "Merge", and
docs/llm_contexts.md, #11 and #11b.
"""

from __future__ import annotations

import json
from contextlib import ExitStack
from unittest.mock import patch

import pytest

from conftest import MockConfig


ROUTER = "qwen2.5:3b"
CHAT = "google/gemini-2.5-flash"

NEW_FACT = "Possessor was directed by Brandon Cronenberg."
KNOWN = "Alien was directed by Ridley Scott."
TWIN = "Alien's director is Ridley Scott."

_PICKER_PROMPT_MARK = "Respond with ONLY the number"


def _only_the_chat_model_answers(*, chat_model, system_prompt, user_content,
                                 thinking=False, **kwargs):
    """Extracts one fact, picks the Cinema node, and rewrites it without
    the duplicate it carried. Extraction may think; the picker and the
    merge get nothing back if they ask to."""
    from jarvis.memory import graph_ops

    if chat_model != CHAT:
        return None
    if _PICKER_PROMPT_MARK in system_prompt:
        return None if thinking else "1"
    if system_prompt == graph_ops._MERGE_SYSTEM_PROMPT:
        if thinking:
            return None
        incoming = [NEW_FACT] if NEW_FACT in user_content else []
        return json.dumps({"facts": [KNOWN, *incoming]})
    return json.dumps([NEW_FACT])


def _config():
    cfg = MockConfig(llm_chat_model=CHAT, tool_router_model=ROUTER)
    cfg.intent_judge_model = ROUTER
    cfg.llm_thinking_enabled = True
    return cfg


def _seed_cinema(store):
    from jarvis.memory.graph import BRANCH_WORLD

    return store.create_node(
        name="Cinema", description="Films and directors",
        data=f"{KNOWN}\n{TWIN}", parent_id=BRANCH_WORLD,
    )


def _assert_filed_and_merged(store, cinema_id):
    from jarvis.memory.graph import BRANCH_WORLD

    lines = store.get_node(cinema_id).data.splitlines()
    assert any("Possessor" in line for line in lines), lines
    assert not any("Alien's director" in line for line in lines), lines
    assert "Possessor" not in (store.get_node(BRANCH_WORLD).data or "")


@pytest.fixture
def graph_db(tmp_path):
    from jarvis.memory.db import Database
    from jarvis.memory.graph import GraphMemoryStore

    db_path = str(tmp_path / "jarvis.db")
    db = Database(db_path, sqlite_vss_path=None)
    store = GraphMemoryStore(db_path)
    yield db_path, db, store
    store.close()
    db.close()


@pytest.mark.unit
def test_a_fact_is_filed_and_merged_when_the_chat_model_thinks(graph_db):
    from jarvis.memory import graph_ops

    _, _, store = graph_db
    cinema = _seed_cinema(store)

    with patch.object(graph_ops, "call_llm_direct", side_effect=_only_the_chat_model_answers):
        graph_ops.update_graph_from_dialogue(
            store=store, summary="We talked about Possessor.", cfg=None,
            chat_model=CHAT, tools_used=["webSearch"], thinking=True,
        )

    _assert_filed_and_merged(store, cinema.id)


@pytest.mark.unit
def test_a_diary_flush_files_and_merges_on_the_chat_model(graph_db, monkeypatch):
    from jarvis import daemon
    from jarvis.memory import graph_ops
    from jarvis.memory.conversation import DialogueMemory

    _, db, store = graph_db
    cinema = _seed_cinema(store)

    memory = DialogueMemory(inactivity_timeout=300, max_interactions=20)
    memory.add_message("user", "Who directed Possessor?")
    memory.record_tool_turn([
        {"role": "tool", "tool_name": "webSearch", "content": "Brandon Cronenberg"},
    ])
    memory.add_message("assistant", "Brandon Cronenberg.")
    monkeypatch.setattr(daemon, "_global_dialogue_memory", memory)

    with patch("jarvis.memory.conversation.generate_conversation_summary",
               return_value=("We talked about Possessor.", "films")), \
         patch.object(graph_ops, "call_llm_direct", side_effect=_only_the_chat_model_answers):
        daemon._check_and_update_diary(db, _config(), force=True)

    _assert_filed_and_merged(store, cinema.id)


try:
    import flask as _flask  # noqa: F401
    _HAS_FLASK = True
except ImportError:
    _HAS_FLASK = False


@pytest.fixture
def viewer(graph_db):
    """The memory viewer's client, pointed at this graph and config."""
    from conftest import ViewerClient
    from src.desktop_app import memory_viewer

    db_path, _, store = graph_db
    with ExitStack() as stack:
        stack.enter_context(patch.object(memory_viewer, "_get_db_path", return_value=db_path))
        stack.enter_context(patch.object(memory_viewer, "get_graph_store", return_value=store))
        stack.enter_context(patch("jarvis.config.load_settings", return_value=_config()))
        stack.enter_context(patch("jarvis.memory.graph_ops.call_llm_direct",
                                  side_effect=_only_the_chat_model_answers))
        yield ViewerClient(memory_viewer)


@pytest.mark.unit
@pytest.mark.skipif(not _HAS_FLASK, reason="Flask not available")
def test_the_diary_import_files_and_merges_on_the_chat_model(graph_db, viewer):
    _, db, store = graph_db
    cinema = _seed_cinema(store)
    db.conn.execute(
        """INSERT INTO conversation_summaries (date_utc, ts_utc, summary, topics, source_app)
           VALUES (?, ?, ?, ?, ?)""",
        ("2026-10-26", "2026-10-26T21:00:00Z", "We talked about Possessor.", "films", "jarvis"),
    )
    db.conn.commit()

    resp = viewer.post("/api/graph/import-diary", json={})
    assert resp.status_code == 200
    resp.get_data()

    _assert_filed_and_merged(store, cinema.id)


@pytest.mark.unit
@pytest.mark.skipif(not _HAS_FLASK, reason="Flask not available")
def test_the_consolidation_sweep_runs_on_the_chat_model(graph_db, viewer):
    _, _, store = graph_db
    cinema = _seed_cinema(store)

    resp = viewer.post("/api/graph/consolidate-all", json={})
    assert resp.status_code == 200
    resp.get_data()

    lines = store.get_node(cinema.id).data.splitlines()
    assert not any("Alien's director" in line for line in lines), lines

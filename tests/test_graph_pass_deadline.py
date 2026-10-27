"""The knowledge graph's pass has one deadline, and its calls share it.

The picker (#11), the merge (#11b), the split (#11c) and the extraction
before them (#10) run on the chat model. A chat model slower than a small
router model must still get to place and merge a fact, so no call is held to
a figure sized for something faster: each is handed what the pass has left.
The pass itself is what is bounded. Once its deadline is spent, or a stop
request arrives, it asks nothing more and finishes on the paths that need no
model, so a fact is filed less well but never lost.

That bound is what keeps shutdown whole. The desktop app gives the daemon
60 s to stop before it kills it, and a kill skips the speaker, the MCP
sessions and the database close. The diary pass at shutdown, summary and
graph together, is held to `SHUTDOWN_DIARY_TIMEOUT_SEC`, even when a call
never comes back.

See src/jarvis/memory/graph.spec.md, "The Pass's Deadline", and
docs/llm_contexts.md, #10, #11, #11b and #11c.
"""

from __future__ import annotations

import json
import threading
import time
from contextlib import ExitStack
from unittest.mock import patch

import pytest

from conftest import MockConfig


CHAT = "google/gemini-2.5-flash"

NEW_FACT = "Possessor was directed by Brandon Cronenberg."
KNOWN = "Alien was directed by Ridley Scott."
TWIN = "Alien's director is Ridley Scott."

# Slower than the 15 s and 20 s the picker and the merge used to be held to,
# and well inside one chat-model timeout.
SLOW = 25.0

_PICKER_PROMPT_MARK = "Respond with ONLY the number"


def _answer(system_prompt, user_content, facts):
    """What a chat model that does the job well says to each graph call."""
    from jarvis.memory import graph_ops

    if _PICKER_PROMPT_MARK in system_prompt:
        return "1"
    if system_prompt == graph_ops._MERGE_SYSTEM_PROMPT:
        incoming = [f for f in facts if f in user_content]
        return json.dumps({"facts": [KNOWN, *incoming]})
    return json.dumps(list(facts))


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class _SlowChatModel:
    """Answers well, but takes ``latency`` seconds a call. Handed less than
    that, it spends the budget and gives nothing back, as a request that
    times out does. On a ``_Clock`` the seconds are simulated; without one
    no real time passes and only the budget is compared."""

    def __init__(self, latency, facts=(NEW_FACT,), clock=None):
        self.latency = latency
        self.facts = facts
        self.clock = clock

    def __call__(self, *, chat_model, system_prompt, user_content,
                 timeout_sec, **kwargs):
        if self.clock is not None:
            self.clock.now += min(self.latency, timeout_sec)
        if timeout_sec < self.latency:
            return None
        return _answer(system_prompt, user_content, self.facts)


class _MergeNeverReturns:
    """Answers the extraction and the picker at once; the merge never comes
    back within any budget, like a provider that keeps the connection open
    past the timeout it was given. ``on_merge`` runs as the merge starts."""

    def __init__(self, on_merge=None):
        self.release = threading.Event()
        self.on_merge = on_merge

    def __call__(self, *, chat_model, system_prompt, user_content, **kwargs):
        from jarvis.memory import graph_ops

        if system_prompt == graph_ops._MERGE_SYSTEM_PROMPT:
            if self.on_merge is not None:
                self.on_merge()
            self.release.wait(30)
        return _answer(system_prompt, user_content, (NEW_FACT,))


def _config(chat_timeout=180.0):
    return MockConfig(llm_chat_model=CHAT, llm_chat_timeout_sec=chat_timeout)


def _seed_cinema(store):
    from jarvis.memory.graph import BRANCH_WORLD

    return store.create_node(
        name="Cinema", description="Films and directors",
        data=f"{KNOWN}\n{TWIN}", parent_id=BRANCH_WORLD,
    )


def _cinema_lines(store, cinema_id):
    return store.get_node(cinema_id).data.splitlines()


def _assert_filed_and_merged(store, cinema_id):
    lines = _cinema_lines(store, cinema_id)
    assert any("Possessor" in line for line in lines), lines
    assert not any("Alien's director" in line for line in lines), lines


def _world_branch_text(store):
    from jarvis.memory.graph import BRANCH_WORLD

    world = store.get_node(BRANCH_WORLD)
    texts = [world.data or ""]
    texts += [child.data or "" for child in store.get_children(BRANCH_WORLD)]
    return "\n".join(texts)


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


@pytest.fixture
def a_window_with_a_lookup(monkeypatch):
    """The daemon's dialogue memory, holding one exchange that looked
    something up, with the summariser answering at once."""
    from jarvis import daemon
    from jarvis.memory.conversation import DialogueMemory

    memory = DialogueMemory(inactivity_timeout=300, max_interactions=20)
    memory.add_message("user", "Who directed Possessor?")
    memory.record_tool_turn([
        {"role": "tool", "tool_name": "webSearch", "content": "Brandon Cronenberg"},
    ])
    memory.add_message("assistant", "Brandon Cronenberg.")
    monkeypatch.setattr(daemon, "_global_dialogue_memory", memory)
    monkeypatch.setattr(daemon, "_global_stop_requested", False)
    with patch("jarvis.memory.conversation.generate_conversation_summary",
               return_value=("We talked about Possessor.", "films")):
        yield memory


# ── A slow chat model still gets to do the job ────────────────────────


@pytest.mark.unit
def test_a_slow_chat_model_files_and_merges_from_the_diary_flush(
        graph_db, a_window_with_a_lookup):
    from jarvis import daemon
    from jarvis.memory import graph_ops

    _, db, store = graph_db
    cinema = _seed_cinema(store)

    with patch.object(graph_ops, "call_llm_direct", side_effect=_SlowChatModel(SLOW)):
        daemon._check_and_update_diary(db, _config(), force=True)

    _assert_filed_and_merged(store, cinema.id)


try:
    import flask as _flask  # noqa: F401
    _HAS_FLASK = True
except ImportError:
    _HAS_FLASK = False


@pytest.fixture
def slow_viewer(graph_db):
    """The memory viewer's client, on this graph, with a slow chat model."""
    from conftest import ViewerClient
    from src.desktop_app import memory_viewer

    db_path, _, store = graph_db
    with ExitStack() as stack:
        stack.enter_context(patch.object(memory_viewer, "_get_db_path", return_value=db_path))
        stack.enter_context(patch.object(memory_viewer, "get_graph_store", return_value=store))
        stack.enter_context(patch("jarvis.config.load_settings", return_value=_config()))
        stack.enter_context(patch("jarvis.memory.graph_ops.call_llm_direct",
                                  side_effect=_SlowChatModel(SLOW)))
        yield ViewerClient(memory_viewer)


@pytest.mark.unit
@pytest.mark.skipif(not _HAS_FLASK, reason="Flask not available")
def test_a_slow_chat_model_files_and_merges_from_the_diary_import(graph_db, slow_viewer):
    _, db, store = graph_db
    cinema = _seed_cinema(store)
    db.conn.execute(
        """INSERT INTO conversation_summaries (date_utc, ts_utc, summary, topics, source_app)
           VALUES (?, ?, ?, ?, ?)""",
        ("2026-10-26", "2026-10-26T21:00:00Z", "We talked about Possessor.", "films", "jarvis"),
    )
    db.conn.commit()

    resp = slow_viewer.post("/api/graph/import-diary", json={})
    assert resp.status_code == 200
    resp.get_data()

    _assert_filed_and_merged(store, cinema.id)


@pytest.mark.unit
@pytest.mark.skipif(not _HAS_FLASK, reason="Flask not available")
def test_a_slow_chat_model_consolidates_in_the_sweep(graph_db, slow_viewer):
    _, _, store = graph_db
    cinema = _seed_cinema(store)

    resp = slow_viewer.post("/api/graph/consolidate-all", json={})
    assert resp.status_code == 200
    resp.get_data()

    assert not any("Alien's director" in line
                   for line in _cinema_lines(store, cinema.id))


# ── The pass is bounded, and loses nothing to the bound ──────────────


@pytest.mark.unit
def test_once_the_deadline_is_spent_the_pass_stops_asking_and_keeps_every_fact(graph_db):
    from jarvis.memory import graph_ops

    _, _, store = graph_db
    _seed_cinema(store)
    facts = (
        NEW_FACT,
        "Crimes of the Future was directed by David Cronenberg.",
        "Infinity Pool was directed by Brandon Cronenberg.",
    )
    clock = _Clock()
    # Room for the extraction and one placement, not for the rest.
    deadline = graph_ops.GraphDeadline(60.0, clock=clock)

    with patch.object(graph_ops, "call_llm_direct",
                      side_effect=_SlowChatModel(SLOW, facts=facts, clock=clock)):
        result = graph_ops.update_graph_from_dialogue(
            store=store, summary="We talked about the Cronenbergs.", cfg=None,
            chat_model=CHAT, tools_used=["webSearch"], deadline=deadline,
        )

    assert clock.now <= 60.0
    world = _world_branch_text(store)
    for fact in facts:
        assert fact.rstrip(".") in world, world
    assert sorted(f for f, _ in result.stored) == sorted(facts)


@pytest.mark.unit
def test_a_call_that_never_returns_does_not_hold_the_shutdown_diary_pass(
        graph_db, a_window_with_a_lookup, monkeypatch):
    from jarvis import daemon
    from jarvis.memory import graph_ops

    _, db, store = graph_db
    cinema = _seed_cinema(store)
    budget = 1.0
    monkeypatch.setattr(daemon, "SHUTDOWN_DIARY_TIMEOUT_SEC", budget)
    # Shutdown is why this pass runs: the stop has already been asked for.
    monkeypatch.setattr(daemon, "_global_stop_requested", True)
    model = _MergeNeverReturns()

    try:
        with patch.object(graph_ops, "call_llm_direct", side_effect=model):
            started = time.monotonic()
            daemon._save_diary_before_shutdown(db, _config())
            elapsed = time.monotonic() - started
    finally:
        model.release.set()

    assert elapsed < budget + 1.5, elapsed
    assert any("Possessor" in line for line in _cinema_lines(store, cinema.id))


@pytest.mark.unit
def test_the_shutdown_diary_pass_still_learns_after_the_stop_request(
        graph_db, a_window_with_a_lookup, monkeypatch):
    from jarvis import daemon
    from jarvis.memory import graph_ops

    _, db, store = graph_db
    cinema = _seed_cinema(store)
    monkeypatch.setattr(daemon, "_global_stop_requested", True)

    with patch.object(graph_ops, "call_llm_direct", side_effect=_SlowChatModel(0.0)):
        daemon._save_diary_before_shutdown(db, _config())

    _assert_filed_and_merged(store, cinema.id)


@pytest.mark.unit
def test_a_stop_request_ends_the_periodic_pass_without_losing_the_fact(
        graph_db, a_window_with_a_lookup):
    from jarvis import daemon
    from jarvis.memory import graph_ops

    _, db, store = graph_db
    cinema = _seed_cinema(store)
    # He quits while the merge is out with the model.
    model = _MergeNeverReturns(on_merge=daemon.request_stop)

    try:
        with patch.object(graph_ops, "call_llm_direct", side_effect=model):
            started = time.monotonic()
            daemon._check_and_update_diary(db, _config(), force=True)
            elapsed = time.monotonic() - started
    finally:
        model.release.set()

    assert elapsed < 2.0, elapsed
    assert any("Possessor" in line for line in _cinema_lines(store, cinema.id))

"""A lookup she has already made does not need making again.

Two doors sit between a question and the graph. The lower one asks the
memory extractor what the query is about; the upper one is the planner,
which decides whether the extractor runs at all. The planner reads a
factual question as an errand for the network, so the lower door never
gets a chance — measured twice, in two shapes:

  (a) the plan is `webSearch ... | Reply`, memory never runs, and she
      pays a search plus a page fetch for a sentence in her own database;
  (b) the plan is `searchMemory ... | webSearch ... | Reply`, the graph
      answers, the model says as much, and the round trip is paid anyway.

Both are closed here without asking the planner to change its mind. The
plan's own arguments are a second key to the graph: the planner composed
them against the user's intent with pronouns resolved to literal names,
so they name the subject better than the utterance does, and they exist
on a turn where no memory was planned. And a read-only step whose every
word is already answered by the memory in front of the model is dropped
before the plan is injected.

The bias is deliberate. Nothing is dropped unless coverage is total, the
tool is never removed from the allow-list, and every drop is printed —
so a wrong one costs a turn, never an answer, and never goes unseen.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class _Node:
    def __init__(self, name, data):
        self.id, self.name, self.data = name, name, data
        self.data_token_count = len(data) // 4


KESTREL = "The Kestrel M3 is a compact workstation with 96 GB of unified memory."


def _turn(tmp_path, plan, *, noeuds=None, petit=False, digest=None,
          extracteur=None, graphe_casse=False):
    """One engine turn. Returns (system prompt, tools dispatched, output)."""
    from src.jarvis.memory.conversation import DialogueMemory
    from src.jarvis.reply.engine import run_reply_engine
    from src.jarvis.tools.types import ToolExecutionResult

    cfg = MagicMock()
    cfg.db_path = str(tmp_path / "t.db")
    cfg.llm_chat_model = "gemma4:e2b" if petit else "test-large-model"
    cfg.mcps = {}
    cfg.voice_debug = False
    cfg.memory_enrichment_source = "all"
    cfg.memory_enrichment_max_results = 3
    cfg.memory_digest_enabled = petit
    cfg.tool_result_digest_enabled = False
    cfg.location_enabled = False
    cfg.llm_thinking_enabled = False
    cfg.agentic_max_turns = 6
    cfg.tool_selection_strategy = "all"
    cfg.tool_carryover_max_turns = 2
    cfg.tool_carryover_per_entry_chars = 1200
    cfg.tool_search_max_calls = 3
    cfg.planner_enabled = True

    store = MagicMock()
    store.search_nodes.return_value = noeuds if noeuds is not None else []
    store.get_ancestors.return_value = [_Node("World", "")]

    appels = []

    def _outil(db, cfg_, tool_name, tool_args, **kw):
        appels.append(tool_name)
        return ToolExecutionResult(success=True, reply_text="No results found.")

    vu = {}

    def _chat(**kw):
        messages = kw.get("messages") or []
        vu.setdefault("system", messages[0].get("content", "") if messages else "")
        return {"message": {"content": "ok"}}

    def _store_factory(*a, **k):
        if graphe_casse:
            raise RuntimeError("database is locked")
        return store

    stack = [
        patch("src.jarvis.reply.engine.plan_query", return_value=list(plan)),
        patch("src.jarvis.memory.graph.GraphMemoryStore", side_effect=_store_factory),
        patch("src.jarvis.reply.engine.run_tool_with_retries", side_effect=_outil),
        patch("src.jarvis.reply.engine.chat_with_messages", side_effect=_chat),
        patch("src.jarvis.reply.engine.extract_text_from_response", return_value="ok"),
    ]
    if extracteur is not None:
        stack.append(patch("src.jarvis.reply.engine.extract_search_params_for_memory",
                           side_effect=extracteur))
    else:
        stack.append(patch("src.jarvis.reply.engine.extract_search_params_for_memory",
                           return_value={}))
    if digest is not None:
        stack.append(patch("src.jarvis.reply.engine.digest_memory_for_query",
                           return_value=digest))

    for p in stack:
        p.start()
    try:
        run_reply_engine(db=MagicMock(), cfg=cfg, tts=None,
                         text="the Kestrel M3, how much memory does it have?",
                         dialogue_memory=DialogueMemory(), origin="chat")
    finally:
        for p in reversed(stack):
            p.stop()
    return vu.get("system", ""), appels, store


def _plan_still_asks(prompt: str, needle: str) -> bool:
    """Is the step still in the ACTION PLAN the model is handed?

    On a large model the plan is advisory, so this is where a dropped
    step becomes observable: the model is simply never told to make the
    call. On a small model the same shortened list drives direct-exec.
    """
    if "ACTION PLAN" not in prompt:
        return False
    return needle.lower() in prompt[prompt.index("ACTION PLAN"):].lower()


WEB = ["webSearch query='Kestrel M3 memory'", "Reply to the user with the findings."]


# ── Shape (a): the plan never asked for memory ────────────────────────


def test_a_plan_that_names_only_a_web_lookup_still_reads_the_graph(tmp_path):
    """The measured case. No searchMemory, so the extractor never ran,
    so there were no keywords, so the graph was never opened."""
    prompt, _, store = _turn(tmp_path, WEB, noeuds=[_Node("Kestrel M3", KESTREL)])

    assert store.search_nodes.called
    assert "96 GB of unified memory" in prompt


def test_the_extractor_is_not_woken_to_do_it(tmp_path):
    """The graph is a SQLite scan. Paying an LLM to reach it would trade
    a network round trip for a local one and call it a saving."""
    def _boom(*a, **k):
        raise AssertionError("the extractor must not run for a plan-term read")

    _turn(tmp_path, WEB, noeuds=[_Node("Kestrel M3", KESTREL)], extracteur=_boom)


def test_she_does_not_go_out_for_something_she_has(tmp_path):
    prompt, _, _ = _turn(tmp_path, WEB, noeuds=[_Node("Kestrel M3", KESTREL)])

    assert not _plan_still_asks(prompt, "webSearch")


def test_the_skip_is_announced(tmp_path, capsys):
    _turn(tmp_path, WEB, noeuds=[_Node("Kestrel M3", KESTREL)])

    assert "already in memory" in capsys.readouterr().out


# ── Coverage must be total ────────────────────────────────────────────


def test_she_still_goes_out_for_the_half_she_does_not_have(tmp_path):
    """A note about the machine's memory does not answer what it costs.
    A rule that accepted the overlap would cancel the very search that
    carries the missing half."""
    plan = ["webSearch query='Kestrel M3 price'", "Reply to the user."]

    prompt, _, _ = _turn(tmp_path, plan, noeuds=[_Node("Kestrel M3", KESTREL)])

    assert _plan_still_asks(prompt, "webSearch")


def test_a_single_content_word_never_covers_a_step(tmp_path):
    plan = ["webSearch query='Kestrel'", "Reply to the user."]

    prompt, _, _ = _turn(tmp_path, plan, noeuds=[_Node("Kestrel M3", KESTREL)])

    assert _plan_still_asks(prompt, "webSearch")


def test_an_empty_graph_drops_nothing(tmp_path):
    prompt, _, _ = _turn(tmp_path, WEB, noeuds=[])

    assert _plan_still_asks(prompt, "webSearch")


# ── What can never be stood in for ────────────────────────────────────


def test_a_step_that_writes_is_never_skipped(tmp_path):
    """Reading a fact twice is waste. Not writing something because a
    similar sentence was already read is losing the user's instruction."""
    plan = ["remember fact='Kestrel M3 has 96 GB of unified memory'",
            "Reply to the user."]

    prompt, _, _ = _turn(tmp_path, plan, noeuds=[_Node("Kestrel M3", KESTREL)])

    assert _plan_still_asks(prompt, "remember")


def test_a_tool_the_catalogue_has_never_heard_of_is_never_skipped(tmp_path):
    """Unclassified is destructive, here as at the tool gate."""
    plan = ["inconnuMachin query='Kestrel M3 memory'", "Reply to the user."]

    prompt, _, _ = _turn(tmp_path, plan, noeuds=[_Node("Kestrel M3", KESTREL)])

    assert _plan_still_asks(prompt, "inconnuMachin")


def test_a_step_with_no_arguments_is_unsuppressible(tmp_path):
    """`getWeather` with no arguments asks about now and here. Nothing
    already read can answer it, and nothing should try."""
    plan = ["getWeather", "Reply to the user."]

    prompt, _, _ = _turn(tmp_path, plan, noeuds=[_Node("World", "It is 14C in Lyon.")])

    assert _plan_still_asks(prompt, "getWeather")


def test_a_step_carrying_a_placeholder_is_left_alone(tmp_path):
    plan = ["webSearch query='specs of <machine from step 1>'", "Reply to the user."]

    prompt, _, _ = _turn(tmp_path, plan, noeuds=[_Node("Kestrel M3", KESTREL)])

    assert _plan_still_asks(prompt, "webSearch")


# ── The silent-degradation holes, each pinned ─────────────────────────


def test_when_the_digest_keeps_nothing_the_step_survives(tmp_path):
    """On a small model the digest replaces the raw blocks entirely,
    including when it keeps nothing. Dropping a step on evidence that
    never reaches the prompt would leave the model with neither the fact
    nor the means to fetch it, and direct-exec has no second turn."""
    prompt, _, _ = _turn(tmp_path, WEB, noeuds=[_Node("Kestrel M3", KESTREL)],
                         petit=True, digest="")

    assert _plan_still_asks(prompt, "webSearch")


def test_a_graph_that_cannot_be_read_says_so(tmp_path, capsys):
    """A graph that has silently stopped answering must not look the
    same as an empty one."""
    prompt, _, _ = _turn(tmp_path, WEB, graphe_casse=True)

    assert _plan_still_asks(prompt, "webSearch")
    assert "could not be read" in capsys.readouterr().out


def test_accents_do_not_hide_a_node(tmp_path):
    """The plan is written in the user's language, the node by an
    extractor working in English."""
    plan = ["webSearch query='Café Rouvière horaires'", "Reply to the user."]
    node = _Node("Cafe Rouviere", "Cafe Rouviere is open 08:00 to 18:00, horaires fixes.")

    prompt, _, _ = _turn(tmp_path, plan, noeuds=[node])

    assert not _plan_still_asks(prompt, "webSearch")

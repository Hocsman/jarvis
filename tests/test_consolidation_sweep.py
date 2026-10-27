"""What the memory viewer's 🧹 sweep does to the graph, and what it says it does.

The sweep re-runs the merge prompt over the graph's populated nodes with
nothing new to add, so the latest rules reach data written before them.
It borrows the per-flush merge, and it is held to the same promises:

- a line the node already carried keeps its source. The rewrite hands
  every line back bare, and a sweep that did not put the suffixes back
  turned every line it touched into `inconnu`;
- it walks the World branch and nothing else. The User and Directives
  subtrees belong to the core (core.spec.md): a node still holding data
  there is waiting on its hand-over, and rewriting it would put a model's
  wording into the user's profile as `migré`, after sending that profile
  to whichever provider runs the chat model. The viewer counts the same
  walk up front, so its progress bar ends where the sweep does;
- the modal and the toolbar describe the rules the prompt applies, which
  do not drop a line for being common knowledge.

See src/jarvis/memory/graph.spec.md, "Consolidate All (🧹)".
"""

from __future__ import annotations

import json
import re
from contextlib import ExitStack
from unittest.mock import patch

import pytest

from conftest import MockConfig, ViewerClient, viewer_page_with_scripts


KNOWN = "Alien was directed by Ridley Scott."
TWIN = "Alien's director is Ridley Scott."
PROFILE = "Lives in Lyon and works nights."
RULE = "Always answer in French."

try:
    import flask as _flask  # noqa: F401
    _HAS_FLASK = True
except ImportError:
    _HAS_FLASK = False


def _a_merge_that_drops_the_twin(sent):
    """Rewrites a node the way the merge does: every line back bare, the
    near-duplicate gone. Records what each call sent out."""
    from jarvis.memory.provenance import fact_text

    def answer(*, user_content, **kwargs):
        sent.append(user_content)
        current = user_content.split(":\n", 1)[1].split("\n")
        kept = [fact_text(line) for line in current if fact_text(line) != TWIN]
        return json.dumps({"facts": list(dict.fromkeys(kept))})

    return answer


@pytest.fixture
def store(tmp_path):
    from jarvis.memory.graph import GraphMemoryStore

    s = GraphMemoryStore(str(tmp_path / "sweep.db"))
    yield s
    s.close()


def _sweep(store, sent):
    from jarvis.memory import graph_ops

    with patch.object(graph_ops, "call_llm_direct",
                      side_effect=_a_merge_that_drops_the_twin(sent)):
        return list(graph_ops.consolidate_world_nodes(
            store=store, cfg=None, chat_model="model"))


@pytest.mark.unit
def test_a_line_the_sweep_hands_back_keeps_its_source(store):
    from jarvis.memory.graph import BRANCH_WORLD
    from jarvis.memory.provenance import SOURCE_TOOL, SOURCE_WEB, fact_line

    web = fact_line("Possessor was directed by Brandon Cronenberg.",
                    SOURCE_WEB, "2026-08-16")
    tool = fact_line(KNOWN, SOURCE_TOOL, "2026-01-02")
    cinema = store.create_node(
        name="Cinema", description="Films and directors",
        data="\n".join([web, tool, TWIN]), parent_id=BRANCH_WORLD,
    )

    _sweep(store, [])

    assert store.get_node(cinema.id).data.split("\n") == [web, tool]


@pytest.mark.unit
def test_the_sweep_leaves_the_core_s_branches_alone(store):
    from jarvis.memory.graph import BRANCH_DIRECTIVES, BRANCH_USER, BRANCH_WORLD

    home = store.create_node(
        name="Home", description="Where he lives",
        data=f"{PROFILE}\n{PROFILE}", parent_id=BRANCH_USER,
    )
    store.update_node(BRANCH_DIRECTIVES, data=f"{RULE}\n{RULE}")
    store.update_node(BRANCH_WORLD, data=f"{KNOWN}\n{TWIN}")
    sent: list = []

    swept = _sweep(store, sent)

    assert store.get_node(home.id).data == f"{PROFILE}\n{PROFILE}"
    assert store.get_node(BRANCH_DIRECTIVES).data == f"{RULE}\n{RULE}"
    assert not any(PROFILE in s or RULE in s for s in sent), sent
    # The World root holds whatever no child fitted, so it is swept too.
    assert store.get_node(BRANCH_WORLD).data == KNOWN
    assert [name for name, _, _ in swept] == ["World"]


@pytest.mark.unit
def test_the_sweep_reaches_every_depth_of_the_world_branch(store):
    from jarvis.memory.graph import BRANCH_WORLD

    cinema = store.create_node(name="Cinema", description="Films",
                               data="", parent_id=BRANCH_WORLD)
    horror = store.create_node(name="Horror", description="Horror films",
                               data=f"{KNOWN}\n{TWIN}", parent_id=cinema.id)

    swept = _sweep(store, [])

    assert store.get_node(horror.id).data == KNOWN
    assert [name for name, _, _ in swept] == ["Horror"]


# ── The viewer: what it counts, and what it says ───────────────────────


@pytest.fixture
def viewer(store):
    from src.desktop_app import memory_viewer

    with ExitStack() as stack:
        stack.enter_context(patch.object(memory_viewer, "get_graph_store",
                                         return_value=store))
        stack.enter_context(patch("jarvis.config.load_settings",
                                  return_value=MockConfig(llm_chat_model="model")))
        stack.enter_context(patch("jarvis.memory.graph_ops.call_llm_direct",
                                  side_effect=_a_merge_that_drops_the_twin([])))
        yield ViewerClient(memory_viewer)


@pytest.mark.unit
@pytest.mark.skipif(not _HAS_FLASK, reason="Flask not available")
def test_the_progress_bar_counts_the_nodes_the_sweep_walks(store, viewer):
    from jarvis.memory.graph import BRANCH_DIRECTIVES, BRANCH_USER, BRANCH_WORLD

    store.create_node(name="Home", description="Where he lives",
                      data=PROFILE, parent_id=BRANCH_USER)
    store.update_node(BRANCH_DIRECTIVES, data=RULE)
    store.update_node(BRANCH_WORLD, data=KNOWN)
    store.create_node(name="Cinema", description="Films",
                      data=f"{KNOWN}\n{TWIN}", parent_id=BRANCH_WORLD)

    resp = viewer.post("/api/graph/consolidate-all", json={})
    events = [json.loads(line) for line in resp.get_data(as_text=True).splitlines()]

    start = next(e for e in events if e["type"] == "start")
    progress = [e for e in events if e["type"] == "progress"]
    complete = next(e for e in events if e["type"] == "complete")
    assert start["total"] == len(progress) == complete["nodes"] == 2


def _the_sweep_s_own_words(html: str) -> list[str]:
    """The toolbar button's title and the confirmation modal's body."""
    title = re.search(r'id="btn-consolidate-all"[^>]*title="([^"]*)"', html)
    modal = re.search(r"function showConsolidateAllModal\(\)(.*?)registerAction\(",
                      html, re.S)
    assert title and modal, "the sweep's button or modal moved"
    return [title.group(1), modal.group(1)]


@pytest.mark.unit
@pytest.mark.skipif(not _HAS_FLASK, reason="Flask not available")
def test_the_sweep_does_not_promise_to_prune_common_knowledge(viewer):
    """Rule 5 of the merge prompt keeps an answer the user asked for
    however obvious it is. The page must not say the opposite."""
    for text in _the_sweep_s_own_words(viewer_page_with_scripts(viewer)):
        assert not re.search(r"prun\w*[^.]*common knowledge", text, re.I), text


@pytest.mark.unit
@pytest.mark.skipif(not _HAS_FLASK, reason="Flask not available")
def test_the_sweep_says_it_covers_the_world_branch(viewer):
    for text in _the_sweep_s_own_words(viewer_page_with_scripts(viewer)):
        assert "World" in text, text

"""A node whose split was refused twice is not asked again until it has grown.

The split guards (graph.spec.md, Split quality safeguards) refuse an
answer that loses the node's facts. On the default small model every
split is refused that way, so a node past `SPLIT_THRESHOLD` keeps its
data, and without a memory of the refusals every flush that touches it
would pay for another full rewrite of the node that is thrown away. A
strong model refuses now and then too, at random (about 1 split in 6 for
gemini-2.5-flash under the current split prompt), and asking it again at
the next flush usually works. So refusals are remembered with the chat
model that gave them and the node's size: after
`_SPLIT_REFUSALS_BEFORE_BACKOFF` of them in a row from the same model, the
split is asked again only once the node has grown by `_SPLIT_RETRY_GROWTH`,
the chat model has changed, or the node has shrunk below the size it was
refused at (someone trimmed it). A split that got no answer (deadline,
timeout, error, an empty reply) was not refused and is asked again at the
next flush. A split that succeeds forgets the refusals.

The splits of a flush come after all of its merges, so a split, which
can always wait for the next flush, never spends the pass's deadline
before a merge that would otherwise fall back to plain append.

See src/jarvis/memory/graph.spec.md, Auto-Split, and docs/llm_contexts.md
#11c.
"""

import itertools
import json
import math
from unittest.mock import patch

import pytest

from src.jarvis.memory import graph_ops
from src.jarvis.memory.graph import BRANCH_WORLD, GraphMemoryStore, SPLIT_THRESHOLD
from src.jarvis.memory.graph_ops import (
    _SPLIT_REFUSALS_BEFORE_BACKOFF,
    _SPLIT_RETRY_GROWTH,
    update_graph_from_dialogue,
)


_EXTRACT_MARK = "You extract EXTERNAL FACTS"
_SPLIT_MARK = "knowledge organiser"
_PICKER_MARK = "Respond with ONLY the number"
_FACTS_MARK = "Facts to organise:\n"

# Worded apart, so a lossy split cannot pass for a faithful one.
_WORDS = ("amber basil cedar dahlia ember fennel garnet hazel indigo juniper "
          "kestrel lilac maple nutmeg olive pepper quartz russet saffron thistle").split()


def _lines(n, start=0):
    return [
        f"The {_WORDS[i % 20]} {_WORDS[(i * 7 + 3) % 20]} stall at market {start + i} "
        f"sells {_WORDS[(i * 3 + 1) % 20]} and {_WORDS[(i * 11 + 5) % 20]} on weekday mornings."
        for i in range(n)
    ]


def _oversized(store, node_id=BRANCH_WORLD, extra=0):
    """A node just past the threshold, plus ``extra`` lines."""
    lines = _lines(1)
    while len("\n".join(lines)) // 4 <= SPLIT_THRESHOLD:
        lines = _lines(len(lines) + 10)
    store.update_node(node_id, data="\n".join(lines + _lines(extra, start=500)))
    return store.get_node(node_id)


def _just_under(store, node_id):
    """A node exactly at the threshold, so the flush merges it and the
    line it appends takes it over."""
    lines = _lines(1)
    while len("\n".join(lines + _lines(1))) // 4 <= SPLIT_THRESHOLD:
        lines = _lines(len(lines) + 1)
    data = "\n".join(lines)
    prefix = "\nNotes "
    data += prefix + "x" * (SPLIT_THRESHOLD * 4 - len(data) - len(prefix))
    store.update_node(node_id, data=data)
    node = store.get_node(node_id)
    assert node.data_token_count == SPLIT_THRESHOLD
    return node


def _grow_to(store, tokens):
    """Append lines to the World root until it holds at least ``tokens``."""
    data = store.get_node(BRANCH_WORLD).data
    extra = 0
    grown = data
    while len(grown) // 4 < tokens:
        extra += 1
        grown = data + "\n" + "\n".join(_lines(extra, start=1000))
    store.update_node(BRANCH_WORLD, data=grown)


# One new fact per flush, never repeated across models: the dedupe would
# drop a repeat before it reached the node.
_NIGHTS = itertools.count(1)


class _Model:
    """Extracts one new fact per flush, declines every merge (so the facts
    are appended) and answers every split: with a lossy split, with
    nothing (``split="none"``), or faithfully (``split="faithful"``). Its
    picker files every fact where it stands (``NONE``). Records what it
    was asked."""

    def __init__(self, split="lossy"):
        self.split = split
        self.asked = []

    def __call__(self, *, system_prompt, user_content, **kw):
        if _EXTRACT_MARK in system_prompt:
            self.asked.append("extract")
            return json.dumps([f"The Rio Cinema screens a silent film on night {next(_NIGHTS)}."])
        if _PICKER_MARK in system_prompt:
            self.asked.append("pick")
            return "NONE"
        if system_prompt == graph_ops._MERGE_SYSTEM_PROMPT:
            self.asked.append("merge")
            return None
        if _SPLIT_MARK in system_prompt:
            self.asked.append("split")
            if self.split == "none":
                return None
            if self.split == "faithful":
                facts = user_content.split(_FACTS_MARK, 1)[1].split("\n")
                half = len(facts) // 2
                groups = (facts[:half], facts[half:])
            else:
                groups = (_lines(2), _lines(2, start=2))
            return json.dumps({
                "categories": [
                    {"name": "Markets", "description": "Markets", "facts": groups[0]},
                    {"name": "Stalls", "description": "Stalls", "facts": groups[1]},
                ],
                "summary": "Markets",
            })
        raise AssertionError(f"unexpected call: {system_prompt[:60]}")


def _flush(store, model, chat_model="model-a"):
    with patch.object(graph_ops, "call_llm_direct", model):
        return update_graph_from_dialogue(
            store=store, summary="He asked about the Rio Cinema.", cfg=None,
            chat_model=chat_model, tools_used=["webSearch"],
        )


def _backed_off(store, model, chat_model="model-a"):
    """Flush until ``chat_model`` has refused the split as often as it
    takes to back off; returns the node's size at the last refusal."""
    for _ in range(_SPLIT_REFUSALS_BEFORE_BACKOFF):
        _flush(store, model, chat_model)
    assert model.asked.count("split") == _SPLIT_REFUSALS_BEFORE_BACKOFF
    return store.get_node(BRANCH_WORLD).data_token_count


@pytest.fixture
def store(tmp_path):
    s = GraphMemoryStore(str(tmp_path / "graph.db"))
    yield s
    s.close()


@pytest.mark.unit
class TestSplitBackoff:

    def test_one_refusal_is_asked_again_at_the_next_flush(self, store):
        # A strong model refuses now and then at random; the next answer
        # usually passes.
        _oversized(store)
        model = _Model()

        _flush(store, model)
        _flush(store, model)

        assert model.asked.count("split") == 2

    def test_refusals_in_a_row_are_not_asked_again_while_the_node_has_barely_grown(self, store):
        _oversized(store)
        model = _Model()
        refused_at = _backed_off(store, model)

        _grow_to(store, math.ceil(refused_at * (1 + _SPLIT_RETRY_GROWTH)) - 50)
        _flush(store, model)

        assert model.asked.count("split") == _SPLIT_REFUSALS_BEFORE_BACKOFF
        assert store.get_children(BRANCH_WORLD) == []

    def test_refusals_in_a_row_are_asked_again_once_the_node_has_grown_enough(self, store):
        _oversized(store)
        model = _Model()
        refused_at = _backed_off(store, model)

        _grow_to(store, math.ceil(refused_at * (1 + _SPLIT_RETRY_GROWTH)))
        _flush(store, model)

        assert model.asked.count("split") == _SPLIT_REFUSALS_BEFORE_BACKOFF + 1

    def test_another_chat_model_is_asked_even_at_the_same_size(self, store):
        _oversized(store)
        model = _Model()
        _backed_off(store, model, chat_model="model-a")

        _flush(store, model, chat_model="model-b")

        assert model.asked.count("split") == _SPLIT_REFUSALS_BEFORE_BACKOFF + 1

    def test_a_node_trimmed_below_its_refused_size_is_asked_again(self, store):
        _oversized(store, extra=20)
        model = _Model()
        _backed_off(store, model)

        _oversized(store)  # someone trimmed it, still past the threshold
        _flush(store, model)

        assert model.asked.count("split") == _SPLIT_REFUSALS_BEFORE_BACKOFF + 1

    def test_a_successful_split_forgets_the_refusals(self, store):
        _oversized(store)
        lossy = _Model()
        refused_at = _backed_off(store, lossy, chat_model="model-a")

        _flush(store, _Model(split="faithful"), chat_model="model-b")
        assert store.get_children(BRANCH_WORLD)
        assert store.get_node(BRANCH_WORLD).data == ""

        # The root fills again past the threshold, but not by a quarter of
        # the size model-a refused it at.
        _oversized(store)
        assert store.get_node(BRANCH_WORLD).data_token_count < refused_at * (1 + _SPLIT_RETRY_GROWTH)
        _flush(store, lossy, chat_model="model-a")

        assert lossy.asked.count("split") == _SPLIT_REFUSALS_BEFORE_BACKOFF + 1

    def test_a_split_that_got_no_answer_is_asked_again_at_the_next_flush(self, store):
        _oversized(store)
        model = _Model(split="none")

        for _ in range(_SPLIT_REFUSALS_BEFORE_BACKOFF + 1):
            _flush(store, model)

        assert model.asked.count("split") == _SPLIT_REFUSALS_BEFORE_BACKOFF + 1

    def test_a_node_past_the_threshold_takes_its_new_facts_without_a_merge(self, store):
        _oversized(store)
        model = _Model()

        result = _flush(store, model)

        assert "merge" not in model.asked
        [(fact, _)] = result.stored
        assert fact.startswith("The Rio Cinema screens a silent film")
        assert fact in store.get_node(BRANCH_WORLD).data

    def test_splits_come_after_every_merge_of_the_flush(self, store):
        # Two children under World: the big one, which the flush's fact takes
        # past the threshold, and a small one. The flush brings one fact to
        # each, and both merges come back with nothing.
        big = store.create_node(name="Markets", description="Markets", data="", parent_id=BRANCH_WORLD)
        small = store.create_node(name="Cinemas", description="Cinemas",
                                  data="The Rio Cinema is in Dalston.", parent_id=BRANCH_WORLD)
        _just_under(store, big.id)
        asked = []

        def model(*, system_prompt, user_content, **kw):
            if _EXTRACT_MARK in system_prompt:
                asked.append("extract")
                return json.dumps(["A new stall at market 999 sells honey on Saturdays.",
                                   "The Rio Cinema screens a silent film on Sundays."])
            if _PICKER_MARK in system_prompt:
                asked.append("pick")
                fact, options = user_content.split("Categories:")
                wanted = "Markets" if "market" in fact else "Cinemas"
                return next(line.split(".")[0] for line in options.strip().split("\n")
                            if line.split(". ", 1)[1].startswith(wanted))
            if system_prompt == graph_ops._MERGE_SYSTEM_PROMPT:
                asked.append("merge")
                return None
            if _SPLIT_MARK in system_prompt:
                asked.append("split")
                return None
            raise AssertionError(system_prompt[:60])

        with patch.object(graph_ops, "call_llm_direct", model):
            update_graph_from_dialogue(
                store=store, summary="Markets and cinemas.", cfg=None,
                chat_model="model", tools_used=["webSearch"],
            )

        assert asked.count("merge") == 2 and asked.count("split") == 1
        assert asked.index("split") > max(i for i, a in enumerate(asked) if a == "merge")
        assert "honey" in store.get_node(big.id).data
        assert "silent film" in store.get_node(small.id).data

"""A node whose split was refused is not asked again until it has grown.

The split guards (graph.spec.md, Split quality safeguards) refuse an
answer that loses the node's facts. On the default small model every
split is refused that way, so a node past `SPLIT_THRESHOLD` keeps its
data, and without a memory of the refusal every flush that touches it
would pay for another full rewrite of the node that is thrown away. So a
refusal is remembered with the chat model that gave it and the node's
size, and the split is asked again only once the node has grown by
`_SPLIT_RETRY_GROWTH` or the chat model has changed. A split that got no
answer (deadline, timeout, error) was not refused and is asked again at
the next flush.

The splits of a flush come after all of its merges, so a split, which
can always wait for the next flush, never spends the pass's deadline
before a merge that would otherwise fall back to plain append.

See src/jarvis/memory/graph.spec.md, Auto-Split, and docs/llm_contexts.md
#11c.
"""

import json
import math
from unittest.mock import patch

import pytest

from src.jarvis.memory import graph_ops
from src.jarvis.memory.graph import BRANCH_WORLD, GraphMemoryStore, SPLIT_THRESHOLD
from src.jarvis.memory.graph_ops import (
    _SPLIT_RETRY_GROWTH,
    update_graph_from_dialogue,
)


_EXTRACT_MARK = "You extract EXTERNAL FACTS"
_SPLIT_MARK = "knowledge organiser"
_PICKER_MARK = "Respond with ONLY the number"

# Worded apart, so a lossy split cannot pass for a faithful one.
_WORDS = ("amber basil cedar dahlia ember fennel garnet hazel indigo juniper "
          "kestrel lilac maple nutmeg olive pepper quartz russet saffron thistle").split()


def _lines(n, start=0):
    return [
        f"The {_WORDS[i % 20]} {_WORDS[(i * 7 + 3) % 20]} stall at market {start + i} "
        f"sells {_WORDS[(i * 3 + 1) % 20]} and {_WORDS[(i * 11 + 5) % 20]} on weekday mornings."
        for i in range(n)
    ]


def _oversized(store, node_id=BRANCH_WORLD):
    lines = _lines(1)
    while len("\n".join(lines)) // 4 <= SPLIT_THRESHOLD:
        lines = _lines(len(lines) + 10)
    store.update_node(node_id, data="\n".join(lines))
    return store.get_node(node_id)


def _just_under(store, node_id):
    """A node a few characters short of the threshold, so the flush merges
    it and the line it appends takes it over."""
    lines = _lines(1)
    while len("\n".join(lines + _lines(1))) // 4 <= SPLIT_THRESHOLD:
        lines = _lines(len(lines) + 1)
    data = "\n".join(lines)
    data += "\n" + "Notes " + "x" * (SPLIT_THRESHOLD * 4 - len(data) - 8)
    store.update_node(node_id, data=data)
    node = store.get_node(node_id)
    assert node.data_token_count == SPLIT_THRESHOLD
    return node


class _Model:
    """Extracts one new fact per flush, declines every merge (so the facts
    are appended) and answers every split with a lossy split, or with
    nothing when ``split_answers`` is False. Records what it was asked."""

    def __init__(self, split_answers=True):
        self.split_answers = split_answers
        self.asked = []
        self.flush = 0

    def __call__(self, *, system_prompt, user_content, **kw):
        if _EXTRACT_MARK in system_prompt:
            self.flush += 1
            self.asked.append("extract")
            return json.dumps([f"The Rio Cinema screens a silent film on night {self.flush}."])
        if _PICKER_MARK in system_prompt:
            self.asked.append("pick")
            return "1"
        if system_prompt == graph_ops._MERGE_SYSTEM_PROMPT:
            self.asked.append("merge")
            return None
        if _SPLIT_MARK in system_prompt:
            self.asked.append("split")
            if not self.split_answers:
                return None
            return json.dumps({
                "categories": [
                    {"name": "Markets", "description": "Markets", "facts": _lines(2)},
                    {"name": "Stalls", "description": "Stalls", "facts": _lines(2, start=2)},
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


@pytest.fixture
def store(tmp_path):
    s = GraphMemoryStore(str(tmp_path / "graph.db"))
    yield s
    s.close()


@pytest.mark.unit
class TestSplitBackoff:

    def test_a_refused_split_is_not_asked_again_while_the_node_has_barely_grown(self, store):
        _oversized(store)
        model = _Model()

        _flush(store, model)
        _flush(store, model)
        _flush(store, model)

        assert model.asked.count("split") == 1
        assert store.get_children(BRANCH_WORLD) == []

    def test_a_refused_split_is_asked_again_once_the_node_has_grown_enough(self, store):
        node = _oversized(store)
        model = _Model()
        _flush(store, model)
        assert model.asked.count("split") == 1

        refused_at = store.get_node(BRANCH_WORLD).data_token_count
        grown = node.data.split("\n")
        extra = 0
        while len("\n".join(grown)) // 4 < math.ceil(refused_at * (1 + _SPLIT_RETRY_GROWTH)):
            extra += 5
            grown = node.data.split("\n") + _lines(extra, start=1000)
        store.update_node(BRANCH_WORLD, data="\n".join(grown))

        _flush(store, model)

        assert model.asked.count("split") == 2

    def test_another_chat_model_is_asked_even_at_the_same_size(self, store):
        _oversized(store)
        model = _Model()

        _flush(store, model, chat_model="model-a")
        _flush(store, model, chat_model="model-b")

        assert model.asked.count("split") == 2

    def test_a_split_that_got_no_answer_is_asked_again_at_the_next_flush(self, store):
        _oversized(store)
        model = _Model(split_answers=False)

        _flush(store, model)
        _flush(store, model)

        assert model.asked.count("split") == 2

    def test_a_node_past_the_threshold_takes_its_new_facts_without_a_merge(self, store):
        _oversized(store)
        model = _Model()

        result = _flush(store, model)

        assert "merge" not in model.asked
        assert [fact for fact, _ in result.stored] == ["The Rio Cinema screens a silent film on night 1."]
        assert "silent film on night 1" in store.get_node(BRANCH_WORLD).data

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

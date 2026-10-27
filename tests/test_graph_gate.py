"""The door to what she already looked up.

The graph holds world facts — things she searched for in earlier
conversations, kept so she does not pay for them twice. Step 1 moved
everything about the user out of it and into the core files, which are
now the sole authority on what she believes about him.

The door did not move with the furniture. It asked the memory extractor
for *implicit personal questions about the user* and crawled the graph
only when it got some. The extractor stopped producing them once there
was nothing personal left to answer, so the branch grew for weeks and was
never once read: measured on a real machine, a question whose answer sat
verbatim in the graph cost a web search and a page fetch.

What the crawl actually needs is content words. Keywords are content
words about the subject, they are produced on every enrichment turn, and
the graph is about subjects. So the door opens on those.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class _Node:
    def __init__(self, name, data):
        self.id, self.name, self.data = name, name, data
        self.data_token_count = len(data) // 4


def _turn(tmp_path, params, *, noeuds=None):
    """One engine turn, handing back what reached the system prompt."""
    from src.jarvis.memory.conversation import DialogueMemory
    from src.jarvis.reply.engine import run_reply_engine

    cfg = MagicMock()
    cfg.db_path = str(tmp_path / "t.db")
    cfg.llm_chat_model = "test-large"
    cfg.mcps = {}
    cfg.voice_debug = False
    cfg.memory_enrichment_source = "all"
    cfg.memory_enrichment_max_results = 3
    cfg.memory_digest_enabled = False
    cfg.tool_result_digest_enabled = False
    cfg.location_enabled = False
    cfg.llm_thinking_enabled = False
    cfg.agentic_max_turns = 8
    cfg.tool_selection_strategy = "all"
    cfg.tool_carryover_max_turns = 2
    cfg.tool_carryover_per_entry_chars = 1200
    cfg.tool_search_max_calls = 3
    cfg.llm_chat_timeout_sec = 45.0
    cfg.llm_tools_timeout_sec = 8.0

    store = MagicMock()
    store.search_nodes.return_value = noeuds if noeuds is not None else []
    store.get_ancestors.return_value = [_Node("World", "")]

    vu = {}

    def _capture(**kw):
        messages = kw.get("messages") or []
        vu.setdefault("system", messages[0].get("content", "") if messages else "")
        return {"message": {"content": "ok"}}

    with patch("src.jarvis.reply.engine.plan_query",
               return_value=["searchMemory topic='x'", "Reply."]), \
         patch("src.jarvis.reply.engine.extract_search_params_for_memory",
               return_value=params), \
         patch("src.jarvis.memory.graph.GraphMemoryStore", return_value=store), \
         patch("src.jarvis.reply.engine.chat_with_messages", side_effect=_capture), \
         patch("src.jarvis.reply.engine.extract_text_from_response", return_value="ok"):
        run_reply_engine(
            db=MagicMock(), cfg=cfg, tts=None,
            text="le DGX Spark, il a combien de mémoire ?",
            dialogue_memory=DialogueMemory(), origin="chat",
        )
    return store, vu.get("system", "")


SPARK = [_Node("World", "DGX Spark is compact with 128 GB unified memory.")]


# ── What the extractor actually produces ──────────────────────────────


def test_keywords_alone_open_the_door(tmp_path, monkeypatch):
    """The measured case. `questions` came back None on every probe, and
    the graph was skipped every time — while holding the answer."""
    store, prompt = _turn(
        tmp_path,
        {"keywords": ["dgx spark", "memory", "specifications"]},
        noeuds=SPARK,
    )

    assert store.search_nodes.called
    assert "128 GB unified memory" in prompt


def test_questions_still_open_it(tmp_path):
    """They are not required, but they are not ignored either: when the
    extractor does produce them, their words search too."""
    store, prompt = _turn(
        tmp_path,
        {"questions": ["what hardware does the user own?"]},
        noeuds=SPARK,
    )

    assert store.search_nodes.called
    assert "128 GB unified memory" in prompt


def test_both_are_searched_together(tmp_path):
    _turn(tmp_path,
          {"keywords": ["spark"], "questions": ["what memory does it have?"]},
          noeuds=SPARK)


# ── And when there is nothing to ask ──────────────────────────────────


def test_neither_means_no_crawl(tmp_path):
    """A utility turn — the time, a sum — has nothing to look up."""
    store, _ = _turn(tmp_path, {})

    assert not store.search_nodes.called


def test_a_single_content_word_is_still_too_thin(tmp_path):
    """One generic term against a LIKE search surfaces noise, and noise
    in the prompt is worse than a search she pays for."""
    store, _ = _turn(tmp_path, {"keywords": ["the"]})

    assert not store.search_nodes.called


def test_nothing_found_adds_nothing_to_the_prompt(tmp_path):
    store, prompt = _turn(tmp_path, {"keywords": ["dgx", "spark"]}, noeuds=[])

    assert store.search_nodes.called
    assert "looked up in earlier conversations" not in prompt


# ── What it is framed as ──────────────────────────────────────────────


def test_it_is_framed_as_the_world_and_not_as_him(tmp_path):
    """Step 1 made the core the sole authority on the user. A world fact
    arriving as something believed about him would undo that."""
    _, prompt = _turn(tmp_path, {"keywords": ["dgx", "spark"]}, noeuds=SPARK)

    assert "describes the world, not the user" in prompt


# ── What of a large node reaches the prompt ───────────────────────────
#
# A node is shown within a fixed budget of characters. A node that never
# splits (on the small tier the split guards refuse every split) grows far
# past it, and the search matches its whole text, so the budget is spent
# on the lines that name what was asked rather than on whichever lines
# happen to come first.


def _large_node(answer_at=30):
    lines = [
        f"The {w} gallery on Wharf Road number {i} opens at {8 + i % 4}am on weekdays."
        for i, w in enumerate(("amber", "basil", "cedar", "dahlia", "ember", "fennel",
                               "garnet", "hazel", "indigo", "juniper") * 4)
    ]
    lines.insert(answer_at, "The Kestrel M3 workstation ships with 96 GB of unified memory.")
    return _Node("World", "\n".join(lines))


def test_a_large_node_shows_the_line_that_matched(tmp_path):
    node = _large_node()
    assert len(node.data) > 1000 and node.data.index("Kestrel") > 1000

    _, prompt = _turn(tmp_path, {"keywords": ["kestrel m3", "memory"]}, noeuds=[node])

    assert "96 GB of unified memory" in prompt
    assert "amber gallery" not in prompt


def test_the_lines_that_match_more_of_the_search_come_first(tmp_path):
    lines = [f"The Kestrel M{i} drone folds to the size of a paperback and weighs {300 + i} g."
             for i in range(1, 30)]
    lines.insert(20, "The Kestrel M3 workstation ships with 96 GB of unified memory.")
    node = _Node("World", "\n".join(lines))

    _, prompt = _turn(tmp_path, {"keywords": ["kestrel", "m3", "memory"]}, noeuds=[node])

    assert "96 GB of unified memory" in prompt


def test_a_search_word_matches_whatever_the_accents(tmp_path):
    lines = _large_node().data.split("\n")
    lines.insert(25, "Café Rouvière in Lyon is known for its Yirgacheffe filter.")
    node = _Node("World", "\n".join(lines))

    _, prompt = _turn(tmp_path, {"keywords": ["cafe rouviere", "filter"]}, noeuds=[node])

    assert "Yirgacheffe" in prompt


def test_a_small_node_reaches_the_prompt_whole(tmp_path):
    node = _Node("World", "DGX Spark is compact with 128 GB unified memory.\n"
                          "It ships with DGX OS, a build of Ubuntu.")

    _, prompt = _turn(tmp_path, {"keywords": ["dgx spark", "memory"]}, noeuds=[node])

    assert "128 GB unified memory" in prompt and "a build of Ubuntu" in prompt


def test_a_large_node_no_line_of_which_matches_shows_its_first_lines(tmp_path):
    # The search matched the node by its name or description, not by a line.
    node = _large_node()

    _, prompt = _turn(tmp_path, {"keywords": ["world", "galleries"]}, noeuds=[node])

    assert "amber gallery" in prompt


def test_the_lines_shown_stay_within_the_budget(tmp_path):
    from src.jarvis.reply.engine import _GRAPH_PREVIEW_CHARS

    lines = [f"The Kestrel M{i} drone folds to the size of a paperback and weighs {300 + i} g."
             for i in range(1, 60)]
    node = _Node("World", "\n".join(lines))

    _, prompt = _turn(tmp_path, {"keywords": ["kestrel", "drone"]}, noeuds=[node])

    shown = [line for line in lines if line in prompt]
    assert shown and len("\n".join(shown)) <= _GRAPH_PREVIEW_CHARS


def test_a_matching_line_longer_than_the_budget_is_cut_not_passed_over(tmp_path):
    from src.jarvis.reply.engine import _GRAPH_PREVIEW_CHARS

    long_line = ("The Kestrel M3 workstation ships with 96 GB of unified memory, "
                 + "and a long list of ports, " * 20).strip()
    assert len(long_line) > _GRAPH_PREVIEW_CHARS
    lines = _large_node().data.split("\n")
    lines.insert(30, long_line)
    node = _Node("World", "\n".join(lines))

    _, prompt = _turn(tmp_path, {"keywords": ["kestrel m3", "memory"]}, noeuds=[node])

    assert "96 GB of unified memory" in prompt
    assert "amber gallery" not in prompt


def test_the_log_says_why_a_node_surfaced(tmp_path, capsys):
    """These lines are read by the user, not by a machine. `· World` with
    nothing after it says a node was pulled in and refuses to say what
    for — which is the same silence that hid the closed door."""
    _turn(tmp_path, {"keywords": ["dgx spark", "memory"]}, noeuds=SPARK)

    assert "· World → dgx spark" in capsys.readouterr().out

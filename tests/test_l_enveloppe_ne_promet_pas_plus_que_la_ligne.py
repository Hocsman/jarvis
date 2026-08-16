"""The prompt stops vouching for lines nobody vouched for.

Graph results were introduced to the model as "Things you looked up in
earlier conversations and kept". That sentence is a claim about
provenance, made over every line in the block, including the ones written
before provenance existed and the ones read off a page the assistant does
not vouch for.

The lines already end with what they rest on — `· web · 2026-08-16` —
because that is how they are stored. What was missing is that the model
had no idea those words meant anything, while the envelope above them
asserted a lookup for all of them at once.

So the envelope explains the vocabulary and claims only what the lines
themselves say. `web` was read somewhere and can be wrong; `inconnu`
predates the record and cannot be claimed as looked up at all.
"""

from __future__ import annotations

import pytest

from src.jarvis.memory.provenance import SOURCE_TOOL, SOURCE_WEB, fact_line
from src.jarvis.reply.engine import _graph_context_block


LIGNE_WEB = "[World] " + fact_line("Le DGX Spark a 128 Go", SOURCE_WEB,
                                   "2026-08-16")
LIGNE_OUTIL = "[World] " + fact_line("Le vol AF447 dure 11 h", SOURCE_TOOL,
                                     "2026-08-16")
LIGNE_ANCIENNE = "[World] Palworld est sorti en janvier 2024."


def test_the_block_no_longer_vouches_for_everything_at_once():
    bloc = _graph_context_block([LIGNE_ANCIENNE])

    assert "you looked up" not in bloc.lower()


def test_a_line_read_on_a_page_is_introduced_as_such():
    bloc = _graph_context_block([LIGNE_WEB])

    assert "web" in bloc
    assert "page" in bloc.lower()


def test_a_line_that_predates_the_record_is_not_claimed():
    bloc = _graph_context_block([LIGNE_ANCIENNE])

    assert "inconnu" in bloc


def test_the_vocabulary_is_only_explained_when_it_is_there():
    """A block of trusted-tool lines should not spend prompt on words
    that appear nowhere in it."""
    bloc = _graph_context_block([LIGNE_OUTIL])

    assert "outil" in bloc
    assert "web" not in bloc


def test_the_facts_themselves_still_reach_the_model():
    """The control. An envelope that lectured about provenance and lost
    the content would pass every test above."""
    bloc = _graph_context_block([LIGNE_WEB, LIGNE_ANCIENNE])

    assert "Le DGX Spark a 128 Go" in bloc
    assert "Palworld est sorti en janvier 2024." in bloc


def test_it_still_says_this_describes_the_world_and_loses_to_the_core():
    """The two guarantees that were already there and must survive: the
    block is about the world, and the user's own files win."""
    bloc = _graph_context_block([LIGNE_WEB])

    assert "world" in bloc.lower()
    assert "user" in bloc.lower()


def test_an_empty_block_is_empty():
    assert _graph_context_block([]) == ""

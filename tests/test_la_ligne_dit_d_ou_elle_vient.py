"""A world fact says what it rests on, on its own line.

The core files already do this: `profil.md` writes
`- il habite à Genève · dit`, and the source travels with the claim
rather than in a table beside it. A world fact does the same:

    Le DGX Spark a 128 Go de mémoire unifiée · web · 2026-08-16

It has to be the line and not a column, because a fact is a line inside
a node's `data` blob and one node accumulates facts from many windows on
many days. A column would describe the node, not the fact.

Two consequences the tests below pin down.

There is no migration. A line written before this has no suffix, and a
line with no suffix *is* `inconnu` — by construction, not by a rule that
marks it. Nothing is rewritten and an existing graph cannot be damaged.

And dedupe has to compare the fact, not the line. The daily summary is
cumulative and re-seeds the same facts on every flush; matching whole
lines would stop recognising a fact re-extracted on a later date and
append it again, once per flush, for ever.
"""

from __future__ import annotations

import pytest

from src.jarvis.memory.graph import GraphMemoryStore
from src.jarvis.memory.provenance import (
    SOURCE_TOOL,
    SOURCE_UNKNOWN,
    SOURCE_WEB,
    fact_line,
    fact_source,
    fact_text,
)


# ── Composing and reading a line ───────────────────────────────────────


def test_a_line_carries_the_fact_the_source_and_the_date():
    ligne = fact_line("Le DGX Spark a 128 Go de mémoire unifiée",
                      SOURCE_WEB, "2026-08-16")

    assert ligne == ("Le DGX Spark a 128 Go de mémoire unifiée "
                     "· web · 2026-08-16")


def test_the_fact_reads_back_out_of_its_line():
    ligne = fact_line("Le DGX Spark a 128 Go", SOURCE_WEB, "2026-08-16")

    assert fact_text(ligne) == "Le DGX Spark a 128 Go"
    assert fact_source(ligne) == SOURCE_WEB


def test_a_line_written_before_any_of_this_is_unknown():
    """The whole migration, and it is the absence of one."""
    ancienne = "The correct name of Nvidia NGX Spark is DGX Spark."

    assert fact_source(ancienne) == SOURCE_UNKNOWN
    assert fact_text(ancienne) == ancienne


def test_a_fact_containing_a_middle_dot_is_not_mistaken_for_a_suffix():
    """Only the vocabulary plus an ISO date at end of line counts as a
    suffix, so a fact that happens to use the character survives whole."""
    texte = "Le menu affiche entrée · plat · dessert"

    assert fact_source(texte) == SOURCE_UNKNOWN
    assert fact_text(texte) == texte


def test_a_suffix_like_word_that_is_not_the_vocabulary_is_left_alone():
    texte = "La réunion a lieu · lundi · 2026-08-16"

    assert fact_source(texte) == SOURCE_UNKNOWN
    assert fact_text(texte) == texte


def test_a_line_with_no_date_is_still_read():
    """Composition always supplies one, but a hand edit might not."""
    assert fact_source("Un fait quelconque · outil") == SOURCE_TOOL
    assert fact_text("Un fait quelconque · outil") == "Un fait quelconque"


# ── Dedupe compares the fact ───────────────────────────────────────────


def _store(tmp_path):
    return GraphMemoryStore(str(tmp_path / "g.db"))


def test_the_same_fact_on_a_later_day_is_recognised(tmp_path):
    """The cumulative diary re-emits it on every flush. Comparing whole
    lines would append a copy each time, for ever."""
    store = _store(tmp_path)
    n = store.create_node(name="Matériel", description="x",
                          data=fact_line("Le DGX Spark a 128 Go",
                                         SOURCE_WEB, "2026-08-16"),
                          parent_id="world")

    trouve = store.node_contains_fact(n.id, "Le DGX Spark a 128 Go")
    store.close()

    assert trouve is True


def test_it_is_recognised_whatever_the_source_that_day(tmp_path):
    store = _store(tmp_path)
    n = store.create_node(name="Matériel", description="x",
                          data=fact_line("Le DGX Spark a 128 Go",
                                         SOURCE_TOOL, "2026-01-02"),
                          parent_id="world")

    trouve = store.node_contains_fact(n.id, "Le DGX Spark a 128 Go")
    store.close()

    assert trouve is True


def test_an_unsuffixed_line_still_dedupes_against_a_new_fact(tmp_path):
    """Facts stored before this must not be duplicated by the first flush
    that re-extracts them."""
    store = _store(tmp_path)
    n = store.create_node(name="Matériel", description="x",
                          data="Le DGX Spark a 128 Go", parent_id="world")

    trouve = store.node_contains_fact(n.id, "Le DGX Spark a 128 Go")
    store.close()

    assert trouve is True


def test_a_genuinely_different_fact_is_not_swallowed(tmp_path):
    """The control. Stripping too much would make every fact match."""
    store = _store(tmp_path)
    n = store.create_node(name="Matériel", description="x",
                          data=fact_line("Le DGX Spark a 128 Go",
                                         SOURCE_WEB, "2026-08-16"),
                          parent_id="world")

    trouve = store.node_contains_fact(n.id, "Le Mac Studio a 512 Go")
    store.close()

    assert trouve is False


# ── The search still reaches the fact ──────────────────────────────────


def test_the_suffix_does_not_hide_a_fact_from_the_search(tmp_path):
    store = _store(tmp_path)
    store.create_node(name="Matériel", description="matériel",
                      data=fact_line("Le DGX Spark a 128 Go de mémoire",
                                     SOURCE_WEB, "2026-08-16"),
                      parent_id="world")

    trouves = store.search_nodes("DGX Spark mémoire", limit=5)
    store.close()

    assert [n.name for n in trouves] == ["Matériel"]

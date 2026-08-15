"""Folding is done once per write, not once per row per keyword per search.

`search_nodes` compares accent-and-case-folded text so a question typed
with accents finds a node an English extractor wrote without them. It did
that by wrapping every column in `fold()`, a Python function registered
on the connection: SQLite then calls it once per column, per keyword, per
row, over a full scan, on blobs up to six thousand characters.

Measured on the real store, a thousand nodes: 86 ms at two keywords,
692 ms at sixteen. The call sits on the reply path, and the reply path is
the VoiceListener's own loop — the one that drains a 64-frame audio queue
at 20 ms a frame. Past 1.28 s the microphone frames are dropped inside a
bare `except`. A search is not supposed to be able to eat his sentence.

The folded text is a property of the row, so it is stored with the row.
"""

from __future__ import annotations

import os
import tempfile
import time

import pytest


def _store(tmp_path):
    from src.jarvis.memory.graph import GraphMemoryStore

    return GraphMemoryStore(str(tmp_path / "g.db"))


def _peuple(store, combien: int, texte: str):
    for i in range(combien):
        store.create_node(name=f"Noeud {i}", description=f"sujet {i}",
                          data=texte, parent_id="world")


def test_a_search_over_a_thousand_nodes_stays_out_of_the_way(tmp_path):
    """The budget is the audio queue: 1.28 s before frames are dropped.
    A single search taking half of that is a search that costs him words."""
    store = _store(tmp_path)
    _peuple(store, 1000, "Le Kestrel M3 est une station compacte. " * 40)

    mots = " ".join(f"mot{j}" for j in range(16))
    t0 = time.perf_counter()
    store.search_nodes(mots, limit=5)
    dt = time.perf_counter() - t0
    store.close()

    assert dt < 0.15, f"{dt*1000:.0f} ms pour seize mots-clés sur mille nœuds"


def test_it_still_finds_a_node_written_without_the_accents(tmp_path):
    """The property the folding exists for, and the reason none of this
    can simply be deleted: the extractor writes English, he asks in
    French."""
    store = _store(tmp_path)
    store.create_node(name="Cafe Rouviere", description="torrefacteur",
                      data="Cafe Rouviere is a coffee roaster in Lyon.",
                      parent_id="world")

    trouves = store.search_nodes("café rouvière", limit=5)
    store.close()

    assert [n.name for n in trouves] == ["Cafe Rouviere"]


def test_the_other_direction_works_too(tmp_path):
    """Accents on the row, none in the question."""
    store = _store(tmp_path)
    store.create_node(name="Café Rouvière", description="torréfacteur",
                      data="Le café Rouvière torréfie à Lyon.", parent_id="world")

    trouves = store.search_nodes("cafe rouviere", limit=5)
    store.close()

    assert [n.name for n in trouves] == ["Café Rouvière"]


def test_a_node_whose_text_changed_is_found_by_its_new_words(tmp_path):
    """Stored folding has to follow an update, or the search answers about
    a version of the node that no longer exists."""
    store = _store(tmp_path)
    n = store.create_node(name="Bar", description="x", data="il sert du maté",
                          parent_id="world")
    store.update_node(n.id, data="il sert du café brûlé")

    assert [x.id for x in store.search_nodes("brule", limit=5)] == [n.id]
    assert store.search_nodes("mate", limit=5) == []
    store.close()


def test_a_database_written_before_the_stored_folding_still_searches(tmp_path):
    """An existing graph must not go blind on the upgrade: the folded
    columns are filled for rows that predate them."""
    import sqlite3

    from src.jarvis.memory.graph import GraphMemoryStore

    chemin = str(tmp_path / "g.db")
    store = _store(tmp_path)
    n = store.create_node(name="Café Rouvière", description="torréfacteur",
                          data="Le café Rouvière torréfie à Lyon.", parent_id="world")
    store.close()

    # Blank whatever the writer stored, as an older file would have.
    brut = sqlite3.connect(chemin)
    for colonne in ("name_fold", "description_fold", "data_fold"):
        try:
            brut.execute(f"UPDATE memory_nodes SET {colonne} = ''")
        except sqlite3.OperationalError:
            pass
    brut.commit()
    brut.close()

    rouvert = GraphMemoryStore(chemin)
    trouves = rouvert.search_nodes("cafe rouviere", limit=5)
    rouvert.close()

    assert [x.id for x in trouves] == [n.id]

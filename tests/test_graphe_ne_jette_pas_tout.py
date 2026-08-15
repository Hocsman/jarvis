"""One misplaced node does not cost him everything he looked up.

`migrate_legacy_shape` exists for a real reason, and the spec states it:
a node sitting directly under root, from before the taxonomy existed, is
unreachable by branch-pinned traversal for ever, and carrying it is dead
weight.

That justifies deleting *those* nodes. The code deleted the whole table.
So a single stray child — a cold-start write, an interrupted import, a
bug in some future placement code — took every correctly-filed fact with
it, and said so only in a debug channel nobody reads.

What is unreachable goes: root's own pre-taxonomy data, and any child of
root that is not a fixed branch, together with everything beneath it.
What is properly filed under `world` stays, because nothing about it is
unreachable and nobody asked for it to go.

And it is said out loud. A knowledge store that quietly empties itself is
indistinguishable from one that never learnt anything.
"""

from __future__ import annotations

import pytest


def _store(tmp_path):
    from src.jarvis.memory.graph import GraphMemoryStore
    return GraphMemoryStore(str(tmp_path / "t.db"))


def _noms(s):
    return {n.name for n in s.get_all_nodes()}


def test_a_stray_child_does_not_take_the_rest_with_it(tmp_path):
    s = _store(tmp_path)
    s.create_node(name="Cinéma", description="films",
                  data="Possessor is a 2020 film by Brandon Cronenberg.",
                  parent_id="world")
    s.create_node(name="Égaré", description="", data="", parent_id=s.get_root().id)

    assert s.migrate_legacy_shape() is True

    restants = _noms(s)
    assert "Cinéma" in restants, "correctly filed knowledge was destroyed"
    assert "Égaré" not in restants
    s.close()


def test_the_stray_subtree_goes_with_it(tmp_path):
    """Its children are unreachable for the same reason it is."""
    s = _store(tmp_path)
    egare = s.create_node(name="Égaré", description="", data="", parent_id=s.get_root().id)
    s.create_node(name="Sous-égaré", description="", data="x", parent_id=egare.id)
    s.create_node(name="Cinéma", description="", data="un fait", parent_id="world")

    s.migrate_legacy_shape()

    restants = _noms(s)
    assert "Sous-égaré" not in restants
    assert "Cinéma" in restants
    s.close()


def test_pre_taxonomy_data_on_root_is_cleared_and_the_branches_survive(tmp_path):
    s = _store(tmp_path)
    s.update_node(s.get_root().id, data="un fait écrit avant la taxonomie")
    s.create_node(name="Cinéma", description="", data="un fait", parent_id="world")

    assert s.migrate_legacy_shape() is True

    assert not (s.get_node(s.get_root().id).data or "").strip()
    assert "Cinéma" in _noms(s)
    s.close()


def test_a_conforming_graph_is_left_alone(tmp_path):
    s = _store(tmp_path)
    s.create_node(name="Cinéma", description="", data="un fait", parent_id="world")

    assert s.migrate_legacy_shape() is False
    assert "Cinéma" in _noms(s)
    s.close()


def test_the_fixed_branches_are_never_treated_as_strays(tmp_path):
    from src.jarvis.memory.graph import FIXED_BRANCH_IDS

    s = _store(tmp_path)
    s.create_node(name="Égaré", description="", data="", parent_id=s.get_root().id)

    s.migrate_legacy_shape()

    assert {n.id for n in s.get_all_nodes()} >= set(FIXED_BRANCH_IDS)
    s.close()


def test_it_says_what_it_removed(tmp_path, capsys):
    """A store that quietly empties itself looks exactly like one that
    never learnt anything."""
    s = _store(tmp_path)
    s.create_node(name="Égaré", description="", data="", parent_id=s.get_root().id)

    s.migrate_legacy_shape()

    sortie = capsys.readouterr().out
    assert sortie.strip(), "the removal was silent"
    s.close()

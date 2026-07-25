"""Behavioural tests for handing the graph's user knowledge over to the core.

The core is the authority for what the assistant believes about its user,
so whatever the graph accumulated under its user and directives branches
before has to end up in the files the user can read.
"""

import pytest

from src.jarvis.memory.core import SECTION_PROFILE, SECTION_RULES, MemoryCore
from src.jarvis.memory.graph import (
    BRANCH_DIRECTIVES,
    BRANCH_USER,
    BRANCH_WORLD,
    GraphMemoryStore,
)
from src.jarvis.memory.graph_ops import migrate_graph_branches_into_core


@pytest.fixture
def store(tmp_path):
    s = GraphMemoryStore(str(tmp_path / "test.db"))
    yield s
    s.close()


@pytest.fixture
def core(tmp_path):
    return MemoryCore(tmp_path / "yuba")


def test_user_knowledge_becomes_readable_profile_entries(store, core):
    store.create_node(
        name="Identity",
        description="Who the user is",
        data="Le nom de l'utilisateur est Hocine.",
        parent_id=BRANCH_USER,
    )

    migrate_graph_branches_into_core(store, core)

    assert [e.text for e in core.active(SECTION_PROFILE)] == [
        "Le nom de l'utilisateur est Hocine.",
    ]


def test_standing_instructions_become_readable_rules(store, core):
    store.create_node(
        name="Style",
        description="How to answer",
        data="Toujours répondre en français.",
        parent_id=BRANCH_DIRECTIVES,
    )

    migrate_graph_branches_into_core(store, core)

    assert [e.text for e in core.active(SECTION_RULES)] == [
        "Toujours répondre en français.",
    ]


def test_migrated_entries_say_where_they_came_from(store, core):
    store.create_node(
        name="Identity",
        description="Who the user is",
        data="Le nom de l'utilisateur est Hocine.",
        parent_id=BRANCH_USER,
    )

    migrate_graph_branches_into_core(store, core)

    assert core.active(SECTION_PROFILE)[0].source == "migré"


def test_world_knowledge_stays_in_the_graph(store, core):
    store.create_node(
        name="Films",
        description="Films looked up",
        data="Possessor est sorti en 2020.",
        parent_id=BRANCH_WORLD,
    )

    migrate_graph_branches_into_core(store, core)

    assert core.active(SECTION_PROFILE) == []
    assert core.active(SECTION_RULES) == []


def test_each_stored_fact_becomes_its_own_entry(store, core):
    store.create_node(
        name="Identity",
        description="Who the user is",
        data="Le nom de l'utilisateur est Hocine.\nIl vit à Lyon.",
        parent_id=BRANCH_USER,
    )

    migrate_graph_branches_into_core(store, core)

    assert [e.text for e in core.active(SECTION_PROFILE)] == [
        "Le nom de l'utilisateur est Hocine.",
        "Il vit à Lyon.",
    ]


def test_knowledge_nested_deeper_in_the_branch_is_not_missed(store, core):
    parent = store.create_node(
        name="Tastes",
        description="What the user likes",
        data="",
        parent_id=BRANCH_USER,
    )
    store.create_node(
        name="Films",
        description="Films the user likes",
        data="Il aime la science-fiction sombre.",
        parent_id=parent.id,
    )

    migrate_graph_branches_into_core(store, core)

    assert [e.text for e in core.active(SECTION_PROFILE)] == [
        "Il aime la science-fiction sombre.",
    ]


def test_the_node_survives_but_no_longer_holds_the_facts(store, core):
    """A hand-over, not a copy. Leaving the text in the graph would let
    query-driven recall put back what a correction just retired."""
    node = store.create_node(
        name="Identity",
        description="Who the user is",
        data="Le nom de l'utilisateur est Hocine.",
        parent_id=BRANCH_USER,
    )

    migrate_graph_branches_into_core(store, core)

    assert store.get_node(node.id) is not None
    assert store.get_node(node.id).data == ""


def test_a_branch_root_holding_facts_is_emptied_too(store, core):
    store.update_node(BRANCH_USER, data="Le nom de l'utilisateur est Hocine.")

    migrate_graph_branches_into_core(store, core)

    assert store.get_node(BRANCH_USER).data == ""


def test_world_knowledge_is_never_emptied(store, core):
    node = store.create_node(
        name="Films",
        description="Films looked up",
        data="Possessor est sorti en 2020.",
        parent_id=BRANCH_WORLD,
    )

    migrate_graph_branches_into_core(store, core)

    assert store.get_node(node.id).data == "Possessor est sorti en 2020."


def test_a_line_the_user_deletes_does_not_come_back(store, core):
    """The core files are the user's to prune. A migration that re-ran
    over the same nodes would undo that on every restart."""
    store.create_node(
        name="Health",
        description="Sensitive",
        data="Il prend un traitement pour la tension.",
        parent_id=BRANCH_USER,
    )
    migrate_graph_branches_into_core(store, core)

    path = core.path_for(SECTION_PROFILE)
    kept = [
        line for line in path.read_text(encoding="utf-8").splitlines()
        if "tension" not in line
    ]
    path.write_text("\n".join(kept) + "\n", encoding="utf-8")

    migrate_graph_branches_into_core(store, core)

    assert core.active(SECTION_PROFILE) == []


def test_a_failed_write_leaves_the_facts_in_the_graph(store, core, monkeypatch):
    """Clearing a node before its facts are safely in the core would
    lose them outright."""
    node = store.create_node(
        name="Identity",
        description="Who the user is",
        data="Le nom de l'utilisateur est Hocine.",
        parent_id=BRANCH_USER,
    )

    def _boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("pathlib.Path.write_text", _boom)

    migrate_graph_branches_into_core(store, core)

    assert store.get_node(node.id).data == "Le nom de l'utilisateur est Hocine."


def test_running_it_again_adds_nothing(store, core):
    store.create_node(
        name="Identity",
        description="Who the user is",
        data="Le nom de l'utilisateur est Hocine.",
        parent_id=BRANCH_USER,
    )

    first = migrate_graph_branches_into_core(store, core)
    second = migrate_graph_branches_into_core(store, core)

    assert first == 1
    assert second == 0
    assert len(core.active(SECTION_PROFILE)) == 1


def test_something_the_user_retired_is_not_brought_back(store, core):
    store.create_node(
        name="Identity",
        description="Who the user is",
        data="Il vit à Paris.",
        parent_id=BRANCH_USER,
    )
    migrate_graph_branches_into_core(store, core)
    core.retire(SECTION_PROFILE, "Il vit à Paris.", reason="corrigé")

    migrate_graph_branches_into_core(store, core)

    assert core.active(SECTION_PROFILE) == []


def test_an_empty_graph_migrates_nothing(store, core):
    assert migrate_graph_branches_into_core(store, core) == 0
    assert not core.path_for(SECTION_PROFILE).exists()

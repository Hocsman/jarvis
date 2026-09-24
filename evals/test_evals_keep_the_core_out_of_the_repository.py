"""The eval harness keeps the memory core out of the repository.

Evals run the real reply engine against ``MockConfig``, whose database is
``:memory:``. ``MemoryCore.for_config`` places the core beside the database,
and the parent of ``:memory:`` is ``.``, so an eval that remembers something
would write a profile into the directory the run started from. Every later
eval run from there would then read it back into its prompt, and measure a
different assistant from the one under test.

This is not an eval: it calls no model, and it runs wherever the evals do,
which is exactly where the guard has to hold.
"""

from __future__ import annotations

from pathlib import Path

import pytest

DEPOT = Path(__file__).resolve().parent.parent


def _hors_du_depot(chemin: Path) -> bool:
    cible = chemin.resolve()
    return cible != DEPOT and DEPOT not in cible.parents


@pytest.mark.unit
def test_what_an_eval_remembers_is_written_outside_the_repository(mock_config):
    from jarvis.memory.core import MemoryCore, SECTION_PROFILE

    core = MemoryCore.for_config(mock_config)
    core.remember(SECTION_PROFILE, "écrit par une évaluation")
    fichier = core.path_for(SECTION_PROFILE)

    assert core.directory.is_absolute(), core.directory
    assert fichier.exists(), fichier
    assert _hors_du_depot(fichier), fichier
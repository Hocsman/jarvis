"""The eval harness keeps the memory core out of the repository, and out
of the user's own data.

Evals run the real reply engine against ``MockConfig``, whose database is
``:memory:``. ``MemoryCore.for_config`` places the core beside the database,
and the parent of ``:memory:`` is ``.``, so an eval that remembers something
would write a profile into the directory the run started from. Every later
eval run from there would then read it back into its prompt, and measure a
different assistant from the one under test.

The evals that need the real chat model take ``real_model_config``: the
user's settings with the database moved into the sandbox. The engine reads
the core beside the database into its system prompt, and a real ``db_path``
would send the user's own ``profil.md`` and ``regles.md`` to whichever
provider the settings name. The model is the thing under test; the user's
life is not part of it.

These are not evals: they call no model, and they run wherever the evals
do, which is exactly where the guards have to hold.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

DEPOT = Path(__file__).resolve().parent.parent


def _hors_du_depot(chemin: Path) -> bool:
    cible = chemin.resolve()
    return cible != DEPOT and DEPOT not in cible.parents


def _foyer() -> Path:
    """Where the user's own database lives when nothing pins a path."""
    from jarvis.config import _default_db_path

    return Path(_default_db_path()).expanduser().resolve().parent


def _hors_du_foyer(chemin: Path) -> bool:
    cible = chemin.resolve()
    foyer = _foyer()
    return cible != foyer and foyer not in cible.parents


@pytest.fixture
def _settings_from_an_empty_config(monkeypatch, tmp_path):
    """Load the settings from an empty config file, so the guard reads
    nothing of the developer's own configuration. The database then takes
    its default, the user's data home, which is the case that matters."""
    vide = tmp_path / "config.json"
    vide.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("JARVIS_CONFIG_PATH", str(vide))


@pytest.mark.unit
def test_what_an_eval_remembers_is_written_outside_the_repository(mock_config):
    from jarvis.memory.core import MemoryCore, SECTION_PROFILE

    core = MemoryCore.for_config(mock_config)
    core.remember(SECTION_PROFILE, "écrit par une évaluation")
    fichier = core.path_for(SECTION_PROFILE)

    assert core.directory.is_absolute(), core.directory
    assert fichier.exists(), fichier
    assert _hors_du_depot(fichier), fichier


@pytest.mark.unit
def test_the_real_model_config_reads_no_core_of_the_users_own(
    _settings_from_an_empty_config, real_model_config
):
    from jarvis.memory.core import MemoryCore, SECTION_PROFILE, SECTION_RULES

    core = MemoryCore.for_config(real_model_config)

    assert _hors_du_foyer(Path(real_model_config.db_path)), real_model_config.db_path
    assert _hors_du_foyer(core.directory), core.directory
    assert _hors_du_depot(core.directory), core.directory
    for section in (SECTION_PROFILE, SECTION_RULES):
        assert not core.path_for(section).exists(), (
            "the model under test must start from an empty memory"
        )


@pytest.mark.unit
def test_the_real_model_config_keeps_the_real_model(
    _settings_from_an_empty_config, real_model_config
):
    """Only the database moves: the provider and the models are the point."""
    from jarvis.config import load_settings

    reference = load_settings()
    assert real_model_config.llm_provider == reference.llm_provider
    assert real_model_config.llm_chat_model == reference.llm_chat_model
    assert real_model_config.ollama_chat_model == reference.ollama_chat_model


@pytest.mark.unit
def test_a_config_pinned_to_the_users_data_home_is_kept_out_of_it():
    """Belt and braces: an eval that builds its own config around the real
    default path still gets a sandboxed core."""
    from jarvis.config import _default_db_path
    from jarvis.memory.core import MemoryCore

    core = MemoryCore.for_config(SimpleNamespace(db_path=_default_db_path()))

    assert core.directory.is_absolute(), core.directory
    assert _hors_du_foyer(core.directory), core.directory
    assert _hors_du_depot(core.directory), core.directory


@pytest.mark.unit
def test_an_absolute_path_elsewhere_keeps_the_real_resolution(tmp_path):
    from jarvis.memory.core import MemoryCore, CORE_DIRNAME

    core = MemoryCore.for_config(SimpleNamespace(db_path=str(tmp_path / "jarvis.db")))

    assert core.directory == tmp_path / CORE_DIRNAME


# ── The engine's own read, watched ───────────────────────────────────
#
# The guards above check where the core resolves. This one watches the
# file system while the engine's read of the core runs, so the proof does
# not rest on the resolution alone: nothing under the user's data home is
# opened, listed or written, whatever path the read takes.

_FILE_EVENTS = {
    "open", "os.listdir", "os.scandir", "os.mkdir", "os.rename", "os.remove",
    "os.rmdir", "shutil.rmtree", "shutil.copyfile", "shutil.move",
}
_armed: list = []
_hook_installed = False


def _audit(event, args):
    if not _armed or event not in _FILE_EVENTS:
        return
    wire = _armed[0]
    for arg in args:
        if isinstance(arg, (str, bytes, os.PathLike)):
            # No file-system call in here: the hook would hear itself.
            path = os.path.normcase(os.path.abspath(os.fsdecode(arg)))
            if path == wire.foyer or path.startswith(wire.foyer + os.sep):
                wire.touched.append((event, path))


@pytest.fixture
def data_home_tripwire():
    global _hook_installed
    if not _hook_installed:
        # An audit hook cannot be removed, so it is installed once and stays
        # inert while nothing is armed.
        sys.addaudithook(_audit)
        _hook_installed = True
    wire = SimpleNamespace(foyer=os.path.normcase(str(_foyer())), touched=[])
    _armed.append(wire)
    try:
        yield wire
    finally:
        _armed.clear()


@pytest.mark.unit
def test_the_engines_read_of_the_core_touches_nothing_of_the_users_own(
    _settings_from_an_empty_config, real_model_config, data_home_tripwire
):
    """``run_reply_engine`` reads the core with ``MemoryCore.for_config`` and
    ``build_core_profile`` before every reply; run with the real-model
    settings, that read lands in the sandbox and finds it empty."""
    from jarvis.memory.core import MemoryCore, build_core_profile

    profile = build_core_profile(MemoryCore.for_config(real_model_config))

    assert profile == {"user": "", "directives": ""}, profile
    assert data_home_tripwire.touched == [], data_home_tripwire.touched

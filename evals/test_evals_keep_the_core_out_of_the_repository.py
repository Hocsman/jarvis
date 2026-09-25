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


# Values no default carries, so a fixture that dropped them would show.
_MARKER_CONFIG = {
    "ollama_chat_model": "guard-chat-model",
    "ollama_base_url": "http://127.0.0.1:1",
    "intent_judge_model": "guard-judge-model",
}


@pytest.fixture
def _settings_from_a_marker_config(monkeypatch, tmp_path):
    """Load the settings from a config file whose values match no default."""
    import json

    marque = tmp_path / "config.json"
    marque.write_text(json.dumps(_MARKER_CONFIG), encoding="utf-8")
    monkeypatch.setenv("JARVIS_CONFIG_PATH", str(marque))


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
def test_the_real_model_config_keeps_the_user_out_of_the_prompt(
    _settings_from_an_empty_config, real_model_config
):
    """The engine's live line names no place when location is off, and
    reaches for no cache to say so."""
    from jarvis.reply.engine import _live_time_location_string

    assert real_model_config.location_enabled is False
    assert real_model_config.location_auto_detect is False
    assert real_model_config.location_ip_address is None
    assert "Location: Disabled" in _live_time_location_string(real_model_config)


@pytest.mark.unit
def test_the_real_model_config_keeps_the_real_model(
    _settings_from_a_marker_config, real_model_config
):
    """Only the database and the location move: the models are the point.

    Measured against values no default carries, so a fixture that rebuilt
    the settings from scratch would fail here."""
    assert real_model_config.ollama_chat_model == _MARKER_CONFIG["ollama_chat_model"]
    assert real_model_config.llm_chat_model == _MARKER_CONFIG["ollama_chat_model"]
    assert real_model_config.ollama_base_url == _MARKER_CONFIG["ollama_base_url"]
    assert real_model_config.intent_judge_model == _MARKER_CONFIG["intent_judge_model"]


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
_NETWORK_EVENTS = {"socket.getaddrinfo", "socket.gethostbyname", "socket.connect"}
_LOOPBACK = {None, "", b"", "localhost", b"localhost", "::1", b"::1"}
_armed: list = []
_hook_installed = False


def _is_loopback(host) -> bool:
    if host in _LOOPBACK:
        return True
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    return isinstance(host, str) and host.startswith("127.")


def _audit(event, args):
    if not _armed:
        return
    wire = _armed[0]
    if event in _NETWORK_EVENTS:
        # The network is closed to the guard, so nothing leaves and the turn
        # is the same on every machine. Only a host beyond this machine is
        # a leak: the engine's small chain looking for a local model is not.
        host = args[1] if event == "socket.connect" else args[0]
        if isinstance(host, tuple):
            host = host[0] if host else None
        if not _is_loopback(host):
            wire.network.append((event, host))
        raise ConnectionRefusedError("the network is closed to this guard")
    if event not in _FILE_EVENTS:
        return
    for arg in args:
        if isinstance(arg, (str, bytes, os.PathLike)):
            # No file-system call in here: the hook would hear itself.
            path = os.path.normcase(os.path.abspath(os.fsdecode(arg)))
            if any(path == f or path.startswith(f + os.sep) for f in wire.foyers):
                wire.touched.append((event, path))


def _wire(*foyers):
    """A tripwire around the given directories, by every spelling given."""
    return SimpleNamespace(
        foyers=[os.path.normcase(os.path.abspath(str(f))) for f in foyers],
        touched=[],
        network=[],
    )


def _arm(wire):
    global _hook_installed
    if not _hook_installed:
        # An audit hook cannot be removed, so it is installed once and stays
        # inert while nothing is armed.
        sys.addaudithook(_audit)
        _hook_installed = True
    _armed.append(wire)


@pytest.fixture
def data_home_tripwire():
    """Watches the user's data home, spelled both as the default path says
    it and as the file system resolves it: a home moved behind a link is
    opened through the link."""
    from jarvis.config import _default_db_path

    wire = _wire(_foyer(), Path(_default_db_path()).expanduser().parent)
    _arm(wire)
    try:
        yield wire
    finally:
        _armed.clear()


@pytest.mark.unit
def test_the_tripwire_hears_a_file_opened_where_it_watches(tmp_path):
    """The positive control: an empty ``touched`` means something only if
    the hook reports what it is pointed at."""
    wire = _wire(tmp_path)
    _arm(wire)
    try:
        (tmp_path / "profil.md").write_text("témoin", encoding="utf-8")
        (tmp_path / "profil.md").read_text(encoding="utf-8")
    finally:
        _armed.clear()

    assert [event for event, _ in wire.touched].count("open") >= 2, wire.touched
    assert all(p.endswith("profil.md") for _, p in wire.touched), wire.touched


@pytest.mark.unit
def test_the_core_the_engine_reads_is_empty_and_elsewhere(
    _settings_from_an_empty_config, real_model_config, data_home_tripwire
):
    """The two calls the engine makes for the core, ``MemoryCore.for_config``
    then ``build_core_profile``, land in the sandbox and find it empty."""
    from jarvis.memory.core import MemoryCore, build_core_profile

    profile = build_core_profile(MemoryCore.for_config(real_model_config))

    assert profile == {"user": "", "directives": ""}, profile
    assert data_home_tripwire.touched == [], data_home_tripwire.touched


@pytest.mark.unit
def test_a_real_model_turn_touches_nothing_of_the_users_own(
    _settings_from_an_empty_config, real_model_config, eval_dialogue_memory,
    data_home_tripwire,
):
    """The turn the honesty eval performs, with the model faked and the
    network closed, opens nothing under the user's data home: whatever the
    engine reads on the way to the prompt, it is not the user's."""
    import dataclasses
    from unittest.mock import patch

    from helpers import ToolCallCapture, create_mock_llm_response
    from jarvis.memory.db import Database
    from jarvis.reply.engine import run_reply_engine
    from jarvis.tools.types import ToolExecutionResult

    cfg = dataclasses.replace(real_model_config, tool_selection_strategy="all")
    capture = ToolCallCapture()

    def failing_tool(db, cfg, tool_name, tool_args, **kwargs):
        capture.record(tool_name, tool_args or {})
        return ToolExecutionResult(success=False, reply_text="I couldn't auto-detect your location.")

    def faked_model(cfg, messages, **kwargs):
        return create_mock_llm_response("Je n'ai pas pu obtenir la météo.")

    db = Database(":memory:", sqlite_vss_path=None)
    try:
        with patch("jarvis.reply.engine.run_tool_with_retries", side_effect=failing_tool), \
             patch("jarvis.reply.engine.chat_with_messages", side_effect=faked_model), \
             patch("jarvis.reply.engine.extract_search_params_for_memory", return_value={"keywords": []}):
            reply = run_reply_engine(
                db=db, cfg=cfg, tts=None,
                text="quelle est la météo ?", dialogue_memory=eval_dialogue_memory,
            )
    finally:
        db.close()

    assert reply, "the faked model's reply must come back"
    assert data_home_tripwire.touched == [], data_home_tripwire.touched
    assert data_home_tripwire.network == [], (
        f"the guard reached beyond this machine: {data_home_tripwire.network}"
    )

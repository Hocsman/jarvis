"""The daemon's shutdown runs to its end (`src/desktop_app/desktop_app.spec.md`,
"Shutdown and the Console").

The desktop app reads the daemon's console through a pipe, and the pipe goes
when the app does. Whatever the daemon prints after that has nowhere to go,
and the shutdown prints before it saves the diary. A console that fails must
cost the console only: the diary is saved, everything is stopped and closed,
and the log file, when one is named, still gets every line, once.
"""

from __future__ import annotations

import errno
import io
import os
import sqlite3
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from conftest import MockConfig

SRC = Path(__file__).resolve().parent.parent / "src"


# ── A console the desktop app has stopped reading ─────────────────────


class _DeadPipe:
    """The daemon's console once the desktop app has gone: every write fails."""

    encoding = "utf-8"

    def write(self, text):
        raise OSError(errno.EINVAL, "Invalid argument")

    def flush(self):
        raise OSError(errno.EINVAL, "Invalid argument")


_CHILD = """
    import sys, time
    sys.path.insert(0, {src!r})
    from jarvis.main import _install_sink
    _install_sink()
    from jarvis.debug import debug_log
    print("ready", flush=True)
    sys.stdin.readline()        # the desktop app quits: stdin ends, the pipe goes
    time.sleep(0.3)
    try:
        pass
    finally:
        print("🔄 shutting down", flush=True)
        debug_log("saving the diary", "jarvis")
        open({marker!r}, "w").write("saved")
        print("👋 stopped", flush=True)
"""


@pytest.mark.unit
def test_a_console_that_closes_under_the_daemon_costs_neither_the_shutdown_nor_a_line(tmp_path):
    """The daemon's own entry setup, a real pipe, and a reader that goes away
    before the shutdown starts: what follows the first print still runs, and
    the log file gets each line once."""
    log = tmp_path / "daemon.log"
    marker = tmp_path / "diary-saved"
    env = dict(
        os.environ,
        JARVIS_DEBUG_FILE=str(log),
        JARVIS_VOICE_DEBUG="1",
        JARVIS_CONFIG_PATH=str(tmp_path / "config.json"),
        PYTHONIOENCODING="utf-8",
        PYTHONUTF8="1",
    )
    child = subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent(_CHILD.format(src=str(SRC), marker=str(marker)))],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
    )
    try:
        for line in child.stdout:
            if line.strip() == b"ready":
                break
        child.stdout.close()
        child.stdin.close()
        code = child.wait(timeout=60)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()

    assert marker.exists(), "the work after the first print never ran"
    text = log.read_text(encoding="utf-8")
    assert code == 0, text
    assert text.count("🔄 shutting down") == 1, text
    assert text.count("saving the diary") == 1, text
    assert text.count("👋 stopped") == 1, text
    assert "Exception ignored" not in text, text


# ── The shutdown's steps ───────────────────────────────────────────────


class _Part:
    """A component of the running daemon, which records being stopped and can
    fail to stop."""

    def __init__(self, fails: bool = False):
        self.stopped = False
        self._fails = fails

    def stop(self, *args, **kwargs):
        self.stopped = True
        if self._fails:
            raise RuntimeError("it would not stop")

    def join(self, timeout=None):
        pass


@pytest.fixture(autouse=True)
def _the_process_outlives_the_daemon(monkeypatch):
    """What a shutdown closes for the whole process is the test's own, or is
    re-armed afterwards, as the next daemon run in the same process does."""
    from jarvis import daemon
    from jarvis.projets import delegation
    from jarvis.tools.external.mcp_runtime import reset_shutdown_latch

    monkeypatch.setattr(delegation, "DELEGATIONS", delegation.Delegations())
    for name in ("_global_reminders", "_global_routines", "_global_routine_runner",
                 "_warm_profile_core_listener"):
        monkeypatch.setattr(daemon, name, None)
    yield
    reset_shutdown_latch()


@pytest.fixture
def a_conversation(monkeypatch):
    """The daemon's dialogue memory, holding one exchange not yet in the diary,
    with the summariser answering at once."""
    from jarvis import daemon
    from jarvis.memory.conversation import DialogueMemory

    memory = DialogueMemory(inactivity_timeout=300)
    memory.add_message("user", "When should I water the tomatoes?")
    memory.add_message("assistant", "Early in the morning.")
    monkeypatch.setattr(daemon, "_global_dialogue_memory", memory)
    monkeypatch.setattr(daemon, "_global_stop_requested", True)
    monkeypatch.setattr(daemon, "_global_skip_shutdown_diary_update", False)
    with patch("jarvis.memory.conversation.generate_conversation_summary",
               return_value=("We talked about watering the tomatoes.", "garden")):
        yield memory


@pytest.fixture
def db(tmp_path):
    from jarvis.memory.db import Database

    database = Database(str(tmp_path / "jarvis.db"), sqlite_vss_path=None)
    yield database
    database.close()


def _diary(db_path):
    with sqlite3.connect(db_path) as conn:
        return [row[0] for row in conn.execute("SELECT summary FROM conversation_summaries")]


def _assert_closed(db):
    with pytest.raises(sqlite3.ProgrammingError):
        db.get_all_conversation_summaries()


@pytest.mark.unit
def test_a_step_that_fails_does_not_keep_the_next_ones_from_running(
        a_conversation, db, tmp_path, monkeypatch, capsys):
    from jarvis import daemon

    reminders, routines = _Part(), _Part(fails=True)
    monkeypatch.setattr(daemon, "_global_reminders", reminders)
    monkeypatch.setattr(daemon, "_global_routines", routines)
    dictation, voice, tts, face = _Part(fails=True), _Part(fails=True), _Part(), _Part()

    daemon._shut_down(db, MockConfig(), dictation=dictation, voice_thread=voice, tts=tts, face=face)

    assert dictation.stopped and reminders.stopped and routines.stopped and voice.stopped
    assert _diary(str(tmp_path / "jarvis.db")) == ["We talked about watering the tomatoes."]
    assert tts.stopped and face.stopped
    _assert_closed(db)
    out = capsys.readouterr().out
    assert out.count("it would not stop") == 3, out
    assert out.rstrip().endswith("👋 Daemon stopped"), out


@pytest.mark.unit
def test_a_console_that_fails_during_the_shutdown_does_not_cost_the_diary(
        a_conversation, db, tmp_path, monkeypatch):
    """The entry's stream setup over a console nobody reads any more, as the
    desktop app leaves it when it quits first."""
    from jarvis import daemon
    from jarvis.main import _install_sink

    monkeypatch.delenv("JARVIS_DEBUG_FILE", raising=False)
    monkeypatch.setattr(sys, "stdout", _DeadPipe())
    monkeypatch.setattr(sys, "stderr", _DeadPipe())
    _install_sink()
    tts, face = _Part(), _Part()

    daemon._shut_down(db, MockConfig(), dictation=None, voice_thread=None, tts=tts, face=face)

    assert _diary(str(tmp_path / "jarvis.db")) == ["We talked about watering the tomatoes."]
    assert tts.stopped and face.stopped
    _assert_closed(db)


@pytest.mark.unit
def test_a_fast_stop_skips_the_diary_and_nothing_else(a_conversation, db, tmp_path, monkeypatch):
    from jarvis import daemon

    monkeypatch.setattr(daemon, "_global_skip_shutdown_diary_update", True)
    tts, face = _Part(), _Part()

    daemon._shut_down(db, MockConfig(), dictation=None, voice_thread=None, tts=tts, face=face)

    assert _diary(str(tmp_path / "jarvis.db")) == []
    assert tts.stopped and face.stopped
    _assert_closed(db)


# ── The log file gets each line once ───────────────────────────────────


@pytest.fixture
def debug_on(monkeypatch, tmp_path):
    """Debug lines on, and a log file named for them."""
    from types import SimpleNamespace

    import jarvis.debug as debug

    log = tmp_path / "daemon.log"
    monkeypatch.setattr(debug, "_cached_voice_debug", None)
    monkeypatch.setattr(debug, "_last_check_time", 0.0)
    monkeypatch.setattr(debug, "load_settings", lambda: SimpleNamespace(voice_debug=True))
    monkeypatch.setattr(debug, "_SINK_PATH", str(log))
    return debug, log


@pytest.mark.unit
def test_a_debug_line_reaches_the_file_once_when_the_console_is_mirrored_there(debug_on):
    """The daemon run by the desktop app: its console is mirrored into the
    file, and the line reaches both."""
    from jarvis.main import _Tee

    debug, log = debug_on
    console = io.StringIO()
    with open(log, "a", encoding="utf-8") as handle:
        with patch.object(sys, "stderr", _Tee(console, handle)):
            debug.debug_log("the diary is saved", "jarvis")

    assert log.read_text(encoding="utf-8").count("the diary is saved") == 1
    assert console.getvalue().count("the diary is saved") == 1


@pytest.mark.unit
def test_a_debug_line_reaches_the_file_when_the_console_is_not_mirrored(debug_on, capsys):
    """The desktop app's own process, and the daemon run in its thread, print
    to a console that is not mirrored: the file still gets their lines."""
    debug, log = debug_on

    debug.debug_log("the tray is ready", "desktop")

    assert log.read_text(encoding="utf-8").count("the tray is ready") == 1
    assert "the tray is ready" in capsys.readouterr().err


@pytest.mark.unit
def test_the_mirror_keeps_writing_the_file_once_the_console_is_gone(tmp_path):
    from jarvis.main import _Tee

    log = tmp_path / "daemon.log"
    with open(log, "a", encoding="utf-8") as handle:
        tee = _Tee(_DeadPipe(), handle)
        for word in ("one", "two", "three"):
            print(word, file=tee, flush=True)

    lines = log.read_text(encoding="utf-8").splitlines()
    assert [l for l in lines if l in ("one", "two", "three")] == ["one", "two", "three"]


@pytest.mark.unit
def test_a_mirror_without_a_file_still_survives_the_console(tmp_path):
    from jarvis.main import _Tee

    tee = _Tee(_DeadPipe(), None)

    print("nobody hears this", file=tee, flush=True)
    tee.flush()

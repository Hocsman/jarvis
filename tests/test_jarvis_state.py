"""What the assistant is doing right now, as one value the whole machine can read.

The voice pipeline publishes it (idle, listening, thinking, speaking, ...) and
the desktop app reads it to drive the orb, the dashboard and the face. It is
the core's own state, so it lives in ``jarvis`` and the desktop app imports it
from there. It crosses processes through a small file, because the daemon runs
as a subprocess in a development checkout and as a thread in the bundled app.

Importing it must pull no Qt: a headless run publishes state like any other.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

from conftest import ROOT
from jarvis.state import JarvisState, JarvisStateManager, get_jarvis_state


@pytest.fixture
def state_file(tmp_path):
    return str(tmp_path / "jarvis_state")


class TestTheSharedState:
    @pytest.mark.unit
    def test_a_fresh_manager_starts_asleep_and_says_so_in_the_file(self, state_file):
        manager = JarvisStateManager(state_file)

        assert manager.state == JarvisState.ASLEEP
        with open(state_file, encoding="utf-8") as handle:
            assert handle.read().strip() == JarvisState.ASLEEP.value

    @pytest.mark.unit
    def test_a_stale_file_from_a_previous_session_does_not_survive_a_launch(self, state_file):
        with open(state_file, "w", encoding="utf-8") as handle:
            handle.write(JarvisState.SPEAKING.value)

        assert JarvisStateManager(state_file).state == JarvisState.ASLEEP

    @pytest.mark.unit
    @pytest.mark.parametrize("state", list(JarvisState))
    def test_what_one_side_publishes_the_other_side_reads(self, state_file, state):
        """Two managers over one file stand for the daemon process and the
        desktop process: the reader never saw the writer's memory, only the
        file."""
        desktop = JarvisStateManager(state_file)
        daemon = JarvisStateManager(state_file)
        daemon.set_state(state)

        assert desktop.state == state

    @pytest.mark.unit
    def test_a_change_made_by_another_process_is_picked_up(self, state_file):
        manager = JarvisStateManager(state_file)
        manager.set_state(JarvisState.SPEAKING)

        with open(state_file, "w", encoding="utf-8") as handle:
            handle.write(JarvisState.THINKING.value)

        assert manager.state == JarvisState.THINKING

    @pytest.mark.unit
    def test_unreadable_content_falls_back_to_the_last_state_set_here(self, state_file):
        manager = JarvisStateManager(state_file)
        manager.set_state(JarvisState.LISTENING)

        with open(state_file, "w", encoding="utf-8") as handle:
            handle.write("a value no release ever wrote")

        assert manager.state == JarvisState.LISTENING

    @pytest.mark.unit
    def test_a_missing_file_falls_back_to_the_last_state_set_here(self, state_file):
        manager = JarvisStateManager(state_file)
        manager.set_state(JarvisState.THINKING)
        os.remove(state_file)

        assert manager.state == JarvisState.THINKING

    @pytest.mark.unit
    def test_a_location_that_cannot_be_written_never_breaks_the_pipeline(self, tmp_path):
        """The voice path publishes state on the way to speaking. A disk that
        refuses the write costs the desktop its view of the state, never the
        user their answer."""
        unwritable = str(tmp_path / "no" / "such" / "directory" / "jarvis_state")

        manager = JarvisStateManager(unwritable)
        manager.set_state(JarvisState.SPEAKING)

        assert manager.state == JarvisState.SPEAKING

    @pytest.mark.unit
    def test_a_machine_with_no_usable_temp_directory_still_publishes_in_process(self, monkeypatch):
        import jarvis.state as state_module

        def no_temp_directory():
            raise FileNotFoundError("No usable temporary directory found")

        monkeypatch.setattr(state_module, "_state_file_path", no_temp_directory)

        manager = JarvisStateManager()
        manager.set_state(JarvisState.THINKING)

        assert manager.state == JarvisState.THINKING

    @pytest.mark.unit
    def test_the_process_holds_a_single_manager(self):
        assert get_jarvis_state() is get_jarvis_state()

    @pytest.mark.unit
    def test_the_pipeline_and_the_process_manager_share_state(self):
        get_jarvis_state().set_state(JarvisState.LISTENING)

        assert get_jarvis_state().state == JarvisState.LISTENING


class TestConcurrentPublishersLeaveTheLatestState:
    """The pipeline publishes from timer threads, the reply thread and the
    speech threads. Whichever publish started last is what a viewer must end
    up reading, however the threads interleave on the way to the disk."""

    @pytest.mark.unit
    def test_a_publish_held_up_mid_write_cannot_overwrite_a_later_one(self, state_file):
        import threading
        import time

        manager = JarvisStateManager(state_file)
        real_write = manager._write_state
        reached_the_disk = threading.Event()
        let_it_through = threading.Event()

        def held_up_write(state):
            if state == JarvisState.LISTENING:
                reached_the_disk.set()
                let_it_through.wait(timeout=5)
            real_write(state)

        manager._write_state = held_up_write

        first = threading.Thread(target=manager.set_state, args=(JarvisState.LISTENING,))
        first.start()
        assert reached_the_disk.wait(timeout=5), "the first publish never reached the disk"

        second = threading.Thread(target=manager.set_state, args=(JarvisState.IDLE,))
        second.start()
        time.sleep(0.1)  # the second publish is under way while the first is held up
        let_it_through.set()
        first.join(timeout=5)
        second.join(timeout=5)

        assert manager.state == JarvisState.IDLE


class TestPublishingNeedsNoDesktop:
    """The headless guarantee, checked in a clean interpreter: nothing the
    voice pipeline imports to publish its state may drag in Qt or the desktop
    package."""

    @pytest.mark.unit
    def test_importing_the_publishers_loads_neither_qt_nor_the_desktop_app(self, tmp_path):
        script = textwrap.dedent(
            """
            import sys

            import jarvis.state
            import jarvis.listening.state_manager
            import jarvis.listening.listener
            import jarvis.output.tts
            import jarvis.reply.engine
            import jarvis.daemon

            leaked = sorted(
                name for name in sys.modules
                if name.split(".")[0] in ("PyQt6", "PyQt5", "PySide6", "desktop_app")
            )
            print("LEAKED:" + ",".join(leaked))
            """
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT / "src")
        # Nothing imported here should reach the user's own data; point every
        # place it could look at the sandbox all the same.
        env["HOME"] = env["USERPROFILE"] = str(tmp_path)
        env["JARVIS_CONFIG_PATH"] = str(tmp_path / "config.json")

        done = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, env=env, cwd=str(tmp_path), timeout=240,
        )

        assert done.returncode == 0, done.stderr[-2000:]
        assert "LEAKED:" in done.stdout, done.stdout[-2000:]
        leaked = done.stdout.split("LEAKED:", 1)[1].strip()
        assert leaked == "", f"importing the core loaded {leaked}"

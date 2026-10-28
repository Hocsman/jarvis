"""Labelling a dictation hotkey loads no audio or keyboard library.

The settings window and the setup wizard fill their hotkey dropdown when they
are built. The labels are plain string work, so building them must not load
sounddevice, numpy or pynput: those belong to the dictation engine, which runs
in the daemon.
"""

import os
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
HEAVY = ("sounddevice", "numpy", "pynput")


def _loaded_after(code: str) -> set:
    """The heavy modules a fresh interpreter holds after running ``code``."""
    probe = (
        f"import sys\nsys.path.insert(0, {str(SRC)!r})\n{code}\n"
        f"print('loaded:' + ','.join(m for m in {HEAVY!r} if m in sys.modules))"
    )
    env = dict(os.environ, PYTHONIOENCODING="utf-8", QT_QPA_PLATFORM="offscreen")
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=120, env=env, encoding="utf-8"
    )
    assert result.returncode == 0, result.stderr
    (line,) = [line for line in result.stdout.splitlines() if line.startswith("loaded:")]
    return {m for m in line.removeprefix("loaded:").split(",") if m}


def test_labelling_the_presets_loads_nothing_heavy():
    code = (
        "from jarvis.dictation.hotkeys import dictation_hotkey_choices, format_hotkey_display\n"
        "assert dictation_hotkey_choices()\n"
        "assert format_hotkey_display('ctrl+shift+d')"
    )
    assert _loaded_after(code) == set()


def test_opening_the_settings_module_loads_nothing_heavy():
    assert _loaded_after("import desktop_app.settings_window") == set()


def test_building_the_wizards_dictation_page_loads_nothing_heavy():
    code = (
        "from PyQt6.QtWidgets import QApplication\n"
        "app = QApplication([])\n"
        "from desktop_app.setup_wizard import DictationPage\n"
        "DictationPage()"
    )
    assert _loaded_after(code) == set()

"""``qapp`` tears down a window that only a reference cycle holds.

A test that ends with a window only a cycle holds (a chat window and its orb,
say) leaves it shown, with a paint pending. If the cyclic garbage collector
runs while Qt paints it, it frees the window while its child is still in its
paint event, and the process dies with an access violation that no test can
fail on. Whether that happens depends only on when the collector fires, which
is any allocation, so a change elsewhere in the suite is enough to start it.

The case in ``qapp_teardown_case.py`` makes the collection happen during the
paint every time, and runs in a session of its own.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
CASE = TESTS_DIR / "qapp_teardown_case.py"


@pytest.mark.unit
def test_a_window_only_a_cycle_holds_does_not_end_the_session():
    pytest.importorskip("PyQt6.QtWidgets")
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "PYTHONDONTWRITEBYTECODE": "1"}

    session = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", str(CASE)],
        cwd=TESTS_DIR.parent, env=env, capture_output=True, text=True, timeout=180,
    )

    assert session.returncode == 0, (
        f"the session ended with {session.returncode}\n"
        f"--- stdout ---\n{session.stdout[-3000:]}\n--- stderr ---\n{session.stderr[-3000:]}"
    )
    assert "1 passed" in session.stdout, session.stdout[-3000:]

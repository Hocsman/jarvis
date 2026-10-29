"""The dashboard's web view shows its own page and no other.

The view is handed the bridge over ``QWebChannel``, and Qt gives the
channel's transport to whatever document the view shows. A foreign page
there could call ``submitQuery``, which drives the daemon and its tools,
or ``openSettings``. So the view navigates to the dashboard's own ``index.html`` alone (Qt's
internal ``about:`` states aside), in every frame, and a link or file
dropped on it opens nothing.

QtWebEngine is built in a process of its own (``dashboard_page_child.py``):
Chromium aborts the process when the QApplication has no program name,
which the suite's shared one may lack, and an abort here would end the
whole run. The child drives the real window, page, channel and bridge, and
reports what happened. A drop cannot be simulated there (a synthetic drop
navigates no view, even one that allows it), so the drop is read from the
page's own setting.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("PyQt6.QtWebEngineWidgets")

pytestmark = pytest.mark.unit

TESTS_DIR = Path(__file__).resolve().parent
CHILD = TESTS_DIR / "dashboard_page_child.py"

FOREIGN = ("another local file", "a remote page", "the memory viewer's loopback", "a data document")
OWN = ("own index", "own index with a fragment", "about:blank")


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    home = tmp_path_factory.mktemp("dashboard-page")
    env = {
        **os.environ,
        "QT_QPA_PLATFORM": "offscreen",
        # Offscreen GPU contexts are lost at random; the page needs no GPU.
        "QTWEBENGINE_CHROMIUM_FLAGS": "--disable-gpu",
        # The Linux runners refuse the user namespaces Chromium's sandbox
        # needs; the child loads only files it wrote and the dashboard's own.
        "QTWEBENGINE_DISABLE_SANDBOX": "1",
        # The assistant's state file lives in the temp directory, and a new
        # state manager writes to it: the child's must not reach the live one.
        "TMP": str(home), "TEMP": str(home), "TMPDIR": str(home),
        "XDG_CONFIG_HOME": str(home / "config"),
        "JARVIS_CONFIG_PATH": str(home / "config" / "config.json"),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    child = subprocess.run(
        [sys.executable, str(CHILD)],
        cwd=home, env=env, capture_output=True, text=True, encoding="utf-8", timeout=180,
    )
    lines = [line for line in child.stdout.splitlines() if line.startswith("REPORT:")]
    assert child.returncode == 0 and lines, (
        f"the child ended with {child.returncode}\n"
        f"--- stdout ---\n{child.stdout[-3000:]}\n--- stderr ---\n{child.stderr[-3000:]}"
    )
    return json.loads(lines[-1].removeprefix("REPORT:"))


class TestItsOwnPage:

    def test_the_dashboard_shows_its_own_page(self, report):
        assert report["index_loaded"]
        assert report["shown_url_is_index"]

    def test_its_own_page_reaches_the_bridge(self, report):
        assert report["asked_from_own_page"] == ["from the dashboard"]

    @pytest.mark.parametrize("frame", ["main_frame", "sub_frame"])
    @pytest.mark.parametrize("address", OWN)
    def test_its_own_page_and_qts_internal_states_are_accepted(self, report, frame, address):
        assert report[frame][address] is True


class TestNoOtherPage:

    def test_a_link_to_another_page_opens_nothing(self, report):
        assert report["after_link"]["url_is_index"]
        assert report["after_link"]["asked"] == ["from the dashboard"]
        assert report["after_link"]["settings_opened"] is False

    @pytest.mark.parametrize("frame", ["main_frame", "sub_frame"])
    @pytest.mark.parametrize("address", FOREIGN)
    def test_any_other_address_is_refused(self, report, frame, address):
        assert report[frame][address] is False

    def test_a_drop_never_navigates(self, report):
        assert report["navigates_on_drop"] is False

    def test_a_refusal_is_logged_with_the_scheme_and_host_alone(self, report):
        (line,) = report["refusal_logged"]
        assert "https://example.invalid" in line
        for part in ("private", "notes.html", "token", "s3cret", "top"):
            assert part not in line

"""The memory viewer's web view navigates to the loopback names alone.

The page the viewer shows holds the launch token, and the embedded view
must never become a way for the machine to reach a remote host. So the
page the window installs accepts a navigation, in any frame, only to a
host named ``localhost`` / ``127.0.0.1`` / ``::1`` (it checks the host
alone, so any scheme and port on those names pass), or to any ``about:``
or ``data:`` address: Qt's internal states, and the self-contained
document ``setHtml`` loads, which carries the error page. It refuses
every other one and logs the refusal with the address's scheme and host
alone. A link or a script that opens a new window opens nothing: the view
creates no window (``desktop_app.spec.md``, "Memory Viewer").

QtWebEngine is built in a process of its own (``memory_viewer_page_child.py``):
Chromium aborts the process when the QApplication has no program name,
which the suite's shared one may lack, and an abort here would end the
whole run. The child builds the real ``MemoryViewerWindow`` with its server
replaced, so the real memory server never starts, and reports what the
window showed and what its page decided. Nothing it loads leaves the
machine: the error page, pages the child serves itself on 127.0.0.1, and
the attempts to reach 127.0.0.2, a loopback address the guard does not
name, where a second server counts what reaches it. Every remote address
is a decision asked of the page, which loads nothing, and Chromium is
given no proxy, so it looks for none on the network.
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
CHILD = TESTS_DIR / "memory_viewer_page_child.py"

LOOPBACK_NAMES = ("localhost", "127.0.0.1", "::1")
LOCAL = (
    "the viewer's server", "a path on 127.0.0.1", "the IPv6 loopback",
    "about:blank", "a data: document",
)
FOREIGN = (
    "a remote page", "a font CDN", "a LAN address", "a file path with no host",
    "a host named after localhost", "a remote host behind loopback credentials",
    "a loopback address the guard does not name",
)


def _tail(text):
    return (text or "")[-3000:]


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    home = tmp_path_factory.mktemp("memory-viewer-page")
    env = {
        **os.environ,
        "QT_QPA_PLATFORM": "offscreen",
        # Offscreen GPU contexts are lost at random; the page needs no GPU.
        "QTWEBENGINE_CHROMIUM_FLAGS": "--disable-gpu",
        # The Linux runners refuse the user namespaces Chromium's sandbox
        # needs; the child loads only pages it wrote itself.
        "QTWEBENGINE_DISABLE_SANDBOX": "1",
        # The child's work directory, which it never removes, and anything
        # else the app puts in the temp directory stay in this test's home.
        "TMP": str(home), "TEMP": str(home), "TMPDIR": str(home),
        "XDG_CONFIG_HOME": str(home / "config"),
        "JARVIS_CONFIG_PATH": str(home / "config" / "config.json"),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    try:
        child = subprocess.run(
            [sys.executable, str(CHILD)],
            cwd=home, env=env, capture_output=True, text=True, encoding="utf-8", timeout=180,
        )
    except subprocess.TimeoutExpired as stalled:
        pytest.fail(
            f"the child ran past {stalled.timeout} s\n"
            f"--- stdout ---\n{_tail(stalled.stdout)}\n--- stderr ---\n{_tail(stalled.stderr)}"
        )
    lines = [line for line in child.stdout.splitlines() if line.startswith("REPORT:")]
    assert child.returncode == 0 and lines, (
        f"the child ended with {child.returncode}\n"
        f"--- stdout ---\n{_tail(child.stdout)}\n--- stderr ---\n{_tail(child.stderr)}"
    )
    return json.loads(lines[-1].removeprefix("REPORT:"))


def _one_refusal_of_the_host_off_the_list(report, refused):
    (line,) = refused
    assert report["foreign_host"] in line


class TestLocalPages:

    def test_the_error_page_shows_when_the_server_fails(self, report):
        shown = report["error_page"]
        assert shown["loaded"]
        assert shown["scheme"] == "data"
        assert shown["heading"]
        assert shown["refused"] == []

    def test_the_servers_page_shows_when_it_runs(self, report):
        shown = report["served"]
        assert shown["loaded"]
        assert shown["host"] in LOOPBACK_NAMES
        assert shown["on_the_servers_port"]
        assert shown["shows_its_page"]
        assert shown["refused"] == []

    def test_a_link_to_another_page_of_the_server_opens_it(self, report):
        followed = report["server_link"]
        assert followed["clicked"]
        assert followed["shown"]
        assert followed["refused"] == []

    @pytest.mark.parametrize("frame", ["main_frame", "sub_frame"])
    @pytest.mark.parametrize("address", LOCAL)
    def test_a_local_address_is_accepted(self, report, frame, address):
        assert report[frame][address] is True

    def test_an_accepted_address_logs_nothing(self, report):
        assert report["accepted_logged"] == []


class TestNoOtherPage:

    @pytest.mark.parametrize("frame", ["main_frame", "sub_frame"])
    def test_every_address_asked_is_checked(self, report, frame):
        assert set(report[frame]) == set(LOCAL) | set(FOREIGN)

    @pytest.mark.parametrize("frame", ["main_frame", "sub_frame"])
    @pytest.mark.parametrize("address", FOREIGN)
    def test_any_other_address_is_refused(self, report, frame, address):
        assert report[frame][address] is False

    def test_a_remote_address_is_refused_however_it_is_reached(self, report):
        decisions = report["remote_by_navigation_type"]
        assert decisions
        assert set(decisions.values()) == {False}

    def test_a_link_to_a_host_off_the_list_opens_nothing(self, report):
        followed = report["foreign_link"]
        assert followed["clicked"]
        _one_refusal_of_the_host_off_the_list(report, followed["refused"])
        assert followed["stayed"]

    def test_a_redirect_to_a_host_off_the_list_is_refused(self, report):
        redirected = report["foreign_redirect"]
        _one_refusal_of_the_host_off_the_list(report, redirected["refused"])
        assert redirected["host"] != report["foreign_host"]

    def test_a_frame_of_a_host_off_the_list_stays_empty(self, report):
        framed = report["foreign_frame"]
        assert framed["loaded"]
        _one_refusal_of_the_host_off_the_list(report, framed["refused"])
        assert framed["frame_shows"] == "about:blank"

    def test_a_new_window_opens_nothing(self, report):
        opened = report["new_window"]
        assert opened["asked"]
        assert opened["stayed"]

    def test_nothing_reaches_a_host_off_the_list(self, report):
        if report["foreign_requests"] is None:
            pytest.skip(f"{report['foreign_host']} cannot be listened on here")
        assert report["foreign_requests"] == []

    def test_a_refusal_is_logged_with_the_scheme_and_host_alone(self, report):
        (line,) = report["refusal_logged"]
        assert "https://example.invalid" in line
        for part in ("someone", "hunter2", "8443", "private", "notes.html", "token", "s3cret", "frag"):
            assert part not in line

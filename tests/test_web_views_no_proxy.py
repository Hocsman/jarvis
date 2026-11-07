"""The app's web views look for no proxy.

QtWebEngine takes its proxy from Qt, and Qt follows the system by
default. Windows' default "Automatically detect settings" then makes
Chromium send a DHCP query and a DNS lookup for ``wpad`` on the first
HTTP load, a loopback one included, so opening the memory viewer would
announce itself on the network. The views show local pages only, so the
tray gives Qt no proxy and no system lookup at start-up, before it builds
any window (``desktop_app.spec.md``, "Memory Viewer", "No proxy lookup").

QtWebEngine is built in a process of its own (``web_views_proxy_child.py``):
Chromium aborts the process when the QApplication has no program name,
which the suite's shared one may lack, and an abort here would end the
whole run. The child takes the tray's start-up step, loads the real
dashboard and the real memory viewer, and Chromium writes its net log:
the proxy configuration it was given, every proxy lookup it ran, every
name it asked a resolver for. The memory viewer's HTTP load is what puts
these in the log: the dashboard's ``file:`` and ``qrc:`` loads never reach
Chromium's proxy resolution, so its half shows only that the page still
renders with no proxy. The tray's own order is checked here, in the
suite's process, without a web view.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("PyQt6.QtWebEngineWidgets")

from PyQt6.QtNetwork import QNetworkProxy, QNetworkProxyFactory  # noqa: E402

import desktop_app.app as app_mod  # noqa: E402

pytestmark = pytest.mark.unit

TESTS_DIR = Path(__file__).resolve().parent
CHILD = TESTS_DIR / "web_views_proxy_child.py"
NET_LOG = "netlog.json"


def _tail(text):
    # A timeout's output is bytes on POSIX, whatever ``text`` asked for.
    if isinstance(text, bytes):
        text = text.decode("utf-8", "replace")
    return (text or "")[-3000:]


def _read_net_log(path):
    """The events of a Chromium net log, as ``(type name, params)``.

    The constants sit on the first line and each event on a line of its
    own, so a log an abrupt exit cut short still reads line by line.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    constants = json.loads(lines[0].rstrip().rstrip(",") + "}")["constants"]
    names = {number: name for name, number in constants["logEventTypes"].items()}
    events = []
    for line in lines[1:]:
        try:
            event = json.loads(line.rstrip().rstrip(","))
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and "type" in event and "phase" in event:
            events.append((names.get(event["type"], str(event["type"])), event.get("params") or {}))
    return events


@pytest.fixture(scope="module")
def child_run(tmp_path_factory):
    home = tmp_path_factory.mktemp("web-views-proxy")
    env = {
        **os.environ,
        "QT_QPA_PLATFORM": "offscreen",
        # Offscreen GPU contexts are lost at random; the pages need no GPU.
        # The net log lands in the child's working directory: a relative
        # path, because these flags are split on spaces.
        "QTWEBENGINE_CHROMIUM_FLAGS": f"--disable-gpu --log-net-log={NET_LOG}",
        # The Linux runners refuse the user namespaces Chromium's sandbox
        # needs; the child loads only the dashboard and its own loopback page.
        "QTWEBENGINE_DISABLE_SANDBOX": "1",
        # The assistant's state file lives in the temp directory, and the
        # dashboard's bridge reads it: the child's must not reach the live one.
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
    net_log = home / NET_LOG
    assert net_log.exists(), f"Chromium wrote no net log\n--- stderr ---\n{_tail(child.stderr)}"
    return json.loads(lines[-1].removeprefix("REPORT:")), _read_net_log(net_log)


@pytest.fixture(scope="module")
def report(child_run):
    return child_run[0]


@pytest.fixture(scope="module")
def net_log(child_run):
    return child_run[1]


class TestThePagesLoad:

    def test_the_dashboard_shows_its_page(self, report):
        shown = report["dashboard"]
        assert shown["loaded"]
        assert shown["scheme"] == "file"

    def test_the_memory_viewer_shows_its_servers_page(self, report):
        shown = report["memory_viewer"]
        assert shown["loaded"]
        assert shown["host"] == "localhost"
        assert shown["shows_its_page"]

    def test_the_net_log_holds_the_memory_viewers_load(self, net_log):
        urls = [params.get("url", "") for name, params in net_log if name == "URL_REQUEST_START_JOB"]
        assert any(url.startswith("http://localhost:") for url in urls), urls


class TestNoProxyLookup:

    def test_chromium_is_given_no_proxy(self, net_log):
        configs = [params.get("new_config") for name, params in net_log if name == "PROXY_CONFIG_CHANGED"]
        assert configs
        assert all(config == {} for config in configs), configs

    def test_no_proxy_discovery_runs(self, net_log):
        discovery = sorted({
            name for name, _ in net_log
            if name.startswith(("PAC_FILE_", "WPAD_")) or name.endswith("_WAITING_FOR_INIT_PAC")
        })
        assert discovery == []

    def test_no_resolver_is_asked_for_a_proxy(self, net_log):
        lookups = sorted({
            str(params.get("host")) for name, params in net_log
            if name.startswith("HOST_RESOLVER_") and str(params.get("host", "")).lower().startswith("wpad")
        })
        assert lookups == []

    @pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Chromium reads auto_proxy on Linux alone")
    def test_the_systems_proxy_script_is_never_fetched(self, report):
        assert report["proxy_script_requests"] == []


class _FirstWindow(Exception):
    """Stops the tray at the first window it builds."""


def _follow_the_system():
    """Qt's proxy as a new process has it. Setting an application proxy
    turns the system lookup off, so the lookup goes back on last."""
    QNetworkProxy.setApplicationProxy(QNetworkProxy())
    QNetworkProxyFactory.setUseSystemConfiguration(True)


def _qt_proxy():
    """Whether Qt follows the system, and its application proxy's type
    when it does not. Reading the proxy while Qt follows the system asks
    the system for one (WinHTTP's discovery on Windows), so it is not read."""
    if QNetworkProxyFactory.usesSystemConfiguration():
        return True, None
    return False, QNetworkProxy.applicationProxy().type()


@pytest.fixture
def qt_proxy_restored(qapp):
    """Qt's proxy and the app's quit rule, as the suite had them."""
    uses_system = QNetworkProxyFactory.usesSystemConfiguration()
    proxy = None if uses_system else QNetworkProxy.applicationProxy()
    quits = qapp.quitOnLastWindowClosed()
    yield
    if uses_system:
        _follow_the_system()
    else:
        QNetworkProxy.setApplicationProxy(proxy)
    qapp.setQuitOnLastWindowClosed(quits)


class TestTheTray:

    def test_qt_has_no_proxy_before_the_trays_first_window(self, qt_proxy_restored, monkeypatch):
        _follow_the_system()
        assert _qt_proxy() == (True, None)
        seen = []

        def first_window(*args, **kwargs):
            seen.append(_qt_proxy())
            raise _FirstWindow

        # The windows the constructor builds first. One built for real here
        # with a web view in it would abort the suite (see the module docstring).
        for window in ("LogViewerWindow", "MemoryViewerWindow", "FaceWindow"):
            monkeypatch.setattr(app_mod, window, first_window)
        # It kills every process whose command line names the daemon.
        monkeypatch.setattr(app_mod.JarvisSystemTray, "cleanup_orphaned_processes", lambda self: None)

        with pytest.raises(_FirstWindow):
            app_mod.JarvisSystemTray()

        assert seen == [(False, QNetworkProxy.ProxyType.NoProxy)]

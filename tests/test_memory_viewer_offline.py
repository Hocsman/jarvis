"""The memory viewer loads nothing from anywhere.

The desktop's promise is a fully local app: the viewer page is served
by a loopback server and read by an embedded WebEngine view. A remote
stylesheet or font CDN in that page turns every opening of the window
into a call to a third party, contradicting the offline promise, and
it is the only remote content the embedded Chromium would ever parse.
So the served HTML must reference no remote URL — the fonts come from
stacks the machine already has — and the embedded view must refuse to
navigate anywhere but the loopback server.
"""

from __future__ import annotations

import re
from types import SimpleNamespace
from urllib.parse import urlparse

import pytest

try:
    import flask  # noqa: F401

    _HAS_FLASK = True
except ImportError:
    _HAS_FLASK = False

# Every absolute URL in the served page, in attributes, scripts or CSS.
URL_RE = re.compile(r"https?://[^\s\"'`<>()]+")

LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")


@pytest.fixture()
def viewer(tmp_path, monkeypatch):
    """The viewer app on an isolated database, core and ledger."""
    from desktop_app import memory_viewer
    from jarvis.memory.core import MemoryCore
    from jarvis.memory.db import Database

    db_path = str(tmp_path / "viewer.db")
    seed = Database(db_path)
    seed.close()
    monkeypatch.setattr(memory_viewer, "_get_db_path", lambda: db_path)
    monkeypatch.setattr(memory_viewer, "_core", MemoryCore(tmp_path / "yuba"))
    monkeypatch.setattr(memory_viewer, "_db_conn", None)
    monkeypatch.setattr(memory_viewer, "_activity_db", None)
    monkeypatch.setattr(memory_viewer, "_graph_store", None)
    memory_viewer.app.config["TESTING"] = True

    yield SimpleNamespace(
        mv=memory_viewer,
        client=memory_viewer.app.test_client(),
        token=memory_viewer.get_launch_token(),
    )

    conn = memory_viewer._db_conn
    if conn is not None:
        conn.close()


def _served_html(viewer) -> str:
    from conftest import ViewerClient

    resp = ViewerClient(viewer.mv).get("/")
    assert resp.status_code == 200
    return resp.get_data(as_text=True)


@pytest.mark.unit
@pytest.mark.skipif(not _HAS_FLASK, reason="Flask not available")
def test_the_served_page_references_no_remote_url(viewer):
    """Opening the viewer contacts nobody: every URL in the page is the
    loopback server itself."""
    remote = [
        u for u in URL_RE.findall(_served_html(viewer))
        if urlparse(u).hostname not in LOCAL_HOSTS
    ]
    assert remote == [], (
        "the served page references remote URLs; opening the viewer "
        f"contacts third parties: {remote}"
    )


@pytest.mark.unit
@pytest.mark.skipif(not _HAS_FLASK, reason="Flask not available")
def test_the_page_carries_no_other_remote_shape(viewer):
    """URL_RE only matches absolute http(s) URLs; a protocol-relative
    reference (//host/path) is fetched over https exactly like one, and
    WebSocket endpoints bypass the http scan entirely."""
    html = _served_html(viewer)
    assert re.search(r"""(?:src|href)\s*=\s*["']\s*//""", html) is None
    assert "ws://" not in html
    assert "wss://" not in html


@pytest.mark.unit
@pytest.mark.skipif(not _HAS_FLASK, reason="Flask not available")
def test_fonts_come_from_local_stacks(viewer):
    """The webfont families and their CDN are gone; the page styles text
    with fonts the machine already has."""
    html = _served_html(viewer)
    assert "fonts.googleapis.com" not in html
    assert "fonts.gstatic.com" not in html
    assert "JetBrains" not in html
    assert "Outfit" not in html
    assert "monospace" in html
    assert "sans-serif" in html


@pytest.mark.unit
def test_loopback_and_internal_urls_pass_the_guard():
    from PyQt6.QtCore import QUrl

    from desktop_app.app import _is_local_navigation

    assert _is_local_navigation(QUrl("http://localhost:5050/"))
    assert _is_local_navigation(QUrl("http://127.0.0.1:5050/api/graph/node/3"))
    assert _is_local_navigation(QUrl("http://[::1]:5050/"))
    assert _is_local_navigation(QUrl("about:blank"))
    # setHtml is a navigation to a self-contained data: document under
    # Qt 6; the error page rides it, so the guard must let it through.
    assert _is_local_navigation(QUrl("data:text/html;charset=UTF-8,<b>x</b>"))


@pytest.mark.unit
def test_remote_urls_are_refused_by_the_guard():
    from PyQt6.QtCore import QUrl

    from desktop_app.app import _is_local_navigation

    assert not _is_local_navigation(QUrl("https://fonts.googleapis.com/css2"))
    assert not _is_local_navigation(QUrl("https://evil.example/"))
    assert not _is_local_navigation(QUrl("http://192.168.1.5/"))
    assert not _is_local_navigation(QUrl("file:///etc/passwd"))


@pytest.mark.unit
def test_the_embedded_page_class_wires_the_guard():
    """When WebEngine is available, the page the viewer window installs
    overrides acceptNavigationRequest with the local-only decision, and
    the window really installs it. WebEngine pages are not instantiated
    in tests (CI-lightness), so the wiring is pinned structurally."""
    import inspect

    from desktop_app import app as desktop_app_mod

    if not desktop_app_mod.HAS_WEBENGINE:
        pytest.skip("QtWebEngine not installed")

    from PyQt6.QtWebEngineCore import QWebEnginePage

    assert issubclass(desktop_app_mod._LocalOnlyPage, QWebEnginePage)
    assert "acceptNavigationRequest" in desktop_app_mod._LocalOnlyPage.__dict__
    init_src = inspect.getsource(desktop_app_mod.MemoryViewerWindow.__init__)
    assert "setPage(_LocalOnlyPage(" in init_src, (
        "the viewer window does not install the local-only page; the "
        "guard exists but never runs"
    )

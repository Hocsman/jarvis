"""The memory viewer server answers the local app and nobody else.

The server binds to 127.0.0.1, but the bind is not the boundary: a
DNS-rebinding page reaches it under an attacker's Host name, and any
website can send it cross-site form POSTs. The body-less bulk sweeps are
the sharpest edge — each one starts an LLM rewrite of the whole diary or
graph, and with a remote provider that ships the diary to the cloud with
no user action.

The gate these tests pin:

1. a Host name that is not localhost / 127.0.0.1 is refused, reads and
   writes alike;
2. a write whose Origin is present and not the viewer's own is refused;
3. every mutating request carries a per-launch token, which only the
   served page holds — the server embeds it in the index and no response
   carries a CORS allowance, so a foreign page can never read it;
4. the four bulk sweeps additionally require application/json, a shape
   no HTML form can produce and a content type that forces a CORS
   preflight the server never grants.

Reads stay token-free for the embedded view, except the node view,
which writes an access score and carries the token like a mutation;
and the desktop app asks whatever holds the port to identify itself
before pointing a window at it.
"""

from __future__ import annotations

import socket
import threading
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

try:
    import flask  # noqa: F401

    _HAS_FLASK = True
except ImportError:
    _HAS_FLASK = False


# The test client's default Host is "localhost", so the viewer's own
# origin, as the gate compares it, is http://localhost.
VIEWER_ORIGIN = "http://localhost"

SWEEP_PATHS = (
    "/api/graph/import-diary",
    "/api/graph/consolidate-all",
    "/api/diary/scrub-deflections",
    "/api/diary/optimise-topics",
)


def _example_path(rule) -> str:
    """A concrete path for a URL rule: every converter gets a stand-in
    that matches its shape, so the request routes and the gate answers
    before any handler runs."""
    parts = []
    for segment in rule.rule.strip("/").split("/"):
        if segment.startswith("<") and segment.endswith(">"):
            converter = segment[1:-1]
            parts.append("1" if converter.startswith("int:") else "x")
        else:
            parts.append(segment)
    return "/" + "/".join(parts)


@contextmanager
def _served(wsgi_app):
    """Run a WSGI app on an ephemeral loopback port for one probe."""
    from werkzeug.serving import make_server

    srv = make_server("127.0.0.1", 0, wsgi_app)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield srv.server_port
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=5)


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


@pytest.mark.unit
@pytest.mark.skipif(not _HAS_FLASK, reason="Flask not available")
class TestHostGate:
    """The Host header names who the request believes it is talking to.
    A rebinding page names the attacker's domain; the gate refuses it
    before any route runs."""

    def test_a_foreign_host_cannot_read_the_core(self, viewer):
        resp = viewer.client.get("/api/core", headers={"Host": "attacker.example:5050"})

        assert resp.status_code == 400

    def test_a_foreign_host_cannot_write(self, viewer):
        resp = viewer.client.put(
            "/api/core/profile",
            json={"raw": "- injecté\n"},
            headers={
                "Host": "attacker.example:5050",
                "Origin": "http://attacker.example:5050",
                "X-Jarvis-Token": viewer.token,
            },
        )

        assert resp.status_code == 400

    def test_a_foreign_host_never_reaches_the_page(self, viewer):
        resp = viewer.client.get("/", headers={"Host": "attacker.example:5050"})

        assert resp.status_code == 400

    @pytest.mark.parametrize("host", ["localhost:5050", "127.0.0.1:5050", "localhost", "127.0.0.1"])
    def test_the_loopback_hosts_are_served(self, viewer, host):
        resp = viewer.client.get("/api/health", headers={"Host": host})

        assert resp.status_code == 200


@pytest.mark.unit
@pytest.mark.skipif(not _HAS_FLASK, reason="Flask not available")
class TestOriginGate:
    """A browser stamps Origin on every cross-site POST it sends. One
    that is not the viewer's own is an attack, whatever token or content
    type travels with it."""

    def test_a_cross_site_json_write_is_refused(self, viewer):
        resp = viewer.client.post(
            "/api/diary/scrub-deflections",
            json={},
            headers={"Origin": "https://evil.example", "X-Jarvis-Token": viewer.token},
        )

        assert resp.status_code == 403

    def test_a_cross_site_form_post_to_every_sweep_is_refused(self, viewer):
        for path in SWEEP_PATHS:
            resp = viewer.client.post(
                path,
                data="",
                content_type="application/x-www-form-urlencoded",
                headers={"Origin": "https://evil.example"},
            )

            assert resp.status_code == 403, path
            assert resp.mimetype == "application/json", path

    def test_a_localhost_page_on_another_port_is_still_foreign(self, viewer):
        resp = viewer.client.post(
            "/api/diary/scrub-deflections",
            json={},
            headers={"Origin": "http://localhost:9999", "X-Jarvis-Token": viewer.token},
        )

        assert resp.status_code == 403

    def test_a_null_origin_does_not_bypass_the_token(self, viewer):
        resp = viewer.client.post(
            "/api/diary/scrub-deflections",
            data="",
            content_type="application/x-www-form-urlencoded",
            headers={"Origin": "null"},
        )

        assert resp.status_code == 403

    def test_the_viewers_own_origin_reaches_the_sweep(self, viewer):
        resp = viewer.client.post(
            "/api/diary/scrub-deflections",
            json={},
            headers={"Origin": VIEWER_ORIGIN, "X-Jarvis-Token": viewer.token},
        )

        assert resp.status_code == 200
        assert resp.mimetype == "application/x-ndjson"


@pytest.mark.unit
@pytest.mark.skipif(not _HAS_FLASK, reason="Flask not available")
class TestTokenGate:
    """The per-launch token is a browser-boundary control: a web page
    that is not the served page can never read it, and so can never
    write. (A local non-browser process can scrape it from GET / — it
    could equally read the database files directly — the token defends
    the browser, not the machine.) Reads stay open, except the node
    view, which writes: the embedded view loads them, and no response
    carries a CORS allowance, so a foreign page can request but never
    read."""

    def test_a_write_without_the_token_is_refused(self, viewer):
        resp = viewer.client.put("/api/core/profile", json={"raw": "- x\n"})

        assert resp.status_code == 403

    def test_a_write_with_a_wrong_token_is_refused(self, viewer):
        resp = viewer.client.put(
            "/api/core/profile",
            json={"raw": "- x\n"},
            headers={"X-Jarvis-Token": "pas-le-token"},
        )

        assert resp.status_code == 403

    def test_a_non_ascii_token_is_refused_without_a_crash(self, viewer):
        resp = viewer.client.put(
            "/api/core/profile",
            json={"raw": "- x\n"},
            headers={"X-Jarvis-Token": "jeton-éé"},
        )

        assert resp.status_code == 403

    def test_a_delete_without_the_token_is_refused(self, viewer):
        resp = viewer.client.delete("/api/activity")

        assert resp.status_code == 403

    @pytest.mark.parametrize("path", SWEEP_PATHS)
    def test_every_sweep_refuses_an_untokened_json_post(self, viewer, path):
        resp = viewer.client.post(path, json={})

        assert resp.status_code == 403

    def test_every_mutating_route_refuses_a_tokenless_write(self, viewer):
        """The mechanism, not a list of endpoints: the gate keys on the
        method, so every mutating rule in the app, present and future,
        must refuse a tokenless request before its handler runs."""
        checked = 0
        for rule in viewer.mv.app.url_map.iter_rules():
            for method in sorted(rule.methods & {"POST", "PUT", "PATCH", "DELETE"}):
                resp = viewer.client.open(_example_path(rule), method=method, json={})

                assert resp.status_code == 403, f"{method} {rule.rule}"
                checked += 1

        assert checked >= len(SWEEP_PATHS)

    def test_a_refused_write_leaves_the_core_untouched(self, viewer):
        viewer.client.put("/api/core/profile", json={"raw": "- injecté\n"})

        assert viewer.client.get("/api/core").get_json()["profile"]["entries"] == []

    def test_reads_need_no_token(self, viewer):
        resp = viewer.client.get("/api/core")

        assert resp.status_code == 200

    def test_the_token_is_a_secret_worth_stealing(self, viewer):
        assert len(viewer.token) >= 32

    def test_the_served_page_carries_the_token_and_the_header_name(self, viewer):
        from conftest import viewer_scripts

        html = viewer.client.get("/").get_data(as_text=True)
        script = " ".join(viewer_scripts(viewer.client))

        assert viewer.token in html
        assert "X-Jarvis-Token" in script


@pytest.mark.unit
@pytest.mark.skipif(not _HAS_FLASK, reason="Flask not available")
class TestSweepsRequireJson:
    """The four bulk sweeps start an LLM rewrite of everything; they
    only accept application/json. No HTML form can produce it, and a
    cross-origin fetch that sets it triggers a preflight the server
    never grants."""

    @pytest.mark.parametrize("path", SWEEP_PATHS)
    def test_a_tokened_form_post_is_still_refused(self, viewer, path):
        resp = viewer.client.post(
            path,
            data="x=1",
            content_type="application/x-www-form-urlencoded",
            headers={"X-Jarvis-Token": viewer.token},
        )

        assert resp.status_code == 415


@pytest.mark.unit
@pytest.mark.skipif(not _HAS_FLASK, reason="Flask not available")
class TestFramingAndCache:
    """The served page holds the launch token, so it must never render
    inside a foreign frame (a framed genuine page is same-origin with a
    valid token: clickjacking it sidesteps the whole gate), and it must
    never be cached (a cached page wields a dead token after a server
    restart, and every write fails 403 until a hard reload)."""

    def test_the_page_refuses_to_be_framed(self, viewer):
        resp = viewer.client.get("/")

        assert resp.headers["X-Frame-Options"] == "DENY"
        assert "frame-ancestors 'none'" in resp.headers["Content-Security-Policy"]

    def test_the_token_bearing_page_is_not_cached(self, viewer):
        resp = viewer.client.get("/")

        assert resp.headers["Cache-Control"] == "no-store"

    def test_refusals_carry_the_same_headers(self, viewer):
        resp = viewer.client.get("/api/core", headers={"Host": "attacker.example:5050"})

        assert resp.status_code == 400
        assert resp.headers["X-Frame-Options"] == "DENY"


@pytest.mark.unit
@pytest.mark.skipif(not _HAS_FLASK, reason="Flask not available")
class TestViewerClientCarriesTheToken:
    """The shared test helper must send the token on every mutating verb
    the gate covers, PATCH included, or an endpoint test using it would
    silently test the gate instead of the endpoint."""

    def test_patch_carries_the_token(self, viewer):
        from conftest import ViewerClient

        client = ViewerClient(viewer.mv)

        # No PATCH route exists: past the gate the answer is 405.
        # Without the injected token the gate would answer 403 first.
        assert client.patch("/api/health").status_code == 405

    def test_the_context_manager_delegates(self, viewer):
        from conftest import ViewerClient

        with ViewerClient(viewer.mv) as client:
            assert client.get("/api/health").status_code == 200


@pytest.mark.unit
@pytest.mark.skipif(not _HAS_FLASK, reason="Flask not available")
class TestHealthIdentity:
    """The desktop app treats an occupied port as "our server already
    runs" — a foreign program holding the port would then be pointed at
    by the window. The health marker tells them apart."""

    def test_the_health_endpoint_identifies_the_server(self, viewer):
        payload = viewer.client.get("/api/health").get_json()

        assert payload["app"] == "jarvis-memory-viewer"

    def test_the_app_recognises_a_live_viewer_server(self, viewer):
        from desktop_app.app import _port_holds_viewer_server

        with _served(viewer.mv.app) as port:
            assert _port_holds_viewer_server(port) is True

    def test_the_app_recognises_a_foreign_server(self):
        from flask import Flask, jsonify

        from desktop_app.app import _port_holds_viewer_server

        other = Flask("other")

        @other.route("/api/health")
        def other_health():
            return jsonify({"app": "something-else"})

        with _served(other) as port:
            assert _port_holds_viewer_server(port) is False

    def test_the_probe_reaches_loopback_past_a_configured_proxy(self, viewer, monkeypatch):
        """The machine has a proxy configured; the probe must still
        reach its own loopback directly. A proxy cannot route to the
        user's loopback, and a healthy viewer must not be reported as
        foreign on a proxy-configured machine."""
        from desktop_app.app import _port_holds_viewer_server

        monkeypatch.setenv("http_proxy", "http://127.0.0.1:1")
        monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
        with _served(viewer.mv.app) as port:
            assert _port_holds_viewer_server(port) is True

    def test_the_probe_does_not_follow_a_redirect(self, viewer):
        """Whoever holds the port must answer for itself. A listener
        that forwards the probe elsewhere is not identifying itself,
        and the app must not chase it: that would turn the identity
        check into a request to an arbitrary destination."""
        from flask import Flask, redirect

        from desktop_app.app import _port_holds_viewer_server

        with _served(viewer.mv.app) as real_port:
            forwarder = Flask("forwarder")

            @forwarder.route("/api/health")
            def forward():
                return redirect(f"http://127.0.0.1:{real_port}/api/health")

            with _served(forwarder) as fwd_port:
                assert _port_holds_viewer_server(fwd_port) is False

    def test_a_dead_port_is_not_ours(self):
        from desktop_app.app import _port_holds_viewer_server

        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]

        assert _port_holds_viewer_server(port) is False

"""The memory viewer window, driven for real in a process of its own.

``test_memory_viewer_navigation.py`` runs this script and reads the one
``REPORT:`` line it prints. It builds the real ``MemoryViewerWindow``,
so the real web view and the page it installs, with ``start_server``
replaced: the real memory server never starts. One window is told the
server failed and shows its error page. Another is pointed at a server
this script runs on loopback, serving pages it wrote itself, and from
there a link and a redirect try to reach another loopback address the
guard does not name: refused, they load nothing, and followed they would
still stay on this machine. Every remote address is a decision asked of
the page without loading anything, so nothing leaves the machine.

The name keeps it out of the suite's collection.
"""

import http.server
import json
import os
import socketserver
import sys
import tempfile
import threading
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PyQt6.QtWebEngineWidgets import QWebEngineView  # noqa: E402,F401  (before the QApplication)
from PyQt6.QtCore import QUrl  # noqa: E402
from PyQt6.QtWebEngineCore import QWebEnginePage  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

# Chromium aborts the process when the application has no program name.
app = QApplication(["memory-viewer-page-child"])

import desktop_app.app as app_mod  # noqa: E402

logged = []
app_mod.debug_log = lambda message, category="debug": logged.append(message)


def refused():
    return [line for line in logged if "navigation refused" in line]


def drain(until, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.processEvents()
        if until():
            return True
        time.sleep(0.01)
    app.processEvents()
    return until()


def run_js(page, script, timeout=5.0):
    result = []
    page.runJavaScript(script, 0, result.append)
    drain(lambda: result, timeout)
    return result[0] if result else None


def load(window):
    """Show the window and wait for the page its ``showEvent`` loads."""
    loads = []
    window.web_view.loadFinished.connect(loads.append)
    window.resize(1000, 700)
    window.show()
    drain(lambda: loads, 30.0)
    return bool(loads and loads[-1])


# Loopback addresses outside the guard's list: a refusal loads nothing,
# and a navigation the guard let through would still stay on this machine.
FOREIGN_HOST = "127.0.0.2"

workdir = Path(tempfile.mkdtemp(prefix="memory-viewer-page-"))
(workdir / "index.html").write_text(
    "<html><head><title>the viewer's own page</title></head><body></body></html>",
    encoding="utf-8",
)
(workdir / "second.html").write_text(
    "<html><head><title>another page of the viewer</title></head><body></body></html>",
    encoding="utf-8",
)
local_file = workdir / "local.html"
local_file.write_text("<html><body>a file on disk</body></html>", encoding="utf-8")


class _Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(workdir), **kwargs)

    def do_GET(self):
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", f"http://{FOREIGN_HOST}:{port}/index.html")
            self.end_headers()
            return
        super().do_GET()

    def log_message(self, *args):
        pass


class _Server(http.server.ThreadingHTTPServer):
    def server_bind(self):
        # HTTPServer's own looks its address up by name; this one asks no resolver.
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = "localhost", self.server_address[1]


server = _Server(("127.0.0.1", 0), _Handler)
port = server.server_address[1]
threading.Thread(target=server.serve_forever, daemon=True).start()

report = {"foreign_host": FOREIGN_HOST}

# The server fails: the window shows its error page, a data: document.
failed = app_mod.MemoryViewerWindow()
failed.start_server = lambda: False
loaded = load(failed)
report["error_page"] = {
    "loaded": loaded,
    "scheme": failed.web_view.url().scheme(),
    "heading": run_js(failed.web_view.page(), "document.querySelector('h1').textContent"),
    "refused": refused(),
}

# The server runs: the window shows what it serves on loopback.
del logged[:]
served = app_mod.MemoryViewerWindow()
served.start_server = lambda: True
served.MEMORY_VIEWER_PORT = port
view = served.web_view
page = view.page()
loaded = load(served)
report["served"] = {
    "loaded": loaded,
    "host": view.url().host(),
    "on_the_servers_port": view.url().port() == port,
    "title": page.title(),
    "refused": refused(),
}


def click_link(href):
    return run_js(
        page,
        "const a = document.createElement('a');"
        f"a.href = {json.dumps(href)};"
        "document.body.appendChild(a); a.click(); true",
    )


# A link to another page of the server opens it. The title can change
# before the view's address does, so the wait is for both.
second = QUrl(f"http://127.0.0.1:{port}/second.html")
clicked = click_link(second.toString())
drain(lambda: view.url() == second and page.title() == "another page of the viewer", 10.0)
report["loopback_link"] = {
    "clicked": clicked is True,
    "shown": view.url() == second,
    "title": page.title(),
    "refused": refused(),
}

# A link to a host off the list. A refusal is logged the moment the page is
# asked; a navigation let through would change the shown address instead.
del logged[:]
shown = view.url()
clicked = click_link(f"http://{FOREIGN_HOST}:{port}/index.html")
drain(lambda: refused() or view.url() != shown, 5.0)
drain(lambda: view.url() != shown, 1.0)
report["foreign_link"] = {
    "clicked": clicked is True,
    "refused": refused(),
    "stayed": view.url() == shown,
    "title": page.title(),
}

# The server answers with a redirect to a host off the list.
del logged[:]
view.setUrl(QUrl(f"http://localhost:{port}/redirect"))
drain(lambda: refused() or view.url().host() == FOREIGN_HOST, 10.0)
drain(lambda: view.url().host() == FOREIGN_HOST, 1.0)
report["foreign_redirect"] = {"refused": refused(), "host": view.url().host()}

# The page's answer for each address, asked directly. Nothing is loaded.
link = QWebEnginePage.NavigationType.NavigationTypeLinkClicked
local = {
    "the viewer's server": QUrl("http://localhost:5050/"),
    "a path on 127.0.0.1": QUrl("http://127.0.0.1:5050/api/graph/node/3"),
    "the IPv6 loopback": QUrl("http://[::1]:5050/"),
    "about:blank": QUrl("about:blank"),
    "the error page": QUrl("data:text/html;charset=UTF-8,<h1>Connection Failed</h1>"),
}
foreign = {
    "a remote page": QUrl("https://example.invalid/"),
    "a font CDN": QUrl("https://fonts.googleapis.com/css2"),
    "a LAN address": QUrl("http://192.168.1.5/"),
    "a local file": QUrl.fromLocalFile(str(local_file)),
    "a host named after localhost": QUrl("http://localhost.example.invalid/"),
    "a remote host behind loopback credentials": QUrl("http://127.0.0.1@example.invalid/"),
    "a loopback address the guard does not name": QUrl(f"http://{FOREIGN_HOST}:5050/"),
}
addresses = {**local, **foreign}
report["main_frame"] = {name: page.acceptNavigationRequest(url, link, True) for name, url in addresses.items()}
report["sub_frame"] = {name: page.acceptNavigationRequest(url, link, False) for name, url in addresses.items()}

del logged[:]
for url in local.values():
    page.acceptNavigationRequest(url, link, True)
    page.acceptNavigationRequest(url, link, False)
report["accepted_logged"] = refused()

remote = QUrl("https://example.invalid/")
report["remote_by_navigation_type"] = {
    kind.name: page.acceptNavigationRequest(remote, kind, True) for kind in QWebEnginePage.NavigationType
}

del logged[:]
page.acceptNavigationRequest(
    QUrl("https://someone:hunter2@example.invalid:8443/private/notes.html?token=s3cret#frag"), link, True
)
report["refusal_logged"] = refused()

print("REPORT:" + json.dumps(report), flush=True)
failed.close()
served.close()
server.shutdown()
app.processEvents()

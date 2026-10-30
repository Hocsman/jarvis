"""The memory viewer window, driven for real in a process of its own.

``test_memory_viewer_navigation.py`` runs this script and reads the one
``REPORT:`` line it prints. It builds the real ``MemoryViewerWindow``,
so the real web view and the page it installs, with ``start_server``
replaced: the real memory server never starts. One window is told the
server failed and shows its error page. Another is pointed at a server
this script runs on 127.0.0.1, serving pages it wrote itself. From there
a link, a redirect, a frame and two new windows try to reach 127.0.0.2,
a loopback address the guard does not name, where a second server counts
every request it receives: refused, they reach it never, and let through
they would still stay on this machine. Every remote address is a decision
asked of the page without loading anything, and Chromium is given no
proxy, so nothing leaves the machine.

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
from PyQt6.QtNetwork import QNetworkProxy, QNetworkProxyFactory  # noqa: E402
from PyQt6.QtWebEngineCore import QWebEnginePage  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

# Chromium aborts the process when the application has no program name.
app = QApplication(["memory-viewer-page-child"])

# Chromium takes its proxy from Qt, and Qt from the system. Windows asks
# for discovery by default, which sends a DHCP query and a DNS lookup for
# "wpad" on the first load, loopback included. With no proxy there is none.
QNetworkProxyFactory.setUseSystemConfiguration(False)
QNetworkProxy.setApplicationProxy(QNetworkProxy(QNetworkProxy.ProxyType.NoProxy))

import desktop_app.app as app_mod  # noqa: E402

logged = []
app_mod.debug_log = lambda message, category="debug": logged.append(message)


def refused():
    """The guard's refusal lines among everything the app logged."""
    return [line for line in logged if "navigation refused" in line]


def drain(until, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
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


def show(window):
    """Show the window and wait for the page its ``showEvent`` loads."""
    loads = []
    window.web_view.loadFinished.connect(loads.append)
    window.resize(1000, 700)
    window.show()
    drain(lambda: loads, 30.0)
    return bool(loads and loads[-1])


def load(view, url):
    loads = []
    finished = loads.append
    view.loadFinished.connect(finished)
    view.setUrl(url)
    drain(lambda: loads, 10.0)
    view.loadFinished.disconnect(finished)
    return bool(loads and loads[-1])


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


class _Server(http.server.ThreadingHTTPServer):
    def server_bind(self):
        # HTTPServer's own looks its address up by name; this one asks no resolver.
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = "localhost", self.server_address[1]


def serve(host, handler):
    server = _Server((host, 0), handler)
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    return server


FOREIGN_HOST = "127.0.0.2"
INDEX_TITLE = "the viewer's own page"
SECOND_TITLE = "another page of the viewer"

# What reaches the host off the list. Each attempt asks for a path of its own.
foreign_requests = []


class _Counter(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        foreign_requests.append(self.path)
        body = b"<html><head><title>a page off the list</title></head></html>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


workdir = Path(tempfile.mkdtemp(prefix="memory-viewer-page-"))


class _Viewer(_Quiet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(workdir), **kwargs)

    def do_GET(self):
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", f"{foreign}/from-a-redirect")
            self.end_headers()
            return
        super().do_GET()


viewer_server = serve("127.0.0.1", _Viewer)
port = viewer_server.server_address[1]
try:
    foreign_server = serve(FOREIGN_HOST, _Counter)
    foreign = f"http://{FOREIGN_HOST}:{foreign_server.server_address[1]}"
except OSError:
    # macOS gives its loopback interface 127.0.0.1 alone.
    foreign_server = None
    foreign = f"http://{FOREIGN_HOST}:{port}"

(workdir / "index.html").write_text(
    f"<html><head><title>{INDEX_TITLE}</title></head><body></body></html>", encoding="utf-8"
)
(workdir / "second.html").write_text(
    f"<html><head><title>{SECOND_TITLE}</title></head><body></body></html>", encoding="utf-8"
)
(workdir / "framed.html").write_text(
    f"<html><body><iframe src='{foreign}/from-a-frame'></iframe></body></html>", encoding="utf-8"
)
local_file = workdir / "local.html"
local_file.write_text("<html><body>a file on disk</body></html>", encoding="utf-8")

report = {"foreign_host": FOREIGN_HOST}

# The server fails: the window shows its error page, a data: document.
failed = app_mod.MemoryViewerWindow()
failed.start_server = lambda: False
loaded = show(failed)
report["error_page"] = {
    "loaded": loaded,
    "scheme": failed.web_view.url().scheme(),
    "heading": run_js(failed.web_view.page(), "document.querySelector('h1').textContent"),
    "refused": refused(),
}
failed.close()

# The server runs: the window shows what it serves on loopback.
del logged[:]
served = app_mod.MemoryViewerWindow()
served.start_server = lambda: True
served.MEMORY_VIEWER_PORT = port
view = served.web_view
page = view.page()
loaded = show(served)
report["served"] = {
    "loaded": loaded,
    "host": view.url().host(),
    "on_the_servers_port": view.url().port() == port,
    "shows_its_page": page.title() == INDEX_TITLE,
    "refused": refused(),
}


def click_link(href, target=""):
    return run_js(
        page,
        "const a = document.createElement('a');"
        f"a.href = {json.dumps(href)}; a.target = {json.dumps(target)};"
        "document.body.appendChild(a); a.click(); true",
    )


def nothing_moved(since):
    """Wait out a navigation the guard might have let through: it would
    move the view, or reach the host off the list within milliseconds."""
    drain(lambda: view.url() != since or foreign_requests, 0.5)
    return view.url() == since


# A link to another page of the server opens it. The title can change
# before the view's address does, so the wait is for both.
second = QUrl(f"http://127.0.0.1:{port}/second.html")
clicked = click_link(second.toString())
drain(lambda: view.url() == second and page.title() == SECOND_TITLE, 10.0)
report["server_link"] = {
    "clicked": clicked is True,
    "shown": view.url() == second and page.title() == SECOND_TITLE,
    "refused": refused(),
}

# A link to the host off the list: refused the moment the page is asked.
del logged[:]
shown, title = view.url(), page.title()
clicked = click_link(f"{foreign}/from-a-link")
drain(lambda: refused() or view.url() != shown or foreign_requests, 5.0)
report["foreign_link"] = {
    "clicked": clicked is True,
    "refused": refused(),
    "stayed": nothing_moved(shown) and page.title() == title,
}

# The server answers with a redirect to the host off the list.
del logged[:]
view.setUrl(QUrl(f"http://localhost:{port}/redirect"))
drain(lambda: refused() or view.url().host() == FOREIGN_HOST or foreign_requests, 10.0)
drain(lambda: view.url().host() == FOREIGN_HOST or foreign_requests, 0.5)
report["foreign_redirect"] = {"refused": refused(), "host": view.url().host()}

# A page of the server frames the host off the list.
del logged[:]
loaded = load(view, QUrl(f"http://localhost:{port}/framed.html"))
drain(lambda: refused() or foreign_requests, 5.0)
drain(lambda: foreign_requests, 0.5)
report["foreign_frame"] = {
    "loaded": loaded,
    "refused": refused(),
    "frame_shows": run_js(
        page,
        "(() => { try { return document.querySelector('iframe').contentWindow.location.href; }"
        " catch (e) { return 'a page of another origin'; } })()",
    ),
}

# A link and a script that open a new window, to the host off the list.
del logged[:]
shown = view.url()
clicked = click_link(f"{foreign}/from-a-new-tab", target="_blank")
opened = run_js(page, f"window.open({json.dumps(foreign + '/from-window-open')}); true")
report["new_window"] = {
    "asked": clicked is True and opened is True,
    "stayed": nothing_moved(shown),
}

report["foreign_requests"] = list(foreign_requests) if foreign_server else None

# The page's answer for each address, asked directly. Nothing is loaded.
link = QWebEnginePage.NavigationType.NavigationTypeLinkClicked
local = {
    "the viewer's server": QUrl("http://localhost:5050/"),
    "a path on 127.0.0.1": QUrl("http://127.0.0.1:5050/api/graph/node/3"),
    "the IPv6 loopback": QUrl("http://[::1]:5050/"),
    "about:blank": QUrl("about:blank"),
    "a data: document": QUrl("data:text/html;charset=UTF-8,<h1>x</h1>"),
}
outside = {
    "a remote page": QUrl("https://example.invalid/"),
    "a font CDN": QUrl("https://fonts.googleapis.com/css2"),
    "a LAN address": QUrl("http://192.168.1.5/"),
    "a file path with no host": QUrl.fromLocalFile(str(local_file)),
    "a host named after localhost": QUrl("http://localhost.example.invalid/"),
    "a remote host behind loopback credentials": QUrl("http://127.0.0.1@example.invalid/"),
    "a loopback address the guard does not name": QUrl(f"http://{FOREIGN_HOST}:5050/"),
}
addresses = {**local, **outside}
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
served.close()
viewer_server.shutdown()
if foreign_server:
    foreign_server.shutdown()
app.processEvents()

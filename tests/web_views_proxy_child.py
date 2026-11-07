"""The app's two web views, loaded for real in a process of their own.

``test_web_views_no_proxy.py`` runs this script, reads the one ``REPORT:``
line it prints, then reads the net log Chromium wrote. The script takes
the step the tray takes at start-up, before any window, then builds the
real ``DashboardWindow`` and the real ``MemoryViewerWindow``, with the
memory server replaced by one this script runs on 127.0.0.1, and waits
for both pages. The memory viewer's HTTP load is the one that goes
through Chromium's proxy resolution; the dashboard's ``file:`` and
``qrc:`` loads never do, so its page shows only that it still renders.

Chromium follows the system's proxy configuration unless Qt is given
another. On Windows that is the system's own setting, whose default asks
for discovery. On Linux it comes from the environment when no desktop
names one, so the script sets ``auto_proxy`` to a proxy script on a second
loopback server, which counts every request it receives: a view that
followed the system would fetch it.

The name keeps it out of the suite's collection.
"""

import http.server
import json
import os
import socketserver
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class _Server(http.server.ThreadingHTTPServer):
    def server_bind(self):
        # HTTPServer's own looks its address up by name; this one asks no resolver.
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = "localhost", self.server_address[1]


def serve(handler):
    server = _Server(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    return server


class _Page(http.server.BaseHTTPRequestHandler):
    body = b""
    content_type = "text/html"

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", self.content_type)
        self.send_header("Content-Length", str(len(self.body)))
        self.end_headers()
        self.wfile.write(self.body)

    def log_message(self, *args):
        pass


VIEWER_TITLE = "the memory viewer's page"
script_requests = []


class _Viewer(_Page):
    body = f"<html><head><title>{VIEWER_TITLE}</title></head><body></body></html>".encode()


class _ProxyScript(_Page):
    body = b'function FindProxyForURL(url, host) { return "DIRECT"; }'
    content_type = "application/x-ns-proxy-autoconfig"

    def do_GET(self):
        script_requests.append(self.path)
        super().do_GET()


viewer_server = serve(_Viewer)
script_server = serve(_ProxyScript)
os.environ["auto_proxy"] = f"http://127.0.0.1:{script_server.server_address[1]}/wpad.dat"

from PyQt6.QtWebEngineWidgets import QWebEngineView  # noqa: E402,F401  (before the QApplication)
from PyQt6.QtNetwork import QNetworkProxy, QNetworkProxyFactory  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

# Chromium aborts the process when the application has no program name.
app = QApplication(["web-views-proxy-child"])

import jarvis.config  # noqa: E402

# No config: the bridge then has no location, so no weather is fetched.
jarvis.config.load_settings = lambda *args, **kwargs: None

import desktop_app.app as app_mod  # noqa: E402
import desktop_app.dashboard_window as dashboard_window  # noqa: E402

# Qt follows the system, as a new process does wherever Qt is built to.
# Setting it asks the system nothing; an application proxy turns it off,
# so it goes on last.
QNetworkProxy.setApplicationProxy(QNetworkProxy())
QNetworkProxyFactory.setUseSystemConfiguration(True)

# The step the tray takes at start-up, before it builds any window.
app_mod._give_qt_no_proxy()


def drain(until, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if until():
            return True
        time.sleep(0.01)
    app.processEvents()
    return until()


def show(window, view):
    """Show the window and wait for the page its view loads."""
    loads = []
    view.loadFinished.connect(loads.append)
    window.resize(1000, 700)
    window.show()
    drain(lambda: loads, 30.0)
    return bool(loads and loads[-1])


report = {}

dashboard = dashboard_window.DashboardWindow()
loaded = show(dashboard, dashboard.view)
report["dashboard"] = {"loaded": loaded, "scheme": dashboard.view.url().scheme()}

viewer = app_mod.MemoryViewerWindow()
viewer.start_server = lambda: True
viewer.MEMORY_VIEWER_PORT = viewer_server.server_address[1]
loaded = show(viewer, viewer.web_view)
report["memory_viewer"] = {
    "loaded": loaded,
    "host": viewer.web_view.url().host(),
    "shows_its_page": viewer.web_view.page().title() == VIEWER_TITLE,
}

report["proxy_script_requests"] = list(script_requests)

print("REPORT:" + json.dumps(report), flush=True)
dashboard.close()
viewer.close()
viewer_server.shutdown()
script_server.shutdown()
app.processEvents()

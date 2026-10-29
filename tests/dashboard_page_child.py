"""The dashboard window, driven for real in a process of its own.

``test_dashboard_own_page.py`` runs this script and reads the one
``REPORT:`` line it prints. It builds the real ``DashboardWindow``, so the
real page, channel and bridge, shows the dashboard's own ``index.html``,
then tries to reach another page the ways a page can be left. Every
address it hands the view is a file it wrote itself or a decision asked of
the page without loading anything, so nothing leaves the machine.

The name keeps it out of the suite's collection.
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PyQt6.QtWebEngineWidgets import QWebEngineView  # noqa: E402,F401  (before the QApplication)
from PyQt6.QtCore import QUrl  # noqa: E402
from PyQt6.QtWebEngineCore import QWebEnginePage, QWebEngineSettings  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

# Chromium aborts the process when the application has no program name.
app = QApplication(["dashboard-page-child"])

import jarvis.config  # noqa: E402

# No config: the bridge then has no location, so no weather is fetched.
jarvis.config.load_settings = lambda *args, **kwargs: None

import desktop_app.dashboard_window as dashboard_window  # noqa: E402

logged = []
dashboard_window.debug_log = lambda message, category="debug": logged.append(message)


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


workdir = Path(tempfile.mkdtemp(prefix="dashboard-page-"))
foreign = workdir / "foreign.html"
foreign.write_text(
    "<html><head><title>foreign</title></head><body>"
    "<script src='qrc:///qtwebchannel/qwebchannel.js'></script><script>"
    "if (window.qt && qt.webChannelTransport) {"
    "  new QWebChannel(qt.webChannelTransport,"
    "    ch => ch.objects.jarvis.submitQuery('from a foreign page'));"
    "}"
    "</script></body></html>",
    encoding="utf-8",
)
foreign_url = QUrl.fromLocalFile(str(foreign))
index = dashboard_window._dashboard_index()
index_url = QUrl.fromLocalFile(str(index))

asked = []
window = dashboard_window.DashboardWindow(submit_fn=asked.append)
window.resize(1200, 800)
window.show()
view = window.view
page = view.page()
report = {}


def shows_index():
    return view.url().adjusted(QUrl.UrlFormattingOption.RemoveFragment) == index_url


loads = []
view.loadFinished.connect(loads.append)
drain(lambda: loads, 30.0)
report["index_loaded"] = bool(loads and loads[0])
report["shown_url_is_index"] = view.url() == index_url

# The page wires the bridge once the channel answers; then a message typed
# in its chat reaches the submit hook.
drain(lambda: run_js(page, "jarvis !== null", timeout=1.0) is True, 15.0)
run_js(
    page,
    "document.getElementById('chat-input').value = 'from the dashboard';"
    "document.getElementById('chat-send').click(); true",
)
drain(lambda: asked, 5.0)
report["asked_from_own_page"] = list(asked)

# A link to another page, clicked inside the dashboard. A page that loads
# answers in well under a second here; the wait covers a slow machine.
run_js(
    page,
    "const a = document.createElement('a');"
    f"a.href = {json.dumps(foreign_url.toString())};"
    "document.body.appendChild(a); a.click(); true",
)
drain(lambda: len(asked) > 1 or not shows_index(), 4.0)
drain(lambda: len(asked) > 1, 1.0)
report["after_link"] = {
    "url_is_index": shows_index(),
    "asked": list(asked),
}

# The page's answer for each address, asked directly. Nothing is loaded.
link = QWebEnginePage.NavigationType.NavigationTypeLinkClicked
addresses = {
    "own index": index_url,
    "own index with a fragment": QUrl(index_url.toString() + "#chat"),
    "about:blank": QUrl("about:blank"),
    "another local file": foreign_url,
    "a remote page": QUrl("https://example.invalid/"),
    "the memory viewer's loopback": QUrl("http://127.0.0.1:5050/"),
    "a data document": QUrl("data:text/html,<b>x</b>"),
}
report["main_frame"] = {name: page.acceptNavigationRequest(url, link, True) for name, url in addresses.items()}
report["sub_frame"] = {name: page.acceptNavigationRequest(url, link, False) for name, url in addresses.items()}

del logged[:]
page.acceptNavigationRequest(QUrl("https://example.invalid/private/notes.html?token=s3cret#top"), link, True)
report["refusal_logged"] = list(logged)

report["navigates_on_drop"] = page.settings().testAttribute(
    QWebEngineSettings.WebAttribute.NavigateOnDropEnabled
)

print("REPORT:" + json.dumps(report), flush=True)
window.close()
app.processEvents()

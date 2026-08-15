"""fetchWebPage does not reach inside the machine it runs on.

The URL it is handed comes out of a model, and that model has been
reading the web all turn — a page it fetched a moment ago can propose
the next address. So the URL is untrusted input, and this tool is `read`
and therefore free by default: no card, no question, nothing between the
sentence on a web page and the request.

`webSearch` next door already knew this. `_is_public_url` there rejects
non-http schemes and anything resolving to loopback, private, link-local
or reserved space, re-checks every redirect hop, and refuses a hostname
whose DNS returns even one non-public address. This tool had none of it,
and it was reproducible on the developer's own machine: a fetch of
`http://127.0.0.1:11434/api/tags` came back with his Ollama model list
and `success=True`.

The guard belongs to both, so it lives in one place and both import it.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _tool():
    from src.jarvis.tools.builtin.fetch_web_page import FetchWebPageTool
    return FetchWebPageTool()


def _ctx():
    from src.jarvis.tools.base import ToolContext
    return ToolContext(db=MagicMock(), cfg=MagicMock(), system_prompt="",
                       original_prompt="", redacted_text="", max_retries=0,
                       user_print=lambda *a, **k: None)


@pytest.mark.parametrize("url", [
    "http://127.0.0.1:11434/api/tags",          # his own model server
    "http://localhost:5050/api/core",           # the memory viewer
    "http://169.254.169.254/latest/meta-data/",  # cloud metadata
    "http://10.0.0.1/",
    "http://192.168.1.1/",
    "http://[::1]:8080/",
])
def test_it_refuses_to_reach_inside_the_machine(url):
    """Each of these was reachable. The first two are services this app
    itself runs, and their contents would land in the agentic loop as
    ordinary page text."""
    with patch("src.jarvis.tools.builtin.fetch_web_page.requests.get") as get:
        r = _tool().run({"url": url}, _ctx())

    assert not r.success, f"{url} was fetched"
    assert not get.called, f"a request was actually issued to {url}"


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com/x",
                                 "gopher://example.com/"])
def test_it_refuses_a_scheme_that_is_not_the_web(url):
    with patch("src.jarvis.tools.builtin.fetch_web_page.requests.get") as get:
        r = _tool().run({"url": url}, _ctx())

    assert not r.success
    assert not get.called


def test_the_refusal_says_why_without_teaching_the_next_attempt():
    """The model reads this reply. It needs to know the address was
    refused, not a recipe for finding one that is not."""
    r = _tool().run({"url": "http://127.0.0.1:11434/api/tags"}, _ctx())

    texte = (r.reply_text or "").lower()
    assert texte
    assert "127.0.0.1" not in texte or "refus" in texte or "not" in texte


def test_a_redirect_into_the_machine_is_refused_too():
    """The first address can be public and the second not. `webSearch`
    re-checks every hop; so does this."""
    import requests as _rq

    reponse = MagicMock()
    reponse.is_redirect = True
    reponse.is_permanent_redirect = False
    reponse.headers = {"Location": "http://127.0.0.1:11434/api/tags"}
    reponse.__enter__ = lambda s: s
    reponse.__exit__ = lambda s, *a: None

    with patch("src.jarvis.tools.builtin.fetch_web_page.requests.get",
               return_value=reponse) as get:
        r = _tool().run({"url": "https://example.com/go"}, _ctx())

    assert not r.success
    # It must not have followed the redirect itself.
    for appel in get.call_args_list:
        assert appel.kwargs.get("allow_redirects") is False


def test_an_ordinary_page_still_works():
    """The guard is a hole in the permission, not a new permission."""
    reponse = MagicMock()
    reponse.is_redirect = False
    reponse.is_permanent_redirect = False
    reponse.content = b"<html><body><p>bonjour</p></body></html>"
    reponse.encoding = "utf-8"
    reponse.iter_content = lambda chunk_size=8192: iter([reponse.content])
    reponse.text = "<html><body><p>bonjour</p></body></html>"
    reponse.headers = {"Content-Type": "text/html"}
    reponse.raise_for_status = lambda: None
    reponse.__enter__ = lambda s: s
    reponse.__exit__ = lambda s, *a: None

    with patch("src.jarvis.tools.builtin.fetch_web_page.requests.get",
               return_value=reponse), \
         patch("src.jarvis.tools.builtin.fetch_web_page._is_public_url",
               return_value=True):
        r = _tool().run({"url": "https://example.com/page"}, _ctx())

    assert r.success
    assert "bonjour" in (r.reply_text or "")


def test_both_tools_share_one_guard():
    """Two copies drift, and the copy that drifts is the one nobody is
    looking at. This tool had no copy at all for months."""
    from src.jarvis.tools.builtin import fetch_web_page, web_search

    assert fetch_web_page._is_public_url is web_search._is_public_url

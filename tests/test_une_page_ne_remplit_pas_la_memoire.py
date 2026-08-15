"""A page she fetches cannot be allowed to fill the machine's memory.

`fetchWebPage` is `lecture`, so it runs free, and the URL can come from a
link the previous fetched page supplied — the file's own comment says so.
It read `response.content` and then `response.text`, both unbounded and
both in memory at once, and only truncated afterwards.

`webSearch` already streams with a byte cap for exactly this reason. The
two tools fetch the same web with the same trust in it, so they get the
same ceiling — the same argument that moved `_is_public_url` into a place
both could use.

What a runaway page costs is not just this tool: the daemon holds the
reminder thread and the routine runner, so an exhausted process takes
promises down with it.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class _ReponseSansFin:
    """A server that never stops sending."""

    status_code = 200
    is_redirect = False
    is_permanent_redirect = False
    headers = {"content-type": "text/html"}
    url = "https://exemple.test/page"
    encoding = "utf-8"

    def __init__(self, morceau: bytes = b"<p>" + b"a" * 8192 + b"</p>", tours: int = 10_000):
        self._morceau = morceau
        self._tours = tours
        self.lus = 0

    def iter_content(self, chunk_size=8192):
        for _ in range(self._tours):
            self.lus += len(self._morceau)
            yield self._morceau

    @property
    def content(self):
        return b"".join(self.iter_content())

    @property
    def text(self):
        return self.content.decode("utf-8", "replace")

    def raise_for_status(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _fetch(reponse):
    from src.jarvis.tools.builtin.fetch_web_page import FetchWebPageTool

    outil = FetchWebPageTool()
    with patch("src.jarvis.tools.builtin.fetch_web_page.requests.get",
               return_value=reponse), \
         patch("src.jarvis.tools.builtin.fetch_web_page._is_public_url",
               return_value=True):
        contexte = MagicMock()
        contexte.cfg = MagicMock()
        return outil.run({"url": "https://exemple.test/page"}, contexte)


def test_an_endless_page_is_read_only_up_to_the_ceiling():
    reponse = _ReponseSansFin()

    _fetch(reponse)

    assert reponse.lus <= 2 * 1024 * 1024, f"{reponse.lus} octets lus"


def test_an_ordinary_page_still_comes_back_whole():
    """The control. A ceiling set at zero would pass the test above and
    make the tool useless."""
    from src.jarvis.tools.builtin.fetch_web_page import FetchWebPageTool

    reponse = _ReponseSansFin(
        morceau=b"<html><title>Le titre</title><body><p>Bonjour Lyon.</p></body></html>",
        tours=1,
    )

    resultat = _fetch(reponse)

    assert resultat.success
    assert "Bonjour Lyon" in (resultat.reply_text or "")

"""The fence a tool puts round text it did not write.

A web page, or whatever a screen is showing, is somebody else's text and
can say anything, "ignore your instructions" included. Such text reaches
the model between two markers and under a line saying it is data. This
module holds what the markers are, how one cut off by a truncation is put
back, and how text is kept from carrying a marker of its own.
"""

from __future__ import annotations

import re

WEB_EXTRACT = "WEB EXTRACT"
SCREEN_TEXT = "SCREEN TEXT"

# Every kind of fence a tool here emits. A closing marker is only ever
# composed from one of these names, never from text found between markers:
# that text is the untrusted part.
KINDS = (WEB_EXTRACT, SCREEN_TEXT)

# Three or more angle brackets in a row, plain or full-width: what a marker
# opens and ends with. Text with no such run left holds no marker, whole,
# nested or rebuilt by removing another. Two in a row are left alone: they
# are a shift, a redirection or the end of a generic type far more often
# than anything else.
_OPENING_RUN_RE = re.compile(r"[<＜]{3,}")
_CLOSING_RUN_RE = re.compile(r"[>＞]{3,}")


def opening(kind: str) -> str:
    return f"<<<BEGIN UNTRUSTED {kind}>>>"


def closing(kind: str) -> str:
    return f"<<<END UNTRUSTED {kind}>>>"


def without_markers(text: str) -> str:
    """`text` with every run of three or more angle brackets, which is
    what a marker is made of, turned into a single guillemet.

    Removing the markers found would not do: `<<<END UNTRUSTED <<<END
    UNTRUSTED X>>>SCREEN TEXT>>>` loses its inner marker and becomes a real
    one. What is rewritten is the bracket run itself, so nothing that is
    left can open or close a fence.
    """
    return _CLOSING_RUN_RE.sub("›", _OPENING_RUN_RE.sub("‹", text))


def close_cut_fences(cut: str) -> str:
    """`cut` with the closing marker put back on a fence a truncation left
    open, so what was fenced as data stays fenced downstream.

    A fence is open when its last opening marker comes after its last
    closing one.
    """
    for kind in KINDS:
        if cut.rfind(opening(kind)) > cut.rfind(closing(kind)):
            cut = f"{cut}\n{closing(kind)}"
    return cut

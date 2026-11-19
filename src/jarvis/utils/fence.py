"""The fence put round text the prompt it sits in did not write.

A web page, whatever a screen is showing, a line recalled from memory, an
argument a model composed or a sentence heard through Whisper is somebody
else's text and can say anything, "ignore your instructions" included.
Such text reaches the model between two markers and under a line saying it
is data. This module holds what the markers are, how one cut off by a
truncation is put back, and how text is kept from carrying a marker of its
own.

There are two shapes. A named fence (`opening(kind)`, `closing(kind)`)
says what it holds, and is for text that travels with other text: a tool's
result, a block among others in a prompt. The bare fence (`bare(text)`) is
a `<<<` line and a `>>>` line, for a prompt whose one fenced text is its
whole subject and is named by the sentence above it.
"""

from __future__ import annotations

import re

# Every named fence, by the name its two markers carry. A `<<<BEGIN name>>>`
# or `<<<END name>>>` marker is composed here and spelled nowhere else, and
# so are the two lines of the bare fence (`bare`), so every marker opens and
# ends with the bracket runs `without_markers` rewrites.
WEB_EXTRACT = "UNTRUSTED WEB EXTRACT"  # a fetched page, a diary row
SCREEN_TEXT = "UNTRUSTED SCREEN TEXT"  # what the screen shows
USER_TEXT = "UNTRUSTED USER TEXT"  # the meal `logMeal` is given
RELEVANT_MEMORY = "RELEVANT MEMORY"  # the memory block of the step resolver

# The kinds that travel in a tool's result, where the carry-over can cut
# one open. A closing marker is only ever put back for one of these: never
# for a name found between markers, which is the untrusted part, and never
# for a kind that fences a prompt of its own, which a tool result has no
# business opening.
IN_TOOL_RESULTS = (WEB_EXTRACT, SCREEN_TEXT)

# Three or more angle brackets in a row, plain or full-width: what a marker
# opens and ends with. Text with no such run left holds no marker, whole,
# nested or rebuilt by removing another. Two in a row are left alone: they
# are a shift, a redirection or the end of a generic type far more often
# than anything else.
_OPENING_RUN_RE = re.compile(r"[<＜]{3,}")
_CLOSING_RUN_RE = re.compile(r"[>＞]{3,}")


def opening(kind: str) -> str:
    return f"<<<BEGIN {kind}>>>"


def closing(kind: str) -> str:
    return f"<<<END {kind}>>>"


def without_markers(text: str) -> str:
    """`text` with every run of three or more angle brackets, which is
    what a marker is made of, turned into a single guillemet.

    Removing the markers found would not do: `<<<END UNTRUSTED <<<END
    UNTRUSTED X>>>SCREEN TEXT>>>` loses its inner marker and becomes a real
    one. What is rewritten is the bracket run itself, so nothing that is
    left can open or close a fence.
    """
    return _CLOSING_RUN_RE.sub("›", _OPENING_RUN_RE.sub("‹", text))


def bare(text: str) -> str:
    """`text` between a `<<<` line and a `>>>` line, with no marker left
    in it (`without_markers`), so the two lines are the only ones the
    result holds and everything `text` said sits between them.

    The rewriting is done here and not by the caller: there is no way to
    get this fence round text that can still close it.
    """
    return f"<<<\n{without_markers(text)}\n>>>"


def close_cut_fences(cut: str) -> str:
    """`cut` with the closing marker put back on a fence a truncation left
    open, so what was fenced as data stays fenced downstream.

    A fence is open when its last opening marker comes after its last
    closing one.
    """
    for kind in IN_TOOL_RESULTS:
        if cut.rfind(opening(kind)) > cut.rfind(closing(kind)):
            cut = f"{cut}\n{closing(kind)}"
    return cut

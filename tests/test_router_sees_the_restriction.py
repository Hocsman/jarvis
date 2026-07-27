"""What the tool router actually reads.

The router is given one line per tool: the name, and the first 120
characters of the description. Everything past that is invisible to it —
so a description whose opening sentence invites and whose restriction
comes second is, to the router, an unrestricted invitation.

That is not hypothetical. `remember`'s description used to open with
"Save something to long-term memory so it survives this conversation.
Call this ONLY when the user explicitly asks you to" — cut mid-word,
exactly before the clause that would have restrained it. In production a
user said "can't you write to me in the conversation?", the router
matched on the one word the two had in common, and the model filed her
own previous answer as a permanent fact about him.

The lesson generalises past that one tool: the first 120 characters have
to carry the discrimination, because they are all the router gets.
"""

from __future__ import annotations

import pytest

from src.jarvis.tools.registry import BUILTIN_TOOLS

def _router_slice(description: str) -> str:
    """Exactly what the router is shown, from the router's own code."""
    from src.jarvis.tools.selection import _router_summary

    return _router_summary(description)


def _seen(name: str) -> str:
    return _router_slice(BUILTIN_TOOLS[name].description)


# ── The write path into the user's memory ─────────────────────────────


def test_the_router_sees_that_remember_is_restricted():
    """If the restriction is past the cut, the router is choosing from a
    description that says only "save things"."""
    seen = _seen("remember").lower()

    assert "only" in seen or "ask" in seen, (
        f"The router reads {seen!r} and nothing in it says this tool is "
        f"for explicit requests."
    )


def test_the_word_that_matched_the_complaint_is_gone():
    """A single shared word put this tool in front of the model on a turn
    that needed no tool at all."""
    assert "conversation" not in _seen("remember").lower()


def test_the_router_slice_is_a_whole_thought():
    """Cut mid-word, the last clause the router reads is whatever
    survived the truncation — here, the opening half of a restriction
    that reversed its meaning."""
    seen = _seen("remember")

    assert seen.rstrip().endswith((".", "!", "?")), (
        f"The router's view of `remember` ends mid-sentence: …{seen[-40:]!r}"
    )


def test_forget_is_restricted_within_the_slice_too():
    """Its sibling writes to the same files and has the same failure
    shape."""
    seen = _seen("forget").lower()

    assert "only" in seen or "ask" in seen


# ── A drift guard over the whole catalogue ────────────────────────────


@pytest.mark.parametrize("name", sorted(BUILTIN_TOOLS))
def test_no_tool_is_described_to_the_router_mid_word(name: str):
    """A description cut mid-word tells the router something its author
    did not write. Where the full text is shorter than the cut there is
    nothing to check; where it is longer, the opening has to stand on its
    own."""
    full = BUILTIN_TOOLS[name].description
    seen = _router_slice(full)
    if len(full) <= len(seen):
        return

    assert seen.rstrip().endswith((".", "!", "?")), (
        f"{name}: the router reads …{seen[-50:]!r}, which stops mid-thought"
    )

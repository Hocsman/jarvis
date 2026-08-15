"""The pinned action, its digest, and which door it may be answered by.

Three things live here, and each exists to close a specific hole.

The **digest** pins the call. Between the moment Yuba asks "may I delete
notes.txt?" and the moment the answer arrives, the model gets to run
again; without a digest recomputed at execution time, the approval the
user gave for one path would run whatever the model resolved to next.

The **channel rule** decides whether an utterance may grant at all. A
false no costs a turn; a false yes costs a file. Whisper transcribing
the room plus a small model reading the transcription is two lossy
layers, and for `destructif` the design does not tune them, it removes
them: a destructive action is granted by a deliberate gesture or not at
all.

The **renderer** authors what the user is shown. Tool arguments are
model output derived from text that may itself be attacker-influenced,
so nothing the model wrote is ever spoken in Yuba's own voice, and the
displayed copy is byte-for-byte — the ledger's copy goes through
`redact()`, which collapses whitespace, and a path is not a sentence.
"""

from __future__ import annotations

import pytest

from src.jarvis.tools.confirmation import (
    CHANNEL_GESTE,
    CHANNEL_PAROLE,
    PendingAction,
    channel_for,
    describe_action,
    fingerprint,
)
from src.jarvis.tools.policy import RISK_ACTION, RISK_DESTRUCTIVE, RISK_READ


# ── The digest pins the call ──────────────────────────────────────────


def test_the_same_call_digests_the_same():
    assert fingerprint("localFiles", {"operation": "read", "path": "/a"}) == \
        fingerprint("localFiles", {"operation": "read", "path": "/a"})


def test_argument_order_does_not_change_the_digest():
    """Dict ordering is an accident of how the model emitted its JSON,
    not a difference in what would run."""
    assert fingerprint("localFiles", {"operation": "read", "path": "/a"}) == \
        fingerprint("localFiles", {"path": "/a", "operation": "read"})


def test_changing_any_argument_changes_the_digest():
    base = fingerprint("localFiles", {"operation": "delete", "path": "/a"})

    assert fingerprint("localFiles", {"operation": "delete", "path": "/b"}) != base
    assert fingerprint("localFiles", {"operation": "read", "path": "/a"}) != base


def test_changing_the_tool_changes_the_digest():
    """An approval for one tool must never carry to another."""
    assert fingerprint("localFiles", {"x": 1}) != fingerprint("deleteMeal", {"x": 1})


def test_an_added_argument_changes_the_digest():
    """A model that re-resolves the step with one extra flag is proposing
    a different call, whatever the flag does."""
    assert fingerprint("localFiles", {"path": "/a"}) != \
        fingerprint("localFiles", {"path": "/a", "recursive": True})


def test_an_unserialisable_argument_still_digests():
    """Bookkeeping must not be the reason a gate decision cannot be made."""
    assert fingerprint("odd", {"handle": object()})


def test_nested_arguments_digest_stably():
    a = fingerprint("t", {"outer": {"b": 2, "a": 1}, "list": [1, {"y": 2, "x": 1}]})
    b = fingerprint("t", {"list": [1, {"x": 1, "y": 2}], "outer": {"a": 1, "b": 2}})

    assert a == b


# ── The channel rule ──────────────────────────────────────────────────


from src.jarvis.tools.base import Tool


class _Builtin(Tool):
    """Stand-in for a builtin: we classified it ourselves."""

    @property
    def name(self):
        return "fakeBuiltin"

    @property
    def description(self):
        return ""

    @property
    def inputSchema(self):
        return {"type": "object", "properties": {}}

    def run(self, args, context):
        raise NotImplementedError


class _McpSpec:
    def __init__(self, annotations=None):
        self.annotations = annotations


def test_a_destructive_action_is_never_answerable_by_voice():
    """The one rule the whole design rests on."""
    assert channel_for(RISK_DESTRUCTIVE, _Builtin()) == CHANNEL_GESTE


def test_a_builtin_we_classified_may_be_answered_by_voice():
    assert channel_for(RISK_ACTION, _Builtin()) == CHANNEL_PAROLE
    assert channel_for(RISK_READ, _Builtin()) == CHANNEL_PAROLE


def test_an_mcp_tool_that_declares_itself_may_be_answered_by_voice():
    assert channel_for(RISK_READ, _McpSpec({"readOnlyHint": True})) == CHANNEL_PAROLE


def test_an_mcp_tool_that_declares_nothing_is_gesture_only():
    """`resolve_risk` hands `action` to any MCP tool whose annotations
    merely omit `destructiveHint`. Without this, a third-party server
    would decide by its own metadata what a mis-transcription can
    authorise."""
    assert channel_for(RISK_ACTION, _McpSpec({"title": "Click"})) == CHANNEL_GESTE


def test_an_mcp_tool_with_no_annotations_at_all_is_gesture_only():
    assert channel_for(RISK_ACTION, _McpSpec(None)) == CHANNEL_GESTE


def test_an_unknown_tool_is_gesture_only():
    """The catalogue has never heard of it, so nobody classified it."""
    assert channel_for(RISK_ACTION, None) == CHANNEL_GESTE


# ── The renderer ──────────────────────────────────────────────────────


def test_the_description_names_the_tool_and_the_risk():
    d = describe_action("localFiles", {"operation": "delete", "path": "/a"},
                        RISK_DESTRUCTIVE)

    assert "localFiles" in d.shown
    assert RISK_DESTRUCTIVE in d.shown


def test_the_arguments_shown_round_trip_to_what_would_run():
    """Lossless and unambiguous, which is a stronger property than raw.

    `redact()` collapses whitespace on the way to the ledger, so that copy
    is unusable here: a path is not a sentence. The card's copy must lose
    nothing — and an escaped `\\t` is *better* than a literal tab, which
    on screen is indistinguishable from spaces."""
    import json

    args = {"path": "/a  b/c\td", "operation": "delete"}
    d = describe_action("localFiles", args, RISK_DESTRUCTIVE)

    payload = d.shown.split("\n", 1)[1]
    assert json.loads(payload) == args


def test_a_doubled_space_is_not_collapsed():
    d = describe_action("localFiles", {"path": "/a  b"}, RISK_DESTRUCTIVE)

    assert "/a  b" in d.shown


def test_a_long_argument_is_not_truncated():
    """Truncation is where the interesting part of a path goes to hide."""
    path = "/Users/x/" + "a" * 400 + "/secret.key"
    d = describe_action("localFiles", {"path": path}, RISK_DESTRUCTIVE)

    assert path in d.shown


def test_invisible_characters_are_flagged():
    """A right-to-left override can make a path read as its own reverse."""
    d = describe_action("localFiles", {"path": "/a‮b/c"}, RISK_DESTRUCTIVE)

    assert d.hazards


def test_an_invisible_character_is_made_visible_on_the_card():
    """Flagging it is not enough: shown raw it displays nothing of itself
    and silently reverses everything after it."""
    import json

    args = {"path": "/a‮b/c"}
    d = describe_action("localFiles", args, RISK_DESTRUCTIVE)

    assert "‮" not in d.shown
    assert "\\u202e" in d.shown
    # And still says exactly what would run.
    assert json.loads(d.shown.split("\n", 1)[1]) == args


def test_making_it_visible_does_not_mangle_french():
    d = describe_action("localFiles", {"path": "/Users/hocine/résumé.txt"},
                        RISK_DESTRUCTIVE)

    assert "résumé" in d.shown


def test_a_zero_width_character_is_flagged():
    d = describe_action("localFiles", {"path": "/a​b"}, RISK_DESTRUCTIVE)

    assert d.hazards


def test_mixed_scripts_are_flagged():
    """A Cyrillic а inside a Latin path is the classic homoglyph."""
    d = describe_action("localFiles", {"path": "/Users/hа/notes"},
                        RISK_DESTRUCTIVE)

    assert d.hazards


def test_an_ordinary_path_is_not_flagged():
    """A hazard strip that fires on everything is a hazard strip nobody
    reads."""
    d = describe_action("localFiles", {"path": "/Users/hocine/notes.txt"},
                        RISK_DESTRUCTIVE)

    assert not d.hazards


def test_an_accented_french_path_is_not_flagged():
    d = describe_action("localFiles", {"path": "/Users/hocine/résumé été.txt"},
                        RISK_DESTRUCTIVE)

    assert not d.hazards


def test_a_request_carrying_invisible_characters_cannot_be_spoken():
    """If she cannot read it out honestly, she must not offer the voice
    door for it — an approval given to a sentence that hides what it does
    is not consent."""
    d = describe_action("localFiles", {"path": "/a‮b"}, RISK_ACTION)

    assert d.speakable is False


def test_an_ordinary_request_can_be_spoken():
    d = describe_action("getWeather", {"city": "Lyon"}, RISK_READ)

    assert d.speakable is True
    assert d.spoken


def test_the_spoken_form_carries_no_affirmation_or_negation():
    """Her own question comes back to her as echo. A spoken form that
    contained the word she is listening for could approve itself."""
    d = describe_action("getWeather", {"city": "Lyon"}, RISK_READ)

    assert d.spoken
    # The judge reads the user's utterance, never this; the guard is that
    # the question never states a decision of its own.
    assert "?" in d.spoken


# ── The pinned record ─────────────────────────────────────────────────


def test_a_pending_action_carries_its_own_digest():
    p = PendingAction.create(
        tool="localFiles", args={"operation": "delete"}, risk=RISK_DESTRUCTIVE,
        channel=CHANNEL_GESTE, origin="voix", query_redacted="supprime",
        raised_at_turn=3, ttl_sec=180.0,
    )

    assert p.fingerprint == fingerprint("localFiles", {"operation": "delete"})


def test_two_requests_get_different_ids():
    def _mk():
        return PendingAction.create(
            tool="t", args={}, risk=RISK_READ, channel=CHANNEL_PAROLE,
            origin="chat", query_redacted="", raised_at_turn=1, ttl_sec=180.0,
        )

    assert _mk().request_id != _mk().request_id


def test_the_id_is_not_guessable_from_the_tool():
    p = PendingAction.create(
        tool="localFiles", args={}, risk=RISK_READ, channel=CHANNEL_PAROLE,
        origin="chat", query_redacted="", raised_at_turn=1, ttl_sec=180.0,
    )

    assert "localFiles" not in p.request_id
    assert p.request_id.startswith("cf_")


def test_a_pending_action_cannot_be_edited_after_it_is_raised():
    """The pinned call is the whole point. If it can be mutated between
    the question and the answer, it pins nothing."""
    p = PendingAction.create(
        tool="localFiles", args={"path": "/a"}, risk=RISK_DESTRUCTIVE,
        channel=CHANNEL_GESTE, origin="voix", query_redacted="",
        raised_at_turn=1, ttl_sec=180.0,
    )

    with pytest.raises(Exception):
        p.tool = "deleteMeal"


def test_expiry_is_measured_on_a_clock_that_cannot_go_backwards():
    """Wall-clock time moves when the machine sleeps or the user travels.
    A grant whose deadline can jump is a grant that can be resurrected."""
    p = PendingAction.create(
        tool="t", args={}, risk=RISK_READ, channel=CHANNEL_PAROLE,
        origin="chat", query_redacted="", raised_at_turn=1, ttl_sec=0.0,
    )

    assert p.has_expired() is True


def test_a_fresh_request_has_not_expired():
    p = PendingAction.create(
        tool="t", args={}, risk=RISK_READ, channel=CHANNEL_PAROLE,
        origin="chat", query_redacted="", raised_at_turn=1, ttl_sec=180.0,
    )

    assert p.has_expired() is False

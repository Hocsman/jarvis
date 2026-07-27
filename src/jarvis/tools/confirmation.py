"""
🙋 Confirmation — turning a `demande` verdict into a question someone answers.

The gate knows a tool needs the user's say-so. This module is what a
question is made of: the call pinned so it cannot drift, the door the
answer may arrive through, and the words the user is shown.

Nothing here blocks and nothing here executes. The gate raises a
question and the turn ends; the answer arrives later, on its own thread,
through `daemon.resolve_confirmation` or through the next utterance. That
is not a stylistic choice: for voice, `run_reply_engine` runs on the
listener's own audio thread, so a gate that waited for a spoken answer
would silence the microphone that has to hear it.

See policy.spec.md for the full contract.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .policy import RISK_DESTRUCTIVE

# Which door an answer may arrive through.
CHANNEL_GESTE = "geste"   # a deliberate click on a card showing the call
CHANNEL_PAROLE = "parole"  # something the user said or typed, read by a model

# The user's decision, once made.
GRANTED = "oui"
DENIED = "non"
UNCLEAR = "flou"


# ─────────────────────────────────────────────────────────────────────────────
# Pinning the call
# ─────────────────────────────────────────────────────────────────────────────

def fingerprint(tool: str, args: Optional[Dict[str, Any]]) -> str:
    """A digest of exactly what would run.

    Recomputed at execution time and compared against the digest pinned
    when the question was asked. Between those two moments the model gets
    to run again — the planner re-resolves steps, the loop re-emits tool
    calls — so without this an approval given for one path would run
    whatever the model produced next under the same tool name.

    Canonical rather than a dict comparison: key order is an accident of
    how the model emitted its JSON, not a difference in what would run.
    """
    try:
        payload = json.dumps(
            {"tool": tool, "args": args or {}},
            sort_keys=True, ensure_ascii=False, default=str,
        )
    except Exception:
        # Never the reason a gate decision cannot be made. An unhashable
        # shape digests as its repr, which still changes when it changes.
        payload = f"{tool}:{args!r}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PendingAction:
    """One question, waiting. Never written to disk.

    A deletion proposed before a crash and approved after it is an
    approval given without the context that produced it, so a restart
    erases every pending request rather than resurrecting it.
    """

    request_id: str
    tool: str
    args: Dict[str, Any]
    fingerprint: str
    risk: str
    channel: str
    origin: Optional[str]
    query_redacted: str
    raised_at_turn: int
    ttl_sec: float
    created_monotonic: float = field(default_factory=time.monotonic)

    @classmethod
    def create(
        cls, *, tool: str, args: Optional[Dict[str, Any]], risk: str,
        channel: str, origin: Optional[str], query_redacted: str,
        raised_at_turn: int, ttl_sec: float,
    ) -> "PendingAction":
        return cls(
            request_id=f"cf_{uuid.uuid4().hex[:10]}",
            tool=tool,
            args=dict(args or {}),
            fingerprint=fingerprint(tool, args),
            risk=risk,
            channel=channel,
            origin=origin,
            query_redacted=query_redacted,
            raised_at_turn=raised_at_turn,
            ttl_sec=float(ttl_sec),
        )

    def has_expired(self) -> bool:
        """Measured on the monotonic clock.

        Wall-clock time moves when the machine sleeps or the user
        travels; a deadline that can jump backwards is a grant that can
        be resurrected.
        """
        return (time.monotonic() - self.created_monotonic) >= self.ttl_sec

    def matches(self, tool: str, args: Optional[Dict[str, Any]]) -> bool:
        """Whether the call about to run is the one that was approved."""
        return self.fingerprint == fingerprint(tool, args)


@dataclass(frozen=True)
class Approval:
    """A grant, good for exactly one execution."""

    request_id: str
    fingerprint: str


# ─────────────────────────────────────────────────────────────────────────────
# Which door
# ─────────────────────────────────────────────────────────────────────────────

def risk_is_declared(tool: Any) -> bool:
    """Whether this tool's risk was classified by someone we trust.

    True for a builtin, which we classified ourselves, and for an MCP
    tool that ships an explicit `readOnlyHint` or `destructiveHint`.

    False for everything else — including an MCP tool whose annotations
    exist but mention neither hint, which `resolve_risk` hands `action`.
    Without this distinction a third-party server would decide, by its
    own metadata, what a mis-transcription is allowed to authorise.
    """
    from .base import Tool

    if isinstance(tool, Tool):
        return True

    annotations = getattr(tool, "annotations", None)
    if isinstance(annotations, dict):
        return "destructiveHint" in annotations or "readOnlyHint" in annotations
    return False


def channel_for(risk: str, tool: Any) -> str:
    """The only door this action may be answered through.

    `destructif` is a gesture, always. A false no costs a turn; a false
    yes costs a file. Whisper transcribing a room and a small model
    reading that transcription are two lossy layers in front of an
    irreversible action, and here the design does not tune them, it
    removes them.

    Anything else may be answered by voice, but only when its risk was
    declared by us or by an explicit hint — an unclassified tool arrives
    at the gate as destructive anyway, and one that merely says nothing
    should not be trusted to have said the right nothing.
    """
    if risk == RISK_DESTRUCTIVE:
        return CHANNEL_GESTE
    return CHANNEL_PAROLE if risk_is_declared(tool) else CHANNEL_GESTE


# ─────────────────────────────────────────────────────────────────────────────
# What the user is shown
# ─────────────────────────────────────────────────────────────────────────────

# Characters that can make a string read as something other than what it
# is: bidirectional overrides, zero-width joiners and spaces, and the
# other formatting controls.
_INVISIBLE = re.compile(r"[​-‏‪-‮⁠-⁤⁪-⁯﻿]")

_LATIN = re.compile(r"[A-Za-z]")


def _script_of(ch: str) -> Optional[str]:
    """A coarse script name for a letter, or None for anything else."""
    if not ch.isalpha():
        return None
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return None
    return name.split(" ")[0]


def _escape_invisibles(text: str) -> str:
    """Make characters that show nothing of themselves visible as escapes."""
    return _INVISIBLE.sub(lambda m: f"\\u{ord(m.group()):04x}", text)


def _hazards_in(text: str) -> list:
    """Ways this string could read as something it is not."""
    found = []
    if _INVISIBLE.search(text):
        found.append("caractères invisibles")

    scripts = {s for s in (_script_of(c) for c in text) if s}
    # Accented French is LATIN throughout; a Cyrillic а among Latin
    # letters is the classic homoglyph and shows up as a second script.
    if len(scripts) > 1:
        found.append(f"écritures mélangées ({', '.join(sorted(scripts))})")
    return found


@dataclass(frozen=True)
class ActionDescription:
    """Everything any surface needs to present one question.

    Authored here, by code. Tool arguments are chat-model output derived
    from text that may itself be attacker-influenced — `webSearch`
    already fences web content as data, which is an admission that pages
    reach the model — so nothing the model wrote is ever relayed as
    Yuba's own words.
    """

    tool: str
    risk: str
    shown: str        # for the card: byte-for-byte, never truncated
    spoken: str       # for TTS: short, and stating no decision of its own
    hazards: list
    speakable: bool


def describe_action(
    tool: str, args: Optional[Dict[str, Any]], risk: str,
) -> ActionDescription:
    """Render one pending action for every surface that shows it."""
    try:
        rendered = json.dumps(args or {}, ensure_ascii=False, default=str,
                              sort_keys=True)
    except Exception:
        rendered = repr(args)

    hazards = _hazards_in(rendered)

    # Lossless and unambiguous, which is stronger than raw. Nothing is
    # truncated, no whitespace is collapsed and no Unicode is folded —
    # the ledger's copy goes through `redact()`, which collapses
    # whitespace, and a path is not a sentence.
    #
    # But `ensure_ascii=False` leaves a right-to-left override raw, and
    # raw is precisely how it lies: on screen it reverses the text after
    # it and shows nothing of itself. Those characters, and only those,
    # are escaped so the user sees that something is there. Accented
    # French stays readable, and `\uXXXX` is valid JSON, so the card's
    # copy still parses back to exactly what would run.
    shown = f"{tool} · {risk}\n{_escape_invisibles(rendered)}"
    speakable = not hazards

    # No affirmation and no negation in any language: her own question
    # comes back to her as echo, and a spoken form that contained the
    # word the judge is listening for could approve itself.
    spoken = f"{tool} ?"

    return ActionDescription(
        tool=tool, risk=risk, shown=shown, spoken=spoken,
        hazards=hazards, speakable=speakable,
    )

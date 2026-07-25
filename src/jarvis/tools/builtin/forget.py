"""Forget tool — drop one entry from the core.

The counterpart to ``remember``. Without it the assistant can only
retire a belief by asserting a replacement, so "forget where I live"
either invents a negation that adds more personal data than it removes,
or is answered with a "done" that changes nothing.

Matching is deliberately conservative. The entries are visible to the
model in its system prompt, so it can quote one; when it paraphrases
instead, a single clear candidate is dropped and anything less certain
is handed back rather than guessed at. Retiring the wrong belief is
worse than retiring none, because the user is told it worked.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from ...debug import debug_log
from ...memory.core import (
    SECTION_PROFILE,
    SECTION_RULES,
    CoreEntry,
    MemoryCore,
)
from ..base import Tool, ToolContext
from ..types import ToolExecutionResult

# How much of an ENTRY the target must account for before that entry is a
# candidate. Scoring against the entry, rather than against whichever side
# is shorter, is what stops a single shared word from reading as a match:
# "tout" would otherwise score a perfect 1.0 against "Il mange de tout."
# and drop a dietary fact when the user asked to wipe everything, and
# "son adresse" would land on "Son adresse mail est ..." rather than the
# postal address a person means by it. Both were reproduced against this
# tool before the denominator changed.
#
# The cost is that a bare topic word no longer drops anything. That is the
# right trade: the entries are in the model's context, so it can name one,
# and a near-miss now comes back with the entries listed so the next call
# lands. A missed drop costs a turn; a wrong drop costs a memory the user
# was told was safe.
_MATCH_THRESHOLD = 0.6

# Word extraction, not language detection: ``\w{3,}`` with re.UNICODE
# keeps accented and non-Latin scripts intact and drops the short
# function words that would otherwise match everything. Same approach as
# the recall gate.
_WORD_RE = re.compile(r"\w{3,}", re.UNICODE)

# How many candidates to name back when the request is ambiguous. Enough
# for the model to repeat the call with the right wording, bounded so a
# vague target cannot flood the prompt with the whole core.
_MAX_CANDIDATES = 6


def _words(text: str) -> set:
    return set(_WORD_RE.findall((text or "").lower()))


def _score(target_words: set, entry_words: set) -> float:
    """How much of the entry the target accounts for, from 0 to 1."""
    if not target_words or not entry_words:
        return 0.0
    shared = len(target_words & entry_words)
    if not shared:
        return 0.0
    return shared / len(entry_words)


def _candidates(
    core: MemoryCore, target: str, sections: List[str],
) -> List[Tuple[str, CoreEntry]]:
    """Return the active entries that plausibly match ``target``.

    An exact match short-circuits: when the model quotes an entry back
    verbatim there is nothing to weigh up.
    """
    target_words = _words(target)
    normalised = " ".join(target.split()).casefold()

    scored: List[Tuple[float, str, CoreEntry]] = []
    for section in sections:
        for entry in core.active(section):
            if " ".join(entry.text.split()).casefold() == normalised:
                return [(section, entry)]
            score = _score(target_words, _words(entry.text))
            if score >= _MATCH_THRESHOLD:
                scored.append((score, section, entry))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [(section, entry) for _score_, section, entry in scored]


def _held_entries(core: MemoryCore) -> List[str]:
    """Every active entry, so a near-miss can be answered with the real
    wording instead of a dead end."""
    return [
        entry.text
        for section in (SECTION_PROFILE, SECTION_RULES)
        for entry in core.active(section)
    ]


class ForgetTool(Tool):
    """Drop something the user no longer wants the assistant to hold."""

    @property
    def name(self) -> str:
        return "forget"

    @property
    def description(self) -> str:
        return (
            "Remove something from long-term memory when the user asks you "
            "to forget it, or tells you a rule no longer applies. Use this "
            "instead of saving a negation: 'he no longer lives in Paris' "
            "adds information, it does not remove any. The entries you hold "
            "are listed in your context, so pass the one the user means "
            "exactly as it appears there. Drops one entry per call; if the "
            "user wants several gone, call it once for each. Never call it "
            "on your own initiative."
        )

    @property
    def inputSchema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": (
                        "The entry to drop, worded as it appears in the "
                        "memory sections of your context. Pass the whole "
                        "entry, not a topic word from it: nothing is dropped "
                        "unless the wording accounts for a stored entry, and "
                        "a near miss comes back with the entries listed so "
                        "you can call again with one of them."
                    ),
                },
            },
            "required": ["target"],
        }

    def run(
        self, args: Optional[Dict[str, Any]], context: ToolContext,
    ) -> ToolExecutionResult:
        raw_target = args.get("target") if isinstance(args, dict) else None
        target = str(raw_target).strip() if raw_target else ""
        if not target:
            debug_log("    ⚠️ forget called without a target", "tools")
            return ToolExecutionResult(
                success=False,
                reply_text=(
                    "Nothing was dropped: no entry was named. Ask the user "
                    "which of the things you know they want removed."
                ),
            )

        core = MemoryCore.for_config(context.cfg)

        try:
            matches = _candidates(core, target, [SECTION_PROFILE, SECTION_RULES])
        except OSError as e:
            debug_log(f"    ⚠️ forget could not read the core: {e}", "tools")
            return ToolExecutionResult(
                success=False,
                reply_text=(
                    f"Nothing was dropped: the memory file could not be read "
                    f"({e}). Tell the user their memory is unchanged."
                ),
            )

        if not matches:
            debug_log(f"    🪨 forget: nothing matches '{target[:60]}'", "tools")
            held = _held_entries(core)
            if not held:
                return ToolExecutionResult(
                    success=False,
                    reply_text=(
                        "Nothing was dropped: there is nothing in long-term "
                        "memory yet. Tell the user you hold nothing about "
                        "them, rather than claiming to have forgotten it."
                    ),
                )
            listed = "; ".join(f'"{text}"' for text in held[:_MAX_CANDIDATES])
            return ToolExecutionResult(
                success=False,
                reply_text=(
                    f'Nothing was dropped: "{target}" does not account for any '
                    f"stored entry, and dropping one on a partial word match "
                    f"would risk the wrong memory. What is held: {listed}. If "
                    f"one of those is what the user meant, call forget again "
                    f"with that exact wording; otherwise tell them you hold "
                    f"nothing like that."
                ),
            )

        if len(matches) > 1:
            listed = "; ".join(f'"{entry.text}"' for _s, entry in matches[:_MAX_CANDIDATES])
            debug_log(
                f"    🪨 forget: {len(matches)} candidates for '{target[:40]}'",
                "tools",
            )
            return ToolExecutionResult(
                success=False,
                reply_text=(
                    f'Nothing was dropped: "{target}" could mean more than one '
                    f"stored entry, and dropping the wrong one is worse than "
                    f"dropping none. Candidates: {listed}. Ask the user which "
                    f"they mean, then call forget again with that exact wording."
                ),
            )

        section, entry = matches[0]
        try:
            dropped = core.retire(
                section, entry.text, reason="oublié à la demande de l'utilisateur",
            )
        except OSError as e:
            debug_log(f"    ⚠️ forget write failed: {e}", "tools")
            context.user_print("⚠️ Impossible d'écrire dans la mémoire.")
            return ToolExecutionResult(
                success=False,
                reply_text=(
                    f"Nothing was dropped: the memory file could not be "
                    f"written ({e}). Tell the user their memory is unchanged."
                ),
            )

        if not dropped:
            return ToolExecutionResult(
                success=False,
                reply_text=(
                    f'Nothing was dropped: "{entry.text}" is no longer active '
                    f"in memory."
                ),
            )

        label = "rule" if section == SECTION_RULES else "fact"
        context.user_print(f"🪨 Oublié : {entry.text}")
        debug_log(f"    🪨 forget: dropped {label} — '{entry.text[:60]}'", "tools")

        return ToolExecutionResult(
            success=True,
            reply_text=(
                f'Dropped from long-term memory: "{entry.text}" ({label}). '
                f"It will not reach any future conversation. The line stays "
                f"struck through in the user's file so they can see what "
                f"changed."
            ),
        )

"""Remember tool — the only write path into the core.

The core is the assistant's readable memory of its user (see
``memory/core.spec.md``). Nothing infers its way in: an entry appears
because the user asked for it, or because they corrected something.
"""

from typing import Any, Dict, Optional

from ...debug import debug_log
from ...memory.core import SECTION_PROFILE, SECTION_RULES, MemoryCore
from ...utils.redact import contains_redaction_placeholder
from ..base import Tool, ToolContext
from ..types import ToolExecutionResult


class RememberTool(Tool):
    """Record a fact about the user, or a rule they want followed."""

    def risk_for(self, args):
        """Writes, but only into Yuba's own files and database: text the
        user can reopen and correct by hand, never their machine and
        never anything outward."""
        from ..policy import RISK_READ

        return RISK_READ

    @property
    def name(self) -> str:
        return "remember"

    @property
    def description(self) -> str:
        # The router is shown whole sentences, within a budget — so the
        # opening ones have to carry the restriction. An earlier wording
        # led with "Save something to long-term memory so it survives
        # this conversation", and that is all the router saw: an
        # unrestricted invitation, matching a user who had said
        # "conversation" while complaining about something else entirely.
        return (
            "Store a lasting fact about the user, or a standing "
            "instruction, ONLY when they explicitly ask for it "
            "('remember that...', 'from now on...', 'note that I...') or "
            "when they CORRECT something you believe about them. Anything "
            "they want right now — said, written out, shown, repeated, "
            "looked up — is not this tool. Never call it on your own "
            "initiative: do not save conclusions you drew, advice you "
            "gave, what was discussed, or anything the user did not state "
            "about themselves. A wrong memory is worse than no memory. "
            "Repeat the user's own statement rather than paraphrasing it."
        )

    @property
    def inputSchema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": (
                        "What to remember, as a self-contained statement, "
                        "written in the SAME LANGUAGE the user just spoke. "
                        "Never translate it: the user reads these entries in "
                        "a file of their own and corrects them by hand. For a "
                        "fact, write about the user in the third person ('He "
                        "is vegetarian' / 'Il est végétarien'). For a rule, "
                        "write it as an instruction ('Always answer in "
                        "French' / 'Toujours répondre en français')."
                    ),
                },
                "kind": {
                    "type": "string",
                    "enum": ["fact", "rule"],
                    "description": (
                        "'fact' when it describes the user, 'rule' when it "
                        "tells you how to behave. Defaults to 'fact'."
                    ),
                },
                "replaces": {
                    "type": "string",
                    "description": (
                        "OPTIONAL. Only when this corrects something already "
                        "in memory: the outdated statement, worded as it is "
                        "currently stored. It stops being used but stays "
                        "visible to the user."
                    ),
                },
            },
            "required": ["text"],
        }

    def run(
        self, args: Optional[Dict[str, Any]], context: ToolContext,
    ) -> ToolExecutionResult:
        if not isinstance(args, dict):
            debug_log("    ⚠️ remember called with no arguments", "tools")
            return ToolExecutionResult(
                success=False,
                reply_text=(
                    "Nothing was saved: no text was given. Ask the user what "
                    "they want you to remember."
                ),
            )

        raw_text = args.get("text")
        text = str(raw_text).strip() if raw_text else ""
        if not text:
            debug_log("    ⚠️ remember called with empty text", "tools")
            return ToolExecutionResult(
                success=False,
                reply_text=(
                    "Nothing was saved: the text was empty. Ask the user what "
                    "they want you to remember."
                ),
            )

        # The model only ever sees redacted text, so "remember my email
        # is x@y.com" arrives here as "[REDACTED_EMAIL]". Storing that
        # keeps nothing worth having and tells the user their email was
        # saved, which is the one answer worse than refusing.
        if contains_redaction_placeholder(text):
            debug_log("    ⚠️ remember refused a redacted value", "tools")
            context.user_print("🪨 Non enregistré : donnée sensible.")
            return ToolExecutionResult(
                success=False,
                reply_text=(
                    "Nothing was saved. That statement's substance was a "
                    "sensitive value (an address, a card, a key, a password) "
                    "which is removed before it ever reaches you, so storing "
                    "it would keep only a placeholder. Tell the user plainly "
                    "that you do not keep sensitive values in memory, and "
                    "offer to remember the part that is not sensitive."
                ),
            )

        # Anything that isn't explicitly a rule is a fact about the user.
        # Small models reach for invented enum values often enough that
        # failing on one would cost the user a memory over a typo.
        kind = str(args.get("kind") or "").strip().lower()
        section = SECTION_RULES if kind == "rule" else SECTION_PROFILE

        raw_replaces = args.get("replaces")
        replaces = str(raw_replaces).strip() if raw_replaces else ""

        core = MemoryCore.for_config(context.cfg)

        retired = False
        if replaces:
            try:
                # The correction may move an entry across sections (a
                # misfiled rule corrected into a fact), so if it isn't
                # where this call says it should be, look in the other half
                # before giving up.
                other = SECTION_PROFILE if section == SECTION_RULES else SECTION_RULES
                retired = core.retire(
                    section, replaces, reason="corrigé par l'utilisateur",
                ) or core.retire(
                    other, replaces, reason="corrigé par l'utilisateur",
                )
            except OSError as e:
                debug_log(f"    ⚠️ core retire failed: {e}", "tools")
                context.user_print("⚠️ Impossible d'écrire dans la mémoire.")
                return ToolExecutionResult(
                    success=False,
                    reply_text=(
                        f"Nothing was saved: the memory file could not be "
                        f"written ({e}). Tell the user their memory was not "
                        f"updated."
                    ),
                )

        try:
            result = core.remember(section, text, corrected=bool(replaces))
        except (OSError, ValueError) as e:
            debug_log(f"    ⚠️ core write failed: {e}", "tools")
            context.user_print("⚠️ Impossible d'écrire dans la mémoire.")
            return ToolExecutionResult(
                success=False,
                reply_text=(
                    f"Nothing was saved: the memory file could not be written "
                    f"({e}). Tell the user their memory was not updated."
                ),
            )

        label = "rule" if section == SECTION_RULES else "fact"

        if result.already_known:
            # Echo the entry as stored, not as the model just phrased it —
            # what the user should see is what is actually in their file.
            known = result.entry.text
            context.user_print(f"🪨 Déjà en mémoire : {known}")
            debug_log(f"    🪨 remember: already known — '{known[:60]}'", "tools")
            return ToolExecutionResult(
                success=True,
                reply_text=(
                    f"Already in long-term memory, nothing changed: "
                    f'"{known}" ({label}).'
                ),
            )

        context.user_print(f"🪨 Retenu : {text}")
        debug_log(f"    🪨 remember: stored {label} — '{text[:60]}'", "tools")

        lines = [f'Saved to long-term memory as a {label}: "{text}".']
        if replaces:
            if retired:
                lines.append(f'It replaces the previous entry: "{replaces}".')
            else:
                lines.append(
                    f'No stored entry matched "{replaces}", so nothing was '
                    f"replaced — the new entry was still saved."
                )
        lines.append("It will be available in every future conversation.")

        return ToolExecutionResult(success=True, reply_text=" ".join(lines))

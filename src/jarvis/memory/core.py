"""
🪨 Core — the assistant's readable, portable memory.

Two Markdown files the user can open, read, correct, and back up: what the
assistant knows about them (``profil.md``) and how they have told it to
behave (``regles.md``). Models are swappable; the core is what stays.

Only two things write here: an explicit request from the user ("remember
that...") and a correction. Nothing is inferred, and nothing is erased —
superseding an entry strikes it through and leaves it in place.

See core.spec.md for the full contract.
"""

from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Union

from ..debug import debug_log

# The two halves of the core. Each maps to one file and one labelled
# section of the system prompt.
SECTION_PROFILE = "profile"
SECTION_RULES = "rules"

# How an entry came to be known. Written into the file for the user's
# benefit; never shown to the model.
SOURCE_SAID = "dit"
SOURCE_CORRECTED = "corrigé"
SOURCE_MIGRATED = "migré"

CORE_DIRNAME = "yuba"

_FILENAMES = {
    SECTION_PROFILE: "profil.md",
    SECTION_RULES: "regles.md",
}

# The preamble written when a file is first created. Deliberately in the
# user's language and addressed to them: these files exist to be opened
# and edited by hand, so they explain themselves.
_HEADERS = {
    SECTION_PROFILE: (
        "# Profil\n"
        "\n"
        "<!--\n"
        "  Ce que l'assistant sait de toi.\n"
        "\n"
        "  Rien n'est écrit ici par déduction. Une entrée apparaît quand tu\n"
        "  le demandes (« souviens-toi que... ») ou quand tu corriges une\n"
        "  erreur, jamais parce que quelque chose a été supposé.\n"
        "\n"
        "  Format d'une ligne : - AAAA-MM-JJ · source : texte\n"
        "  Une ligne barrée (~~...~~) est une entrée retirée : elle n'est\n"
        "  plus utilisée, et reste visible pour garder la trace de ce qui a\n"
        "  été cru, quand, et pourquoi ça a changé.\n"
        "\n"
        "  Ce fichier est à toi. Édite-le à la main quand tu veux : toute\n"
        "  ligne commençant par « - » est lue comme une entrée, même sans\n"
        "  date, et rien de ce que tu écris ici n'est réécrit ou supprimé.\n"
        "-->\n"
    ),
    SECTION_RULES: (
        "# Règles\n"
        "\n"
        "<!--\n"
        "  Les consignes que tu as données à l'assistant : ton, langue,\n"
        "  style, ce qu'il faut faire et ne pas faire. Elles s'appliquent à\n"
        "  chaque réponse, sans que tu aies à les rappeler.\n"
        "\n"
        "  Rien n'est écrit ici par déduction. Une règle apparaît quand tu\n"
        "  la donnes explicitement ou quand tu corriges un comportement.\n"
        "\n"
        "  Format d'une ligne : - AAAA-MM-JJ · source : texte\n"
        "  Une ligne barrée (~~...~~) est une règle retirée : elle n'est\n"
        "  plus suivie, et reste visible pour garder la trace de ce qui a\n"
        "  été demandé, quand, et pourquoi ça a changé.\n"
        "\n"
        "  Ce fichier est à toi. Édite-le à la main quand tu veux : toute\n"
        "  ligne commençant par « - » est lue comme une entrée, même sans\n"
        "  date, et rien de ce que tu écris ici n'est réécrit ou supprimé.\n"
        "-->\n"
    ),
}

# Callbacks fired after a successful write. The daemon registers one to
# drop the conversation-scoped profile cache, so a fact remembered
# mid-conversation is in the prompt on the very next turn instead of the
# next conversation.
_MUTATION_LISTENERS: list[Callable[..., None]] = []


def register_core_mutation_listener(cb: Callable[..., None]) -> None:
    """Register a callback invoked after every core write.

    The callback receives keyword arguments ``action`` (``"remember"`` or
    ``"retire"``) and ``section``.
    """
    if cb not in _MUTATION_LISTENERS:
        _MUTATION_LISTENERS.append(cb)


def unregister_core_mutation_listener(cb: Callable[..., None]) -> None:
    """Remove a previously registered mutation listener (idempotent)."""
    try:
        _MUTATION_LISTENERS.remove(cb)
    except ValueError:
        pass


def _notify_core_mutation(action: str, section: str) -> None:
    for cb in list(_MUTATION_LISTENERS):
        try:
            cb(action=action, section=section)
        except Exception as exc:
            debug_log(f"core mutation listener failed (non-fatal): {exc}", "memory")


_DATE_RE = r"\d{4}-\d{2}-\d{2}"

# The date and attribution an entry carries: "2026-07-25 · dit : ".
_PREFIX_RE = re.compile(
    rf"^(?P<date>{_DATE_RE})\s+·\s+(?P<source>\S+)\s*:\s*(?P<text>.+?)\s*$"
)

# An active entry: "- 2026-07-25 · dit : Il s'appelle Hocine."
_ACTIVE_RE = re.compile(rf"^-\s+{_PREFIX_RE.pattern[1:]}")

# A retired one. The strikethrough alone is what retires an entry — the
# header in every core file tells the user exactly that, and striking a
# line out is the obvious way to drop a belief when editing by hand. The
# stamp that follows is bookkeeping the assistant adds; a line without it
# is just as retired.
_STRUCK_RE = re.compile(r"^-\s+~~(?P<inner>.+?)~~\s*(?P<stamp>.*?)\s*$")

# "· retiré le 2026-07-25 : corrigé par l'utilisateur", reason optional.
_STAMP_RE = re.compile(
    rf"^·?\s*retiré le\s+(?P<retired_on>{_DATE_RE})\s*(?::\s*(?P<reason>.*?))?\s*$"
)

# Anything else starting with a bullet. Hand-written entries land here:
# the text is whatever follows the bullet, with no date or source.
_LOOSE_RE = re.compile(r"^-\s+(?P<text>\S.*?)\s*$")


@dataclass(frozen=True)
class CoreEntry:
    """One line of the core, as it currently stands on disk."""

    text: str
    date: Optional[str] = None
    source: Optional[str] = None
    retired: bool = False
    retired_on: Optional[str] = None
    retired_reason: Optional[str] = None
    # The line exactly as written, so a rewrite can find and replace it
    # without reformatting anything the user typed themselves.
    raw: str = ""


@dataclass(frozen=True)
class RememberResult:
    """Outcome of a write, so callers can tell a new fact from a known one."""

    entry: CoreEntry
    already_known: bool


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _normalise(text: str) -> str:
    """Fold a fact down to its comparable shape for duplicate detection.

    ``NFKC`` + ``casefold`` rather than ``lower`` because the core holds
    whatever language the user speaks: casefold handles German ß and
    Turkish dotted capitals correctly, and NFKC collapses code points
    that render identically (a precomposed é against e + combining
    accent), which a hand-edited file will contain sooner or later.
    """
    return " ".join(unicodedata.normalize("NFKC", text).split()).casefold()


def _flatten(text: str) -> str:
    """Squash a fact onto a single line and strip the strikethrough marker.

    Entries are one per line, so embedded newlines would silently split a
    fact in two on the next read. A literal ``~~`` inside the text would
    make the line parse as retired, so it goes too.
    """
    return " ".join(text.replace("~~", "").split())


def _parse_line(line: str) -> Optional[CoreEntry]:
    """Read one line of a core file, or return None if it holds no entry."""
    match = _STRUCK_RE.match(line)
    if match:
        inner = match.group("inner").strip()
        prefix = _PREFIX_RE.match(inner)
        # The date and attribution survive when they are there, but a
        # user who struck out their own bare line still gets an entry.
        date = prefix.group("date") if prefix else None
        source = prefix.group("source") if prefix else None
        text = prefix.group("text").strip() if prefix else inner

        stamp = _STAMP_RE.match(match.group("stamp"))
        reason = (stamp.group("reason") or "").strip() if stamp else ""
        return CoreEntry(
            text=text,
            date=date,
            source=source,
            retired=True,
            retired_on=stamp.group("retired_on") if stamp else None,
            retired_reason=reason or None,
            raw=line,
        )

    match = _ACTIVE_RE.match(line)
    if match:
        return CoreEntry(
            text=match.group("text").strip(),
            date=match.group("date"),
            source=match.group("source"),
            raw=line,
        )

    match = _LOOSE_RE.match(line)
    if match:
        # A line the user wrote in their own shape. Believed as-is, with
        # no date and no attribution to invent.
        return CoreEntry(text=match.group("text").strip(), raw=line)

    return None


def _render_active(text: str, date: str, source: str) -> str:
    return f"- {date} · {source} : {text}"


def _render_retired(entry: CoreEntry, on_date: str, reason: str) -> str:
    date = entry.date or "?"
    source = entry.source or SOURCE_SAID
    stamp = f" · retiré le {on_date}"
    if reason.strip():
        stamp += f" : {reason.strip()}"
    return f"- ~~{date} · {source} : {entry.text}~~{stamp}"


class MemoryCore:
    """Read and write the core's two Markdown files.

    Reads fail open (an unreadable file is an empty section, never an
    exception into the reply path); writes fail loudly, because a user
    told "noted" about something that was never saved is worse than an
    error message.
    """

    def __init__(self, directory: Union[str, Path]):
        self.directory = Path(directory).expanduser()

    @classmethod
    def for_config(cls, cfg) -> "MemoryCore":
        """Locate the core beside the configured database."""
        db_path = Path(str(getattr(cfg, "db_path", "")) or "jarvis.db").expanduser()
        return cls(db_path.parent / CORE_DIRNAME)

    # ── Reading ──────────────────────────────────────────────────────

    def path_for(self, section: str) -> Path:
        try:
            return self.directory / _FILENAMES[section]
        except KeyError:
            raise ValueError(f"unknown core section: {section!r}") from None

    def _read_lines_strict(self, section: str) -> list[str]:
        """Read a section, raising if a file that exists cannot be read.

        Writers use this: rewriting a file whose current contents could
        not be read would silently discard everything in it, which is the
        one thing the core must never do.
        """
        path = self.path_for(section)
        if not path.exists():
            return []
        return path.read_text(encoding="utf-8").splitlines()

    def _read_lines(self, section: str) -> list[str]:
        path = self.path_for(section)
        try:
            return self._read_lines_strict(section)
        except OSError as e:
            debug_log(f"core read failed for {path.name} (non-fatal): {e}", "memory")
            return []

    def entries(self, section: str) -> list[CoreEntry]:
        """Every entry in the section, active and retired, in file order."""
        parsed = [_parse_line(line) for line in self._read_lines(section)]
        return [entry for entry in parsed if entry is not None]

    def active(self, section: str) -> list[CoreEntry]:
        """The entries the assistant currently believes, in file order."""
        return [entry for entry in self.entries(section) if not entry.retired]

    def knows(self, section: str, text: str) -> bool:
        """Whether the section holds this text at all, retired included.

        Distinct from a duplicate check against the active entries: a
        bulk import must not resurrect something the user has retired,
        whereas the user asking to remember it again should.
        """
        key = _normalise(_flatten(text))
        return any(_normalise(entry.text) == key for entry in self.entries(section))

    # ── Writing ──────────────────────────────────────────────────────

    def _write(self, section: str, content: str) -> None:
        """Replace a core file atomically, leaving no debris on failure."""
        path = self.path_for(section)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        try:
            tmp.write_text(content, encoding="utf-8")
            os.replace(tmp, path)
        except OSError:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    def write_raw(self, section: str, content: str) -> None:
        """Replace a section's file with exactly this text.

        For surfaces where the user edits the file itself rather than
        asking the assistant to. Nothing is reformatted and no
        unrecognised line is dropped: an editor that rewrites what you
        typed is one you stop trusting with the thing it holds.
        """
        self._write(section, content)
        debug_log(f"core: {section} rewritten by hand", "memory")
        _notify_core_mutation("edit", section)

    def fingerprint(self) -> tuple:
        """A cheap stamp of the files' current state.

        Callers that cache the injected block hold this alongside it and
        rebuild when it changes. Without it an edit made outside the
        running process — from the memory viewer, or from the user's own
        text editor — appears to do nothing until the conversation ends,
        which reads exactly like the correction was ignored.
        """
        stamp = []
        for section in (SECTION_PROFILE, SECTION_RULES):
            path = self.path_for(section)
            try:
                stat = path.stat()
                stamp.append((section, stat.st_mtime_ns, stat.st_size))
            except OSError:
                stamp.append((section, None, None))
        return tuple(stamp)

    def remember(
        self,
        section: str,
        text: str,
        *,
        on_date: Optional[str] = None,
        corrected: bool = False,
        source: Optional[str] = None,
    ) -> RememberResult:
        """Append a fact or rule, unless it is already known.

        ``corrected`` marks the entry as coming from the user putting the
        assistant right, which is recorded in the file so the user can see
        which of their statements were fixes.
        """
        flat = _flatten(text)
        if not flat:
            raise ValueError("refusing to remember empty text")

        key = _normalise(flat)
        existing = self._read_lines_strict(section)
        for line in existing:
            entry = _parse_line(line)
            if entry is None or entry.retired:
                continue
            if _normalise(entry.text) == key:
                debug_log(f"core: '{flat[:60]}' already known, not rewriting", "memory")
                return RememberResult(entry=entry, already_known=True)

        stamp = on_date or _today_utc()
        attribution = source or (SOURCE_CORRECTED if corrected else SOURCE_SAID)
        line = _render_active(flat, stamp, attribution)

        if existing:
            body = "\n".join(existing).rstrip("\n")
            content = f"{body}\n{line}\n"
        else:
            content = f"{_HEADERS[section]}\n{line}\n"
        self._write(section, content)

        debug_log(f"core: remembered in {section}: '{flat[:60]}'", "memory")
        _notify_core_mutation("remember", section)
        return RememberResult(
            entry=CoreEntry(
                text=flat, date=stamp, source=attribution, raw=line,
            ),
            already_known=False,
        )

    def retire(
        self,
        section: str,
        text: str,
        *,
        on_date: Optional[str] = None,
        reason: str = "",
    ) -> bool:
        """Strike an entry through so it stops being believed.

        The line stays in the file. Returns False when nothing active
        matches, in which case the file is left untouched.
        """
        key = _normalise(_flatten(text))
        lines = self._read_lines_strict(section)
        stamp = on_date or _today_utc()

        for index, line in enumerate(lines):
            entry = _parse_line(line)
            if entry is None or entry.retired:
                continue
            if _normalise(entry.text) != key:
                continue
            lines[index] = _render_retired(entry, stamp, reason)
            self._write(section, "\n".join(lines) + "\n")
            debug_log(f"core: retired from {section}: '{entry.text[:60]}'", "memory")
            _notify_core_mutation("retire", section)
            return True

        debug_log(f"core: nothing active matches '{text[:60]}' in {section}", "memory")
        return False


# ── The block that reaches the prompt ─────────────────────────────────


def _collect(entries: list[CoreEntry], max_chars: int) -> str:
    """Join entry texts newest-first, stopping at the character budget.

    Newest-first because when the budget bites, the facts the user stated
    most recently are the ones most likely to still hold.
    """
    out: list[str] = []
    used = 0
    for entry in reversed(entries):
        text = entry.text.strip()
        if not text:
            continue
        cost = len(text) + (1 if out else 0)
        if used + cost > max_chars:
            break
        out.append(text)
        used += cost
    return "\n".join(out)


def build_core_profile(
    core: MemoryCore,
    *,
    user_max_chars: int = 1200,
    directives_max_chars: int = 600,
) -> dict[str, str]:
    """Build the profile blob injected into every reply's system prompt.

    Returned as ``{"user": ..., "directives": ...}`` so the caller can
    render the two sections separately: directives want an imperative
    framing, user facts a descriptive one. An empty string on either key
    means that half of the core is empty and the section should be
    omitted entirely rather than rendered as an empty heading.

    Only the fact text crosses into the prompt. Dates and attribution are
    bookkeeping for the user reading the file, and would just be context
    the model has to ignore.
    """
    return {
        "user": _collect(core.active(SECTION_PROFILE), user_max_chars),
        "directives": _collect(core.active(SECTION_RULES), directives_max_chars),
    }


def format_warm_profile_block(profile: dict[str, str]) -> str:
    """Render a core profile dict as a labelled block for the system prompt.

    Returns an empty string when both sections are empty so the caller can
    append unconditionally without introducing whitespace noise on fresh
    installs with no accumulated memory.

    The labels deliberately mirror the denial templates small models
    produce under uncertainty ("I don't have information the user has
    shared in prior conversations"). Naming the section exactly what the
    denial refers to short-circuits the denial pattern — see the CLAUDE.md
    note on denial-template mirroring.
    """
    user = (profile.get("user") or "").strip()
    directives = (profile.get("directives") or "").strip()
    if not user and not directives:
        return ""

    sections: list[str] = []
    if user:
        sections.append(
            "INFORMATION THE USER HAS SHARED IN PRIOR CONVERSATIONS\n"
            "(their identity, location, tastes, preferences, habits, "
            "history — treat this as known context about the user, not "
            "as new information you need to ask about):\n"
            f"{user}"
        )
    if directives:
        sections.append(
            "STANDING INSTRUCTIONS FROM THE USER\n"
            "(rules the user has told you to follow — obey these "
            "verbatim, in every reply, without being reminded):\n"
            f"{directives}"
        )
    return "\n\n".join(sections)

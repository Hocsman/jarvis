"""claudeCode — hand a task on one of the user's projects to Claude Code.

The job runs in the background and its report is delivered when it ends,
so this tool only ever says a job has started, never that it is done. Its
risk follows its arguments, the way `localFiles` does: a task that may
change files costs a click on a card, a question and a progress check
only read.

See src/jarvis/projets/projets.spec.md.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ...debug import debug_log
from ...projets import delegation as dl
from ..base import Tool, ToolContext
from ..types import ToolExecutionResult

# The values of `modifier` that mean "read only". JSON booleans and
# numbers, never words: a word is a language, and anything unreadable
# must land on the side that asks.
_READ_ONLY_VALUES = (False, 0, "false", "0")


def wants_changes(args: Optional[Dict[str, Any]]) -> bool:
    """Whether the call may edit the project, read as the gate reads it."""
    value = (args or {}).get("modifier", True)
    if isinstance(value, str):
        value = value.strip().lower()
    return value not in _READ_ONLY_VALUES


def _task(args: Optional[Dict[str, Any]]) -> str:
    return str((args or {}).get("tache") or "").strip()


def _project(args: Optional[Dict[str, Any]]) -> str:
    return str((args or {}).get("projet") or "").strip()


def _texts(args: Optional[Dict[str, Any]]) -> list:
    return [v for v in (args or {}).values() if isinstance(v, str)]


def _minutes_since(started: float) -> int:
    return max(0, int((time.monotonic() - started) // 60))


def edits_kept(job) -> str:
    """What a stopped job leaves behind: stopping is not undoing."""
    if not job.modifier:
        return "It was only reading, so nothing in the project changed."
    return ("Files it had already changed are not undone, so do not claim "
            "the project is back as it was: the user may want to look at them.")


def status_text() -> str:
    """How the job is going, from memory: nothing here touches a project."""
    job = dl.DELEGATIONS.running()
    if job is not None:
        return (
            f"Claude Code is still working on {job.projet}, started "
            f"{_minutes_since(job.started)} min ago. Its report will be "
            f"delivered when it finishes."
        )
    last = dl.DELEGATIONS.last()
    if last is not None:
        ago = _minutes_since(last.job.started + last.duration_s)
        if last.cancelled:
            return (
                f"Nothing is running. The last task, on {last.job.projet}, "
                f"was stopped at the user's request {ago} min ago. "
                f"{edits_kept(last.job)}"
            )
        return (
            f"Nothing is running. The last task, on {last.job.projet}, "
            f"{'finished' if last.ok else 'failed'} {ago} min ago. "
            f"Its report: « {last.report} »"
        )
    return "Claude Code has not been given any task since Yuba started."


def _holds_yubas_store(cfg, project: Path) -> bool:
    """Whether the project contains Yuba's own files, or lies inside them."""
    try:
        from ...memory.core import MemoryCore

        core = MemoryCore.for_config(cfg).directory.resolve()
        project = project.resolve()
    except Exception:
        return False
    return core == project or project in core.parents or core in project.parents


def vet_call(args: Optional[Dict[str, Any]], cfg) -> Tuple[Optional[str], Optional[Path]]:
    """Decide whether a call carrying a task may start, and where.

    Returns (reason it may not, None) or (None, project folder). A call
    that may edit runs only on a project its `projet` names exactly, the
    name on the card the user approves; a near miss or a project named
    only inside the task is sent back with the exact name to use.
    """
    projet = _project(args)
    root = str(getattr(cfg, "claude_code_projects_dir", "") or "").strip()
    if not root:
        return ("handing projects to Claude Code is switched off: the setting "
                "claude_code_projects_dir names no folder"), None
    root_path = Path(root).expanduser()
    if not root_path.is_dir():
        return f"the projects folder {root} does not exist", None

    found = dl.choose_project(root_path, projet, _texts(args))
    if found.reason == dl.AMBIGUOUS:
        return (f"the project could be any of {', '.join(found.candidates)}. "
                f"Ask the user which one"), None
    if found.path is None:
        named = f"there is no project called '{projet}'" if projet else "no project was named"
        return (f"{named}. {status_text()} If the user asked for work on a "
                f"project, ask which one; the projects are: "
                f"{', '.join(found.candidates) or 'none'}"), None
    if _holds_yubas_store(cfg, found.path):
        return (f"the project {found.path.name} holds or lies inside Yuba's own "
                f"files (profile, rules, tool permissions), which no tool may "
                f"edit"), None
    if wants_changes(args) and not found.exact:
        return (f"to change code, the project must be named exactly as its "
                f"folder, so the user approves the one that runs: call the tool "
                f"again with projet='{found.path.name}'"), None
    return None, found.path


class ClaudeCodeTool(Tool):
    """Give one of the user's projects to Claude Code, and report back."""

    # Its empty call asks how a job is going, which reads. Classified from
    # `{}`, the generated outils.md would free it, and a line there frees
    # every call, a change to the user's code included.
    classified_per_call = True

    def risk_for(self, args):
        """Asking how it is going, or a question Claude Code answers by
        reading only, changes nothing. Anything else may edit the user's
        code, including a call whose `modifier` cannot be read and one
        whose project is only named inside the task."""
        from ..policy import RISK_DESTRUCTIVE, RISK_READ

        if not _task(args) or not wants_changes(args):
            return RISK_READ
        return RISK_DESTRUCTIVE

    def offered(self, cfg) -> bool:
        """Only once a projects folder is named: off, every call refuses."""
        return bool(str(getattr(cfg, "claude_code_projects_dir", "") or "").strip())

    @property
    def name(self) -> str:
        return dl.TOOL_NAME

    @property
    def description(self) -> str:
        # Whole sentences, restriction first: the router reads only the
        # opening of this.
        return (
            "Hand a task on one of the user's code projects to Claude Code, "
            "a coding agent that works on its own in the project folder and "
            "reports back later, ONLY when the user asks for work on a "
            "project (change, fix, add, explain or review its code) or asks "
            "how that work is going. It runs in the background: say it has "
            "started, never that it is done."
        )

    @property
    def inputSchema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "projet": {
                    "type": "string",
                    "description": "The project, as the user named it.",
                },
                "tache": {
                    "type": "string",
                    "description": (
                        "What Claude Code must do, as one complete "
                        "instruction in the user's language with every "
                        "detail they gave. Leave empty to ask how the "
                        "current task is going."
                    ),
                },
                "modifier": {
                    "type": "boolean",
                    "description": (
                        "true when the task changes files (fix, add, "
                        "refactor, update); false when it only asks a "
                        "question, an explanation or a review."
                    ),
                },
            },
        }

    def run(self, args: Optional[Dict[str, Any]],
            context: ToolContext) -> ToolExecutionResult:
        cfg = context.cfg
        tache = _task(args)
        if not tache:
            return ToolExecutionResult(success=True, reply_text=status_text())

        refusal, path = vet_call(args, cfg)
        if refusal:
            return self._refuse(refusal)

        modifier = wants_changes(args)
        try:
            started, job = dl.DELEGATIONS.start(
                cfg=cfg, db=context.db, projet=path.name, path=path,
                tache=tache, modifier=modifier, origin=context.origin,
                risk=self.risk_for(args),
            )
        except dl.ClaudeCodeMissing:
            return self._refuse(
                "Claude Code is not installed on this machine (the command "
                f"'{getattr(cfg, 'claude_code_command', 'claude')}' was not "
                "found). The user can install it, or set "
                "claude_code_command to its full path"
            )
        except dl.DelegationsClosed:
            return self._refuse("Yuba is stopping, so nothing new starts")

        if not started:
            return self._refuse(
                f"Claude Code is already working on {job.projet} "
                f"(for {_minutes_since(job.started)} min); only one task "
                f"runs at a time"
            )

        return ToolExecutionResult(
            success=True,
            reply_text=(
                f"Claude Code has started on the project {job.projet} "
                f"({'it may edit files' if modifier else 'reading only'}). "
                f"It works in the background and its report will be "
                f"delivered when it finishes. Tell the user, in their "
                f"language, that it has started; do not claim it is done "
                f"and do not guess what it will find."
            ),
        )

    @staticmethod
    def _refuse(reason: str) -> ToolExecutionResult:
        debug_log(f"    🛠️ Claude Code not started: {reason}", "tools")
        return ToolExecutionResult(
            success=False,
            reply_text=(
                f"Claude Code was not started: {reason}. Tell the user "
                f"plainly, in their language. Do not claim anything was done."
            ),
        )

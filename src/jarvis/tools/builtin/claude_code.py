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
from typing import Any, Dict, Optional

from ...debug import debug_log
from ...projets import claude_code as cc
from ..base import Tool, ToolContext
from ..types import ToolExecutionResult

# The values of `modifier` that mean "read only". JSON booleans and
# numbers, never words: a word is a language, and anything unreadable
# must land on the side that asks.
_READ_ONLY_VALUES = (False, 0, "false", "0")


def _wants_changes(args: Optional[Dict[str, Any]]) -> bool:
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


class ClaudeCodeTool(Tool):
    """Give one of the user's projects to Claude Code, and report back."""

    def risk_for(self, args):
        """Asking how it is going, or a question Claude Code answers by
        reading only, changes nothing. Anything else may edit the user's
        code, including a call whose `modifier` cannot be read and one
        whose project is only named inside the task."""
        from ..policy import RISK_DESTRUCTIVE, RISK_READ

        if not _task(args) or not _wants_changes(args):
            return RISK_READ
        return RISK_DESTRUCTIVE

    @property
    def name(self) -> str:
        return cc.TOOL_NAME

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
        projet = _project(args)

        if not tache:
            return self._status()

        root = str(getattr(cfg, "claude_code_projects_dir", "") or "").strip()
        if not root:
            return self._refuse(
                "handing projects to Claude Code is switched off: the "
                "setting claude_code_projects_dir names no folder"
            )
        root_path = Path(root).expanduser()
        if not root_path.is_dir():
            return self._refuse(f"the projects folder {root} does not exist")

        found = cc.resolve_project(root_path, projet) if projet else None
        if found is None or found.reason == "unknown":
            # Only what the card shows is searched: the arguments, never
            # the sentence behind them.
            mention = cc.mentioned_project(root_path, *_texts(args))
            if mention.reason in ("ok", "ambiguous"):
                found = mention
            elif found is None:
                return self._refuse(
                    f"no project was named. {self._status().reply_text} If "
                    f"the user asked for work on a project, ask which one; "
                    f"the projects are: {', '.join(mention.candidates) or 'none'}"
                )
        if found.reason == "ambiguous":
            return self._refuse(
                f"the project could be any of {', '.join(found.candidates)}. "
                f"Ask the user which one"
            )
        if found.path is None:
            return self._refuse(
                f"there is no project called '{projet}'. The projects are: "
                f"{', '.join(found.candidates) or 'none'}"
            )

        if self._holds_yubas_store(cfg, found.path):
            return self._refuse(
                f"the project {found.path.name} holds Yuba's own files "
                f"(profile, rules, tool permissions), which no tool may edit"
            )

        modifier = _wants_changes(args)
        try:
            started, job = cc.DELEGATIONS.start(
                cfg=cfg, db=context.db, projet=found.path.name, path=found.path,
                tache=tache, modifier=modifier, origin=context.origin,
                risk=self.risk_for(args),
            )
        except cc.ClaudeCodeMissing:
            return self._refuse(
                "Claude Code is not installed on this machine (the command "
                f"'{getattr(cfg, 'claude_code_command', 'claude')}' was not "
                "found). The user can install it, or set "
                "claude_code_command to its full path"
            )

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

    def _status(self) -> ToolExecutionResult:
        job = cc.DELEGATIONS.running()
        if job is not None:
            return ToolExecutionResult(
                success=True,
                reply_text=(
                    f"Claude Code is still working on {job.projet}, started "
                    f"{_minutes_since(job.started)} min ago. Its report will "
                    f"be delivered when it finishes."
                ),
            )
        last = cc.DELEGATIONS.last()
        if last is not None:
            ago = _minutes_since(last.job.started + last.duration_s)
            return ToolExecutionResult(
                success=True,
                reply_text=(
                    f"Nothing is running. The last task, on {last.job.projet}, "
                    f"{'finished' if last.ok else 'failed'} {ago} min ago. "
                    f"Its report: « {last.report} »"
                ),
            )
        return ToolExecutionResult(
            success=True,
            reply_text="Claude Code has not been given any task since Yuba started.",
        )

    @staticmethod
    def _holds_yubas_store(cfg, project: Path) -> bool:
        try:
            from ...memory.core import MemoryCore

            core = MemoryCore.for_config(cfg).directory.resolve()
            project = project.resolve()
        except Exception:
            return False
        return core == project or project in core.parents

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

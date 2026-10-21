"""stopClaudeCode — stop the running Claude Code job without stopping Yuba.

A wrong job (a misheard project, a change approved by mistake) would
otherwise keep editing until `claude_code_timeout_sec`, holding the only
slot. A tool of its own rather than an argument on `claudeCode`: small
models misfile extra arguments, and a stop misfiled there would be read
as a task.

Its risk is `action`: it changes something outside Yuba's own store, a
process working on the user's project, and asking again restarts the
work. So by default she asks first, by voice or on a card; a user who
wants it immediate writes `- stopClaudeCode` under `## Libre`.

See src/jarvis/projets/projets.spec.md.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ...debug import debug_log
from ...projets import delegation as dl
from ..base import Tool, ToolContext
from ..types import ToolExecutionResult
from .claude_code import edits_kept, status_text


def carried_instruction(args: Optional[Dict[str, Any]]) -> str:
    """The text a call carries, which a stop never does.

    The tool has no properties. Text in a call is an instruction the model
    is passing on: "ask Claude Code to stop the login page from crashing"
    reaches this tool carrying it, and a request to stop the running task
    never did. Any text, in any language: nothing here reads words.
    """
    for value in (args or {}).values():
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


class StopClaudeCodeTool(Tool):
    """Stop the Claude Code job that is running, and say what it leaves."""

    # Its calls differ in risk: the empty one stops the job, one carrying
    # text stops nothing. A line in the generated outils.md would decide
    # both, and under ## Demande the harmless one would ask too.
    classified_per_call = True

    def risk_for(self, args):
        """Stops a process working on the user's project. Recoverable:
        the task can be asked for again. A call carrying an instruction
        stops nothing, so it only reads and costs no question."""
        from ..policy import RISK_ACTION, RISK_READ

        return RISK_READ if carried_instruction(args) else RISK_ACTION

    def offered(self, cfg) -> bool:
        """Only while a job runs: the rest of the time there is nothing to
        stop, and a tool shown to the router is one more place for a bare
        "stop" to land."""
        if not str(getattr(cfg, "claude_code_projects_dir", "") or "").strip():
            return False
        return dl.DELEGATIONS.running() is not None

    @property
    def name(self) -> str:
        return "stopClaudeCode"

    @property
    def description(self) -> str:
        # Restriction first: the router reads only the opening of this,
        # and "stop" alone must find nothing here that claims it.
        return (
            "Stop the Claude Code task running on one of the user's projects, "
            "ONLY when the user asks to stop, cancel or interrupt Claude Code "
            "or its task. A plain stop without Claude Code ends the "
            "conversation instead."
        )

    @property
    def inputSchema(self) -> Dict[str, Any]:
        # One job at a time, so there is nothing to name.
        return {"type": "object", "properties": {}, "required": []}

    def run(self, args: Optional[Dict[str, Any]],
            context: ToolContext) -> ToolExecutionResult:
        if carried_instruction(args):
            debug_log("    🛑 Claude Code not stopped: the call carried an instruction", "tools")
            return ToolExecutionResult(
                success=True,
                reply_text=(
                    "Nothing was stopped: this tool takes no arguments, and "
                    "this call carried text, which makes it an instruction. A "
                    "new instruction for Claude Code goes through claudeCode. "
                    "Only if the user asked to stop the task already running, "
                    "call stopClaudeCode again with no arguments."
                ),
            )

        job, outcome = dl.DELEGATIONS.cancel()

        if job is None:
            return ToolExecutionResult(
                success=True,
                reply_text=(
                    f"Nothing was stopped: no Claude Code task is running. "
                    f"{status_text()} Tell the user, in their language."
                ),
            )

        if outcome is None:
            return ToolExecutionResult(
                success=True,
                reply_text=(
                    f"The stop was asked but is not confirmed yet: Claude Code "
                    f"is still ending its task on {job.projet}. Its end will be "
                    f"announced when it lands. Tell the user exactly that, in "
                    f"their language; do not claim it has stopped."
                ),
            )

        if not outcome.cancelled:
            debug_log(f"    🛑 Claude Code on {job.projet} had already ended", "tools")
            said = (f"It had finished, and its report says: « {outcome.report} »"
                    if outcome.ok else f"It had failed: {outcome.report}")
            return ToolExecutionResult(
                success=True,
                reply_text=(
                    f"Nothing was stopped: Claude Code's task on {job.projet} "
                    f"had already ended before the stop reached it. {said} "
                    f"Tell the user, in their language."
                ),
            )

        return ToolExecutionResult(
            success=True,
            reply_text=(
                f"Claude Code was stopped on {job.projet} after "
                f"{_elapsed(outcome.duration_s)}. {edits_kept(job)} Tell the "
                f"user, in their language."
            ),
        )


def _elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{max(1, int(seconds))} s"
    return f"{int(seconds // 60)} min"

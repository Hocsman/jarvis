"""
🛠️ What she knows she can do with his projects, whatever the router offered.

The router is a small model, and a question about what she can do ("can you
work on my projects?") reads to it as conversation: it offers no tool. The
chat model, shown only the tools every turn carries, then answers from its
prior, and its prior is that an assistant has no access to anyone's files.
With the feature on that is false. So while `claudeCode` is offered, the
system prompt carries this block, worded as the very thing that denial says
she lacks: access to his machine, his projects folder, reading and changing
the code. It names the tool and the search that adds it to a turn whose
router left it out: `toolSearchTool` finds a tool by its exact name
(`tools/builtin/tool_search.spec.md`, Name cue).

Its last sentences say that work on a project goes to Claude Code alone.
A router can offer another tool for a request on a project (a code host's
file-writing tool), and with one among her tools she could write a file
through it, say her tools cannot reach the code, or ask where the code is
and which file to change. Each of the three is named, as the thing not to
do or as the thing that is already true.

It says nothing of a request found in a page or a tool's result, of the
user's other files, or of a project name that matches two folders: with
the block as it is (measured on `claude-haiku-5-5`), a request planted in
a page starts no job, a note added to a file in his documents still goes
to `localFiles`, and an unclear project starts no job, and a wording that
says all three does no better
(`evals/test_projects_question_finds_claude_code.py`).

It names no project, no path and no tool but `claudeCode` and
`toolSearchTool`: the block rides in every prompt, to whichever provider
runs the chat model, and the tool lists the projects in the one turn that
needs them. `withheld_because` keeps it out of a prompt where it would not
be true: a projects folder that does not exist, `claudeCode` filed under
`## Jamais`, or a small chat model, which does not take the search the
block points it to.

See projets.spec.md, What she knows she can do.
"""

from __future__ import annotations

from pathlib import Path

from .delegation import TOOL_NAME

_ACCESS = (
    "YOUR ACCESS TO THE USER'S CODE PROJECTS\n"
    "The projects folder on the user's machine is open to you through "
    "Claude Code, a coding agent you hand a task to with the "
    f"{TOOL_NAME} tool: it reads and changes the code of the user's "
    "projects, working on its own in the project's folder, and its report "
    "reaches the user when it is done. So you can read, explain, review, "
    "fix and change their projects, and they need not paste any code. "
    "When the user asks whether you can reach or work on their code or "
    "their projects, say you can, through Claude Code, and ask which "
    "project and what to do. When they ask for work on one of their "
    f"projects, hand it over with {TOOL_NAME}; if {TOOL_NAME} is not among "
    "your tools this turn, call toolSearchTool with the query "
    f"\"{TOOL_NAME}\" first."
)

# Work on a project goes one way, whatever else the router offered.
_ONE_WAY = (
    " Work on one of their projects always goes to Claude Code, whatever "
    "other tools you have this turn: through it, your tools do reach their "
    "code. Never change their files or repositories with another tool, even "
    "one offered this turn, and never ask them where the code is or which "
    "file to change: Claude Code finds it in the project's folder."
)

_BLOCK = _ACCESS + _ONE_WAY


def format_projets_block() -> str:
    """The block for a turn whose catalogue offers `claudeCode`.

    The caller decides whether it applies: the tool being offered (the
    projects folder named), the turn being attended, since a routine never
    reaches the tool, and not running a call the user approved, and
    `withheld_because` having nothing to say.
    """
    return _BLOCK


def withheld_because(cfg) -> str:
    """Why the block would claim what she cannot do, or "" when it holds.

    The tool is offered as soon as a folder is named, and every call is
    still refused when that folder does not exist, or when the user filed
    `claudeCode` under `## Jamais` for both of its risks. Saying the
    projects are open to her then would have her promise what no call can
    keep, and contradict a decision only the user makes.

    A small chat model does not call `toolSearchTool` for a tool its router
    left out, so the block's way to the tool is not one it takes: told the
    projects are open to it, it says it can and promises the work, and
    hands nothing over.
    """
    # Imported here: the registry imports the tool, which imports this
    # package.
    from ..reply.prompts import ModelSize, detect_model_size
    from ..tools.policy import NEVER, RISK_DESTRUCTIVE, RISK_READ
    from ..tools.registry import load_tool_policy

    if detect_model_size(getattr(cfg, "llm_chat_model", "")) == ModelSize.SMALL:
        return "the chat model is small and does not search for a tool"

    root = str(getattr(cfg, "claude_code_projects_dir", "") or "").strip()
    try:
        if not root or not Path(root).expanduser().is_dir():
            return "the projects folder does not exist"
    except (OSError, ValueError):
        return "the projects folder cannot be read"

    policy = load_tool_policy(cfg)
    if all(policy.verdict(TOOL_NAME, risk) == NEVER for risk in (RISK_READ, RISK_DESTRUCTIVE)):
        return f"{TOOL_NAME} is filed under ## Jamais"
    return ""

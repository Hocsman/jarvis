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

It names no project and no path: the block rides in every prompt, to
whichever provider runs the chat model, and the tool lists the projects in
the one turn that needs them.

See projets.spec.md, What she knows she can do.
"""

from __future__ import annotations

from .delegation import TOOL_NAME

_BLOCK = (
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


def format_projets_block() -> str:
    """The block for a turn whose catalogue offers `claudeCode`.

    The caller decides whether it applies: the tool being offered (the
    projects folder named) and the turn being attended, since a routine
    never reaches the tool.
    """
    return _BLOCK

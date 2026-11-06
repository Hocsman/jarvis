"""toolSearchTool — mid-loop escape hatch for widening the tool allow-list.

Wraps ``select_tools`` so the chat model can re-run the router with a
refined query when the initial routing was too narrow, and finds a tool the
query names by its exact name whatever the router answers. See
``src/jarvis/tools/builtin/tool_search.spec.md``.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional

from ..base import Tool, ToolContext
from ..types import ToolExecutionResult
from ..selection import (
    _ALWAYS_INCLUDED,
    _LLM_MAX_SELECTED,
    router_timeout_sec,
    select_tools,
    ToolSelectionStrategy,
)
from ...debug import debug_log
from ...llm import get_auxiliary_backend, get_embedding_backend

if TYPE_CHECKING:
    from ..registry import ToolSpec

# A tool's name, as a whole word. A name holds letters, digits, `_`, `.`
# and `-` (`naming.PLAIN_NAME`: `claudeCode`, `srv__files.read`,
# `chrome-devtools__take_snapshot`), so it is a match only where neither
# side continues it: no letter or digit of any script (`\w` is Unicode
# here), no `-`, and no `.` before it. A `.` after it ends a sentence
# ("claudeCode.") unless a name character follows ("srv__files.read" does
# not name `srv__files`).
_NAME_BEFORE = r"(?<![\w.\-])"
_NAME_AFTER = r"(?![\w\-]|\.[\w\-])"

# Any word, in any script: what is left of a query once its tool names are
# taken out must hold none for the router to have nothing to add.
_ANY_WORD_RE = re.compile(r"\w")


@lru_cache(maxsize=1024)
def _name_re(name: str) -> "re.Pattern[str]":
    return re.compile(_NAME_BEFORE + re.escape(name) + _NAME_AFTER)


def _names_in(query: str, names: Iterable[str]) -> List[str]:
    return [n for n in names if n and n in query and _name_re(n).search(query)]


def tools_named_in(
    query: Optional[str],
    builtin_tools: Dict[str, Tool],
    mcp_tools: Dict[str, "ToolSpec"],
) -> List[str]:
    """The tools of the catalogue that *query* names by their exact name.

    A name is an identifier, spelt one way, so the match is exact and on
    whole words: "claudeCode" names the tool, while "Claude Code",
    "claudecode", "claudeCodes" or "àclaudeCode" do not. A tool the
    catalogue does not offer is not named. In catalogue order, builtins
    first. See tool_search.spec.md, Name cue.
    """
    return _names_in(query or "", [*builtin_tools, *mcp_tools])


def _names_only(query: str, named: List[str]) -> bool:
    """Whether *query* is nothing but the tools it names, quotes and
    punctuation aside (``claudeCode``, ``"claudeCode"``)."""
    if not named:
        return False
    rest = query
    for name in named:
        rest = _name_re(name).sub(" ", rest)
    return not _ANY_WORD_RE.search(rest)


def _resolve_router_model(cfg) -> str:
    for candidate in (
        getattr(cfg, "tool_router_model", ""),
        getattr(cfg, "intent_judge_model", ""),
        getattr(cfg, "llm_chat_model", ""),
    ):
        if candidate:
            return candidate
    return ""


class ToolSearchTool(Tool):
    """Re-run tool routing mid-loop to widen the allow-list."""

    def risk_for(self, args):
        """Looks at the world without changing it."""
        from ..policy import RISK_READ

        return RISK_READ

    @property
    def name(self) -> str:
        return "toolSearchTool"

    @property
    def description(self) -> str:
        return (
            "Search the full tool registry to discover additional tools. "
            "CALL THIS FIRST, before apologising or refusing, whenever the user "
            "asks for an action and none of your currently-available tools fit. "
            "Never reply 'I can't do that' without first calling toolSearchTool "
            "to check if a tool exists for it. Pass a short self-contained "
            "description of what you are trying to accomplish."
        )

    @property
    def inputSchema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Self-contained natural-language description of the "
                        "subtask needing a tool. Resolve pronouns and ellipsis "
                        "from the conversation before calling."
                    ),
                },
            },
            "required": ["query"],
        }

    def run(self, args: Optional[Dict[str, Any]], context: ToolContext) -> ToolExecutionResult:
        query = ""
        if isinstance(args, dict):
            raw = args.get("query")
            if isinstance(raw, str):
                query = raw.strip()
        if not query:
            return ToolExecutionResult(
                success=False,
                reply_text=None,
                error_message="toolSearchTool requires a non-empty 'query' argument.",
            )

        cfg = context.cfg
        # Local imports to avoid circulars at module load time.
        from ..registry import BUILTIN_TOOLS, get_cached_mcp_tools, offered_builtin_tools

        try:
            strategy = ToolSelectionStrategy(getattr(cfg, "tool_selection_strategy", "llm"))
        except ValueError:
            strategy = ToolSelectionStrategy.LLM

        try:
            mcp_tools = get_cached_mcp_tools() if getattr(cfg, "mcps", {}) else {}
        except Exception as e:
            debug_log(f"toolSearchTool: MCP cache unavailable: {e}", "tools")
            mcp_tools = {}

        # Only what the turn does not have yet is a find: not itself, and not
        # the always-included tools, which every attended turn offers. A
        # search that matched nothing comes back with those alone, and
        # listing them as hits would answer a question about the weather
        # with `remember`.
        already_offered = {"toolSearchTool", *_ALWAYS_INCLUDED}

        builtin_tools = offered_builtin_tools(cfg)
        # A tool named by its exact name is found whatever the router makes
        # of the query: the system prompt can give the model a name to
        # search for (`claudeCode`, projets.spec.md). A query made only of
        # names leaves the router nothing to add, so it is not asked. The
        # names found are capped as the router's picks are: a search is not
        # a way to list the catalogue.
        every_named = tools_named_in(query, builtin_tools, mcp_tools)
        named = [n for n in every_named if n not in already_offered][:_LLM_MAX_SELECTED]
        if _names_only(query, every_named):
            debug_log(
                f"toolSearchTool: the query names {named} and nothing else, "
                f"answered without the router",
                "tools",
            )
            selected: List[str] = named
        else:
            try:
                router_model = _resolve_router_model(cfg)
                routed = select_tools(
                    query=query,
                    builtin_tools=builtin_tools,
                    mcp_tools=mcp_tools,
                    strategy=strategy,
                    llm_backend=get_auxiliary_backend(cfg, router_model),
                    llm_model=router_model,
                    llm_timeout_sec=router_timeout_sec(cfg),
                    embedding_backend=get_embedding_backend(cfg),
                    embed_model=cfg.embedding_model,
                    embed_timeout_sec=float(getattr(cfg, "llm_embedding_timeout_sec", 10.0)),
                )
            except Exception as e:
                debug_log(f"toolSearchTool: select_tools failed: {e}", "tools")
                if not named:
                    return ToolExecutionResult(
                        success=False,
                        reply_text=None,
                        error_message=f"Tool search failed: {e}",
                    )
                # The name the query carries is found without the router.
                debug_log(f"toolSearchTool: answering with the named tools {named}", "tools")
                routed = []
            selected = named + [n for n in routed if n not in named]

        real = [n for n in selected if n and n not in already_offered]
        if not real:
            debug_log(
                f"toolSearchTool: no additional tools found for query={query!r}",
                "tools",
            )
            return ToolExecutionResult(
                success=True,
                reply_text="No additional tools found for that description.",
                error_message=None,
            )

        lines: list[str] = []
        for tname in real:
            desc = ""
            tool_obj = BUILTIN_TOOLS.get(tname)
            if tool_obj is not None:
                desc = (getattr(tool_obj, "description", "") or "").strip()
            else:
                spec = mcp_tools.get(tname)
                if spec is not None:
                    desc = (getattr(spec, "description", "") or "").strip()
            one_line = desc.splitlines()[0].strip() if desc else ""
            lines.append(f"{tname}: {one_line}" if one_line else tname)

        debug_log(
            f"toolSearchTool: surfaced {len(real)} tool(s) for query={query!r}",
            "tools",
        )
        return ToolExecutionResult(
            success=True,
            reply_text="\n".join(lines),
            error_message=None,
        )

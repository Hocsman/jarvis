from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List
import sys
import re
import requests
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import os

from .builtin.screenshot import ScreenshotTool
from .builtin.web_search import WebSearchTool
from .builtin.local_files import LocalFilesTool
from .builtin.fetch_web_page import FetchWebPageTool
from .builtin.nutrition.log_meal import LogMealTool
from .builtin.nutrition.fetch_meals import FetchMealsTool
from .builtin.nutrition.delete_meal import DeleteMealTool
from .builtin.refresh_mcp_tools import RefreshMCPToolsTool
from .builtin.weather import WeatherTool
from .builtin.remember import RememberTool
from .builtin.forget import ForgetTool
from .builtin.stop import StopTool
from .builtin.tool_search import ToolSearchTool
from .types import ToolExecutionResult
from ..config import Settings
from .external.mcp_client import MCPClient
from ..debug import debug_log


# Registry of all builtin tools
BUILTIN_TOOLS = {
    "screenshot": ScreenshotTool(),
    "webSearch": WebSearchTool(),
    "localFiles": LocalFilesTool(),
    "fetchWebPage": FetchWebPageTool(),
    "logMeal": LogMealTool(),
    "fetchMeals": FetchMealsTool(),
    "deleteMeal": DeleteMealTool(),
    "refreshMCPTools": RefreshMCPToolsTool(),
    "getWeather": WeatherTool(),
    "remember": RememberTool(),
    "forget": ForgetTool(),
    "stop": StopTool(),
    "toolSearchTool": ToolSearchTool(),
}

# Global MCP tools cache
_mcp_tools_cache: Dict[str, "ToolSpec"] = {}
_mcp_tools_cache_lock = threading.Lock()
_mcp_config_cache: Dict[str, Any] = {}


def initialize_mcp_tools(mcps_config: Dict[str, Any], verbose: bool = True) -> Tuple[Dict[str, "ToolSpec"], Dict[str, str]]:
    """
    Initialize MCP tools cache at startup.

    Args:
        mcps_config: MCP server configuration
        verbose: Whether to print status messages

    Returns:
        Tuple of (discovered_tools, errors) where errors maps server name to error message.
    """
    global _mcp_tools_cache, _mcp_config_cache

    with _mcp_tools_cache_lock:
        _mcp_config_cache = mcps_config or {}
        _mcp_tools_cache, errors = discover_mcp_tools(mcps_config)

        if verbose and _mcp_tools_cache:
            debug_log(f"MCP tools cache initialized with {len(_mcp_tools_cache)} tools", "mcp")

        return _mcp_tools_cache.copy(), errors


def get_cached_mcp_tools() -> Dict[str, "ToolSpec"]:
    """Get cached MCP tools without rediscovering."""
    with _mcp_tools_cache_lock:
        return _mcp_tools_cache.copy()


def refresh_mcp_tools(verbose: bool = True) -> Tuple[Dict[str, "ToolSpec"], Dict[str, str]]:
    """
    Refresh MCP tools cache by rediscovering all tools.

    Returns:
        Tuple of (discovered_tools, errors) where errors maps server name to error message.
    """
    global _mcp_tools_cache

    with _mcp_tools_cache_lock:
        if not _mcp_config_cache:
            debug_log("No MCP config cached, skipping refresh", "mcp")
            return {}, {}

        if verbose:
            print("🔄 Refreshing MCP tools...", flush=True)

        _mcp_tools_cache, errors = discover_mcp_tools(_mcp_config_cache)

        if verbose:
            print(f"  ✅ Found {len(_mcp_tools_cache)} MCP tools", flush=True)

        debug_log(f"MCP tools cache refreshed with {len(_mcp_tools_cache)} tools", "mcp")
        return _mcp_tools_cache.copy(), errors


def is_mcp_cache_initialized() -> bool:
    """Check if MCP tools cache has been initialized."""
    with _mcp_tools_cache_lock:
        return len(_mcp_config_cache) > 0 or len(_mcp_tools_cache) > 0



# ToolSpec for MCP compatibility
@dataclass(frozen=True)
class ToolSpec:
    name: str  # canonical tool identifier (camelCase)
    description: str  # Human-readable description (matches MCP format)
    inputSchema: Optional[Dict[str, Any]] = None  # JSON Schema for arguments (matches MCP format)
    # What the server says the tool does to the world: ``readOnlyHint``,
    # ``destructiveHint``. Optional because every existing construction
    # site passes three arguments, and because a server may send none —
    # which the policy gate reads as "unclassified", not as "harmless".
    annotations: Optional[Dict[str, Any]] = None


def _spec_from_tool_info(server_name: str, tool_info: Dict[str, Any]) -> ToolSpec:
    """Build the catalogue entry for one discovered MCP tool.

    Namespaced ``server__tool`` so two servers can expose the same name,
    and carrying the server's own annotations through: they are what lets
    the policy gate tell a snapshot from a click without the user
    classifying dozens of tools by hand.
    """
    name = tool_info.get("name")
    return ToolSpec(
        name=f"{server_name}__{name}",
        description=tool_info.get("description") or f"Tool from {server_name} MCP server",
        inputSchema=tool_info.get("inputSchema") or {"type": "object", "properties": {}, "required": []},
        annotations=tool_info.get("annotations"),
    )


def discover_mcp_tools(mcps_config: Dict[str, Any]) -> Tuple[Dict[str, ToolSpec], Dict[str, str]]:
    """Discover all tools from configured MCP servers and create ToolSpec entries for them.

    Returns:
        Tuple of (discovered_tools, errors) where errors maps server name to error message.
    """
    if not mcps_config:
        return {}, {}

    try:
        client = MCPClient(mcps_config)
        discovered_tools = {}
        errors: Dict[str, str] = {}

        for server_name in mcps_config.keys():
            try:
                tools = client.list_tools(server_name)
                for tool_info in tools:
                    tool_name = tool_info.get("name")
                    if not tool_name:
                        continue

                    # Create a unique tool name: server__toolname
                    full_tool_name = f"{server_name}__{tool_name}"

                    discovered_tools[full_tool_name] = _spec_from_tool_info(
                        server_name, tool_info,
                    )

            except BaseException as e:
                # ExceptionGroups (from anyio TaskGroup) wrap the real cause;
                # extract the first sub-exception for a useful error message.
                cause = e
                if hasattr(e, "exceptions") and e.exceptions:
                    cause = e.exceptions[0]
                debug_log(f"Failed to discover tools from MCP server '{server_name}': {cause}", "mcp")
                errors[server_name] = str(cause)
                continue

        return discovered_tools, errors

    except Exception as e:
        debug_log(f"Failed to discover MCP tools: {e}", "mcp")
        return {}, {"_global": str(e)}


def generate_tools_json_schema(allowed_tools: Optional[List[str]] = None, mcp_tools: Optional[Dict[str, ToolSpec]] = None) -> List[Dict[str, Any]]:
    """
    Generate tools in OpenAI-compatible JSON schema format for native tool calling.

    This format is supported by Ollama for models with native tool calling support
    (Llama 3.1+, Llama 3.2, Qwen 3, Mistral, etc.).

    Returns a list of tool definitions in this format:
    [
        {
            "type": "function",
            "function": {
                "name": "toolName",
                "description": "Tool description",
                "parameters": {
                    "type": "object",
                    "properties": {...},
                    "required": [...]
                }
            }
        }
    ]
    """
    names = list(allowed_tools or list(BUILTIN_TOOLS.keys()))
    tools: List[Dict[str, Any]] = []

    # Add built-in tools
    for tool_name in names:
        tool = BUILTIN_TOOLS.get(tool_name)
        if not tool:
            continue

        tool_def = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema or {"type": "object", "properties": {}, "required": []},
            }
        }
        tools.append(tool_def)

    # Add discovered MCP tools
    if mcp_tools:
        for tool_name, spec in mcp_tools.items():
            if tool_name in names:  # Only include if allowed
                tool_def = {
                    "type": "function",
                    "function": {
                        "name": spec.name,
                        "description": spec.description,
                        "parameters": spec.inputSchema or {"type": "object", "properties": {}, "required": []},
                    }
                }
                tools.append(tool_def)

    return tools


def generate_tools_description(allowed_tools: Optional[List[str]] = None, mcp_tools: Optional[Dict[str, ToolSpec]] = None) -> str:
    """Produce a compact tool help string for the system prompt using OpenAI standard format."""
    names = list(allowed_tools or list(BUILTIN_TOOLS.keys()))
    lines: List[str] = []
    lines.append("Tool-use protocol: Use the tool_calls field in your response:")
    lines.append('tool_calls: [{"id": "call_<id>", "type": "function", "function": {"name": "<toolName>", "arguments": "<json_string>"}}]')
    lines.append("\nAvailable tools and when to use them:")

    # Add built-in tools
    for tool_name in names:
        tool = BUILTIN_TOOLS.get(tool_name)
        if not tool:
            continue
        lines.append(f"\n{tool.name}: {tool.description}")
        if tool.inputSchema:
            # Extract a simple parameter summary from the JSON schema
            props = tool.inputSchema.get("properties", {})
            required = tool.inputSchema.get("required", [])
            param_descriptions = []
            for prop_name, prop_def in props.items():
                prop_type = prop_def.get("type", "any")
                is_required = prop_name in required
                req_marker = " (required)" if is_required else ""
                param_descriptions.append(f"{prop_name}: {prop_type}{req_marker}")
            if param_descriptions:
                lines.append(f"Input: {', '.join(param_descriptions)}")

    # Add discovered MCP tools
    if mcp_tools:
        for tool_name, spec in mcp_tools.items():
            if tool_name in names:  # Only include if allowed
                lines.append(f"\n{spec.name}: {spec.description}")
                if spec.inputSchema:
                    # Extract a simple parameter summary from the JSON schema
                    props = spec.inputSchema.get("properties", {})
                    required = spec.inputSchema.get("required", [])
                    param_descriptions = []
                    for prop_name, prop_def in props.items():
                        prop_type = prop_def.get("type", "any")
                        is_required = prop_name in required
                        req_marker = " (required)" if is_required else ""
                        param_descriptions.append(f"{prop_name}: {prop_type}{req_marker}")
                    if param_descriptions:
                        lines.append(f"Input: {', '.join(param_descriptions)}")

    return "\n".join(lines)

def _normalize_time_range(args: Optional[Dict[str, Any]]) -> Tuple[str, str]:
    now = datetime.now(timezone.utc)
    since: Optional[str] = None
    until: Optional[str] = None
    if args and isinstance(args, dict):
        try:
            since_val = args.get("since_utc")
            since = str(since_val) if since_val else None
        except Exception:
            since = None
        try:
            until_val = args.get("until_utc")
            until = str(until_val) if until_val else None
        except Exception:
            until = None
    if since is None and until is None:
        # Default last 24h
        return (now - timedelta(days=1)).isoformat(), now.isoformat()
    if since is None and until is not None:
        # backfill 24h prior to until
        try:
            until_dt = datetime.fromisoformat(until.replace("Z", "+00:00"))
        except Exception:
            until_dt = now
        return (until_dt - timedelta(days=1)).isoformat(), until_dt.isoformat()
    if since is not None and until is None:
        return since, now.isoformat()
    return since or (now - timedelta(days=1)).isoformat(), until or now.isoformat()


# ── Policy gate ──────────────────────────────────────────────────────
#
# Every tool call in the process passes through ``run_tool_with_retries``:
# the planner's direct execution and the agentic loop both land here. The
# gate lives at that single point rather than at either call site, so a
# future third caller cannot bypass it by construction.

_POLICY_CACHE: Dict[str, Any] = {"stamp": None, "policy": None}


def load_tool_policy(cfg: Settings):
    """Read the user's tool policy, cached on the file's mtime.

    Re-read when the file changes so an edit takes effect on the next
    call rather than the next restart: the file is the control surface,
    and a control surface with a restart delay is one people stop
    trusting.
    """
    from .policy import ToolPolicy

    try:
        from ..memory.core import MemoryCore

        path = MemoryCore.for_config(cfg).directory / "outils.md"
        stamp = path.stat().st_mtime_ns if path.exists() else None
    except Exception:
        return ToolPolicy.empty()

    if stamp is None:
        return ToolPolicy.empty()
    if _POLICY_CACHE["stamp"] == stamp and _POLICY_CACHE["policy"] is not None:
        return _POLICY_CACHE["policy"]

    try:
        policy = ToolPolicy.parse(path.read_text(encoding="utf-8"))
    except OSError as e:
        debug_log(f"tool policy unreadable, falling back to defaults: {e}", "tools")
        return ToolPolicy.empty()

    _POLICY_CACHE["stamp"], _POLICY_CACHE["policy"] = stamp, policy
    return policy


def ensure_policy_file(cfg: Settings) -> None:
    """Write ``outils.md`` from the installed catalogue, once.

    Generated rather than shipped so the user opens it and finds their
    own tools by name. Written only when absent: once it exists it is
    theirs, and a file regenerated on startup would quietly undo every
    decision they made in it.

    Call after MCP discovery, so the servers' tools are in the cache and
    land in the file alongside the builtins.

    Never raises. This runs during daemon boot, and one unwritable file
    must not stop Yuba starting.
    """
    from .policy import render_policy_file, resolve_risk

    try:
        from ..memory.core import MemoryCore

        path = MemoryCore.for_config(cfg).directory / "outils.md"
        if path.exists():
            return

        builtin_risks = {
            name: resolve_risk(name, tool, {})
            for name, tool in BUILTIN_TOOLS.items()
        }
        mcp_risks = {
            name: resolve_risk(name, spec, {})
            for name, spec in get_cached_mcp_tools().items()
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            render_policy_file(builtin_risks, mcp_risks), encoding="utf-8",
        )
        debug_log(
            f"policy file written: {len(builtin_risks)} builtins, "
            f"{len(mcp_risks)} MCP tools",
            "tools",
        )
    except Exception as e:
        debug_log(f"policy file not written: {e}", "tools")


def _known_tool(name: str):
    """The builtin or discovered spec for ``name``, or None.

    None is the honest answer for a tool the catalogue has never heard
    of, and the risk resolver treats it as destructive rather than
    guessing it is harmless.
    """
    if name in BUILTIN_TOOLS:
        return BUILTIN_TOOLS[name]
    return get_cached_mcp_tools().get(name)


def _refusal(name: str, verdict: str, risk: str) -> ToolExecutionResult:
    from .policy import NEVER

    if verdict == NEVER:
        text = (
            f'The tool "{name}" is in the user\'s never list and cannot be '
            f"run under any circumstance. Tell them plainly that you are not "
            f"allowed to do this, and do not offer to try again or suggest a "
            f"way around it."
        )
    else:
        text = (
            f'The tool "{name}" needs the user\'s confirmation before it can '
            f"run ({risk}), and there is no way to ask them yet. Nothing was "
            f"done. Tell them what you wanted to do and that they can allow it "
            f'by moving "{name}" into the ## Libre section of their outils.md '
            f"file."
        )
    debug_log(f"    🚪 refused {name} ({risk} → {verdict})", "tools")
    return ToolExecutionResult(
        success=False, reply_text=text, error_message=None, refused=True,
    )


def _log_action(
    db, *, tool, args, risk, verdict, outcome, query, origin,
    duration_ms=None, request_id=None,
):
    """Write one ledger row. Never lets bookkeeping break a tool call."""
    recorder = getattr(db, "record_action", None)
    if not callable(recorder):
        return
    try:
        recorder(
            tool=tool, args=args, risk=risk, verdict=verdict, outcome=outcome,
            duration_ms=duration_ms, origin=origin, query=query,
            request_id=request_id,
        )
    except Exception as e:
        debug_log(f"action log write skipped: {e}", "tools")


def run_tool_with_retries(
    db,
    cfg: Settings,
    tool_name: str,
    tool_args: Optional[Dict[str, Any]],
    system_prompt: str,
    original_prompt: str,
    redacted_text: str,
    max_retries: int = 1,
    language: Optional[str] = None,
    origin: Optional[str] = None,
) -> ToolExecutionResult:
    """Run one tool, subject to the user's policy, and record what happened.

    ``origin`` labels the ledger row with what set this call going —
    ``"voix"``, ``"chat"``, and later the routines that run unattended.
    Passed rather than read from ambient state because those routines run
    on their own threads while the user is mid-conversation, and a shared
    slot would label their rows with whoever spoke last. None is the
    honest value when nothing said.
    """
    # Normalize tool name to canonical camelCase
    raw_name = (tool_name or "").strip()
    name = raw_name

    # The gate, before anything runs. Placed here and not at the call
    # sites because this is the only funnel: both the planner's direct
    # execution and the agentic loop arrive through it.
    from .policy import FREE, OUTCOME_FAILED, OUTCOME_OK, OUTCOME_REFUSED, resolve_risk

    risk = resolve_risk(name, _known_tool(name), tool_args)
    verdict = load_tool_policy(cfg).verdict(name, risk)
    if verdict != FREE:
        _log_action(
            db, tool=name, args=tool_args, risk=risk, verdict=verdict,
            outcome=OUTCOME_REFUSED, query=redacted_text, origin=origin,
        )
        return _refusal(name, verdict, risk)

    # Allowed. The ledger records the outcome once the call returns, so
    # both halves of the gate's decision leave a trace: what was let
    # through and what it did, not only what was stopped.
    _started = time.monotonic()

    def _finish(result: ToolExecutionResult) -> ToolExecutionResult:
        _log_action(
            db, tool=name, args=tool_args, risk=risk, verdict=verdict,
            outcome=(OUTCOME_OK if getattr(result, "success", False) else OUTCOME_FAILED),
            query=redacted_text, origin=origin,
            duration_ms=int((time.monotonic() - _started) * 1000),
        )
        return result

    # Check if tool name is a discovered MCP tool (server__toolname format)
    if "__" in raw_name:
        server_name, mcp_tool_name = raw_name.split("__", 1)
        mcps_config = getattr(cfg, "mcps", {})
        if mcps_config and server_name in mcps_config:
            try:
                if MCPClient is None:
                    return ToolExecutionResult(success=False, reply_text=None, error_message="MCP client not available. Install 'mcp' package.")

                client = MCPClient(mcps_config)
                result = client.invoke_tool(server_name=server_name, tool_name=mcp_tool_name, arguments=tool_args or {})
                is_error = bool(result.get("isError", False))
                text = result.get("text") or None
                return _finish(ToolExecutionResult(
                    success=(not is_error), reply_text=text,
                    error_message=(text if is_error else None),
                ))
            except Exception as e:
                return _finish(ToolExecutionResult(
                    success=False, reply_text=None,
                    error_message=f"MCP tool '{raw_name}' error: {e}",
                ))

    # Friendly user print helper (non-debug only)
    def _user_print(message: str) -> None:
        # 4-space indent: tool messages happen INSIDE an agentic-loop
        # turn. The turn header (`  🔁 Turn N/M`) sits at 2 spaces, so
        # per-tool activity nests one level deeper for visual hierarchy.
        if not getattr(cfg, "voice_debug", False):
            try:
                print(f"    {message}")
            except Exception:
                pass

    # Check builtin tools first
    if name in BUILTIN_TOOLS:
        tool = BUILTIN_TOOLS[name]
        return _finish(tool.execute(
            db=db,
            cfg=cfg,
            tool_args=tool_args,
            system_prompt=system_prompt,
            original_prompt=original_prompt,
            redacted_text=redacted_text,
            max_retries=max_retries,
            user_print=_user_print,
            language=language,
        ))

    # Unknown tool
    debug_log(f"unknown tool requested: {tool_name}", "tools")
    return ToolExecutionResult(success=False, reply_text=None, error_message=f"Unknown tool: {tool_name}")



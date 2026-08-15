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
from .builtin.set_reminder import SetReminderTool
from .builtin.set_routine import SetRoutineTool
from .builtin.cancel_routine import CancelRoutineTool
from .builtin.review_learnings import ReviewLearningsTool
from .builtin.goals import (
    CloseGoalTool,
    ListGoalsTool,
    NoteGoalTool,
    SetGoalTool,
)
from .builtin.stop import StopTool
from .builtin.tool_search import ToolSearchTool
from .types import ToolExecutionResult
from .naming import is_plain_name
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
    "setReminder": SetReminderTool(),
    "setRoutine": SetRoutineTool(),
    "cancelRoutine": CancelRoutineTool(),
    "setGoal": SetGoalTool(),
    "noteGoal": NoteGoalTool(),
    "closeGoal": CloseGoalTool(),
    "listGoals": ListGoalsTool(),
    "reviewLearnings": ReviewLearningsTool(),
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

                    if not is_plain_name(full_tool_name):
                        # Dropped here rather than filtered at each of the
                        # places that write it onto a line, so a name that
                        # cannot be written is also a name that cannot be
                        # called. Said out loud once, at discovery: a
                        # capability quietly missing looks exactly like a
                        # server that never offered it.
                        montre = full_tool_name.encode(
                            "unicode_escape").decode("ascii")[:120]
                        debug_log(
                            f"MCP tool dropped, name outside the plain class: {montre}",
                            "mcp",
                        )
                        print(f"  ⚠️ 🔌 Tool ignored — its name cannot be written "
                              f"on a line: {montre}", flush=True)
                        continue

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
    # ``None`` asks for the whole catalogue; ``[]`` asks for nothing.
    # Collapsing the two would hand every builtin to the two callers that
    # deliberately narrow to nothing — a routine with an empty envelope,
    # and the resume turn that runs one approved call and narrates it.
    names = list(BUILTIN_TOOLS.keys()) if allowed_tools is None else list(allowed_tools)
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
    # Same rule as the schema: ``None`` is everything, ``[]`` is nothing.
    names = list(BUILTIN_TOOLS.keys()) if allowed_tools is None else list(allowed_tools)
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
    except Exception as e:
        # Not just OSError. `read_text` raises UnicodeDecodeError — a
        # ValueError — for a file that is not UTF-8, which a user reaches
        # by reopening their own generated policy in a Windows editor and
        # saving it as ANSI. Escaping here would break every tool call on
        # every turn until the file was repaired, which is the opposite of
        # what the spec promises: a malformed file yields the defaults.
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

        texte, refuses = render_policy_file(builtin_risks, mcp_risks)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(texte, encoding="utf-8")
        debug_log(
            f"policy file written: {len(builtin_risks)} builtins, "
            f"{len(mcp_risks)} MCP tools, {len(refuses)} names refused",
            "tools",
        )
        if refuses:
            print(
                f"  ⚠️ {len(refuses)} outil(s) écarté(s) d'outils.md : leur nom "
                "ne peut pas s'écrire sur une ligne.",
                flush=True,
            )
            for name in refuses:
                # Escaped, and on one line: an unwritable name is
                # unwritable here too, and a newline in one would forge a
                # second line of this very report.
                montre = name.encode("unicode_escape").decode("ascii")
                print(f"     • {montre[:120]}", flush=True)
            print(
                "     ↳ traités comme destructifs : elle demandera avant "
                f"chaque appel. Fichier : {path}",
                flush=True,
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


def _out_of_scope(name, scope, risk, verdict):
    """Whether an unattended call steps outside its routine's envelope.

    Five checks, and the last four are re-run on every call because the
    envelope was written weeks ago and the world moved.

    Returns a refusal, or None to carry on. The refusal deliberately does
    NOT point at `## Libre` the way the ordinary one does: unattended,
    that becomes a paragraph recommending the policy be loosened for
    every origin, written by a thread running while the user sleeps.
    """
    from ..routines.scope import JAMAIS_EN_ROUTINE
    from .policy import FREE, RISK_READ

    tool = _known_tool(name)
    if name in JAMAIS_EN_ROUTINE:
        # Distinct from the line below because the user may well have
        # written this name in the block themselves, and "not in the
        # perimeter" would read as a bug rather than a rule.
        why = "aucune routine ne peut l'atteindre, quel que soit son périmètre"
    elif not scope.allows(name):
        why = "il n'est pas dans le périmètre de cette routine"
    elif verdict != FREE:
        why = "tu as demandé à être consulté pour cet outil"
    elif risk != RISK_READ:
        why = "il ne se contente pas de lire"
    elif getattr(tool, "writes_own_state", False):
        why = "il écrit la mémoire de Yuba, et personne n'est là pour relire"
    elif getattr(tool, "reads_his_life", False):
        # Deliberately not the `writes_own_state` wording: this one may
        # write nothing at all and still be wrong unattended, because
        # what it does is read his days and form an opinion about him.
        why = "il lit ta vie et se fait une idée de toi, sans toi"
    elif getattr(tool, "needs_a_human", False):
        # It would not fail, it would *wait* — holding the runner's one
        # slot, which every other routine queues on, until a restart.
        why = "il attend que quelqu'un fasse un geste, et personne n'est là"
    else:
        return None

    debug_log(f"    🔁 {name} out of scope for {scope.nom}: {why}", "tools")
    return ToolExecutionResult(
        success=False, refused=True, error_message=None,
        reply_text=(
            f'The tool "{name}" is outside what the routine "{scope.nom}" '
            f"may reach ({why}). Nothing was done. Carry on with the rest "
            f"of the routine and say in your write-up what was missing. Do "
            f"not suggest changing any settings."
        ),
    )


def _fingerprint(name, args):
    from .confirmation import fingerprint

    return fingerprint(name, args)


def _ask(db, cfg, confirmation, *, name, args, risk, verdict, origin,
         redacted_text) -> ToolExecutionResult:
    """Raise a question and end the turn. Nothing runs.

    The gate never waits here. For voice, ``run_reply_engine`` runs on
    the listener's own audio thread, so blocking would silence the
    microphone that has to hear the answer — and blocking destroys the
    answer rather than delaying it, because the audio queue overflows
    into a swallowed exception.
    """
    from .confirmation import PendingAction, channel_for_call
    from .policy import OUTCOME_ASKED, OUTCOME_EXPIRED, OUTCOME_REFUSED

    store = confirmation.store
    channel = channel_for_call(risk, _known_tool(name), name, args)

    # Close whatever this question is about to displace. `raise_pending`
    # overwrites a card past its deadline, and the episode would leave
    # with the object: one `demandé` row and nothing after it, which the
    # Activity tab reads as a question still waiting on him.
    perimee = getattr(store, "take_expired_pending", None)
    if callable(perimee):
        try:
            morte = perimee()
        except Exception as e:
            debug_log(f"expired question not claimed: {e}", "tools")
            morte = None
        if morte is not None:
            debug_log(f"    🚪 {morte.tool} expired unanswered", "tools")
            _log_action(
                db, tool=morte.tool, args=morte.args, risk=morte.risk,
                verdict="demande", outcome=OUTCOME_EXPIRED,
                query=morte.query_redacted, origin=morte.origin,
                request_id=morte.request_id,
            )

    action = PendingAction.create(
        tool=name, args=args, risk=risk, channel=channel, origin=origin,
        query_redacted=redacted_text,
        raised_at_turn=store.current_turn(origin),
        ttl_sec=confirmation.ttl_sec,
    )
    held = store.raise_pending(action)
    if held is None:
        # A different question is already waiting. One card at a time, so
        # a three-step plan asks about its first step rather than
        # stacking questions the user has to disentangle.
        #
        # Not `_refusal`: its text tells the model there is no way to ask
        # and to send the user to `## Libre`, which here would advise
        # weakening the policy to work around a question they are in the
        # middle of answering. The truth is a queue, not a wall.
        waiting = store.peek_pending()
        other = getattr(waiting, "tool", None) or "another action"
        debug_log(f"    🚪 {name} not asked: {other} is still waiting", "tools")
        _log_action(
            db, tool=name, args=args, risk=risk, verdict=verdict,
            outcome=OUTCOME_REFUSED, query=redacted_text, origin=origin,
        )
        return ToolExecutionResult(
            success=False, refused=True, error_message=None,
            reply_text=(
                f'The tool "{name}" was not run because the user is still '
                f'being asked about "{other}". Tell them that, and that you '
                f"can come back to this once they have answered. Do not "
                f"suggest changing any settings."
            ),
        )

    if held.request_id == action.request_id:
        # A re-ask keeps the original id and writes no second row: one
        # episode in the ledger, not a queue of duplicates.
        _log_action(
            db, tool=name, args=args, risk=risk, verdict=verdict,
            outcome=OUTCOME_ASKED, query=redacted_text, origin=origin,
            request_id=held.request_id,
        )

    try:
        confirmation.publish(held)
    except Exception as e:
        debug_log(f"confirmation not published: {e}", "tools")

    debug_log(f"    🚪 asked about {name} ({risk} → {channel})", "tools")
    return ToolExecutionResult(
        success=False, reply_text=None, error_message=None,
        pending_id=held.request_id,
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
    confirmation: Optional["Confirmation"] = None,
    scope: Optional["RoutineScope"] = None,
) -> ToolExecutionResult:
    """Run one tool, subject to the user's policy, and record what happened.

    ``origin`` labels the ledger row with what set this call going —
    ``"voix"``, ``"chat"``, and later the routines that run unattended.
    Passed rather than read from ambient state because those routines run
    on their own threads while the user is mid-conversation, and a shared
    slot would label their rows with whoever spoke last. None is the
    honest value when nothing said.

    ``confirmation`` is a channel the gate can raise a question on, and
    the user's answer to a question raised on an earlier turn. Threaded
    in rather than reached for: without one the gate refuses exactly as
    it does today, because a gate that found a channel lying around
    would ask on behalf of code with no way to show the question.

    ``scope`` says this call is running unattended, inside one routine's
    envelope. It narrows what is allowed and never widens it, and its
    absence changes nothing at all — an attended turn sees exactly the
    behaviour it saw before routines existed.
    """
    # Normalize tool name to canonical camelCase
    raw_name = (tool_name or "").strip()
    name = raw_name

    # The gate, before anything runs. Placed here and not at the call
    # sites because this is the only funnel: both the planner's direct
    # execution and the agentic loop arrive through it.
    from .policy import (
        FREE, NEVER, OUTCOME_ASKED, OUTCOME_FAILED, OUTCOME_OK,
        OUTCOME_OUT_OF_SCOPE, OUTCOME_REFUSED, RISK_READ, resolve_risk,
    )

    risk = resolve_risk(name, _known_tool(name), tool_args)
    verdict = load_tool_policy(cfg).verdict(name, risk)
    request_id: Optional[str] = None

    # Running unattended. `jamais` is checked first, below, because a tool
    # the user retired is refused rather than reported out-of-scope —
    # different facts, and their own retirement is the stronger one.
    #
    # Everything here is re-checked on the call rather than trusted from
    # when the routine was written, because the things that change in
    # between are exactly the ones that matter: the user edited
    # `outils.md`, a server update added `destructiveHint`, or the model
    # picked the same tool with different arguments — `localFiles`
    # reading yesterday and deleting today, under one name.
    if scope is not None and verdict != NEVER:
        refusal = _out_of_scope(name, scope, risk, verdict)
        if refusal is not None:
            _log_action(
                db, tool=name, args=tool_args, risk=risk, verdict=verdict,
                outcome=OUTCOME_OUT_OF_SCOPE, query=redacted_text, origin=origin,
            )
            return refusal

    if verdict != FREE:
        # Order is the design.
        #
        # `jamais` first, and before any approval, so a tool the user
        # retired between the question and the answer is refused while a
        # perfectly valid grant sits in hand. `load_tool_policy` re-reads
        # on mtime, so the gate sees the newer decision, and the newer
        # decision wins.
        approval = getattr(confirmation, "approval", None) if confirmation else None

        if verdict == NEVER:
            _log_action(
                db, tool=name, args=tool_args, risk=risk, verdict=verdict,
                outcome=OUTCOME_REFUSED, query=redacted_text, origin=origin,
            )
            return _refusal(name, verdict, risk)

        if approval is not None:
            # The digest is recomputed here, now, against the call that is
            # actually about to run. Between the question and the answer
            # the model got to run again — the planner re-resolves steps,
            # the loop re-emits tool calls — so a grant given for one path
            # must not carry to whatever came back under the same name.
            if approval.fingerprint != _fingerprint(name, tool_args):
                debug_log(f"    🚪 approval does not match {name}", "tools")
                _log_action(
                    db, tool=name, args=tool_args, risk=risk, verdict=verdict,
                    outcome=OUTCOME_REFUSED, query=redacted_text, origin=origin,
                    request_id=approval.request_id,
                )
                return _refusal(name, verdict, risk)
            request_id = approval.request_id
            debug_log(f"    🚪 approved {name} ({request_id})", "tools")

        elif confirmation is None:
            # Nothing can ask. Today's refusal, which is what keeps every
            # caller that predates this parameter safe by default.
            _log_action(
                db, tool=name, args=tool_args, risk=risk, verdict=verdict,
                outcome=OUTCOME_REFUSED, query=redacted_text, origin=origin,
            )
            return _refusal(name, verdict, risk)

        else:
            return _ask(
                db, cfg, confirmation, name=name, args=tool_args, risk=risk,
                verdict=verdict, origin=origin, redacted_text=redacted_text,
            )

    # Allowed. The ledger records the outcome once the call returns, so
    # both halves of the gate's decision leave a trace: what was let
    # through and what it did, not only what was stopped.
    _started = time.monotonic()

    def _finish(result: ToolExecutionResult) -> ToolExecutionResult:
        _log_action(
            db, tool=name, args=tool_args, risk=risk, verdict=verdict,
            outcome=(
                getattr(result, "outcome", None)
                or (OUTCOME_OK if getattr(result, "success", False) else OUTCOME_FAILED)
            ),
            query=redacted_text, origin=origin, request_id=request_id,
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
                    return _finish(ToolExecutionResult(success=False, reply_text=None, error_message="MCP client not available. Install 'mcp' package."))

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
        # Wrapped like the MCP branch above, and for the reason written
        # below: a call the gate allowed and that then went nowhere still
        # happened. Unwrapped, an exception walks out through the whole
        # funnel, kills the turn, and leaves the ledger claiming she never
        # tried — and when the call came from an approval, its `demandé`
        # row never closes either.
        try:
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
                origin=origin,
            ))
        except Exception as e:
            debug_log(f"builtin tool '{name}' raised: {e}", "tools")
            return _finish(ToolExecutionResult(
                success=False, reply_text=None,
                error_message=f"Tool '{name}' raised: {e}",
            ))

    # Unknown tool
    debug_log(f"unknown tool requested: {tool_name}", "tools")
    # Through `_finish` like every other exit past the gate: a call the
    # gate allowed and that then went nowhere still happened, and a
    # ledger silent about it claims she never tried.
    return _finish(ToolExecutionResult(success=False, reply_text=None, error_message=f"Unknown tool: {tool_name}"))



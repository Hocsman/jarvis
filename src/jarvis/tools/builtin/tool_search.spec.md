## toolSearchTool Spec

### Purpose

Expose the reply engine's tool-routing logic as a callable builtin tool so the agentic loop can widen its own allow-list mid-conversation when the initial routing turned out too narrow.

### Problem

Before each reply, `select_tools` runs once outside the loop and narrows the tool allow-list to the model's best guess given only the user's immediate turn. If the model later realises a different tool is needed (e.g. the user's request was ambiguous, or a clarification reshaped the intent), it cannot access any tool outside that pre-picked set — the loop is stuck with whatever the router picked at turn zero.

### Design

`toolSearchTool` is an escape hatch, not a replacement for `select_tools`. Initial narrow routing still happens once, outside the loop; the loop then exposes:

```
allow-list = <router's picks> + stop + toolSearchTool
```

When the model invokes `toolSearchTool(query=...)`, the tool re-runs the same routing logic (`select_tools` from `src/jarvis/tools/selection.py`) against the new query, and the returned tool names are merged into the loop's allow-list for subsequent turns. `stop` and `toolSearchTool` itself always remain in the allow-list.

A query that is, or contains, the exact name of a tool the catalogue offers finds that tool whatever the router answers (Name cue, below). That is how the system prompt's projects block reaches `claudeCode` on a turn whose router left it out: it tells the model to search for `"claudeCode"` (`projets/projets.spec.md`, What she knows she can do).

### Name cue

A `toolSearchTool` query is read for the names of the tools it searches: the offered builtins (`offered_builtin_tools`) and the MCP tools, through `tools_named_in` in `src/jarvis/tools/selection.py`. A name is an identifier and is spelt one way, so the match is exact and on whole words (a run of letters, digits, `_` and `-`): `claudeCode` names the tool, while `Claude Code`, `claudecode`, `claudeCodes` or `myclaudeCode` do not, and are left to the router. A tool the catalogue does not offer (`claudeCode` with no projects folder) is not found by its name.

- **A query made only of names**, quotes and punctuation aside (`claudeCode`, `"claudeCode"`), is answered with those tools without asking the router: it has nothing to add, and its round trip can take up to `tool_router_timeout_sec`.
- **A name among other words** (`claudeCode corrige le bug du projet quillon`): the router reads the whole query as usual, and the answer lists the named tools first, then what the router found, duplicates collapsed. A router that times out or answers nothing readable leaves the keyword fallback, and the named tools are still found.

It exists because the system prompt can give the chat model the name of a tool the router may not have offered (`projets/projets.spec.md`, What she knows she can do), and a search for that name must then find it without depending on the router. When the router answers, it already finds a name: on a 112-tool catalogue (the builtins and the 89 MCP tools of four public servers), `qwen2.5:3b` alone offered `claudeCode` for the search `claudeCode` 20 times in 20, and for four searches made of the name and a task 80 in 80, against 4 in 20 for a sentence that says "Claude Code", the product rather than the tool. The cue is for the turns it does not answer: the keyword fallback that a timeout or an unreadable answer leads to does not find `claudeCode` from its bare name.

Only `toolSearchTool` reads names. The pre-loop router's query is what the user said, and a word of it that happens to be a tool's name ("j'ai vu un screenshot marrant") is no request for the tool: a spoken sentence gets what the router makes of it, and nothing from such a word (`selection.spec.md`).

### Contract

- **Name**: `toolSearchTool`
- **Description** (visible to the model): "Search the full tool registry for tools that can help with a task. Use this if none of the currently-available tools fit what the user actually needs. Pass a short self-contained description of what you are trying to accomplish."
- **Input schema**:
  - `query` (string, required): a self-contained natural-language description of the subtask needing a tool. Subject to the same `SELF-CONTAINED TOOL ARGUMENTS` rule as every other tool (pronouns and ellipsis resolved from conversation).
- **Output**: a newline-separated list of tool names and one-line descriptions for everything routing surfaced for `query`, apart from what every attended turn already offers: itself and the always-included tools (`stop`, `forget`, `remember`, `selection.spec.md`). When nothing else is left: a short honest note, "No additional tools found for that description." A search that matched nothing comes back with the always-included tools alone (`selection.spec.md`, keyword step 5), and listing them as finds would offer a write to the user's memory as the answer to a question about something else.

### Loop integration

The reply engine:
1. Runs `select_tools(text)` once pre-loop → `base_tools`.
2. Exposes `base_tools ∪ {stop, toolSearchTool}` per turn.
3. On a `toolSearchTool` call, dispatches it (running `select_tools(query)` with the same strategy config, unless the query is made only of tool names, Name cue), appends the tool result as normal, and merges the returned tool names into the allow-list for the next turn. Duplicates collapse; the list only grows.
4. Neither `stop` nor `toolSearchTool` is ever removed.
5. Refuses a call to any name outside the allow-list with a tool-role
   error that names what *is* available. `stop` and `toolSearchTool` are
   named in full however long the list is, and the error carries one
   sentence saying `toolSearchTool` can surface a tool that is not
   listed. Ordinary tools are trimmed past five, and the number trimmed
   is stated rather than implied by an ellipsis.

   A model that has just asked for a tool it cannot have is the one model
   that most needs the escape hatch, so the refusal is the wrong message
   to trim it out of. The two control tools are named only when the
   allow-list genuinely carries them: under a routine envelope
   `scope.allows` strips both, and advertising a tool the gate will
   refuse spends the model's next turn on a closed door.

Tools surfaced by `toolSearchTool` take effect from the NEXT turn onwards; the current turn's result is already committed. This is inherent to the agentic-loop rhythm and is not a bug.

The engine caps invocations per reply via `tool_search_max_calls` (default 3). Beyond the cap, further calls get a tool-error result telling the model to decide with the tools already available.

### What toolSearchTool is NOT

- Not a free-form tool discovery surface: it uses the same routing pipeline as the pre-loop call, not a raw "list every tool" dump. The router already applies allow/deny logic and MCP-awareness; reusing it keeps semantics consistent.
- Not a way to bypass authorisation: it searches the catalogue the pre-loop router is shown, so a tool that catalogue leaves out is not surfaced, by its description or by its name.
- Not free: each call is an LLM round-trip, unless its query is made only of tool names. The model is told to use it only when none of the currently-available tools fit.

### Testing

- Unit tests cover the merge-into-allow-list behaviour, the no-results branch, and the name cue: the bare name found from a router that answers `none`, and answered without waiting for the router; a name among other words found together with what the router found, and found when the router fails; only the exact name counting; an MCP tool found by its name; a tool the catalogue does not offer not found. `tests/test_tool_selection.py` holds `select_tools` to offering nothing for a spoken word that happens to be a tool's name.
- An eval scenario covers the "initial routing was too narrow" case: the user starts with a vague question that routes to one tool, then clarifies into a request that needs a different tool. The agent should invoke `toolSearchTool` and then the newly-surfaced tool.

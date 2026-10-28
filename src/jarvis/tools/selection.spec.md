## Tool Selection Spec

Selects a subset of available tools relevant to a given user query, so the LLM receives only tools it is likely to need. Reduces noise for smaller models and lowers token cost.

### ToolSelectionStrategy Enum

```python
class ToolSelectionStrategy(Enum):
    ALL = "all"
    KEYWORD = "keyword"
    EMBEDDING = "embedding"
    LLM = "llm"
```

### Strategies

Controlled by `tool_selection_strategy` in config:

| Value         | Behaviour                                                           | LLM call? | Extra dependency |
|---------------|---------------------------------------------------------------------|-----------|------------------|
| `"all"`       | Pass every registered tool.                                         | No        | None             |
| `"keyword"`   | Score tools by keyword overlap with the query; return top matches.  | No        | None             |
| `"embedding"` | Rank tools by cosine similarity of embeddings via nomic-embed-text. | No        | numpy            |
| `"llm"`       | Ask a lightweight LLM call to pick the top 3–5 relevant tool names (default). | Yes | None |

### The catalogue

The builtins handed to the strategies are those a tool's own settings and state leave on (`offered_builtin_tools`, from `Tool.offered`): `claudeCode` with no projects folder can only refuse, and shown anyway it would cost every routing call its description; `stopClaudeCode` is shown only while a job runs, since otherwise there is nothing to stop and it would only be one more place for a bare "stop" to land. The reply engine and `toolSearchTool` both select from this catalogue, and the router cache key names it, so switching a tool on or off takes effect on the next turn. A call to a tool that was not offered still reaches it, and it says why it cannot act.

### Always-included Tools

Regardless of strategy, these tools are **always** included:
- `stop` — needed so the user can dismiss the assistant at any time.
- `remember` — an explicit "remember that…" must never be silently dropped because the router failed to offer the tool. Being always available is safe here because the tool is inert unless the user asked for it, and because every write lands in a plain-text file the user can read and edit.
- `forget` — same argument, with more at stake: a user asking for something to be removed and being told "done" while it stays in every later prompt is the worst outcome the core can produce.

Those the strategy did not pick itself are appended in a fixed order, `stop`, `forget`, `remember`, after whatever it did pick. The planner reads the catalogue in that order and a small one plans differently when `remember` moves, so the order is part of the behaviour: an order left to a set's hash seed made the same sentence plan differently from one start to the next, and made every eval measure one draw of it. `remember` goes last because it is the tool a small planner reaches for to restate a request it was given ("remember that the user wants a bug fixed"), and that step writes into the core; on `gemma4:e2b` this order did so 12 times in 30 on delegations to Claude Code, the five others 18 to 21.

### URL cue

Independently of any strategy, a query carrying a URL (`http://`, `https://`
or `www.`) unions `fetchWebPage` into the selection, when the catalogue has
it. The router is a small model and a pasted address does not always read as
a fetch to it ("what do you think of https://…" scores as opinion); the cue
is deterministic and language-agnostic, so the page tool never depends on the
router having understood. It applies to every strategy's result, including a
degraded keyword fallback, and mutates the list in place so a `ToolSelection`
keeps its `degraded` flag. A dotted word that is not an address
(`requirements.txt`) does not trigger it.

### Keyword Strategy

1. Build a keyword index per tool from its `name` (camelCase split) and `description` (lowercased, stop-words removed).
2. Tokenise the user query (lowercase, split on whitespace/punctuation).
3. Score each tool: count of query tokens that appear in the tool's keyword set.
4. Return tools with score > 0, plus always-included tools.
5. If no tools score > 0, fall back to returning all tools (query is too vague to filter).

### Embedding Strategy

1. Embed the user query using `get_embedding()` (calls Ollama `/api/embeddings` with the configured embed model).
2. For each tool (excluding always-included), build a summary string from the tool name (camelCase split) and description, then embed it.
3. Compute cosine similarity between the query embedding and each tool embedding.
4. Select tools using a **relative threshold**: keep tools whose similarity >= `top_score * _RELATIVE_THRESHOLD` (0.97 — nomic-embed-text has a high baseline similarity, so a loose threshold lets the entire catalogue through).
5. If fewer than `_MIN_SELECTED` (3) tools pass the threshold, return the top 3 by similarity.
6. Append always-included tools.
7. If the query embedding fails, fall back to returning all tools.

Note: embedding is **not** the default strategy because nomic-embed-text produces tightly clustered similarities across all tools — the filter struggles to separate "good match" from "generic cluster" when a realistic MCP catalogue (20–40 tools) is in play. The `llm` strategy is cheaper in prompt size and more discriminative on small chat models.

### LLM Strategy (default)

1. Build a catalogue of `- name: description` lines (descriptions truncated to 120 chars) for every registered tool except always-included ones.
2. Send to `call_llm_direct` with a system prompt asking for the **top 5 most relevant** tool names as a comma-separated list. The prompt instructs the router to prefer 1–3 tools for narrow queries and to return `"none"` for greetings/small talk.
3. Parse the response, matching tokens against known tool names (unknowns are dropped silently).
4. Apply a hard `_LLM_MAX_SELECTED` (5) cap regardless of what the router returned, to guard against chatty routers that echo the whole catalogue.
5. Append always-included tools.
6. If the router replies `"none"`, return only the always-included tools.
7. When the router cannot answer, fall back to the **keyword strategy**. That covers a timeout (the call waits at most `tool_router_timeout_sec`), an empty response, a parse failure (no token in the response matched a known tool name), no model configured, no backend supplied, and a backend that raises. The pick comes back as a `ToolSelection` (a list) whose `degraded` flag is true, so a caller that caches picks leaves it out and the next turn asks the router again; the reply engine also prints the tools line as `llm->keyword`. The fallback goes to the keyword strategy rather than to the full catalogue. Reasoning: the catalogue can grow to 30–40 tools once an MCP server like `chrome-devtools` is enabled, and exposing all of them to a small chat model (gemma4:e2b class) overwhelms tool selection, producing empty replies. Keyword scoring narrows on query/name overlap deterministically, and the engine's `toolSearchTool` escape hatch still lets the chat model widen mid-loop if the keyword pick missed.

#### Context-aware routing

When the reply engine passes a `context_hint`, it is split into two labelled semantic slots in the router system prompt:

- **KNOWN FACTS** — things the assistant can already see (current time, detected location). If the query is answerable purely from these, the router should return `none`.
- **RECENT DIALOGUE** — recent user/assistant turns. The router is instructed to read the current query as a continuation of this exchange, so short follow-ups (e.g. "I'm in London" after "which city?") route to the tool that answers the combined intent across turns rather than being treated as idle chatter.

The split is the exact marker `"Recent dialogue (short-term memory):"` — any content before it is known facts, content after it is recent dialogue. If no dialogue marker is present, the whole hint is treated as known facts.

### Interface

```python
def select_tools(
    query: str,
    builtin_tools: Dict[str, Tool],
    mcp_tools: Dict[str, ToolSpec],
    strategy: ToolSelectionStrategy = ToolSelectionStrategy.ALL,
    llm_base_url: str = "",
    llm_model: str = "",
    llm_timeout_sec: float = 8.0,
    embed_model: str = "",
    embed_timeout_sec: float = 10.0,
) -> List[str]:
    """Return list of tool names relevant to the query."""
```

### Integration

Called from the reply engine (Step 6) before `generate_tools_json_schema()` and `generate_tools_description()`. The returned list replaces the current `allowed_tools = list(BUILTIN_TOOLS.keys())`.

### Configuration

- Key: `tool_selection_strategy`
- Type: `str` (validated against `ToolSelectionStrategy` enum values)
- Default: `"llm"`
- Valid values: `"all"`, `"keyword"`, `"embedding"`, `"llm"`

- Key: `tool_router_model`
- Type: `str`
- Default: `""` (empty string — resolves to `intent_judge_model`, then `llm_chat_model`)
- Effect: when `tool_selection_strategy == "llm"`, this model is used for the routing call. Resolution order for the empty default: `intent_judge_model` first (small, fast, already warm for wake-word paths and structurally the same classification job), then `llm_chat_model` as a last resort. Override `tool_router_model` explicitly to decouple routing from both — useful when you want routing on a dedicated third model.

- Key: `tool_router_timeout_sec`
- Type: `float`, clamped to 1–120 seconds
- Default: `15.0`
- Effect: the longest the routing call waits for its model before the selection falls back to the keyword strategy. The router is a classification call that stands between the user's sentence and the first token of the reply, so it has a deadline of its own rather than the minutes-long `llm_tools_timeout_sec` shared with other contexts: a provider that accepts the request and then goes silent costs the user this long, not minutes. The reply engine and `toolSearchTool` both read it through `router_timeout_sec(cfg)`, and so does the memory extractor, which rides the router's model chain and backend unless `memory_extractor_model` names a model of its own, and would otherwise wait out the shared ceiling (twice, for its retry) on the same stalled provider. The value is the wait for the server as the HTTP client measures it (the connection and each read), not a wall-clock cap on the whole exchange: a server that trickles bytes defeats it, and a silent one does not. The router's call is not streamed, so the response arrives in one piece. Choose it near three times the model's healthy answer time: shorter turns a slow answer into a keyword selection, longer waits out stalls. The memory extractor's answer is larger than the router's, so keep the value above about three times the slower of the two. Raise it for a large or cold dedicated local router, and under `low_power_mode`, where the start-up warm-ups are skipped and the first call has to load the model.

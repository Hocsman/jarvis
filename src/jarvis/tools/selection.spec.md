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

1. Build a catalogue of `- name: description` lines for every registered tool except always-included ones, each description read as the router reads it (see What the router reads).
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

#### What the router reads

One line per tool: its name and the whole sentences of its description that fit in 200 characters (`_router_summary`), the first sentence alone when that one is longer, the whole text when it has no sentence end, flattened to one line. The planner reads the first 120 characters of the same description. So the sentence that decides routing comes first, and its opening also decides planning. The keyword and embedding strategies read the whole description.

A description says what the tool does in the words the user asks for it with, because the router matches a request against those words, not against what the tool can technically do:

- `localFiles` finds "the user's files and documents on their computer". Described as acting "within your home directory", it is not offered for "trouve la facture PDF sur mon ordinateur": qwen2.5:3b answers `none`. The user's verbs are not its operations, so a call naming one ("search", "open") is answered with the operations there are and how to find a file (`list` with a glob, recursive).
- `fetchWebPage` opens "a web page or article". Described as fetching "a web page URL", a request for an article that gives no address gets `webSearch` alone.
- `setRoutine` opens with "routine", the word the user says when asking for one, and shows what a repeat sounds like ("every morning, every Monday"). Without the examples, a recurring request is routed on what will be fetched each time ("the weather"), never on the fact that it recurs.

The descriptions are read together, so changing one moves requests meant for another. With `fetchWebPage` opening "Fetch and extract text content…", "tous les lundis matin, fais-moi un point sur la météo" goes to `getWeather, fetchWebPage, webSearch`; changing `fetchWebPage` and `localFiles` alone, `setRoutine` as it was, takes it from 8 to 22 of 30 on qwen2.5:3b. The French city-time questions ("quelle heure est-il à Tokyo ?"), where qwen2.5:3b puts `getWeather` first and `getTime` second, lose `getTime` more often after any change to the catalogue. A description change is therefore measured on every router case, the evals' and their French counterparts, both arms interleaved call by call, never only on its own tool's cases.

The shape of the call counts too. The reply engine always sends the live time and location as KNOWN FACTS; `toolSearchTool` and the bare evals send none. With no KNOWN FACTS block, qwen2.5:3b answers `none` to "trouve la facture PDF sur mon ordinateur" whatever `localFiles` says, and it is the prompt's `none` rule that decides it (the same prompt without that sentence picks a tool). With a time and location line, the request reaches `localFiles`.

These descriptions against their earlier wording, arms interleaved call by call, on 62 router cases (27 English from the evals, 35 French), 15 requests for these three tools in French and English (bare, and with a time and location line), and nine requests where they do not belong:

| Cases | qwen2.5:3b, 30 calls an arm: these / earlier | gemma4:e2b, 15 calls an arm: these / earlier |
|---|---|---|
| The 62 router cases | 85.1% / 84.1% (p = 0.11) | 79.6% / 80.0% (p = 0.57) |
| The 15 requests, bare | 63.6% / 38.4% | 84.9% / 83.6% |
| The 15 requests, with the line | 68.7% / 49.8% | 84.0% / 88.9% (p = 0.18) |
| "tous les lundis matin, fais-moi un point sur la météo" → `setRoutine`, bare; with the line | 27 / 8; 19 / 2 of 30 | 0 / 0; 14 / 0 of 15 |
| "every evening at 9, tell me what I ate that day" → `setRoutine`, bare; with the line | 30 / 14; 22 / 7 of 30 | 0 / 8; 2 / 12 of 15 |
| "Jarvis, mets en place une routine quotidienne qui cherche sur le web et me résume l'actualité." → `setRoutine`, with the line | 4 / 2 of 30 | 1 / 15 of 15 |
| "set up a daily routine that searches the web and sums up the news for me" → `setRoutine`, with the line | 3 / 21 of 30 | 15 / 15 of 15 |
| "trouve la facture PDF sur mon ordinateur" → `localFiles`, bare; with the line | 0 / 0; 30 / 0 of 30 | 15 / 15; 15 / 15 of 15 |
| "trouve-moi un article détaillé sur le programme Apollo" → `webSearch` + `fetchWebPage`, bare; with the line | 23 / 7; 30 / 28 of 30 | 15 / 15; 15 / 15 of 15 |
| "quelle heure est-il à Tokyo ?" → `getTime`, bare | 7 / 24 of 30 | 15 / 15 of 15 |
| Requests where these tools do not belong | 270 / 270 of 270 | 119 / 135 of 135 |

The gains are where qwen2.5:3b misses these tools with the earlier wording; gemma4:e2b offers the files and the articles either way. The losses are requests whose routing moves with any change to the catalogue: a repeated request that names the web goes to `webSearch` and `fetchWebPage` (qwen2.5:3b in English, gemma4:e2b in French), on gemma4:e2b one that names the evening goes to `setReminder`, "cherche sur internet des tutos Python" also gets `fetchWebPage` (15 of 15, an extra tool within the cap), and one one-off reminder in 15 also gets `setRoutine`, which still asks before it creates anything.

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

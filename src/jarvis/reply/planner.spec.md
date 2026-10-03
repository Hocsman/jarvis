# Task-list planner

## Purpose

Small chat models (gemma4:e2b class) don't reliably decompose multi-step
queries turn-by-turn. They stop after one tool call when a second is
needed, echo the raw user utterance into tool arguments, or skip tools
entirely and confabulate from training. The planner fixes this by
running a single cheap classification-shaped LLM pass **ahead of memory
search** that emits a short ordered list of sub-tasks.

The planner runs **after the tool router** and **before memory search**.
The router narrows the catalogue first so the planner's tool steps reference
concrete chosen names; the planner then **gates memory enrichment** and
**drives direct execution** for small models.

The engine uses the plan for three things:
1. **Gate memory enrichment** — the planner emits an explicit
   `searchMemory topic='<topic>'` directive on queries that need past
   user context; otherwise we skip the keyword-extraction LLM call and
   the keyword-driven diary / graph lookups. The graph is still read on
   the arguments of the plan's own tool steps, and the memory-digest LLM
   call runs only when the diary or graph produced text.
2. **Confirm the tool allow-list** — the router's picks are
   authoritative; the tool names the planner references are unioned
   in as a safety net. Feeding the planner the narrowed catalogue
   (instead of the full 30+ list) stops small planners from
   paraphrasing ("get the weather") and from defaulting to
   `webSearch` when a more specific tool exists.
3. **Drive direct execution** for small models, as before — each
   planned step is resolved to a concrete tool call without
   round-tripping the chat model for intermediate turns.

## Scope

This spec covers `src/jarvis/reply/planner.py` and the engine
integration in `src/jarvis/reply/engine.py`.

## Behaviour

### When the planner runs

- After the dialogue context is assembled, MCP tools are loaded, and
  the tool router has produced a narrowed catalogue. Memory search
  runs *after* the planner so it can be gated on its output.
- The planner sees the **router-narrowed** tool catalogue (name +
  one-line description), not the full 30+ list. It does not see memory
  content — it decides whether memory is needed, via the
  `searchMemory` directive.
- Only when the query is at least `MIN_QUERY_CHARS` long (default 4).
  Pure noise like "hi" / "ok" still short-circuits.
- Only when `cfg.planner_enabled` is True (default).
- Only when a planner model resolves (see Model resolution).

### Model resolution

1. `cfg.planner_model` (explicit override, for benchmarking)
2. `cfg.llm_chat_model`

The planner must track the chat model. The plan is the scaffolding the
chat model follows; a weaker planner on top of a stronger chat model
produces bad scaffolding the chat model then fights against. The chat
model is also the one the user picked during setup as their quality
target, so upgrading it (through the setup wizard or config) must
automatically upgrade plan quality without requiring a second choice.

Note: the planner pays a cache miss relative to the tool router, which
*does* ride the warm small model. This is the intended trade-off —
plan quality drives everything downstream, router quality only narrows
one turn's allow-list.

### Prompt contract (plan_query)

The planner prompt instructs the model to emit:

- Short imperative sub-tasks, one per line.
- At most `MAX_STEPS` (default 5) steps.
- As the FIRST step, a `searchMemory topic='<topic>'` directive **only
  when** answering requires information the user shared in prior
  conversations. Omit otherwise — every extra directive is an
  avoidable LLM call downstream.
- Tool names from the provided catalog only (exact match), for any
  concrete tool step.
- Concrete arguments composed against dialogue context, not the raw
  utterance. Optional arguments that the user did not supply must be
  omitted, not fabricated from unrelated words.
- Angle-bracket placeholders (e.g. `<director name from step 1>`) for
  entities the lookup will reveal at runtime.
- Pronouns and demonstratives in the user query ("he", "his", "her",
  "their", "it", "that film") must be resolved against the dialogue
  context before emitting the step. Tools never see prior turns, so
  the named entity has to appear literally inside the tool argument
  string — `webSearch query='Harry Styles most famous songs'`, not
  `webSearch query='his most famous songs'`.
- A final synthesis/reply step when any `searchMemory` or tool step
  was planned.
- Steps in the same language the user wrote the query in.

### Parsing and hygiene

- Numbering (`1.`, `1)`), bullets (`-`, `*`, `•`), wrapping quotes,
  and markdown fences are stripped.
- Overlong steps (>200 chars) are truncated with an ellipsis.
- The list is capped at `MAX_STEPS`.
- The planner does not filter out 1-step plans. A single
  `["Reply to the user."]` plan is the planner's *positive* decision
  that no memory or tools are needed — the engine uses that to skip
  the memory extractor, the diary / graph / digest lookups, and the
  direct-exec path entirely. The router has already run by then, so its
  allow-list still stands. A single-step tool plan (e.g.
  `["webSearch query='...'"]`) preserves its executable tool step so the
  direct-exec path dispatches it directly on turn 1. Only an **empty**
  list means "planner failed / disabled": the engine runs memory
  enrichment (subject to the recall gate) and keeps the router's
  allow-list as it is. These states stay distinguishable.

### Engine integration

The engine consumes the plan in two phases.

**Phase 1 — preparation gating (before the turn loop starts):**

- `plan_step_args(step)` — the `key='value'` pairs a step carries. The
  single parser for that shape, shared with the direct-exec fast path.
- `lookup_terms_of(step)` — a step's concrete argument values joined into
  one search string, or `""` when the step carries an angle-bracket
  placeholder or no arguments. Arguments are composed against the user's
  intent with pronouns resolved to literal entities, so they name the
  subject better than the utterance does; the engine uses them to ask the
  graph whether this lookup has already been made.
- `plan_requires_memory(plan)` — true iff any step is a `searchMemory`
  directive. The engine uses it to gate the memory-enrichment block
  (keyword extractor LLM call and the keyword-driven diary / graph
  lookups; the digest LLM call follows only when they found text).
  Optional `memory_topic_of(step)` extracts the directive's
  `topic='...'` hint, threaded into the keyword extractor so it
  anchors on what the planner wanted to look up rather than
  re-deriving from the raw utterance.
- `tool_names_in_plan(plan, known_names)` — ordered de-duped list of
  tool names the planner referenced. The engine unions this into the
  router-selected allow-list (never replaces it). `stop` and
  `toolSearchTool` are always added regardless.
- `plan_has_unresolved_tool_steps(plan, known_names)` — true when the
  plan has non-synthesis steps but names no known tool (e.g. the
  model wrote `get the weather` instead of `getWeather ...`). In
  this state the direct-exec path is skipped — vague step text
  would otherwise force the resolver LLM to guess arguments (e.g.
  emitting `location='Nowhere'` for a bare weather request). The
  chat model takes the turn instead, using the router-selected
  allow-list.
- `strip_memory_directives(plan)` — the engine strips the
  `searchMemory` step from the plan once memory has been fetched, so
  downstream consumers (system-message injection, direct-exec,
  progress nudge) see a plan of pure tool + synthesis steps.

**Phase 2 — loop integration (existing behaviour):**

- `format_plan_block(steps)` renders an `ACTION PLAN:` block that is
  appended to the initial system message. Empty plan renders nothing.
  Single-step reply-only plans are not rendered either — they are
  noise to the chat model since the plan just says "reply". Single-step
  tool plans are rendered so the chat model is aware of the committed
  sub-task during final synthesis.
  The block tells the model a step may be skipped when a prior tool
  result **or the memory already in its system prompt** satisfies it.
  Scoped to prompt memory rather than "what you already know", so it
  cannot be read as licence to answer a chained-research step from
  training priors. The engine drops a step deterministically when every
  word it would look up is covered; this clause is what reaches the
  cases no lexical rule can — chiefly a step and a stored note written
  in different languages.
- `progress_nudge(steps, tool_results_so_far)` produces a remainder
  hint injected after each tool result, naming the next planned step
  and reminding the model to substitute discovered entities and avoid
  duplicate arguments.
- When `use_text_tools` is active and the plan still has unexecuted
  tool steps (whether single-step or multi-step), the engine runs
  `resolve_next_tool_call` to convert the next step into a concrete
  `{name, arguments}` JSON and dispatches the tool directly, bypassing
  the chat model for that turn. This keeps small models on-rails without
  relying on their native tool-call reliability.
- The chat model still runs the final synthesis turn so the reply is
  phrased in the daemon's voice using its own profile and persona.

### resolve_next_tool_call

- **Fast path**: if the step text is fully concrete (tool name in the
  allow-list + `key='value'` / `key="value"` pairs matching the tool's
  declared property keys, and no `<placeholder>`), parse it
  deterministically and return without any LLM call. This removes the
  resolver LLM as a failure surface for the common case — small models
  occasionally flake (timeout, empty, spurious `null`) even on
  trivially-concrete steps like `webSearch query='foo'`, and without the
  fast path such a flake falls back to the chat model, which tends to
  produce a refusal instead of the search. The fast path is purely
  regex-driven, language-agnostic, and never calls the model.
- **LLM path**: when the step contains a `<placeholder>`, uses unknown
  argument keys, or doesn't fit the `key=value` shape, the step is
  passed to the LLM resolver which can substitute entities from prior
  results and remap names. The resolver also receives the turn's
  prompt-memory (`memory_context`: the memory digest, or the raw
  graph-context block when digestion is off or failed — the same value
  the engine's plan-step drop guard keys on) as a background-data block
  fenced by `<<<BEGIN/END RELEVANT MEMORY>>>` markers, so a fact
  recalled by enrichment (e.g. the user's city) can ground an argument
  the step leaves out. The fence is structural, not a label: the block
  can carry web-derived text, and this prompt's output is executed. The
  fast path never sees memory: a fully concrete step is dispatched
  exactly as written.
- Returns `None` for synthesis steps (the LLM emits the literal
  `null`), unknown tools, or invalid JSON. All `None` paths fall back
  to the normal chat-model turn.
- Validates the tool name against the provided schema's allow-list.
- **Substitution guard**: when the step text heads with an allow-listed
  tool name, a resolved call naming a different tool is rejected
  (`None`) — the plan asked for the head tool, and the resolver's
  output is executed without the chat model ever seeing it, so a
  substitution (misresolution, or memory-borne injected text steering
  the choice) must not be dispatched.
- Filters the returned `arguments` against the tool's declared
  JSON-schema property keys; unknown keys are dropped before dispatch.
  Tools that declare no properties keep the args as-is (they are
  free-form by design).
- Tolerates markdown fences the model may add despite instructions.
- Both planner LLM calls (`plan_query` and `resolve_next_tool_call`)
  request `num_ctx=8192` from Ollama so enriched memory and tool
  catalogue don't silently truncate in the 4096-token default window.
  The resolver additionally runs at temperature 0: it composes JSON,
  not prose.

## Fail-open invariants

- Timeout, empty response, or exception in the planner LLM call →
  return `[]`.
- Invalid JSON in the step resolver → return `None` and let the chat
  model handle the turn normally.
- No plan never worsens the baseline; the engine behaves exactly as it
  did pre-planner.

## Configuration

| Key | Default | Purpose |
|-----|---------|---------|
| `planner_enabled` | `True` | Feature gate. |
| `planner_model` | `""` | Explicit planner model override. |
| `planner_timeout_sec` | `6.0` | Timeout for plan and step-resolver LLM calls. |

## Non-goals

- The planner does not re-plan mid-turn. If the emitted plan is wrong,
  the engine still progresses via the chat model's native tool calls.
  When the chat model produces natural-language content the loop
  terminates immediately.
- The planner does not validate semantic correctness of the plan; it
  trusts the model to produce sensible steps and relies on the
  resolver's schema-level guard to reject unknown tools.
- Plans are not cached across turns. Each user utterance gets its own
  plan because the dialogue state and entity references change.

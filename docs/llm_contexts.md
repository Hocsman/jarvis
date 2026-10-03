# LLM Contexts Map

Every distinct LLM call in Jarvis, what feeds it, what consumes it, and how it is gated. This is the reference for optimising the app's main bottleneck (LLM latency). Keep it in sync with the code — see the note at the bottom.

> **Backend abstraction.** Every context below routes through `jarvis.llm` ([spec](../src/jarvis/llm/llm.spec.md)) via `get_llm_backend(cfg)`, `get_auxiliary_backend(cfg, model)`, `get_private_backend(cfg, pinned)` or `get_embedding_backend(cfg)`. Picking `llm_provider: openai_compatible` swaps the wire shape end-to-end without touching call sites. On a remote OpenAI-compatible endpoint, a *bare* local model tag (no vendor slash) handed to `get_auxiliary_backend` routes to local Ollama; namespaced tags stay on the configured provider, and so does everything on a local endpoint. `get_private_backend` is stricter: any non-empty pin routes to local Ollama, whatever the endpoint. The active chat model is read directly from `cfg.llm_chat_model`, the `Settings` field that always carries the resolved value: `ollama_chat_model` on the Ollama provider, `llm_chat_model` (falling back to `ollama_chat_model` when empty) on an OpenAI-compatible one.

---

## 1. Main Reply Loop (agentic messages loop)

- **File**: [src/jarvis/reply/engine.py](src/jarvis/reply/engine.py) — `run_reply_engine()` and its agentic loop; every model call goes through the module-level `chat_with_messages()`.
- **Trigger**: every user message. Runs up to `agentic_max_turns` (default 8) iterations per reply.
- **Model / gating**: `cfg.llm_chat_model` via `get_llm_backend(cfg)`. Not optional. No size branching on the loop itself — size branching affects the digests/evaluator around it.
- **Inputs**:
  - Redacted user query
  - Recent dialogue (last 5 minutes), including in-loop tool-call + tool-role messages from prior replies within the active conversation (tool carryover, `DialogueMemory.record_tool_turn` / `get_recent_turns_with_tools` in [src/jarvis/memory/conversation.py](src/jarvis/memory/conversation.py); per-prompt cap via `cfg.tool_carryover_max_turns` / `tool_carryover_per_entry_chars`; storage cap `_tool_turns_max_storage = 16`; cleared on `stop` signal AND on new-conversation entry; UNTRUSTED WEB EXTRACT fence markers preserved on truncation; both `content` and `tool_calls[*].function.arguments` scrubbed on write)
  - Unified system prompt from [src/jarvis/system_prompt.py](src/jarvis/system_prompt.py) + ASR note + tool-protocol guidance. Its memory paragraph states that long-term memory is written by the model itself through the `remember` tool and by nothing else, makes the call mandatory on an explicit request or a correction (answering "noted" without calling it saves nothing), and forbids it at any other time. Both halves are pinned by `evals/test_remember_tool_is_called.py`, whose negative cases exist because a prompt tuned only to raise the call rate starts writing facts the user never asked to store.
  - **Core profile block** (query-agnostic profile + rules read from the user's core files, composed by `build_core_profile()` / `format_warm_profile_block()` in [src/jarvis/memory/core.py](src/jarvis/memory/core.py) at Step 3.5 of `run_reply_engine()`; no LLM call, two file reads; injected on every attended turn so personalisation is the default, and withheld from a routine turn unless that routine's block in `yuba/routines.md` says `mémoire: oui` — see `src/jarvis/routines/routines.spec.md`; result cached in `DialogueMemory._hot_cache` under `DialogueMemory.WARM_PROFILE_CACHE_KEY` as `(fingerprint, block)` for the lifetime of the active conversation. Invalidated on `stop`, on new-conversation entry, on any core write in this process via the listener registered in [src/jarvis/daemon.py](src/jarvis/daemon.py) against `register_core_mutation_listener`, AND by a `MemoryCore.fingerprint()` comparison on entry, which is what catches an edit made from the memory viewer or the user's own text editor — other processes, where the listener never fires)
  - **Open goals block** (`format_objectifs_block` in [src/jarvis/objectifs/prompt.py](src/jarvis/objectifs/prompt.py); no LLM call, one `stat()` on a file that is usually absent). One line per open goal — its name, what it is, and the last thing the user said about it — so she recognises the subject when it comes up; the history is one `listGoals` call away, on the turns that want it. Only points whose source is `dit` appear, so that the day a pass can write a line of its own, that line does not arrive in an attended prompt looking like something he said. Withheld from a routine turn unless that routine's block in `yuba/routines.md` says `mémoire: oui` (for the same reason the profile is). Measured: 0.011 ms and zero tokens with no open goal, which is the ordinary case; 0.013 ms and ~200 tokens with four.
  - Digested memory enrichment (optional, see #4). Graph recall is framed as things the assistant looked up, describing the world rather than the user, and yielding to the core block above when the two disagree.
  - Time + location context (computed once per reply, placed at the END of the system message's dynamic region: never the head: so every in-loop call sends a byte-identical system message and the server's KV/prefix cache can reuse the whole prompt head; in text-tools mode it sits just before the tool-call syntax guidance so the instruction block stays final)
  - Tool schema: native via `generate_tools_json_schema()` ([src/jarvis/tools/registry.py](src/jarvis/tools/registry.py)) or text fallback via `_text_tool_call_guidance()` in the same engine module
  - Tool results from prior turns (raw or digested — see #5)
- **Output**: OpenAI-style `{content, tool_calls, thinking}`. Consumed by the tool orchestrator and TTS pipeline. Natural-language content is delivered immediately; no post-turn evaluator runs.
- **Limits**: `num_ctx: 8192` (explicit). Timeout `llm_chat_timeout_sec` (180s). Auto-fallback from native to text tool-calls on HTTP 400 (`ToolsNotSupportedError`), sticky for the session. Risk: `fetchWebPage` truncates at 50,000 chars (~37k tokens) — mitigated for SMALL models by tool-result digest (#5) which compresses the payload before it enters the messages history. LARGE models receive the raw payload and may silently see a truncated context.
- **Routine entry**: an unattended run calls this same context with `origin="routine"`, `tts=None`, and a `RoutineScope`. The scope replaces the router entirely — the envelope's tools that exist are the whole catalogue, `stop` and `toolSearchTool` are not appended, and one filter runs after every branch that can add a name. The scope is also handed to all three `run_tool_with_retries` call sites, where the gate re-checks it per call. No new LLM context: the planner and the digests run unchanged, and #7 (tool router) simply does not fire.
- **Text-chat entry**: The desktop `ChatWindow` (see `src/desktop_app/chat_window.spec.md`) submits via `jarvis.daemon.submit_text_query`, which calls this same context on a worker thread with `tts=None` and `language=None` (no Whisper-detected language for typed input). Voice and text share the global `DialogueMemory` so they are one conversation. No new LLM context is introduced — the planner, router, enrichment, and digests all run unchanged. Text chat never speaks; the reply is returned to the UI via callbacks (bundled) or `__CHAT__:` IPC events (subprocess).

## 2. Intent Judge

- **File**: [src/jarvis/listening/intent_judge.py](src/jarvis/listening/intent_judge.py) — `IntentJudge.judge()`.
- **Trigger**: on a speech segment *only if* there is an engagement signal (wake word detected, hot-window active, or TTS playing). Pure ambient speech skips it.
- **Model / gating**: `cfg.intent_judge_model` (default `gemma4:e2b`, ~2B) via `get_auxiliary_backend(cfg, cfg.intent_judge_model).chat(...)`. When `llm_provider` is a remote OpenAI-compatible endpoint and `intent_judge_model` is a bare local tag (e.g. `qwen2.5:3b`), the call routes to local Ollama, eliminating cloud network latency on every vocal utterance. If Ollama is unreachable, `_cloud_safe_model` rescues the pin to `cfg.llm_chat_model`. The backend re-raises `ConnectionError` so the judge can apply a 30s cooldown after the server actively refuses. The cooldown suppresses the backend call, not the fallback: `judge()` returns `None` immediately and the listener takes its no-verdict path — hot-window override inside a hot window, text-based wake detection outside it — and prints the unavailability to standard output.
- **Inputs**:
  - Rolling transcript buffer (last 120s, with timestamps)
  - Wake-word timestamp (if any), normalised aliases
  - Last TTS text + finish time (echo rejection)
  - State flags (wake_word_mode, hot_window_mode, during_tts)
- **System prompt**: `IntentJudge.SYSTEM_PROMPT_TEMPLATE`. Teaches query extraction, echo detection, stop commands, pronoun/topic disambiguation, imperative re-addressing, declaratives to the wake word.
- **Output**: strict JSON `IntentJudgment{directed, query, stop, confidence, reasoning}` (the `IntentJudgment` dataclass in the same file). Consumed by the listening state machine which dispatches to the reply engine.
- **Limits**: `intent_judge_timeout_sec` (15s). `num_ctx: 8192` (explicit; the system prompt is ~2k tokens and the rolling transcript buffer at default `transcript_buffer_duration_sec=120` can reach ~1.5k tokens in chatty multi-speaker scenes; the larger window gives the few-shot examples and TRANSCRIPT NOISE block at the tail of the prompt enough headroom on Ollama). `max_tokens: 1500` (covers reasoning + JSON answer on reasoning models; mapped to `num_predict` on Ollama). Ollama-only knobs (`keep_alive`, `num_ctx`) flow via `extra_options`; OpenAI-compatible backends silently drop them. `keep_alive` is `"30m"` by default and `"1m"` when `low_power_mode` is true.

## 3. Memory Enrichment Extractor

- **File**: [src/jarvis/reply/enrichment.py](src/jarvis/reply/enrichment.py) — `extract_search_params_for_memory()`.
- **Trigger**: once per reply, **only when the pre-flight planner (#12) emitted a `searchMemory` directive or returned an empty plan (fail-open)**. Pure reply-only plans skip this entirely — saves one LLM call per greeting / small-talk turn.
- **Model / gating**: resolved via `resolve_tool_router_model(cfg)` — `tool_router_model → intent_judge_model → llm_chat_model`. Dispatched via `get_auxiliary_backend(cfg, model)` (the enrichment shim): a bare local tag on a remote OpenAI-compatible endpoint routes to local Ollama. Small classification task; rides the same small/warm model as the router. Returns `None` when the pass could not run (no chat model configured — early return, no wasted LLM round-trip; nothing usable after both attempts; unexpected error). `None` is distinct from `{"keywords": []}`, which is the extractor deciding the query needs no memory search, and the engine prints a warning line for it.
- **Inputs**: user query (with the planner's `topic` hint appended when present), optional context hint (live-context compact summary) or UTC-now anchor, both carried in the USER message.
- **System prompt**: inline in `extract_search_params_for_memory()`. Byte-static: no hint block, no timestamp: so the system prompt is identical across every extractor call and stays cacheable; the per-call hint / UTC anchor rides at the end of the user content.
- **Output**: `{keywords, from?, to?, questions?}`. Consumed by memory search in the reply engine. A degenerate `from >= to` pair (the small model echoing its current-time hint) is dropped deterministically before consumption, so the recall stays unfiltered rather than pinned to today. Not the graph's only key: the concrete arguments of the plan's own tool steps also open it, so the graph is reachable on a turn where this context never fired. `keywords` drive both the diary search and the graph crawl; `questions` are optional and additive — they widen the graph's search text and annotate the hit in the log, but the graph is a world-fact index and they ask about the user, so they never gate it.
- **Limits**: up to 2 attempts (one retry on an unusable answer); timeout `llm_tools_timeout_sec` (300s, shared with #7, #8 and the weather place extractor in #14).
- **Caching**: result cached in `DialogueMemory._hot_cache` under a key made of the `enrichment:` prefix and the extractor query (the redacted query, with the planner's topic hint on a second line when there is one) for the lifetime of the active conversation. Identical follow-ups within the same conversation reuse the dict and skip the LLM hop. Cleared by `clear_hot_cache()` on the `stop` signal and on new-conversation entry. A failed extraction is never cached — the hot cache has no age-based expiry, so caching it would make one outage last the whole conversation.

## 3b. Recall Gate (pre-enrichment short-circuit)

- **File**: [src/jarvis/memory/recall_gate.py](src/jarvis/memory/recall_gate.py) — `should_recall()`.
- **Trigger**: once per reply, before diary/graph/digest enrichment runs (after the planner has decided memory is potentially needed).
- **Model / gating**: NO LLM — deterministic keyword-coverage heuristic. Cheap.
- **Inputs**: query, recent dialogue (incl. tool carryover rows).
- **Output**: `False` only if hot-window contains a fresh tool result AND ≥50% of the query's content words appear in the hot-window transcript → skips diary, graph, and memory digest for this reply. Else `True`. Fail-open on any exception. Content-word extraction uses `\w{3,}` with `re.UNICODE`, so the gate works for Latin, Cyrillic, CJK, Arabic, Hebrew, etc. (per CLAUDE.md "no hardcoded language patterns"). Overlap words are run through `redact()` before being written to debug logs.
- **Planner precedence**: when the planner explicitly emitted a `searchMemory` step, the gate is bypassed — the planner has more signal than coverage and overriding it would silently drop intent. The gate only short-circuits the fail-open empty-plan path.
- **Rationale**: prevents re-running diary/graph lookups when the hot window already grounds the follow-up (e.g. "his most famous song" after a Bieber webSearch).

## 4. Memory Digest (optional, auto-on for SMALL models)

- **File**: [src/jarvis/reply/enrichment.py](src/jarvis/reply/enrichment.py) — `digest_memory_for_query()` + `_distil_batch()`.
- **Trigger**: once per reply when enrichment returns hits AND the digest is on. `memory_digest_enabled` defaults to `null`, which means auto-ON for SMALL (≤7B) and OFF for LARGE; `true` / `false` force it. Skipped if raw < `_DIGEST_MIN_CHARS` (400). Batched if raw > `_DIGEST_BATCH_MAX_CHARS` (2000).
- **Model / gating**: `cfg.llm_chat_model` via `get_auxiliary_backend(cfg, model)` (the enrichment shim). Gated by `memory_digest_enabled`; the auto-on path reads the same chat model so model-size detection follows the active provider. The routing rule in the note at the top applies to the chat model's tag too: one with no vendor slash, on a remote endpoint, would route to local Ollama.
- **Inputs**: user query, raw diary entries, raw graph nodes.
- **System prompt**: `_DIGEST_SYSTEM_PROMPT`. Teaches relevance filtering, preference-signal detection, attribution preservation, `NONE` sentinel, identity queries.
- **Output**: text of at most `_DIGEST_MAX_CHARS` (500) per batch, injected as reference-only memory context into the main loop's system message. Empty only when the distil judged nothing relevant, which is what licenses the caller to drop the raw diary and graph blocks. Raises `MemoryDigestError` when a call could not be made; the caller then keeps those blocks and prints a fallback line.
- **Limits**: `llm_digest_timeout_sec` (8s, shared).

## 5. Tool-Result Digest (optional, auto-on for SMALL models)

- **File**: [src/jarvis/reply/enrichment.py](src/jarvis/reply/enrichment.py) — `digest_tool_result_for_query()` + `_distil_tool_batch()`, gated by `_maybe_digest_tool_result()` in [src/jarvis/reply/engine.py](src/jarvis/reply/engine.py).
- **Trigger**: after each tool result in the loop, if `tool_result_digest_enabled` (default `null` = auto-ON for SMALL ≤7B, OFF for LARGE). Primary motivation on small models: prevents `fetchWebPage`'s 50k-char payloads from filling the 8192 num_ctx window. Tools in `_DIGEST_SKIP_TOOLS` (output already short and structured: weather, time, the routine tools) bypass it. Skipped if raw < `_TOOL_DIGEST_MIN_CHARS` (400); batched if > `_TOOL_DIGEST_BATCH_MAX_CHARS` (2500).
- **Model / gating**: `cfg.llm_chat_model` via `get_auxiliary_backend(cfg, model)` (the enrichment shim). Gated by `tool_result_digest_enabled` — auto-on for SMALL via `detect_model_size(cfg.llm_chat_model)`. The routing rule in the note at the top applies to the chat model's tag too: one with no vendor slash, on a remote endpoint, would route to local Ollama.
- **Inputs**: user query, tool name, raw tool result (e.g. webSearch payload inside UNTRUSTED WEB EXTRACT fence).
- **System prompt**: `_TOOL_DIGEST_SYSTEM_PROMPT`. Teaches attributed fact extraction, `NONE` sentinel, no inference.
- **Output**: at most `_TOOL_DIGEST_MAX_CHARS` (600) per batch, replacing the raw payload in the messages stream. Falls back to raw on `NONE` and on any failure.
- **Limits**: `llm_digest_timeout_sec` (8s, shared).

## 6. Max-Turn Loop Digest

- **File**: [src/jarvis/reply/enrichment.py](src/jarvis/reply/enrichment.py) — `digest_loop_for_max_turns()`.
- **Trigger**: when the loop exhausts `agentic_max_turns` without producing a natural-language reply (e.g. pure tool-call loop). The evaluator no longer drives this — termination on content is immediate.
- **Model / gating**: `_resolve_loop_digest_model(cfg)` takes the first of `evaluator_model`, `intent_judge_model` and `ollama_chat_model` that is set. The last link is the Ollama alias, not the provider-aware `llm_chat_model`. Dispatched via `get_auxiliary_backend(cfg, model)` (the enrichment shim): a bare tag on a remote OpenAI-compatible endpoint routes to local Ollama.
- **Inputs**: user query + loop activity (tool calls, results summaries, any prose).
- **System prompt**: `_LOOP_DIGEST_SYSTEM_PROMPT` — caveat-prefixed, user-language, concise.
- **Output**: caveat-prefixed final reply. Fails open to the last raw candidate or generic error.
- **Limits**: `llm_digest_timeout_sec` (8s, shared).

## 7. Tool Router (pre-loop tool selection)

- **File**: [src/jarvis/tools/selection.py](src/jarvis/tools/selection.py) — `_select_llm()`, reached through `select_tools()`.
- **Trigger**: once per reply, **at the very front of the flow before the planner (#12)**. Runs on every attended turn (a hot-window cache hit skips the call, and a routine's envelope replaces it) — the router is the authoritative tool picker, and its narrowed catalogue is what the planner sees. When the planner later references tools, those names are unioned into the router's allow-list but never replace it; small models tend to default to `webSearch` where a dedicated tool like `getWeather` should win, and the router is tuned for that classification. `tool_selection_strategy == "llm"` is the default; other strategies (`all`, `keyword`, `embedding`) also run here.
- **Model / gating**: `resolve_tool_router_model(cfg)` chain — `tool_router_model → intent_judge_model → llm_chat_model`. Factory-dispatched via `get_auxiliary_backend(cfg, model)`. When `llm_provider` is a remote OpenAI-compatible endpoint and the resolved model is a bare local tag (e.g. `qwen2.5:3b`), the router call routes to local Ollama, avoiding cloud network latency ahead of the planner.
- **Inputs**: user query, tool catalogue (builtin + MCP with descriptions), optional narrow-down hint. The catalogue is one tool per line, so a name outside the plain class is omitted and every description is flattened to a single line: a line break in either would offer the model an entry nobody installed. User-prompt order is KV-cache-disciplined: the mostly-static catalogue opens, the dynamic hint (time + dialogue) follows, the query is the final token: consecutive router calls in one conversation share the full catalogue as prefix.
- **System prompt**: inline in `_select_llm()`. Teaches pick up-to-5 tools or `none`.
- **Output**: comma-separated tool names or `none`. Capped at `_LLM_MAX_SELECTED` (5). The always-included tools (`stop`, `remember`, `forget`, from `_ALWAYS_INCLUDED`) are unioned in regardless: an explicit "remember that…" or "forget that" must not be lost to a router miss. The engine then appends `stop` and `toolSearchTool` to every attended allow-list. The "router came back empty" fallback looks past the mandatory names at what actually matched, so adding one never changes when the fallback fires.
- **Limits**: `llm_tools_timeout_sec` (300s). On a timeout, an empty or unreadable answer, or no model at all, the router falls back to the keyword strategy (`_select_keyword`), which itself returns every tool when nothing matches.
- **Caching**: `routed_tools` cached in `DialogueMemory._hot_cache` under a key made of the `router:` prefix, the redacted query, the strategy and the sorted builtin and MCP tool names, joined by `|`, for the lifetime of the active conversation. The catalogue signature lets a mid-conversation MCP refresh invalidate the cache; `context_hint` is intentionally excluded so time/location drift inside one conversation doesn't bust it. An answer equal to the whole catalogue is never cached: it means the router gave up, and re-rolling it next turn is cheap. Cleared by `clear_hot_cache()` on the `stop` signal and on new-conversation entry.
- **Carry-over guard (engine-side overlay)**: after the cache lookup/write, the engine inspects the previous assistant turn's tool calls. When a previous tool reported `success=False` on its `ToolExecutionResult` (read via the `tool_failed` flag stamped onto each recorded tool result), that tool name is unioned back into the local `routed_tools` for this turn only. Compensates for small routers that misroute follow-ups where the user is supplying missing info (e.g. "I'm in London" routing to `webSearch` after a stalled `getWeather` chain). Successful chains do not carry over — a genuine new short ask after a completed chain keeps the router pick clean. The augmentation never touches the cache; replays of the same query in future turns get the raw router output. See `src/jarvis/reply/reply.spec.md` §6 (Tool allow-list per turn) for the full contract.

## 8. Tool Searcher (mid-loop escape hatch)

- **File**: [src/jarvis/tools/builtin/tool_search.py](src/jarvis/tools/builtin/tool_search.py) — `toolSearchTool`.
- **Trigger**: when the model explicitly invokes `toolSearchTool` during the loop. Capped at `tool_search_max_calls` (3) per reply.
- **What prompts the invocation**: the system prompt tells the model to reach for it before ever saying "I cannot", and the engine's refusal for a call outside the allow-list names it explicitly with one line on what it does. That refusal is the highest-signal moment for this context — the model has just proved its routing was too narrow — so `stop` and `toolSearchTool` are always named there whatever the list's length, and only trimmed ordinary tools are counted away. Under a `RoutineScope` neither is in the allow-list and neither is offered.
- **Model**: the tool router's chain and prompt (#7), through `select_tools()`: one router call per invocation, timed out at `llm_tools_timeout_sec` (300s), and no prompt of its own.
- **Inputs**: self-contained query from the model.
- **Output**: newline-separated tool names + one-liners, merged into the allow-list for the next turn.

## 9. Conversation Summariser

- **File**: [src/jarvis/memory/conversation.py](src/jarvis/memory/conversation.py) — `generate_conversation_summary()`, reached through `update_diary_from_dialogue_memory()`; its calls go through the module's `_direct_llm()` / `_stream_llm()`.
- **Trigger**: background, periodic when unsaved dialogue reaches `dialogue_memory_timeout`, plus the normal daemon shutdown path with `force=True`. `⚡ Stop Now (Skip Diary)` sets the daemon's shutdown skip flag, so this final forced pass is skipped while normal periodic saves remain unchanged. One summary is stored per day per `source_app`.
- **Model / gating**: `cfg.llm_chat_model` via `get_llm_backend(cfg)`. Respects `llm_thinking_enabled`. Uses streaming when a token callback is provided, else direct.
- **Inputs**: recent conversation chunks + prior same-day summary (for incremental update).
- **System prompt**: inline in `generate_conversation_summary()`. Hygiene rules per [src/jarvis/memory/summariser.spec.md](src/jarvis/memory/summariser.spec.md): no deflection narration, attribution preservation, topic separation. The deflection rule (rule 6) is enumerated with concrete BAD/GOOD pairs in English plus parallel pairs in Turkish and Spanish so small models don't assume the rule is keyed to English phrasing. Rule 9 sets the output language to the conversation's rather than the prompt's, names no language while doing so, and forbids translation; without it the model answers in the language it was instructed in, measured at two French rows in ten on a French-speaking machine. ≤200 words + 3-5 topic keywords.
- **Output**: `(summary, topics)` → `conversation_summaries` table, embedded for vector search, feeds enrichment (#3) and graph extraction (#10). No post-process scrub — the prompt is single-source-of-truth, language-agnostic, and improves automatically as the chat model upgrades.
- **Deflection rewrite (separate bulk op)**: `rewrite_all_diary_summaries()` (`POST /api/diary/scrub-deflections`) — cleans historical rows. One `cfg.llm_chat_model` call per row (30 s each) with `_REWRITE_DEFLECTION_SYSTEM_PROMPT`, asking the model to drop sentences that narrate the assistant's own failures while keeping everything else verbatim. Diary text is fenced as untrusted data (same fence used by the web tool). Preserves `ts_utc`; re-embeds updated rows best-effort via `get_embedding_backend(cfg)`. Empty-rewrite guard keeps the original if the model would have emptied the row. Fail-open at every layer (LLM call, write-back, embed). User-triggered from the Maintenance section in the diary sidebar.
- **Topic optimisation (separate bulk op)**: `optimise_diary_topics()` (`POST /api/diary/optimise-topics`) — collects all unique tags from `conversation_summaries`, makes one `cfg.llm_chat_model` call (60 s) with `_TOPIC_OPTIMISE_SYSTEM_PROMPT` to propose a normalised taxonomy (merge synonyms, split compound tags), then applies the mapping to every row that needs updating. Preserves `ts_utc`; re-embeds updated rows best-effort. User-triggered from the Maintenance section in the diary sidebar.
- **Limits**: the daemon passes `llm_chat_timeout_sec` (180s) for a periodic save and `SHUTDOWN_DIARY_TIMEOUT_SEC` (45s) for the forced pass at shutdown. The summariser's own default, used by direct callers, is 30 s.

## 10. Knowledge Graph Fact Extraction

- **File**: [src/jarvis/memory/graph_ops.py](src/jarvis/memory/graph_ops.py) — `extract_graph_memories()`.
- **Trigger**: after each daily summary (#9). Background.
- **Gated shut when no tool ran in the summarised window.** `tools_used` is read off `DialogueMemory.tools_in_pending_window()` in the same breath as the pending chunks; an empty sequence means the window held no lookup, so there is nothing to extract and this context does not fire at all. Filtering afterwards would still pay for the call and would leave the decision to the same would-a-different-assistant heuristic that a confidently invented statistic passes. `None` (the memory viewer importing a stored summary, where the tools are unrecoverable) is distinct from empty and does not gate. See [src/jarvis/memory/provenance.spec.md](src/jarvis/memory/provenance.spec.md).
- **Model**: `cfg.llm_chat_model` via `get_auxiliary_backend(cfg, model)` (the graph-ops shim). The routing rule in the note at the top applies to the chat model's tag too: one with no vendor slash, on a remote endpoint, would route to local Ollama.
- **Inputs**: summary text + optional date. The tools that ran reach this context as a gate and a source label, never as payload — tool content stays out of the diary.
- **System prompt**: inline — asks for a JSON array of strings, each an external fact the assistant looked up. The prompt bans anything about the user outright ("NEVER EXTRACT ANYTHING ABOUT THE USER"): identity, tastes, habits and standing instructions belong to the core ([src/jarvis/memory/core.spec.md](src/jarvis/memory/core.spec.md)), which is written only on an explicit request or a correction. The DO-NOT-EXTRACT block hardens two further traps: assistant-generated recommendations (would-a-different-assistant-give-the-same-answer? heuristic separates these from external lookups, which DO count as facts) and transient snapshots like the current weather / time of day (described as "moments not facts" so the model stops conflating ephemera with persistent climate / location knowledge).
- **Output**: list of fact strings → all routed into the World branch via branch-pinned descent. An object-wrapped fact (`{"fact": "..."}`) is still read, since model output is a boundary the codebase does not own.
- **Limits**: 30 s in the daily flow (the diary's timeout, capped at 30 s); `llm_chat_timeout_sec` (180s) in the memory viewer's diary import. Temperature 0: this is rule-following extraction, and a default sampling temperature makes small models drift off the banned-forms list. Failures → empty list.

## 11. Knowledge Graph Best-Child Picker

- **File**: [src/jarvis/memory/graph_ops.py](src/jarvis/memory/graph_ops.py) — `_llm_pick_best_child()`, called level by level from `find_best_node()`.
- **Trigger**: during graph insertion, per fact, to place it under the best existing category: one call per level of the descent from the World branch root, at most `MAX_TRAVERSAL_DEPTH` (8) levels. Background.
- **Model**: the router-chain model the caller hands to `update_graph_from_dialogue` as its optional picker model (the daemon and the memory viewer resolve it via `resolve_tool_router_model(cfg)`, so a small model when one is configured); falls back to `cfg.llm_chat_model`. Dispatched via `get_auxiliary_backend(cfg, model)` (the graph-ops shim): a bare local tag on a remote OpenAI-compatible endpoint routes to local Ollama.
- **Inputs**: fact text + numbered list of candidate child nodes (name + description).
- **System prompt**: inline in `_llm_pick_best_child()` — answer with number or `NONE`.
- **Output**: child node id or `None` (fact still inserted, just not under an optimal parent).
- **Limits**: 15 s per call.

## 11b. Knowledge Graph Node Merge (rewrite-on-write consolidation)

- **File**: [src/jarvis/memory/graph_ops.py](src/jarvis/memory/graph_ops.py) — `merge_node_data()` (system prompt at `_MERGE_SYSTEM_PROMPT`).
- **Trigger**: **once per (node, flush)** during `update_graph_from_dialogue`. The orchestrator first applies the exact-match dedupe fast-path, then groups the remaining facts by their resolved `node_id` so a 5-fact flush hitting the User node fires one rewrite, not five. Cold-start writes (empty target node) skip straight to plain append. Also invoked with `new_facts=[]` by the `consolidate_all_populated_nodes` maintenance op (powering the memory viewer's 🧹 button) to re-apply current rules to historical data.
- **Model**: the same picker model as #11 (small router model when configured, falls back to `cfg.llm_chat_model`), dispatched via `get_auxiliary_backend(cfg, model)` like #11. Temperature 0 — the task is rule-following classification.
- **Inputs**: existing node `data` + the batch of new facts (zero or more) routed to that node in this flush.
- **System prompt**: defines an ordered rule set — contradiction/reversal drops the old version, near-duplicate phrasings collapse to one, repeated daily activities consolidate into patterns, independent attributes coexist (visible contradictions are NOT silently dropped), common-knowledge facts are pruned. Demands a bare `{"facts": [...]}` JSON object. Parser tries direct `json.loads` first, then a scoped regex (no greedy `\{.*\}`) before giving up.
- **Output**: `MergeResult(success: bool, incorporated_indices: list[int])`. The revised fact list is written back as the node's full `data`; `incorporated_indices` tells the orchestrator which inputs survived as new lines (under NFKC + casefold matching) so consolidated-out facts aren't reported as "newly stored". Subsumes per-flush supersession, near-duplicate dedupe, and ongoing consolidation in a single call. Because the latest prompt rewrites the whole node, updated conventions propagate to old data without a separate migration step.
- **Limits**: 20s timeout. **Hallucination guard**: rewrites with more than `len(existing) + len(new) + 2` lines are rejected as runaway output. **Numeric guard**: a rewrite carrying a figure that was in neither the node nor the new facts is rejected too, since the line count cannot see a digit changed inside a line. Fail-open on any error, parse failure, oversized rewrite, ungrounded figure, or empty rewrite → caller falls back to plain `append_to_node` for each new fact so they still land (a contradiction is recoverable; a silent wipe or hallucinated bloat is not).

## 11c. Knowledge Graph Node Split (consolidation when a node outgrows its budget)

- **File**: [src/jarvis/memory/graph_ops.py](src/jarvis/memory/graph_ops.py), `auto_split_node()`.
- **Trigger**: per node, at the end of the placement pass in `update_graph_from_dialogue`, **only when the node's data has grown past `SPLIT_THRESHOLD` (1500) tokens** after the merge (#11b) or the plain append that replaced it. The gate is the node's size, so an ordinary node costs nothing here; `auto_split_node` re-checks the threshold itself and returns without a call when the node is missing or small. Background, like the rest of the graph pass: never on the reply path.
- **Model / gating**: `cfg.llm_chat_model` via `get_auxiliary_backend(cfg, model)` (the graph-ops shim). Unlike #11 and #11b, `auto_split_node` is never given the picker model. Respects `llm_thinking_enabled`. The routing rule in the note at the top applies to the chat model's tag too: one with no vendor slash, on a remote endpoint, would route to local Ollama.
- **Inputs**: the node's name, its description and its full `data` (every fact line, source suffixes included). Nothing else.
- **System prompt**: inline in `auto_split_node()`, byte-static (the node rides in the user message). Asks for 2-5 categories with every fact assigned to exactly one, consolidation of repeated activities into patterns, pruning of common knowledge, and a bare JSON object `{"categories": [{"name", "description", "facts"}], "summary": "..."}`.
- **Output**: read from the first `{` to the last `}` of the answer, and accepted only with at least two categories that each carry a name and at least one fact. Each category becomes a child node under the split node, the parent's `data` is cleared and its description becomes the summary. Facts are written as the model returned them; unlike #11b there is no step that restores a dropped provenance suffix. See `src/jarvis/memory/graph.spec.md` (Auto-Split).
- **Limits**: 45 s, fixed at the call site (the largest fixed budget in the daily graph pass). Default temperature. **Fail-open**: no response, no JSON, unparseable JSON, fewer than two categories or a category with no facts all return `False` before any node is created, so the node keeps its data and the next flush that touches it while it is still over the threshold retries.

## 12. Task-list Planner (pre-flight decomposition, gates the whole turn)

- **File**: [src/jarvis/reply/planner.py](src/jarvis/reply/planner.py) — `plan_query()`.
- **Trigger**: once per reply, **after the tool router and before memory search**. Skipped when `cfg.planner_enabled = False`, when the query is shorter than `MIN_QUERY_CHARS` (4), or when no planner model resolves.
- **Model / gating**: resolution chain `planner_model (override) → cfg.llm_chat_model`. Dispatched via `get_auxiliary_backend(cfg, model)` (the planner shim): a bare `planner_model` tag on a remote OpenAI-compatible endpoint routes to local Ollama. The planner tracks the active chat model so upgrading it (via setup wizard, config, or provider switch) automatically upgrades plan quality.
- **Inputs**: user query, dialogue context, **router-narrowed** tool catalogue (names + one-line descriptions, flattened and filtered like the router's) — not the full 30+ list. When the carry-over guard from #7 fires, the previous turn's failed tool name is unioned into this catalogue before the planner sees it, so the planner can plan a re-call without `toolSearchTool` round-tripping. **No** memory context — the planner decides *whether* memory is needed.
- **System prompt**: `_PROMPT_TEMPLATE` in `planner.py`. Teaches the `searchMemory topic='...'` directive for prior-conversation lookups, short imperative tool steps, angle-bracket entity placeholders, final synthesis step, same-language output, no numbering.
- **Output**: list of plan steps, at most `MAX_STEPS` (5). Gates memory enrichment (#3 / #4) and augments the tool router (#7 — planner's picks are unioned in, not replacing). Single-step `["Reply to the user."]` plans are the planner's positive "no memory, no tool step" signal (the router's allow-list still stands), while single-step tool plans preserve their executable tool step for direct execution. An empty list is fail-open — the engine reverts to running #3, subject to the recall gate (#3b). Consumed further by the engine to build the `ACTION PLAN:` system-message block and drive the direct-exec loop (#13) for small models.
- **Limits**: `planner_timeout_sec` (6s), `num_ctx: 8192`. Fail-open → `[]`.

## 13. Plan Step Resolver (per direct-exec turn, small models)

- **File**: [src/jarvis/reply/planner.py](src/jarvis/reply/planner.py) — `resolve_next_tool_call()`.
- **Trigger**: top of each agentic-loop iteration when `use_text_tools` is True AND the plan from #12 still has unexecuted tool steps (whether single-step or multi-step). Runs instead of the chat model for that turn. **Fast path skips the LLM entirely** when the step is fully concrete (tool name + `key='value'` args, no `<placeholder>`); the LLM call only fires when entity substitution or key remapping is needed.
- **Model**: same chain as #12, dispatched via `get_auxiliary_backend(cfg, model)` like #12. Temperature 0 — the resolver composes JSON, not prose.
- **Inputs**: next planned step text, prior tool calls (name + args + result excerpt), per-turn tool schema, and the turn's prompt-memory (`memory_context`, only when enrichment recalled something; empty otherwise): the memory digest, or the raw graph-context block when digestion is off or failed — the same value the engine's plan-step drop guard keys on, redacted at the source like everything that reaches a prompt. It rides inside a `<<<BEGIN/END RELEVANT MEMORY>>>` fence. The digest never reaches the fast path (fully concrete steps dispatch as written).
- **System prompt**: `_STEP_RESOLVER_SYSTEM` in `planner.py`. Teaches one-JSON-object output, placeholder substitution from prior results, grounding missing argument values from the memory block, `null` for synthesis steps.
- **Output**: `(tool_name, arguments)` tuple or `None`. Unknown tool names are rejected via the allow-list guard.
- **Limits**: `planner_timeout_sec` (6s), `num_ctx: 8192`. Fail-open → `None` (engine falls back to the chat-model turn).

## 14. Tool-specific LLM calls

- **Weather** ([src/jarvis/tools/builtin/weather.py](src/jarvis/tools/builtin/weather.py), `_extract_place_from_user_text()`) — only when the caller left `location` empty and the auto-detected location is unavailable. Dispatched via `get_auxiliary_backend(cfg, model)`: a bare local tag on a remote OpenAI-compatible endpoint routes to local Ollama. Place extractor model resolution: `tool_router_model → intent_judge_model → cfg.llm_chat_model` so small/warm models handle the parse without paging in the chat model. Pulls a single place name out of the redacted utterance (`none` when there is none), timed out at `llm_tools_timeout_sec` (300s).
- **Nutrition log_meal** ([src/jarvis/tools/builtin/nutrition/log_meal.py](src/jarvis/tools/builtin/nutrition/log_meal.py), `extract_and_log_meal()` and `generate_followups_for_meal()`) — dispatched via `get_llm_backend(cfg)`, so no auxiliary routing. Both the nutrition extractor and the follow-up generator use `cfg.llm_chat_model`, timed out at `llm_chat_timeout_sec` (180s), with `llm_thinking_enabled`. Each `logMeal` run extracts (retried within the tool's retry budget, the user text fenced as untrusted data), then asks for follow-ups once a meal has been logged. Extracts nutrients, confirms logging.

## 15. Approval judge (reads a spoken or typed yes/no)

- **File**: [src/jarvis/tools/confirmation.py](src/jarvis/tools/confirmation.py) — `read_approval()`.
- **Trigger**: the turn immediately after the gate raised a `demande` question **and** that question invited a spoken answer (`channel == parole`). `destructif` never reaches this call: it is settled by a click or not at all.
- **Model**: `confirmation_model → tool_router_model → intent_judge_model → cfg.llm_chat_model`, dispatched via `get_auxiliary_backend(cfg, model)`: a bare local tag on a remote OpenAI-compatible endpoint routes to local Ollama. The pin exists so this one reading can be kept on a local model when the rest of the small chain points at a remote endpoint.
- **Inputs**: the user's utterance, and nothing else. The pinned action is deliberately **not** in the prompt — the judge does not need to know what is being approved in order to read whether the sentence approves, and the blindness keeps injected page text out of this context and keeps the action off the network.
- **System prompt**: `_JUDGE_SYSTEM` in the same file. Names no language, because the user may answer in any. Ternary output, one bare token.
- **Output**: `oui` / `non` / `flou`. Only the exact token grants; a conditional yes is `flou`, because the judge cannot see what the condition refers to.
- **Limits**: `confirmation_timeout_sec` (default 8 s, clamped 2–30). **Fails closed**, unlike every other judge here: a timeout, an exception, an empty body, a model that explains itself, or an unrecognised word all come back as `flou`, which grants nothing and is not written to the ledger as a refusal (an empty utterance never reaches the model and reads as `non`). A false no costs a turn; a false yes runs something.

## 16. Reminder time extractor (turns a spoken time into an instant)

- **File**: [src/jarvis/reminders/extract.py](src/jarvis/reminders/extract.py) — `extract_reminder_time()`.
- **Trigger**: inside `setReminder`, once per call. Nowhere else.
- **Model**: `reminder_model → tool_router_model → intent_judge_model → cfg.llm_chat_model`. Unlike the auxiliary models, `reminder_model` is deliberately **not** passed through `_cloud_safe_model` (which replaces a bare pin with the chat model only when `llm_base_url` is a remote endpoint and local Ollama is unreachable, and prints the substitution): this prompt carries the user's own sentence about their own life, and a pin is the only way to keep it off the network. It is dispatched via `get_private_backend`, which routes an explicitly pinned model to the local backend: a model name alone never decides the destination, because `get_llm_backend` called without a model follows `llm_provider` alone. With no pin the ordinary provider applies: on a remote OpenAI-compatible endpoint the request goes to the remote provider while the model is the first link of the chain that is set. When local Ollama is reachable, `_cloud_safe_model` keeps the bare `intent_judge_model` default (`gemma4:e2b`), so that bare local tag is what the remote provider receives; when Ollama is not reachable, the chain's bare tags were already replaced by the chat model at load. Setting the pin is what keeps the call local. Contexts #17, #18 and #19 share the helper and the promise.
- **Inputs**: the reference moment (weekday, date and wall clock — no timezone name, because `%Z` is ambiguous and carries no offset) and the user's utterance, fenced as data.
- **System prompt**: `_EXTRACT_SYSTEM` in the same file. Names no day, no month, no temporal adverb, in any language, and carries no worked example — an example containing "tomorrow" is a hardcoded language pattern smuggled in by demonstration.
- **Output**: one JSON object, three kinds. `relative` returns the units it was given (`{"minutes": 20}`) and never a timestamp — the caller does the arithmetic, because a small model that cannot add twenty minutes to 12:47 can still copy the number 20. `absolute` returns `date` and/or `time`, and an omitted field is the statement: a day with no hour means the caller applies `reminder_default_hour` and says so aloud. `none` means nothing was asked for.
- **Limits**: `reminder_timeout_sec` (default 8 s, clamped 2–30). **Never invents**: a timeout, an exception, an unknown kind, an unparseable date, an instant in the past, one beyond 400 days, an empty subject or one carrying a redaction placeholder all raise with their reason, which is then spoken. That last guard is sharper here than on the core — a placeholder stored as a fact is merely useless, while one in a reminder is read aloud twenty minutes later.

## 17. Routine recurrence extractor (turns "every Monday morning" into a rule)

- **File**: [src/jarvis/routines/extract.py](src/jarvis/routines/extract.py) — `extract_routine_rule()`.
- **Trigger**: inside `setRoutine`, once per call. Nowhere else.
- **Model**: the same chain and the same pin as #16 (`reminder_model → tool_router_model → intent_judge_model → cfg.llm_chat_model`), because this prompt carries the same thing — the user's own sentence about their own life — and one privacy decision beats two things to get wrong.
- **Inputs**: the reference moment and the user's utterance, fenced as data.
- **System prompt**: `_EXTRACT_SYSTEM` in the same file. Names no day, no month, no adverb, in any language, and carries no worked example. States the weekday convention explicitly (0 = the first day of the working week): leaving it implicit is how a Monday routine runs on Sunday.
- **Output**: a `Lecture(regle, quoi, heure_supposee)` — `daily` or `weekly` with a wall-clock hour, or `none`. `heure_supposee` travels because `setRoutine` has to say it out loud: a rhythm named with no time of day gets one filled in, and for a routine that is a claim about every morning from now on. Never a cron expression — a model can be subtly wrong about one in a way nobody notices until a routine fires at 3am on the 31st, and this runs while the user is asleep. **There is deliberately no shape finer than a day**: a routine that fires on a tick empties a rate limit and a wallet overnight, so the vocabulary cannot express it.
- **Limits**: `reminder_timeout_sec` (8s, shared with #16 and #18). Refuses rather than guessing — an unknown kind, an hour out of range, a weekly rule with no day, an empty subject, a redaction placeholder, a timeout. That last guard bites hardest here: a placeholder in a routine's sentence is replayed to the model every single morning.

## 18. Goal completion judge (whether a goal is worth one question)

- **File**: [src/jarvis/objectifs/juge.py](src/jarvis/objectifs/juge.py) — `peut_etre_fini()`.
- **Trigger**: inside `noteGoal`, once, immediately after a dated line lands. Nowhere else, and never on a turn that has nothing to do with a goal — the state has just changed, somebody is in the room, and an ordinary turn pays nothing.
- **Model**: the same chain and the same pin as #16 and #17 (`reminder_model → tool_router_model → intent_judge_model → cfg.llm_chat_model`), because this prompt carries the same thing: the user's own sentences about their own life.
- **Inputs**: the goal's sentence, the ending condition **the user gave**, and his own dated lines (the last `_DERNIERS` (8) of them), fenced as data. It never reads prose a model wrote — there is none in this slice, and the contract is stated so that the day an unattended pass exists its write-up is already excluded.
- **System prompt**: `_SYSTEM` in the same file. Names no language and carries no worked example.
- **Output**: `peut-etre` or `pas-encore`, and **there is deliberately no third value**. The vocabulary contains nothing meaning "finished", enforced at the parser rather than in the prompt: an answer this code does not recognise becomes `pas-encore`, so no sentence a model can emit and no page it could have read closes a goal. Closing one is `closeGoal`, which costs a card, and it is the user's.
- **The verdict has no writer.** It travels in `noteGoal`'s reply and dies with the turn. Nothing durable records that she wondered, which is what keeps the fork's one rule intact: she may think it, she may say it, and only he can make it a fact. His answer either way is written as an ordinary note, which is also what stops her asking the same question every day — the next judgement then sees changed inputs.
- **Limits**: `reminder_timeout_sec` (8s, shared with #16 and #17). Every failure is quiet: no model, a timeout, unreadable JSON, an unknown word, a goal with no notes, a goal with no ending condition, or a redaction placeholder in the premise all return `pas-encore`. A question asked too early is asked again every day, and then nobody listens.

## 19. Journal proposal reader (things she noticed, for him to confirm)

- **File**: [src/jarvis/appris/propose.py](src/jarvis/appris/propose.py) — `propositions()`.
- **Trigger**: inside `reviewLearnings`, once per invocation. `reviewLearnings` runs only when the router picks it, and the router picks it only when the user asks what she has noticed. No schedule, no background pass, no start-up sweep, no call from any other tool. An ordinary turn pays nothing.
- **Model**: `appris_model → reminder_model → tool_router_model → intent_judge_model → llm_chat_model`. Deliberately **not** passed through `_cloud_safe_model`, like #16 and #17, because what it reads is a fortnight of his life. The last link being the chat model is honest rather than reassuring: LLM #9 wrote these summaries there and #10 already reads them there, so pinning nothing changes nothing — and pinning something is the only way to make this local. No model resolvable at all means the journal was not read, and the tool says so.
- **Inputs**: the `conversation_summaries` rows inside `appris_jours` whose sha256 does not match the digest in `journal_lu`, newest first, capped at 10 rows and 12,000 characters, each prefixed with its date and fenced `<<<…>>>` as data rather than instructions. The cap is announced, never applied quietly. **Nothing from `profil.md`, `regles.md`, `outils.md`, `appris.md` or the ledger reaches this prompt**: every suppression check runs afterwards, in Python, so his existing beliefs never cross the wire.
- **System prompt**: `_SYSTEM` in the same file. The mirror of #10: that one takes the world out of a summary and refuses everything about the person, this one takes the person out and refuses everything about the world. The template names no language: the proposal is written in `response_language` when that setting is set, and otherwise in the language of the note it was taken from, never translated. It requires a character-for-character citation from the notes.
- **Output**: a JSON array of `{genre, texte, citation}`, `genre ∈ {fait, regle}`. Consumed by `page.render_proposition` and written to `appris.md` and **nowhere else**. **It has no writer into the core.** There is no sentence this model can emit that reaches `profil.md`; the only path there is `recolte.recolter`, which reads a checkbox and calls no model.
- **Limits**: `appris_timeout_sec` (30 s, clamped 5–120), `num_ctx: 8192`, `temperature=0.0` for #10's reason — this is rule-following classification, and a default sampling temperature makes small models drift off a banned-forms list. At most `appris_max_propositions` survivors.
- **Failure**: `appelee=False` on no model, a timeout, an answer with no bracket, a JSON error, or a non-list. Then **nothing is recorded as read**, so a single failure never skips those days permanently, and the tool's reply forbids saying there was nothing new. Seven deterministic guards drop items afterwards, each counted and each reported out loud.
- **Cost**: one call per ask. Asking twice in a day costs one call the first time and none the second, because every row is already digested. The rate limit is a property of the artefact, not a timer.

---

## Frequency / Size Summary

| # | Context | Per reply | Optional? | Model tier |
|---|---------|-----------|-----------|------------|
| 1 | Main chat loop | 1-8 | No | LARGE |
| 2 | Intent judge | 1 (voice only) | fallback available | SMALL |
| 3 | Memory enrichment extract | 0-1 | gated by planner | SMALL (via router chain) |
| 4 | Memory digest | 0-N | auto by size | chat model, runs only when it is SMALL (unless forced) |
| 5 | Tool-result digest | 0-N | auto by size | chat model, runs only when it is SMALL (unless forced) |
| 6 | Max-turn digest | 0-1 | No | SMALL |
| 7 | Tool router | 1 | every attended turn; planner picks unioned in | SMALL |
| 8 | Tool searcher | 0-3 | model-initiated | SMALL (reuses #7) |
| 9 | Summariser | ~1/session | No (background) | LARGE |
| 10 | Graph extraction | ~1/session | No (background) | LARGE |
| 11 | Graph best-child | 0-N | No (background) | SMALL (via router chain) |
| 11b | Graph node merge | 0-N (per node, batched) | No (background) | SMALL (via router chain) |
| 11c | Graph node split | 0-N (only past `SPLIT_THRESHOLD`) | No (background) | LARGE (chat model, never the picker model) |
| 12 | Planner (plan_query) | 1 | yes (planner_enabled) | LARGE/SMALL (tracks chat model) |
| 13 | Plan step resolver | 0-N (SMALL only) | auto by size + plan | LARGE/SMALL (tracks chat model, same as #12) |
| 14 | Tool-specific | per-tool | n/a | weather: SMALL (via router chain); logMeal: LARGE |
| 15 | Approval judge | 0-1 | only after a `parole` question | SMALL (via router chain) |
| 16 | Reminder time extractor | 0-1 | only inside setReminder | SMALL (via router chain) |
| 17 | Routine recurrence extractor | 0-1 | only inside setRoutine | SMALL (reuses #16's chain) |
| 18 | Goal completion judge | 0-1 | only inside noteGoal | SMALL (reuses #16's chain) |
| 19 | Journal proposal reader | 0-1 | only inside reviewLearnings | SMALL (appris_model → reminder chain; the chat model is only the last link) |
| 20 | Dictation cleanup | 0-1 | only when `dictation_filler_removal` is on | chat model, pinned to the local backend |

`LARGE` names the chat model (`llm_chat_model`), whatever its size class: the default `gemma4:e2b` is SMALL under `detect_model_size`, so out of the box every `LARGE` row runs on a SMALL model. `via router chain` means `resolve_tool_router_model(cfg)`: `tool_router_model → intent_judge_model → llm_chat_model`.

## 20. Dictation Cleanup (optional, opt-in)

- **File**: [src/jarvis/dictation/dictation_engine.py](src/jarvis/dictation/dictation_engine.py) — `_llm_clean_dictation()`.
- **Trigger**: once per dictation, only when `dictation_filler_removal` is true (default false).
- **Model / gating**: `cfg.llm_chat_model`, dispatched through the **local** backend regardless of `llm_provider`. Unlike #16-#19, which honour a pin and otherwise follow the provider, this one does not get that choice: it carries everything the user dictates, all day, into applications that have nothing to do with the assistant. Nothing local answering costs the cleanup and nothing else.
- **Inputs**: the raw transcript.
- **System prompt**: inline — remove fillers, hesitations and false starts, keep the meaning and the language.
- **Output**: the cleaned text, or the raw transcript on any failure.
- **Limits**: 5 s, hard-coded.

## Size-aware auto switches

Driven by `detect_model_size(model_name) → SMALL (≤7B) | LARGE (8B+)`:

| Feature | SMALL | LARGE |
|---------|-------|-------|
| Memory digest | ON | OFF |
| Tool-result digest | ON | OFF |
| Text-based tool calling | ON | OFF (native) |
| Planner direct-exec | ON | OFF |

The size class comes from the model name, and the default chat model is SMALL, so the SMALL column is what a fresh install runs.

## Config keys

- Models: `llm_chat_model`, `intent_judge_model`, `tool_router_model`, `planner_model`, `evaluator_model`, `confirmation_model`, `reminder_model`, `appris_model`, `embedding_model` (`ollama_chat_model` is the authoritative model key on the Ollama provider, and the fallback for an empty `llm_chat_model` on an OpenAI-compatible one)
- Provider and destination: `llm_provider`, `llm_base_url`, `llm_api_key`, `llm_api_key_env`, `llm_extra_body`, `embedding_provider`, `embedding_base_url`, `auto_redact_before_cloud`. `llm_base_url` also decides local from remote: a bare auxiliary pin is honoured on a loopback or private-network endpoint, and on a remote one while local Ollama is reachable; it is replaced by the chat model only on a remote endpoint with local Ollama unreachable.
- Flags: `memory_digest_enabled`, `tool_result_digest_enabled`, `llm_thinking_enabled`, `intent_judge_thinking_enabled`, `tool_selection_strategy`, `low_power_mode`, `planner_enabled`, `reminders_enabled`, `routines_enabled`, `dictation_filler_removal`, `dictation_thinking_enabled`, `response_language`
- Timeouts: `llm_chat_timeout_sec` (180s; #1, #9, logMeal in #14), `llm_tools_timeout_sec` (300s; #3, #7, #8, the weather extractor in #14), `llm_digest_timeout_sec` (8s, shared across #4/#5/#6), `planner_timeout_sec` (6s; #12, #13), `intent_judge_timeout_sec` (15s), `confirmation_timeout_sec` (8s), `reminder_timeout_sec` (8s; #16-#18), `appris_timeout_sec` (30s), `confirmation_ttl_sec` (180s)
- Caps: `agentic_max_turns` (8), `tool_search_max_calls` (3), `_LLM_MAX_SELECTED` (5), `MAX_STEPS` (5), `_DIGEST_MAX_CHARS` (500), `_TOOL_DIGEST_MAX_CHARS` (600), `SPLIT_THRESHOLD` (1500)
- Runtime residency: `low_power_mode` skips startup LLM warmups and shortens Ollama `keep_alive` for intent judge and warmup calls from `"30m"` to `"1m"`. It does not change prompts, model selection, timeouts, or context limits.
- Startup warm-ups: unless `low_power_mode` is on, the listener pages the chat, intent-judge and (under the `llm` strategy, when distinct) router models in with one minimal request each (`warm_up_chat_model()`), overlapping Whisper's initialisation, each given at least 60 s. They carry no prompt and are not contexts.

## KV-cache discipline (prompt construction rules)

Every context is built against servers (Ollama, vLLM, SGLang, llama.cpp llama-server, LM Studio) that reuse the KV state of the longest matching prompt prefix. The first diverging token decides how much compute is saved, so these rules are load-bearing:

1. **System prompts are byte-static**: in auxiliary contexts (#2-#19), system prompts contain no timestamps, hints, or per-call data; per-call data lives in the user message. In Context #1 (main loop), memoised per-reply context is appended strictly at the tail of the system message so the large prompt prefix remains stable across multi-turn exchanges.
2. **Dynamic blocks go to the tail**: anything that changes per call (context line, hint blocks) is appended at the END of its message, never at the head.
3. **Stable-before-dynamic ordering**: the mostly-static block (persona, tool catalogue) opens the prompt; per-query blocks (digest, plan, hint) follow; the user query is the final token.
4. **Per-reply memoisation**: the main loop's time/location context string is computed once per reply, so all in-loop calls of one reply are byte-identical from token 1; the KV prefix extends through the whole history, not just the system message.
5. **Ollama payloads set `cache_prompt: true` explicitly** on `chat()`, `direct()`, and `streaming()` so the server always retains the request's KV state.

Anything that reorders messages between calls, injects a changing value at the head of a prompt, or rebuilds a system prompt with per-call content breaks prefix reuse for every token after the divergence point.

## Flow

```
user input
  └─▶ [2] Intent Judge            (voice only, SMALL)
        ├─ a spoken question was left waiting → [15] Approval judge reads the answer
        └─▶ [7] Tool router (narrows catalogue for the planner)
              └─▶ [12] Planner (gates memory; advisory for the router allow-list)
                    ├─ plan requests searchMemory  → [3] Enrichment extract → [4] Memory digest (optional)
                    ├─ plan empty (fail-open)      → [3] Enrichment extract (unless the recall gate [3b] skips it) → [4] Memory digest
                    ├─ plan has tool steps only    → skip #3; a graph crawl keyed on the steps' arguments can still feed [4]
                    └─ plan reply-only             → skip #3 and #4 entirely
                    └─▶ AGENTIC LOOP  (≤ agentic_max_turns)
                                      ├─ [13] Plan step resolver (SMALL, direct-exec)
                                      ├─ [1] Main chat turn
                                      ├─ tool execution
                                      │    └─ [5] Tool-result digest (optional)
                                      │    └─ [8] Tool searcher (model-initiated)
                                      │    └─ [14] weather place extractor / logMeal
                                      │    └─ [16] inside setReminder, [17] inside setRoutine,
                                      │       [18] inside noteGoal, [19] inside reviewLearnings
                                      └─ content → deliver immediately
                                      └─ if max turns → [6] Max-turn digest
                          └─▶ TTS / output
                          └─▶ background: [9] summariser → [10] graph extract → [11] best-child
                                            → [11b] node merge → [11c] node split (only past SPLIT_THRESHOLD)

dictation (hold-to-dictate, outside the reply flow) → [20] cleanup
```

## Optimisation ideas (seed list)

1. Batch multi-chunk memory digests (#4) into a single call with explicit markers.
2. Parallelise multiple tool-result digests (#5) when several results land at once.
3. Pre-warm the intent-judge model before TTS finishes.
4. Keep the tool-router (#7) cache beyond one conversation: today it is cleared whenever a new conversation starts.
5. Give each digest its own timeout rather than sharing `llm_digest_timeout_sec` (the memory, tool-result and max-turn digests all use the same value today).
6. Consider single-model deployments: the router and the extractors prefer `intent_judge_model`, so loading a second model hurts cold-start latency on small hardware.
7. Narrow `llm_thinking_enabled` to the main chat loop, not every context that reads it (digests, summariser, graph passes, logMeal).
8. Reduce `intent_judge_timeout_sec` (15s) or race it against text-based wake detection to avoid blocking the audio loop.

---

## Measuring

`tests/performance/test_pipeline_timings.py` times the reply pipeline against a live Ollama. Run:

```
pytest tests/performance/ -v -m performance -s
```

It dumps a JSON report to `tests/performance/reports/`. A micro-benchmark with a tiny fixed prompt runs alongside to give a per-call floor — if that floor moves, every context's total moves with it, so hardware/model drift is visible immediately. How calls are bucketed per context is described in `tests/performance/README.md`.

Baseline on a local gemma4:e2b (as of 2026-04-22, 3 queries × 3 runs): main chat turn p50 ~4.5s, enrichment extract p50 ~0.9s (small-model chain), micro-prompt floor ~0.15s. Sample sizes: main 25 calls, enrichment 9. Use these as rough reference points — the assertions in the test are relative-shape (router ≤ 1.5× main chat turn), not absolute.

## Keep this doc in sync

This graph is the reference for LLM-latency optimisation. Treat it as authoritative: whenever code changes affect an LLM call — a new context, a removed one, a changed model/timeout/cap/gating/prompt source, or a new data-flow edge — update this file in the same PR. If the update would be more than a one-line tweak, reflect it in the relevant `*.spec.md` too. Refer to code by symbol name, never by line number: a line number is wrong by the next commit. `tests/test_llm_contexts_doc.py` holds the settings, defaults and caps named here to the code, so a renamed key or a moved default fails CI until this file follows.

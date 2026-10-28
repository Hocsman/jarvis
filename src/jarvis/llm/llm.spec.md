# LLM Backend Specification

The `jarvis.llm` package owns every LLM HTTP call Jarvis makes and lets the same reply engine, planner, intent judge, memory pipeline, and tools run against Ollama, an OpenAI-compatible server (LM Studio, oMLX, llama.cpp's `llama-server`, vLLM, LocalAI, or a hosted endpoint), or Anthropic's Messages API. Those are the three providers the factory builds.

## Goals

1. **Pluggable.** New backends drop in by subclassing `LLMBackend` and being registered in `factory.get_llm_backend`. Call sites stay unchanged.
2. **Privacy-first.** Backends never send data anywhere the user has not chosen: a remote provider's URL, or the Anthropic provider, whose endpoint is fixed. Defaults remain `127.0.0.1:11434`.
3. **Single source of truth.** Every call site dispatches through `get_llm_backend(cfg)` / `get_embedding_backend(cfg)`. The `Settings` object carries provider, base URL, API key, and model fields; the factory reads them.

## Public surface

```python
from jarvis.llm import (
    LLMBackend,                  # provider-agnostic ABC
    OllamaBackend,               # implementation: Ollama
    OpenAICompatibleBackend,     # implementation: OpenAI-compatible servers
    AnthropicBackend,            # implementation: Anthropic Messages API
    ToolsNotSupportedError,
    get_llm_backend,             # factory: settings → chat backend
    get_auxiliary_backend,       # factory: settings + model → auxiliary backend
    get_private_backend,         # factory: settings + pinned + model → private backend
    get_embedding_backend,       # factory: settings → embedding backend
    chat_is_remote,              # settings → whether the reply leaves home
    call_llm_direct,             # base-URL helper (see below)
    call_llm_streaming,
    chat_with_messages,
    extract_text_from_response,
)
```

Two interchangeable styles dispatch to the same backend:

- **Object-style** (preferred): `get_llm_backend(cfg).direct(...)`. The factory dispatches on `cfg.llm_provider` so swapping providers does not touch call sites. Every site under `src/jarvis/` uses this.
- **Function-style**: `call_llm_direct(base_url, ...)`. A thin wrapper that constructs an `OllamaBackend(base_url)` and delegates. Used by the performance-recording shims in `tests/performance/` and the eval scripts under `evals/`, where only a base URL is in scope.

## `LLMBackend` interface

| Method | Returns | Contract |
|--------|---------|----------|
| `direct(model, system, user, *, timeout_sec, thinking, num_ctx, temperature, max_tokens)` | `Optional[str]` | Single-shot system+user. Returns assistant text, or `None` on timeout / error / empty content. `max_tokens` caps the generation (`num_predict` on Ollama, `max_tokens` on the OpenAI and Anthropic shapes) and is unset by default (Anthropic requires a cap, so its backend sends 8192 when none is given); when a caller sets it, it should be generous (2-3x the expected answer) so a slightly long reply is never truncated mid-JSON. |
| `streaming(model, system, user, *, on_token, timeout_sec, thinking)` | `Optional[str]` | Streams tokens via `on_token`; returns the concatenated full text or `None` if no content was produced. |
| `chat(model, messages, *, timeout_sec, extra_options, tools, thinking, on_token)` | `Optional[Dict]` | Arbitrary messages array. Returns the raw response dict so callers (today: the reply engine) can inspect `content` and `tool_calls`. Raises `ToolsNotSupportedError` when the model rejects native tools. Re-raises `requests.ConnectionError` so callers can distinguish "server unreachable" from a transient HTTP failure. `on_token` receives content deltas during generation; **the return value is identical with or without it** — streaming is an extra output channel, not a different result. Backends that cannot stream must still accept the argument and may ignore it (Ollama buffers today: its JSON-lines stream needs its own reassembly path). |
| `embed(text, model, *, timeout_sec)` | `Optional[List[float]]` | Vector embedding. Returns `None` on error or when the runtime does not expose embeddings. |
| `list_models(*, timeout_sec)` | `List[str]` | Names of models the runtime has available. Returns `[]` on error. |
| `warm_up(model, *, timeout_sec, keep_alive)` | `bool` | Page `model` into resident memory ahead of the first real request. Default impl returns `True` (no-op for runtimes without per-call unloading). Ollama overrides this to issue a minimal `/api/generate` ping with the caller-provided `keep_alive` duration, defaulting to `"30m"`; the remote backends send one minimal, billed request that opens the connection. |

`direct()` and `streaming()` take a system prompt and one user message, so callers running classification-shaped passes (planner, intent judge, enrichment extractor) never build a messages array. Each backend builds their requests itself rather than going through `chat()`, and they never carry tools. `chat()` is the low-level primitive for arbitrary message arrays — multi-turn dialogue, native tool calls, and anything that needs custom roles.

### Tool calling

The `tools` parameter accepts the OpenAI-compatible JSON-schema format produced by `jarvis.tools.registry.generate_tools_json_schema()`. Ollama 0.4+ adopts that exact format, so no translation layer is needed for the Ollama backend; the OpenAI-compatible backend normalises responses inside its `chat()` method, and the Anthropic backend translates tools, calls and results both ways, so the reply engine sees a single shape.

When a model rejects the `tools` parameter (Ollama returns HTTP 400 in that case), the backend raises `ToolsNotSupportedError`. The reply engine catches it and falls back to text-based tool calling for the rest of the session.

### Streaming

Each backend parses its own stream format internally (Ollama JSON lines, OpenAI-compatible SSE, Anthropic's typed SSE events through the SDK's stream helper). The public `on_token(str)` contract is identical across backends.

### Embeddings

`embed()` is part of the same backend interface so the same provider can serve both chat and embeddings when capable. The `embedding_provider` config key lets users on runtimes without embeddings (e.g. some oMLX builds) route embeddings through Ollama while keeping chat on their preferred runtime. Anthropic serves no embeddings: with `llm_provider: anthropic` and `embedding_provider` unset, embeddings run on Ollama, and `AnthropicBackend.embed()` returns `None`. Every embedding call site under `src/jarvis/` resolves through `get_embedding_backend(cfg)`.

## Configuration

Provider-aware fields in `Settings` (see [src/jarvis/config.py](../config.py)):

| Key | Default | Meaning |
|-----|---------|---------|
| `llm_provider` | `"ollama"` | `"ollama"`, `"openai_compatible"` or `"anthropic"` (`LLM_PROVIDERS` in `config.py`). Unknown values fall back to `"ollama"`. |
| `llm_base_url` | (OpenAI-compatible only) | The OpenAI-compatible server's URL, e.g. `http://localhost:1234/v1` (LM Studio default). Read only when `llm_provider == openai_compatible`; the Ollama path always uses `ollama_base_url`, and the Anthropic path always `https://api.anthropic.com` (`ANTHROPIC_BASE_URL`), so a URL left from an earlier OpenAI-compatible setup never receives the Anthropic key. It is also what tells a local OpenAI-compatible server apart from a remote one: auxiliary model pins are honoured on a loopback or private-network endpoint (running on local Ollama when it lists them, on that server otherwise), kept on a remote one while local Ollama is reachable, and replaced by the chat model on a remote one only when local Ollama is unreachable, which is announced. A private context's pin runs on local Ollama behind a remote endpoint (Anthropic always is one); behind a local one, on local Ollama when it lists the tag and on the configured server otherwise. |
| `llm_api_key` | `""` | The remote provider's key: a bearer token on the OpenAI-compatible path, the Anthropic API key on the Anthropic path. Sent only when non-empty; with no key the Anthropic backend sends nothing. |
| `llm_api_key_env` | `""` | Name of an environment variable holding the key. When it names a set variable, its value replaces `llm_api_key`, so the secret never has to sit in `config.json`. For Anthropic, name a variable of your own (e.g. `YUBA_ANTHROPIC_KEY`) rather than `ANTHROPIC_API_KEY`: a headless Claude Code started by the `claudeCode` tool inherits the environment and bills the API instead of the user's subscription when that variable is set. |
| `llm_chat_model` | (remote providers only) | The model name the remote provider serves (`claude-haiku-5-5` on Anthropic). Read when `llm_provider` is not `ollama` (falling back to `ollama_chat_model` if blank); the Ollama path uses `ollama_chat_model`. |
| `llm_extra_body` | `{}` | Provider-specific request fields merged into every OpenAI-compatible chat payload (`direct` / `streaming` / `chat`). Example: `{"provider": {"sort": "throughput"}}` to pin the fastest upstream for an OpenRouter model served by several providers. Never overrides the keys the backend owns (`model` / `messages` / `stream`). No effect on the Ollama and Anthropic paths. |
| `embedding_provider` | inherits `llm_provider` | `"ollama"` / `"openai_compatible"`. Override for runtimes without embeddings. Inheriting `anthropic` means Ollama. |
| `embedding_base_url` | inherits from llm config | Override per-provider URL. Never inherited from the Anthropic chat provider. |
| `embedding_api_key` | inherits `llm_api_key` | Override per-provider key. The Anthropic key is never inherited. |
| `embedding_model` | (OpenAI-compatible only) | The OpenAI-compatible embedding model. Read only when the effective embedding provider is `openai_compatible` (falling back to `ollama_embed_model` if blank); the Ollama path uses `ollama_embed_model`. |
| `low_power_mode` | `false` | When enabled, voice startup skips LLM warmup and Ollama keep-alive windows used by warmup and the intent judge are short. |

The `ollama_base_url` / `ollama_chat_model` / `ollama_embed_model` keys hold the Ollama configuration and are authoritative whenever the active (chat or embedding) provider is Ollama. `_load_settings` resolves `cfg.llm_chat_model` and `cfg.embedding_model` per-provider — the Ollama keys win on the Ollama path, the provider-aware keys win on a remote path — so the codebase reads a single resolved field (`cfg.llm_chat_model`) while each provider keeps its own on-disk model name. The v1 → v2 migration promotes any explicitly-set `ollama_*` values into the provider-aware keys; per-provider resolution means a promoted value never shadows the Ollama picker.

Reasoning is set per role by the flags `llm_thinking_enabled` and `intent_judge_thinking_enabled`, both off by default; every other call that can reach a remote provider passes `thinking=False` (dictation always runs on local Ollama). One request escapes the flag: on Ollama, a call carrying native tools leaves the choice to the model (see the Ollama wire shape below). On the Anthropic path those flags are what decides it: off is `{"type": "disabled"}` (about 0.5 s less per intent judgement on Claude Haiku 5.5, which otherwise thinks by default), on is adaptive thinking.

### Splitting roles across models

`intent_judge_model` / `tool_router_model` / `memory_extractor_model` / `planner_model` exist so the classification-shaped calls can run on a different model from the reply. On a cloud provider this is not a micro-optimisation — it dominates latency.

Measured on the memory extractor (same prompt, same query, OpenRouter):

| Model | Median | Tokens emitted |
|-------|--------|----------------|
| `deepseek/deepseek-v4-flash` | 12.3 s | 632 |
| `openai/gpt-oss-120b` | **0.4 s** | 281 |
| `qwen/qwen3.5-flash` | 40.9 s | 6198 |
| `mistralai/ministral-8b` | 0.8 s | 50 |

All returned valid JSON: the spread is verbosity, not capability. Reasoning-style models narrate before answering, which is worth paying for in a reply and pure latency in an extractor. Routing the auxiliary roles to a terse model took a weather+news query from 26.9 s to 11.2 s end to end — auxiliary calls fell from 21.0 s to 1.4 s — with identical answers.

Capping generation instead (`max_tokens`) does **not** work here: the JSON arrives after the narration, so every cap tested (220/400/600) truncated before it, failing the parse and triggering the extractor's retry — which made the whole reply slower, not faster. Pick a terse model for these roles rather than trying to silence a verbose one.

### Factory dispatch

- `get_llm_backend(cfg, model=None)` reads `llm_provider`. For `openai_compatible` it resolves `llm_base_url` (falling back to `ollama_base_url`); for `anthropic` it uses `ANTHROPIC_BASE_URL`; for `ollama` it uses `ollama_base_url` directly so a stale `llm_base_url` from a previous OpenAI-compatible config cannot leak into another backend. When `model` lives on local Ollama rather than on the configured server (`_lives_on_ollama`), it routes directly to local Ollama via `_SurLaMachine`: behind a remote endpoint that is a local tag rather than a name the provider serves, and behind a local OpenAI-compatible endpoint a tag local Ollama lists (`GET /api/tags`, an untagged name matching `name:latest`, the answer cached 30 s and cleared with the backend cache; Ollama not answering lists nothing). A local address is not proof of a local model: LM Studio may run beside Ollama, and a gateway (LiteLLM) or a tunnel there forwards wherever it is set to. `names_provider_model(provider, model)` in `config.py` draws that line: on OpenAI-compatible endpoints a provider's name is namespaced `vendor/model`, on Anthropic it is a Claude model ID (`claude-…`, which carries no slash, so the slash test cannot apply there). The provider's own chat model is the exception: `model == llm_chat_model` stays on the provider whatever its spelling, because some OpenAI-compatible providers (api.openai.com, DeepSeek, Groq) name their models without a slash and the chat model is the one name known to exist there. Unless `model == ollama_chat_model` too: an empty `llm_chat_model` inherits that name, and it is a local tag, so it routes like one. `llm_api_key` is read regardless (sent only when non-empty). Remote backends (OpenAI-compatible and Anthropic) are wrapped in `RedactingBackend` when `auto_redact_before_cloud` is on. The key fields are shared by the two remote providers, so a provider switch can leave one provider's key configured for the other; Anthropic keys carry a fixed prefix (`sk-ant-`), so the factory never hands one to an OpenAI-compatible backend (chat or embeddings) other than Anthropic's own OpenAI-compatible endpoint, and never hands any other key to the Anthropic backend. Each refusal prints one line.
- `get_auxiliary_backend(cfg, model=None)` routes auxiliary classification passes (intent judge, tool router, planner, enrichment digests, graph picker/merge, approval judge). Local model tags on remote providers execute via local Ollama without leaving the machine. On an OpenAI-compatible endpoint the test is literally the absence of a slash, the provider's own chat model aside: a call that lands on `llm_chat_model` (the planner's default, or a judge or router `_cloud_safe_model` replaced by it) stays on the provider, unless that name is the inherited `ollama_chat_model`. A local tag that *contains* one (e.g. an `hf.co/...` Ollama reference) reads as a namespaced cloud ID, stays on the remote endpoint, and 400s there — such a model cannot ride the auxiliary route. On Anthropic everything but a `claude-…` name is local. Behind a local OpenAI-compatible endpoint a model local Ollama lists runs there, the chat model aside, and anything else stays on that server.
- `chat_is_remote(cfg)` says whether the reply goes to a provider off home, the reading `get_llm_backend(cfg)` makes for the reply (`_on_a_remote_provider`): Anthropic, or `openai_compatible` at an address `is_local_endpoint` does not recognise. The Ollama provider counts as home whatever its URL, as it does for `get_private_backend`. The reminder and routine extractors use it: with the chat remote, the turn already sends their sentence to the chat model, so they read it with that model on `get_llm_backend(cfg, model=llm_chat_model)`, which keeps the chat model on the provider however it is spelled, rather than through `get_private_backend`.
- `get_private_backend(cfg, pinned, model)` routes privacy-sensitive contexts carrying user life data (reminders and routines with the chat at home, goals, learning propositions). On a remote endpoint (Anthropic always is one) a non-empty pin routes to local Ollama whatever its shape, `hf.co/...` tags included. On a local OpenAI-compatible endpoint (LM Studio, `llama-server`, on this machine or its network, as `is_local_endpoint` reads it) the pin runs on local Ollama when Ollama lists that tag, even when it is also the chat model's name, so a gateway at that address never receives it, and on that server otherwise, where its model ID lives. With no pin, `model` (the one the context resolved and is about to send) decides the destination exactly as on the auxiliary route: a local tag on a remote provider, or a tag Ollama lists behind a local one, runs on local Ollama, and a name the provider serves, its own chat model included, stays on it. Both are passed because a context's model chain runs past its pin, and a local tag reached through a fallback must not be sent to the remote endpoint. A slashed local tag stays local only as a pin: unpinned, the slash test reads it as a cloud ID, exactly as on the auxiliary route.
- `get_embedding_backend(cfg)` reads `embedding_provider` (falls back to `llm_provider` when unset, and an inherited `anthropic` becomes `ollama`), resolves `embedding_base_url` (falls back per-provider: `llm_base_url` for OpenAI-compatible, `ollama_base_url` for Ollama), and `embedding_api_key` (falls back to `llm_api_key`). Neither fallback reads the chat settings when the chat provider is Anthropic: its key belongs to Anthropic alone.
- Construction is fail-soft: an unset URL becomes the default Ollama URL, so `get_*_backend` never raises. Errors surface at request time, not construction time.
- Backends are **memoised** by their resolved connection parameters (provider, URL, key, redaction flag, `llm_extra_body`). A single reply makes many `get_llm_backend` calls; returning one cached instance lets them share the backend's persistent HTTP session (keep-alive) instead of each paying a fresh TLS handshake and the provider's cold-start. `clear_backend_cache()` drops the cache after a config reload so the next call rebuilds against the new settings.

### v1 → v2 config migration

The migration in `_migrate_config` runs once when `_config_version < 2`:

1. If `llm_provider` is unset, default to `"ollama"`.
2. Promote `ollama_base_url` → `llm_base_url`, `ollama_chat_model` → `llm_chat_model`, `ollama_embed_model` → `embedding_model` (only when the new key is empty).
3. Bump `_config_version` to `2` and persist via `_save_json` (which restricts the file to `0o600` on POSIX so credentials are not world-readable).

## Wire-shape specifics

### Ollama (`OllamaBackend`)

- Endpoints: `POST /api/chat`, `POST /api/embeddings`, `GET /api/tags`, `POST /api/generate` (used by `warm_up`).
- Streaming: JSON-lines (`{...}\n`).
- Tool calls: native `tools` parameter (Ollama 0.4+); arguments returned as a Python dict.
- Prompt caching: every chat payload (`chat()`, `direct()`, `streaming()`) sets `cache_prompt: true` explicitly so the server retains the request's KV state and reuses it when the next request shares the same prefix. Callers keep prefixes cacheable by keeping system prompts byte-static and pushing per-call data (time, hints) to the tail of the prompt (see `docs/llm_contexts.md` "KV-cache discipline").
- A request carrying `tools` never sends `think: false`; the key is omitted instead. `gemma4:e2b` has a thinking mode, and an explicit false that switches it off also stops its native tool calls. Measured on Ollama 0.40.12 with native tools forced, Yuba's persona prompt and 20 interleaved calls an arm across `getWeather` and `remember`: 16/20 tool calls with the key absent, 15/20 with `think: true`, 0/20 with `think: false`. Under a one-line system prompt the three arms all score 20/20, so a probe with a toy prompt cannot see the effect. The price is that a request carrying tools runs at the model's own default whatever `llm_thinking_enabled` says, and `gemma4:e2b` thinks by default. In the reply loop only large models pay it: a small one gets text tools and no `tools` key, so its turns send `think: false`. Tool-free calls keep `think: false`, because there the flag earns its place: applied blanket, the same model stopped extracting anything from a summary and the knowledge extractor fell from 5/5 to 3/5 eval cases. `extra_options` can still force the flag either way.
- `extra_options` keys map onto the wire shape: `keep_alive` / `format` / `think` go to the payload root; everything else (incl. `temperature`, `num_ctx`, `num_predict`) folds into the nested `options` object. Callers can also pass an explicit `options` sub-dict for explicit nesting.
- `warm_up(model, keep_alive="30m")` issues `POST /api/generate` with an empty prompt and the requested `keep_alive`; the model stays resident for that duration after the call.

### OpenAI-compatible (`OpenAICompatibleBackend`)

- Endpoints: `POST /chat/completions`, `POST /embeddings`, `GET /models`.
- Streaming: Server-Sent Events. Lines start with `data:` and an empty payload terminator is `data: [DONE]`. Comment lines (`: ping`) and malformed payloads are skipped.
- Tool calls: native `tools` parameter; OpenAI returns `tool_calls[*].function.arguments` as a JSON-encoded string. The backend decodes them to a dict so the reply engine sees a single shape, and re-encodes any assistant `tool_calls` arguments dicts back to JSON strings when serialising messages for subsequent turns.
- Response normalisation: `_normalise_response` lifts `choices[0].message` to top-level `message` so callers do not branch on provider. Servers that already return Ollama-shaped responses pass through unchanged.
- `extra_options` lifts sampling fields (`temperature`, `max_tokens`, `top_p`, `stop`, …) to the payload root and silently drops Ollama-only knobs (`keep_alive`, `num_ctx`, `num_predict`, `think`) that have no equivalent in the OpenAI shape.
- `llm_extra_body` fields are merged into the chat payload root via `setdefault` (the backend-owned `model` / `messages` / `stream` keys always win). Embeddings are not affected.
- `chat(..., on_token=...)` streams: the request is sent with `stream: true`, content deltas are forwarded to the callback as they arrive, and tool-call deltas are reassembled by their `index` (a single call's `arguments` JSON arrives split across chunks). The reassembled result is returned in the **same normalised shape as the buffered path**, so callers — notably the reply engine's tool loop — cannot tell the two apart. Without `on_token` nothing changes: `stream: false`, one buffered response. A raising callback is swallowed (logged) so a UI fault never costs the generated reply.
- One persistent `requests.Session` per backend: the TCP + TLS connection to the provider is kept alive across calls (HTTP keep-alive), so each request skips the handshake. On a remote cloud endpoint that handshake is a large share of a short request's latency.
- `warm_up(model)` fires one minimal (`max_tokens: 1`) chat request to establish the session's connection and warm the provider before the first user query. There is nothing to page into memory on a remote endpoint, but the first call to a cold connection pays the TLS handshake and the provider's routing/cold-start (several seconds on a cloud endpoint); absorbing that at startup keeps the first reply on a warm path. Returns `True` on success, `False` on error or empty model.
- Authentication: `Authorization: Bearer <api_key>` header sent only when `api_key` is non-empty.
- Error logs do not echo URLs or API keys: HTTP errors print only the status code, generic exceptions print only the class name, connection errors print a fixed string and re-raise so callers can apply their own back-off.

### Anthropic (`AnthropicBackend`)

- Calls go through the official `anthropic` Python SDK (pinned in `requirements.txt`) to the native Messages API: `POST /v1/messages`, `GET /v1/models`. Never the OpenAI-compatible entry point, which rejects the sampling values Jarvis sends and has no prompt caching. The SDK is imported on first use, so other providers never load it.
- Destination and key: always `ANTHROPIC_BASE_URL`, with the configured key passed explicitly to the client. An explicit key stops the SDK reading `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN` or a stored `ant` profile, so the only credential that can leave is the one the user configured; with none, no request is made and one line says so. Headers named in `ANTHROPIC_CUSTOM_HEADERS` (set for gateways of other tools, which the SDK would copy onto every request after its own, an `Authorization` line replacing the key) are removed, as are the SDK's headers describing the machine (`X-Stainless-OS`, `-Arch`, `-Runtime`, `-Runtime-Version`); the SDK's tracing is off.
- Messages: every system message joins the top-level `system` field. An assistant `tool_calls` entry becomes a `tool_use` block (JSON-string arguments decoded), a `tool` message a `tool_result` block in the following user turn, and a user message the engine tagged with `tool_name` answers the pending call of that tool (the approval path replays a granted call that way). The API rejects a call with no result and a result with no call, so a call left unanswered is not sent and a result whose call is gone is sent as text. Consecutive same-role turns are merged, and the conversation is trimmed to start and end on a user turn. Trimming can orphan a neighbour (tool carry-over outlives the text it followed, so past five minutes a history can open on an old call whose turn is then trimmed and whose result is left), so these passes repeat until nothing changes.
- Tools: `{"type": "function", "function": {...}}` becomes `{"name", "description", "input_schema"}`. `tool_choice` is `{"type": "auto", "disable_parallel_tool_use": true}`: the engine runs the first call of a turn and answers it, and a second parallel call would be left without a result.
- Response normalisation: text blocks join into `message.content`, `tool_use` blocks become `tool_calls` with dict arguments, so callers do not branch on provider. A tool call cut off by `max_tokens` is dropped rather than run on a truncated input. A refusal (`stop_reason: "refusal"`) comes back with empty content and no reasoning, which every caller treats as no answer (reasoning with no answer would read to the engine as a step to continue, re-sending the refused request). The response also carries `stop_reason` and `usage` (input, output, cache-read and cache-write tokens); one debug line per call reports them.
- Sampling: `temperature`, `top_p` and `top_k` are never sent: API version 2026-10-21, which SDK 2.x speaks, rejects them on every model. The intent judge, graph extraction and merge, step resolver and journal reader still pass `temperature=0.0`, which the other backends honour.
- `extra_options`: only the generation cap is read (`max_tokens`, else `num_predict`, top-level or under `options`); the Ollama knobs have no equivalent.
- Thinking: `thinking=False` sends `{"type": "disabled"}`. A model that answers that thinking cannot be turned off gets adaptive thinking at `output_config.effort: "low"` instead, the closest it allows, and is remembered for the backend's lifetime so later calls skip the refused round-trip. `thinking=True` sends `{"type": "adaptive"}` at the model's default effort.
- Thinking across a tool loop: when a tool call came with thinking blocks, the backend keeps the assistant turn as the API returned it (bounded to the last 32 turns, keyed by the call's id) and replays it unchanged, signatures included, when the engine sends that turn back on a request that also thinks. Such a request sets `thinking.block_binding.prefix_mismatch_behavior: "drop_block"` with the `thinking-binding-controls-2026-08-01` beta header, so a block whose history changed (the engine widens the tool list mid-loop through `toolSearchTool`) is dropped instead of failing the request.
- Prompt caching: the system prompt carries a `cache_control` breakpoint (it holds the tools and the system prompt, which render first), so auxiliary contexts whose system prompts are byte-static (`docs/llm_contexts.md`, "KV-cache discipline") read it back from the second call. A `chat()` that offers tools (the agentic loop) also sets top-level automatic caching, so each loop turn reads the conversation the previous one wrote; a single-shot call does not, since its unique tail would be written and never read.
- Streaming: `chat(..., on_token=...)` and `streaming()` use the SDK's `messages.stream`; text deltas go to the callback and the final message is normalised exactly like the buffered path. A raising callback is swallowed (logged). Claude often writes a sentence before a tool call, and a tool-calling turn speaks nothing (`src/jarvis/output/streaming.spec.md`): so when tools are offered, text is held until it is known not to precede a call. It is dropped if a `tool_use` block follows, released when the message ends, and streamed live from the moment it outgrows 240 characters (`_HELD_BEFORE_TOOLS_CHARS`), longer than the one-sentence preamble Claude writes before a call; a longer preamble would be spoken. Short answers on a tool-offering turn therefore arrive whole at the end of generation. The returned content keeps the preamble either way.
- Retries: the SDK's own retries are off, because they re-run a timed-out request and could triple the caller's deadline. Instead one retry follows a rate limit or an overloaded or failing server (an HTTP status, or the same error reported inside a stream that had answered 200), only when no token has been streamed yet, the wait (`retry-after` when the server sends one) is a second or less, and at least a second is left before the deadline. A longer wait is not spent: the voice callers have a fallback that answers now.
- One persistent SDK client per backend: its connection pool keeps an idle connection open for five minutes (the SDK's own default closes it after five seconds, shorter than the gap between two utterances), so a voice turn skips the TLS handshake. `warm_up(model)` sends one 1-token request to open it at startup.
- Errors: a timeout prints a fixed line and returns `None`; a connection failure prints a fixed line and re-raises as `requests.ConnectionError` without the SDK's message (it names the host), so the intent judge's back-off applies; an HTTP 400 about the tool definitions (the API's message starts with `tools` or `tool_choice`) raises `ToolsNotSupportedError`, and no other 400 does, since the engine's text-tools fallback re-sends the same conversation and cannot get past an error in it; any other HTTP error prints its status and returns `None`, with the API's error body in the debug log (it carries no key). `direct()` and `streaming()` return `None` on every failure.
- `embed()` returns `None` (no embeddings endpoint; the factory routes embeddings to Ollama), and `list_models()` reads `GET /v1/models`.

## Module-local LLM wrappers

Each migrated module exposes a single intercept point so tests can patch one symbol per module instead of reaching into the backend ABC:

- `jarvis.reply.engine.chat_with_messages(cfg, messages, ...)` — agentic-loop chat boundary.
- `jarvis.reply.planner.call_llm_direct(*, cfg, chat_model, ...)` — planner + step resolver.
- `jarvis.reply.enrichment.call_llm_direct(*, cfg, chat_model, ...)` — memory enrichment extractor + digest passes.
- `jarvis.memory.graph_ops.call_llm_direct(*, cfg, chat_model, ...)` — knowledge graph extraction, best-child picker, node merge.
- `jarvis.memory.conversation._direct_llm(cfg, system_prompt, user_content, ...)` — diary summary, deflection rewrite, topic optimisation.
- `jarvis.tools.builtin.nutrition.log_meal.call_llm_direct(*, cfg, chat_model, ...)` — nutrition extractor + follow-up generator.
- `jarvis.tools.builtin.weather.get_llm_backend` — hoisted to module scope so the place extractor's backend lookup is patchable.

A factory-dispatch wiring guard at `tests/test_factory_dispatch_wiring.py` parametrises across each migrated module and asserts the wrapper actually reaches `OpenAICompatibleBackend` for `llm_provider: openai_compatible`, `AnthropicBackend` for `anthropic` (with a Claude chat model) and `OllamaBackend` for `ollama`. A regression that drops `get_llm_backend(cfg)` from a wrapper would bypass every unit test but trip this guard.

`import requests` is re-exported from the package `__init__.py` so tests that patch `jarvis.llm.requests.post` keep working without reaching into the per-backend modules.

## File layout

```
src/jarvis/llm/
├── __init__.py             # public re-exports + function-style helpers
├── backend.py              # LLMBackend ABC + ToolsNotSupportedError
├── ollama.py               # OllamaBackend + extract_text_from_response
├── openai_compatible.py    # OpenAICompatibleBackend + _normalise_response
├── anthropic_backend.py    # AnthropicBackend (Messages API through the anthropic SDK)
├── redacting.py            # RedactingBackend, the scrub before remote egress
├── factory.py              # get_llm_backend(cfg) + get_embedding_backend(cfg)
└── llm.spec.md             # this file
```

## Failure handling

Backends fail soft for transient issues so the reply engine can degrade gracefully: timeouts and HTTP errors return `None` (or `[]` for `list_models`); HTTP 400 with `tools` set raises `ToolsNotSupportedError`; any other unexpected error is logged via `debug_log("...", "llm")` and returns `None`. The one exception is `requests.ConnectionError` (server unreachable), which `chat()` re-raises so callers like the intent judge can apply their own back-off — voice, for example, wants a 30s cooldown after a connection-refused error so it stops hammering an unresponsive Ollama between wake words.

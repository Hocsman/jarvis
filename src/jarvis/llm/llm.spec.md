# LLM Backend Specification

The `jarvis.llm` package owns every LLM HTTP call Jarvis makes and lets the same reply engine, planner, intent judge, evaluator, memory pipeline, and tools run against any local runtime: Ollama, an OpenAI-compatible server (LM Studio, oMLX, llama.cpp's `llama-server`, vLLM, LocalAI), or an Anthropic-compatible server.

## Goals

1. **Pluggable.** New backends drop in by subclassing `LLMBackend` and being registered in `factory.get_llm_backend`. Call sites stay unchanged.
2. **Privacy-first.** Backends never send data anywhere unless the user has explicitly configured the URL. Defaults remain `127.0.0.1:11434`.
3. **Single source of truth.** Every call site dispatches through `get_llm_backend(cfg)` / `get_embedding_backend(cfg)`. The `Settings` object carries provider, base URL, API key, and model fields; the factory reads them.

## Public surface

```python
from jarvis.llm import (
    LLMBackend,                  # provider-agnostic ABC
    OllamaBackend,               # implementation: Ollama
    OpenAICompatibleBackend,     # implementation: OpenAI-compatible servers
    ToolsNotSupportedError,
    get_llm_backend,             # factory: settings → chat backend
    get_embedding_backend,       # factory: settings → embedding backend
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
| `direct(model, system, user, *, timeout_sec, thinking, num_ctx, temperature)` | `Optional[str]` | Single-shot system+user. Returns assistant text, or `None` on timeout / error / empty content. |
| `streaming(model, system, user, *, on_token, timeout_sec, thinking)` | `Optional[str]` | Streams tokens via `on_token`; returns the concatenated full text or `None` if no content was produced. |
| `chat(model, messages, *, timeout_sec, extra_options, tools, thinking, on_token)` | `Optional[Dict]` | Arbitrary messages array. Returns the raw response dict so callers (today: the reply engine) can inspect `content` and `tool_calls`. Raises `ToolsNotSupportedError` when the model rejects native tools. Re-raises `requests.ConnectionError` so callers can distinguish "server unreachable" from a transient HTTP failure. `on_token` receives content deltas during generation; **the return value is identical with or without it** — streaming is an extra output channel, not a different result. Backends that cannot stream must still accept the argument and may ignore it (Ollama buffers today: its JSON-lines stream needs its own reassembly path). |
| `embed(text, model, *, timeout_sec)` | `Optional[List[float]]` | Vector embedding. Returns `None` on error or when the runtime does not expose embeddings. |
| `list_models(*, timeout_sec)` | `List[str]` | Names of models the runtime has available. Returns `[]` on error. |
| `warm_up(model, *, timeout_sec, keep_alive)` | `bool` | Page `model` into resident memory ahead of the first real request. Default impl returns `True` (no-op for runtimes without per-call unloading). Ollama overrides this to issue a minimal `/api/generate` ping with the caller-provided `keep_alive` duration, defaulting to `"30m"`. |

`direct()` and `streaming()` are convenience methods over `chat()`: they construct the `[system, user]` messages array internally so callers running classification-shaped passes (planner, intent judge, evaluator, enrichment extractor) do not have to. `chat()` is the low-level primitive for arbitrary message arrays — multi-turn dialogue, native tool calls, and anything that needs custom roles.

### Tool calling

The `tools` parameter accepts the OpenAI-compatible JSON-schema format produced by `jarvis.tools.registry.generate_tools_json_schema()`. Ollama 0.4+ adopts that exact format, so no translation layer is needed for the Ollama backend; OpenAI-compatible and Anthropic-compatible backends translate inside their `chat()` methods so the reply engine sees a single shape.

When a model rejects the `tools` parameter (Ollama returns HTTP 400 in that case), the backend raises `ToolsNotSupportedError`. The reply engine catches it and falls back to text-based tool calling for the rest of the session.

### Streaming

Each backend parses its own stream format internally (Ollama JSONL, OpenAI SSE, Anthropic SSE event blocks). The public `on_token(str)` contract is identical across backends.

### Embeddings

`embed()` is part of the same backend interface so the same provider can serve both chat and embeddings when capable. The `embedding_provider` config key lets users on runtimes without embeddings (e.g. some oMLX builds) route embeddings through Ollama while keeping chat on their preferred runtime. Every embedding call site under `src/jarvis/` resolves through `get_embedding_backend(cfg)`.

## Configuration

Provider-aware fields in `Settings` (see [src/jarvis/config.py](../config.py)):

| Key | Default | Meaning |
|-----|---------|---------|
| `llm_provider` | `"ollama"` | `"ollama"` or `"openai_compatible"`. Unknown values fall back to `"ollama"`. |
| `llm_base_url` | (OpenAI-compatible only) | The OpenAI-compatible server's URL, e.g. `http://localhost:1234/v1` (LM Studio default). Read only when `llm_provider == openai_compatible`; the Ollama path always uses `ollama_base_url`. It is also what tells a local OpenAI-compatible server apart from a remote one: auxiliary model pins are honoured on a loopback or private-network endpoint and replaced by the chat model on a remote one, which is announced. |
| `llm_api_key` | `""` | Optional bearer token. Sent only when non-empty. |
| `llm_chat_model` | (OpenAI-compatible only) | The model name the OpenAI-compatible server exposes. Read only when `llm_provider == openai_compatible` (falling back to `ollama_chat_model` if blank); the Ollama path uses `ollama_chat_model`. |
| `llm_extra_body` | `{}` | Provider-specific request fields merged into every OpenAI-compatible chat payload (`direct` / `streaming` / `chat`). Example: `{"provider": {"sort": "throughput"}}` to pin the fastest upstream for an OpenRouter model served by several providers. Never overrides the keys the backend owns (`model` / `messages` / `stream`). No effect on the Ollama path. |
| `embedding_provider` | inherits `llm_provider` | `"ollama"` / `"openai_compatible"`. Override for runtimes without embeddings. |
| `embedding_base_url` | inherits from llm config | Override per-provider URL. |
| `embedding_api_key` | inherits `llm_api_key` | Override per-provider key. |
| `embedding_model` | (OpenAI-compatible only) | The OpenAI-compatible embedding model. Read only when the effective embedding provider is `openai_compatible` (falling back to `ollama_embed_model` if blank); the Ollama path uses `ollama_embed_model`. |
| `low_power_mode` | `false` | When enabled, voice startup skips LLM warmup and Ollama keep-alive windows used by warmup and the intent judge are short. |

The `ollama_base_url` / `ollama_chat_model` / `ollama_embed_model` keys hold the Ollama configuration and are authoritative whenever the active (chat or embedding) provider is Ollama. `_load_settings` resolves `cfg.llm_chat_model` and `cfg.embedding_model` per-provider — the Ollama keys win on the Ollama path, the provider-aware keys win on the OpenAI-compatible path — so the codebase reads a single resolved field (`cfg.llm_chat_model`) while each provider keeps its own on-disk model name. The v1 → v2 migration promotes any explicitly-set `ollama_*` values into the provider-aware keys; per-provider resolution means a promoted value never shadows the Ollama picker.

### Splitting roles across models

`intent_judge_model` / `tool_router_model` / `planner_model` / `evaluator_model` exist so the classification-shaped calls can run on a different model from the reply. On a cloud provider this is not a micro-optimisation — it dominates latency.

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

- `get_llm_backend(cfg)` reads `llm_provider`. For `openai_compatible` it resolves `llm_base_url` (falling back to `ollama_base_url`); for `ollama` it uses `ollama_base_url` directly so a stale `llm_base_url` from a previous OpenAI-compatible config cannot leak into the Ollama backend. `llm_api_key` is read regardless (sent only when non-empty).
- `get_embedding_backend(cfg)` reads `embedding_provider` (falls back to `llm_provider` when unset), resolves `embedding_base_url` (falls back per-provider: `llm_base_url` for OpenAI-compatible, `ollama_base_url` for Ollama), and `embedding_api_key` (falls back to `llm_api_key`).
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
- A request carrying `tools` never sends `think: false`; the key is omitted instead. Ollama reads an explicit false as an instruction about the model's reasoning mode, and on a model that has no such mode it suppresses native tool calls outright. Measured on `gemma4:e2b`: an identical request scores 3/3 tool calls with the key absent and 0/3 with the flag, for `remember` and `getWeather` alike. Tool-free calls keep it, because there the flag earns its place: applied blanket, the same model stopped extracting anything from a summary and the knowledge extractor fell from 5/5 to 3/5 eval cases. `extra_options` can still force the flag either way.
- `extra_options` keys map onto the wire shape: `keep_alive` / `format` / `think` go to the payload root; everything else (incl. `temperature`, `num_ctx`, `num_predict`) folds into the nested `options` object. Callers can also pass an explicit `options` sub-dict for explicit nesting.
- `warm_up(model, keep_alive="30m")` issues `POST /api/generate` with an empty prompt and the requested `keep_alive`; the model stays resident for that duration after the call.

### OpenAI-compatible (`OpenAICompatibleBackend`)

- Endpoints: `POST /chat/completions`, `POST /embeddings`, `GET /models`.
- Streaming: Server-Sent Events. Lines start with `data:` and an empty payload terminator is `data: [DONE]`. Comment lines (`: ping`) and malformed payloads are skipped.
- Tool calls: native `tools` parameter; OpenAI returns `tool_calls[*].function.arguments` as a JSON-encoded string. The backend decodes them to a dict so the reply engine sees a single shape.
- Response normalisation: `_normalise_response` lifts `choices[0].message` to top-level `message` so callers do not branch on provider. Servers that already return Ollama-shaped responses pass through unchanged.
- `extra_options` lifts sampling fields (`temperature`, `max_tokens`, `top_p`, `stop`, …) to the payload root and silently drops Ollama-only knobs (`keep_alive`, `num_ctx`, `num_predict`, `think`) that have no equivalent in the OpenAI shape.
- `llm_extra_body` fields are merged into the chat payload root via `setdefault` (the backend-owned `model` / `messages` / `stream` keys always win). Embeddings are not affected.
- `chat(..., on_token=...)` streams: the request is sent with `stream: true`, content deltas are forwarded to the callback as they arrive, and tool-call deltas are reassembled by their `index` (a single call's `arguments` JSON arrives split across chunks). The reassembled result is returned in the **same normalised shape as the buffered path**, so callers — notably the reply engine's tool loop — cannot tell the two apart. Without `on_token` nothing changes: `stream: false`, one buffered response. A raising callback is swallowed (logged) so a UI fault never costs the generated reply.
- One persistent `requests.Session` per backend: the TCP + TLS connection to the provider is kept alive across calls (HTTP keep-alive), so each request skips the handshake. On a remote cloud endpoint that handshake is a large share of a short request's latency.
- `warm_up(model)` fires one minimal (`max_tokens: 1`) chat request to establish the session's connection and warm the provider before the first user query. There is nothing to page into memory on a remote endpoint, but the first call to a cold connection pays the TLS handshake and the provider's routing/cold-start (several seconds on a cloud endpoint); absorbing that at startup keeps the first reply on a warm path. Returns `True` on success, `False` on error or empty model.
- Authentication: `Authorization: Bearer <api_key>` header sent only when `api_key` is non-empty.
- Error logs do not echo URLs or API keys: HTTP errors print only the status code, generic exceptions print only the class name, connection errors print a fixed string and re-raise so callers can apply their own back-off.

## Module-local LLM wrappers

Each migrated module exposes a single intercept point so tests can patch one symbol per module instead of reaching into the backend ABC:

- `jarvis.reply.engine.chat_with_messages(cfg, messages, ...)` — agentic-loop chat boundary.
- `jarvis.reply.planner.call_llm_direct(*, cfg, chat_model, ...)` — planner + step resolver.
- `jarvis.reply.evaluator.call_llm_direct(*, cfg, chat_model, ...)` — terminal evaluator.
- `jarvis.reply.enrichment.call_llm_direct(*, cfg, chat_model, ...)` — memory enrichment extractor + digest passes.
- `jarvis.memory.graph_ops.call_llm_direct(*, cfg, chat_model, ...)` — knowledge graph extraction, best-child picker, node merge.
- `jarvis.memory.conversation._direct_llm(cfg, system_prompt, user_content, ...)` — diary summary, deflection rewrite, topic optimisation.
- `jarvis.tools.builtin.nutrition.log_meal.call_llm_direct(*, cfg, chat_model, ...)` — nutrition extractor + follow-up generator.
- `jarvis.tools.builtin.weather.get_llm_backend` — hoisted to module scope so the place extractor's backend lookup is patchable.

A factory-dispatch wiring guard at `tests/test_factory_dispatch_wiring.py` parametrises across each migrated module and asserts the wrapper actually constructs `OpenAICompatibleBackend` for `llm_provider: openai_compatible` and `OllamaBackend` for `ollama`. A regression that drops `get_llm_backend(cfg)` from a wrapper would bypass every unit test but trip this guard.

`import requests` is re-exported from the package `__init__.py` so tests that patch `jarvis.llm.requests.post` keep working without reaching into the per-backend modules.

## File layout

```
src/jarvis/llm/
├── __init__.py             # public re-exports + function-style helpers
├── backend.py              # LLMBackend ABC + ToolsNotSupportedError
├── ollama.py               # OllamaBackend + extract_text_from_response
├── openai_compatible.py    # OpenAICompatibleBackend + _normalise_response
├── factory.py              # get_llm_backend(cfg) + get_embedding_backend(cfg)
└── llm.spec.md             # this file
```

## Failure handling

Backends fail soft for transient issues so the reply engine can degrade gracefully: timeouts and HTTP errors return `None` (or `[]` for `list_models`); HTTP 400 with `tools` set raises `ToolsNotSupportedError`; any other unexpected error is logged via `debug_log("...", "llm")` and returns `None`. The one exception is `requests.ConnectionError` (server unreachable), which `chat()` re-raises so callers like the intent judge can apply their own back-off — voice, for example, wants a 30s cooldown after a connection-refused error so it stops hammering an unresponsive Ollama between wake words.

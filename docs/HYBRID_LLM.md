# Hybrid LLM Mode

Jarvis runs 100% local by default. **Hybrid mode is an opt-in extension** that lets you route a small slice of harder queries (code, multi-step reasoning, tool-chain composition) to Anthropic's Claude API while every other call continues to hit your local Ollama install. This document explains what it does, how to turn it on, the privacy and cost trade-offs, and how to turn it off again.

> If you only ever want local processing, do nothing. The default mode is `local_only` and every LLM call stays on your machine.

---

## The privacy contract

Jarvis ships with three opt-in stances for the LLM router:

| Mode          | Default? | What it does                                                                 |
| ------------- | -------- | ---------------------------------------------------------------------------- |
| `local_only`  | **Yes**  | Every LLM call hits your local Ollama. The router code short-circuits before any classification call. The cloud SDK is never imported. No telemetry table is written. |
| `hybrid`      | No       | A small warm classifier (the same model Jarvis already uses for intent / tool routing) labels each query. Labels in `cloud_intents` go to the cloud; everything else stays local. |
| `cloud_only`  | No       | Every LLM call goes to the cloud. Intended for evals and benchmarking, not normal use. |

In `hybrid` and `cloud_only` modes:

1. The auto-redaction pipeline that scrubs your diary (emails, AWS / Stripe / GitHub / OpenAI keys, JWTs, password / token / secret keyword pairs) runs on every prompt before it leaves the machine. You can disable this with `auto_redact_before_cloud: false` if you really want raw prompts to go out, but the default is on.
2. A local SQLite table (`llm_router_stats` in the same `jarvis.db` Jarvis already uses) records one row per LLM call: timestamp, provider, model, intent label, token counts, estimated cost in USD, and latency. **Prompt and response text are never stored.**
3. If the cloud call fails (rate limit, transient outage, missing API key), Jarvis transparently falls back to your local Ollama so you still get an answer. The fallback row in the stats table is labelled `local_fallback` so you can measure cloud reliability separately from native local calls.

If you switch back to `local_only`, the cloud SDK stops being imported on the next restart and the telemetry table stops growing. Existing rows stay until you delete them (see [Purging the local stats](#purging-the-local-stats)).

---

## Switching it on

1. Get a key at [console.anthropic.com](https://console.anthropic.com/) and export it in your environment:

   ```bash
   export ANTHROPIC_API_KEY="sk-..."
   ```

2. Edit `~/.config/jarvis/config.json` (or open the Settings window and switch to JSON view) and add the `llm_router` section:

   ```json
   {
     "llm_router": {
       "mode": "hybrid",
       "anthropic_api_key_env": "ANTHROPIC_API_KEY",
       "anthropic_model": "claude-sonnet-4-6",
       "cloud_intents": ["code_complex", "multi_step_reasoning", "tool_use_chain"],
       "auto_redact_before_cloud": true,
       "fallback_to_local_on_error": true,
       "anthropic_cache_threshold_chars": 8000
     }
   }
   ```

3. Restart Jarvis (tray menu, Quit, then relaunch).

The classifier reuses the small warm model Jarvis already keeps resident (`tool_router_model` -> `intent_judge_model` -> `ollama_chat_model`), so there is no extra model to pull.

### Tunable knobs

| Key                              | Default                                     | What it does |
| -------------------------------- | ------------------------------------------- | ------------ |
| `mode`                           | `local_only`                                | One of `local_only` / `hybrid` / `cloud_only`. Anything else falls back to `local_only`. |
| `anthropic_api_key_env`          | `ANTHROPIC_API_KEY`                         | Name of the env var holding the key. Useful if your key manager exposes it under a different name. |
| `anthropic_model`                | `claude-sonnet-4-6`                         | The cloud model ID. See [docs.claude.com/en/docs/about-claude/pricing](https://docs.claude.com/en/docs/about-claude/pricing) for the catalogue. |
| `cloud_intents`                  | `["code_complex", "multi_step_reasoning", "tool_use_chain"]` | The labels that route to the cloud in `hybrid` mode. |
| `auto_redact_before_cloud`       | `true`                                      | Apply the project's auto-redaction to every prompt before egress. Turn off only if you have a hard need for raw prompts. |
| `fallback_to_local_on_error`     | `true`                                      | On any cloud failure (timeout, rate limit, missing key) the call falls back to local Ollama. Disable for strict cloud-only evals. |
| `anthropic_cache_threshold_chars`| `8000`                                      | Above this length, the provider attaches `cache_control: ephemeral` on the system prompt to use Anthropic's prompt cache. Below this length the marker is omitted (Anthropic's minimum cacheable prefix is 2048 tokens on Sonnet 4.6; marking shorter prompts is wasted overhead). |

---

## What goes to the cloud, and what stays local

The classifier labels each query with one of six intents. The default `cloud_intents` set is:

| Intent label          | Default routing | Examples                                                       |
| --------------------- | --------------- | -------------------------------------------------------------- |
| `casual_chat`         | Local           | "hey what's up", small-talk, persona questions                 |
| `simple_query`        | Local           | "what time is it", weather, a single factual lookup            |
| `code_complex`        | **Cloud**       | refactor a class, debug a stack trace, design an API           |
| `multi_step_reasoning`| **Cloud**       | compare options across criteria, plan a multi-stage task       |
| `tool_use_chain`      | **Cloud**       | search then summarise, fetch then analyse, plan then act       |
| `stop_command`        | Local           | "stop", "cancel", "nevermind"                                  |

You can tighten the set (e.g. only `code_complex` to cloud) or widen it. The classifier is cached on a 256-entry LRU keyed on the prompt + classifier model, so the same query inside one conversation only pays the classification cost once.

The labels above are also visible in the local telemetry table, so you can audit what was actually sent and how often.

---

## Cost estimate

Pricing snapshot from [docs.claude.com/en/docs/about-claude/pricing](https://docs.claude.com/en/docs/about-claude/pricing) per million tokens.

| Model              | Input $/1M | Output $/1M | Typical Jarvis turn¹   |
| ------------------ | ---------: | ----------: | ---------------------: |
| Claude Sonnet 4.6  |       3.00 |       15.00 | ~$0.001 - $0.005       |
| Claude Opus 4.7    |       5.00 |       25.00 | ~$0.003 - $0.012       |
| Claude Haiku 4.5   |       1.00 |        5.00 | ~$0.0004 - $0.002      |

¹ Based on ~200 input tokens (system prompt with cache hit) + 100 output tokens for a short conversational reply. Multi-turn coding sessions with tool chains can be 5-10x this. Real per-turn cost is logged in `llm_router_stats.cost_estimate_usd`.

Local Ollama calls are billed at zero in the stats table. Unknown / future model IDs also bill at zero (best-effort, so a model rollout never accidentally over-charges).

**Cache savings.** The provider applies a `cache_control: ephemeral` marker on system prompts longer than `anthropic_cache_threshold_chars`. After the first call, repeated turns within the cache TTL pay ~10% of the input rate on the system-prompt portion. The savings are visible in `cache_read_input_tokens` on the underlying Anthropic response (not currently surfaced in the stats table).

---

## The kill switch

Switching back to fully local takes two changes and one restart:

```json
{ "llm_router": { "mode": "local_only" } }
```

Restart Jarvis. From the next call onward, the router short-circuits before any classification or cloud import.

If you also want to drop the existing stats rows, see below.

---

## Inspecting the local stats

Telemetry lives in the same SQLite file as your diary (`~/.local/share/jarvis/jarvis.db` by default), but in a separate table that the rest of the app does not read. The schema is:

```sql
CREATE TABLE llm_router_stats (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts_utc TEXT NOT NULL,
  provider TEXT NOT NULL,              -- "local" | "cloud" | "local_fallback"
  model TEXT NOT NULL,
  intent TEXT,
  tokens_in INTEGER NOT NULL DEFAULT 0,
  tokens_out INTEGER NOT NULL DEFAULT 0,
  cost_estimate_usd REAL NOT NULL DEFAULT 0.0,
  latency_ms INTEGER NOT NULL DEFAULT 0
);
```

Quick lookups from a shell:

```bash
# Total spend in the last 24h
sqlite3 ~/.local/share/jarvis/jarvis.db "
  SELECT printf('$%.4f', SUM(cost_estimate_usd))
  FROM llm_router_stats
  WHERE ts_utc > datetime('now', '-1 day');
"

# Counts by provider this week
sqlite3 ~/.local/share/jarvis/jarvis.db "
  SELECT provider, COUNT(*), printf('$%.4f', SUM(cost_estimate_usd))
  FROM llm_router_stats
  WHERE ts_utc > datetime('now', '-7 day')
  GROUP BY provider;
"

# Cloud-reliability sanity check (local_fallback rows are cloud failures)
sqlite3 ~/.local/share/jarvis/jarvis.db "
  SELECT provider, COUNT(*) FROM llm_router_stats
  WHERE provider IN ('cloud', 'local_fallback')
  GROUP BY provider;
"
```

### Purging the local stats

```bash
sqlite3 ~/.local/share/jarvis/jarvis.db "DELETE FROM llm_router_stats;"
```

Or, programmatically:

```python
from jarvis.config import load_settings
from jarvis import llm_router_telemetry
llm_router_telemetry.clear(load_settings().db_path)
```

---

## Frequently asked

**Q: I never enabled hybrid mode. Is anything different?**
A: No. With `mode: local_only` (the default), the router code short-circuits before any classification, the cloud SDK is never imported, and the stats table is never created. Behaviour is byte-identical to a router-free build.

**Q: Can I run hybrid mode without an API key, just to see what the classifier picks?**
A: Yes. If the API key env var is missing or empty, every call falls back to local. The intent labels still get written to the stats table so you can see what *would* have gone to the cloud.

**Q: Will my conversation history be uploaded?**
A: Only the portion of a turn that the classifier labels as cloud-bound, and only after the redaction pass. The system prompt, the current user turn, and any recent tool results that come along with the conversation context are subject to Anthropic's standard API privacy terms.

**Q: What about the diary, the knowledge graph, my memories?**
A: They stay on your machine. The router does not export memory content. Memory recall results that get *injected* into a chat turn travel with that turn if the turn is cloud-bound, but redaction runs on the assembled prompt the same way.

**Q: Does this work with my custom MCP servers?**
A: Yes. MCP routing is unaffected by the LLM router. Tool definitions are translated to the Anthropic schema when the cloud path is chosen, and tool results round-trip through the same provider abstraction.

**Q: How do I know which calls actually went to the cloud?**
A: Look at `provider` in `llm_router_stats`: `"cloud"` means Anthropic answered, `"local_fallback"` means the cloud call failed and Ollama handled it, `"local"` means the router stayed local for that turn.

**Q: What happens if Anthropic adds a new model I haven't configured pricing for?**
A: The telemetry layer records the row with `cost_estimate_usd=0.0` rather than crash. Update `providers/pricing.py` with the new rate when you want accurate cost accounting.

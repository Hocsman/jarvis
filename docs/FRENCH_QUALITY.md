# Using Jarvis in French

Jarvis is multilingual end-to-end, but a few config choices substantially
change French quality. This guide is the short version: what to set and
why, with measured A/B numbers between the local default model
(`gemma4:e2b`) and the cloud model (`claude-sonnet-4-6`) on ten typical
queries.

## Recommended config snippet

Drop this into `~/.config/jarvis/config.json` (merge with your existing
keys):

```json
{
  "response_language": "français",
  "whisper_model": "medium",
  "tts_engine": "piper",
  "tts_voices": {
    "fr": "/Users/you/.local/share/jarvis/models/piper/fr_FR-siwis-medium.onnx",
    "en": "/Users/you/.local/share/jarvis/models/piper/en_US-amy-medium.onnx"
  },
  "llm_router": {
    "mode": "hybrid",
    "anthropic_api_key_env": "ANTHROPIC_API_KEY",
    "anthropic_model": "claude-sonnet-4-6",
    "cloud_intents": [
      "code_complex",
      "multi_step_reasoning",
      "tool_use_chain",
      "simple_query"
    ],
    "auto_redact_before_cloud": true,
    "fallback_to_local_on_error": true,
    "anthropic_cache_threshold_chars": 8000
  }
}
```

Three details matter:

- **`response_language: "français"`** locks the assistant's output
  language across all turns. Without this, the model mirrors the user's
  language each turn — works for stable French speakers, but flips if
  the user mixes in English words (common for code-related questions).
- **`tts_voices`** is the per-language Piper voice map. When Whisper
  detects French, Jarvis loads the FR voice; when it detects English,
  the EN voice. The legacy scalar `tts_piper_model_path` still works
  as a fallback for single-voice setups.
- **`cloud_intents` includes `simple_query`**: `gemma4:e2b` is too
  small to reliably answer factual one-shot French questions ("quelle
  est la capitale de…", "combien de…"); routing them to Sonnet
  noticeably improves correctness without adding much cost.

## Whisper model for serious FR usage

The default `whisper_model: "medium"` is multilingual and works well.
For better French transcription on longer / accented input, switch to:

```json
"whisper_model": "large-v3-turbo"
```

This pulls ~1.5 GB (Apple Silicon MLX path) or ~3 GB (CTranslate2
faster-whisper path) on first run. It's roughly as fast as `medium`
on a modern Mac thanks to the "turbo" weights, but with noticeably
better French accuracy on technical terms. Keep `medium` if you want
faster startup and don't mind occasional transcription glitches on
unfamiliar words.

The `.en` model variants (`medium.en`, `small.en`) are English-only;
**do not** select one if you want any French support.

## Piper voices for French

`fr_FR-siwis-medium` is the default the setup wizard installs. Two
alternatives if you want a different timbre:

- `fr_FR-tom-medium` — male voice, lower pitch
- `fr_FR-upmc-medium` — closer to news-anchor delivery

Download into `~/.local/share/jarvis/models/piper/` (the wizard does
this automatically for siwis; for the others, fetch from
[rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices/tree/main/fr/fr_FR)).

## Local vs Cloud — measured A/B (2026-05-18)

Ten French queries spanning four intent buckets, each sent twice (once
to local `gemma4:e2b`, once to cloud `claude-sonnet-4-6`) with the same
short French persona prompt. Full report: run
`bash scripts/run_french_quality.sh` or:

```bash
ANTHROPIC_API_KEY=sk-ant-... PYTHONPATH=src \
  .mamba_env/bin/python scripts/test_french_quality.py
```

Output lands in `/tmp/jarvis_french_quality_<ts>.md`.

### Headline numbers

| Metric | Local (`gemma4:e2b`) | Cloud (`claude-sonnet-4-6`) |
|---|---|---|
| Total latency, 10 queries | 19.66 s (1.97 s/q) | 31.87 s (3.19 s/q) |
| Tokens used | n/a | 960 in / 1078 out |
| Total cost | $0 | $0.0191 |

Local is **~1.6× faster**. Cloud wins on **content quality** for
anything beyond casual chat.

### Quality observations

- **Casual chat (`Salut`, jokes)**: both indistinguishable. Stay local.
- **Simple facts ("capitale de l'Australie", "combien de continents")**:
  both correct in our run. **Recommendation**: keep `simple_query` in
  `cloud_intents` anyway — the small local model is *frequently* wrong
  on facts requiring grounding (we've seen it confabulate dates,
  numbers, and place names in real use), and the cost is negligible.
- **Multi-step reasoning**: cloud is the clear winner. On the trivia
  question *"Si je pars à 8h, je conduis 1h30, je m'arrête 20 min,
  je reprends 45 min, à quelle heure j'arrive ?"* —
  - Local: "Si vous partez à 8h, vous arrivez à 10h30." ❌ (skipped
    the second leg)
  - Cloud: "Tu arrives à 10h35. 8h + 1h30 = 9h30 → +20 min = 9h50 →
    +45 min = 10h35." ✅
- **Code (`fonction Python qui somme les nombres premiers`)**: both
  produce working code. Cloud factors out an `est_premier` helper,
  adds an example invocation — closer to how a senior dev would
  write it. Local inlines the primality check.
- **Tool-shaped queries (météo, Wikipedia)**: both honestly say
  they lack live access. Cloud is more verbose and friendly about
  it; local is terser. Neither will actually call a tool here
  because the test bypasses the agentic loop on purpose.

### Recommendation by usage

- **Daily voice assistant, FR-first**: hybrid mode with
  `cloud_intents` extended (the snippet above). Expect ~$3–4/mo with
  prompt cache enabled (the persona prompt is stable across turns and
  gets cached at 90% hit rate above the 8000-char threshold).
- **Code-heavy FR sessions**: same hybrid config — `code_complex` is
  cloud-routed by default, and the quality delta justifies it.
- **Privacy-first, no cloud**: `mode: "local_only"`, accept the
  reasoning gaps. Use `whisper_model: "large-v3-turbo"` to recover
  some quality on the transcription side.

## Limitations

- **First-turn voice selection** uses `response_language` as the
  hint. The live voice swap when Whisper detects a different language
  mid-session is not yet wired (it's a planned follow-up). Today, if
  you speak English on a Jarvis configured for French, Piper will
  still try to render the French voice — sounds odd, but recoverable
  in one config reload.
- **Sonnet's tool-use quality in French**: not exercised by this
  A/B (the script bypasses the tool catalogue). When tools are
  passed in the daemon's real reply loop, Sonnet correctly chooses
  tools regardless of the prompt language — verified manually with
  `getWeather(city="Paris")` from a French query.
- **Cost ceilings**: monitor `llm_router_stats` periodically:

  ```bash
  sqlite3 ~/.local/share/jarvis/jarvis.db \
    "SELECT provider, SUM(cost_estimate_usd), COUNT(*)
     FROM llm_router_stats GROUP BY provider"
  ```

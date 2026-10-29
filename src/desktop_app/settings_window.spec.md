# Settings Window Specification

Auto-generated settings UI that dynamically builds its interface from config field metadata.

## Overview

The Settings Window provides a graphical interface for editing `config.json` without requiring users to manually edit JSON. It reads the current config, presents categorised fields with appropriate input widgets, and saves changes back.

## Design Principles

1. **Metadata-driven**: All fields are defined in a `FIELD_METADATA` registry. Adding a new config parameter to the settings UI requires only adding a `FieldMeta` entry — no widget code changes.
2. **Minimal config files**: Only non-default values are written to `config.json`. Removing a field from the config reverts it to the default.
3. **Preserves unknown keys and values**: Keys not managed by the UI (e.g. `mcps`, `_config_version`, future additions) are preserved when saving, and so is a value a `choice` field has no option for, and a microphone set by name (a `device` field still drops an index that names no listed microphone, see "Hardware Device Selection").
4. **Theme-consistent**: Uses the shared Jarvis theme from `themes.py`.

## Architecture

```
FieldMeta (dataclass)
  ├── key: str           # config.json key name
  ├── label: str         # Human-readable label
  ├── description: str   # Tooltip text
  ├── category: str      # Tab grouping key
  ├── field_type: str    # "bool" | "int" | "float" | "str" | "choice" | "device" | "list" | "map"
  ├── choices            # For "choice"/"device": [(value, display), ...]
  ├── min_val / max_val  # Numeric bounds
  ├── step               # Increment step
  ├── suffix             # Unit label (e.g. "s", "ms", "WPM")
  ├── nullable           # Whether None is valid (shows placeholder)
  └── example            # For "map": a sample row shown when adding one
```

## Widget Mapping

| field_type | Widget | Notes |
|-----------|--------|-------|
| `bool` | QCheckBox | |
| `int` | QSpinBox | With bounds, step, suffix |
| `int` (nullable) | QCheckBox + QSpinBox | Checkbox enables/disables the spinbox |
| `float` | QDoubleSpinBox | With bounds, step, suffix |
| `str` | QLineEdit | Placeholder if nullable |
| `password` | QLineEdit (EchoMode.Password) | Masked input for API keys; same value extraction as `str` |
| `choice` | QComboBox | Pre-defined options. A current value the field does not list (a model served by a remote provider, such as an `intent_judge_model` pin, or a hand-edited value) is added as its own option, shown as the raw value, and selected |
| `device` | QComboBox | Dynamically populated from sounddevice, one option per input device name, each saving that name. A name the list lacks is added as its own option and selected; an index selects the microphone it names (see "Hardware Device Selection") |
| `list` | QListWidget + Add/Edit/Remove buttons | Stores as JSON array in config |
| `map` | QListWidget + Add/Edit/Remove buttons | One `key -> value` row per entry; stores as a JSON object in config. A row that names no key or no value is refused when added, and dropped on save if it got in some other way. Only the first arrow separates, so a value may contain one; a key listed twice keeps its last value |

## Layout

The settings window uses a sidebar navigation pattern: a fixed-width `QListWidget` on the left lists categories, and a `QStackedWidget` on the right shows the selected category's form. This avoids horizontal overflow from too many tabs.

## Categories (Sidebar Order)

1. LLM & AI Models
2. LLM Provider
3. Text-to-Speech
4. Piper TTS
5. Kokoro TTS
6. Kyutai TTS
7. Chatterbox TTS
8. Voice Input (includes microphone device selection)
9. Wake Word
10. Speech Recognition (Whisper)
11. Voice Activity Detection
12. Timing & Windows
13. Memory & Dialogue
14. Location
15. Features (includes web search, Wikipedia fallback, low-power mode, startup tune, the startup update check, and dictation toggles)
16. 🙋 Permissions
17. ⏰ Rappels
18. 🌅 Routines
19. 🛠️ Projets
20. MCP Servers
21. Advanced

### 🙋 Permissions

What the gate asks about before doing it, and how long a card waits.

### ⏰ Rappels

The master switch for a subsystem that speaks unprompted, plus the model
of the reminder chain. Both belong here rather than in `config.json`:
one is the only way to stop it, and the other is the pin that keeps the
goal judge and the journal reader on his own machine or network wherever
the chat runs, since the reminder chain deliberately skips the cloud-safe
rewrite. With the chat at home it also reads the hour out of the
sentence; with the chat in the cloud the chat model reads it, because the
turn has already sent it the sentence. The field's description says both.

What a reminder says and when is not settable here — that is one
sentence about one promise, not a number that applies to all of them.

### 🌅 Routines

The master switch and the two limits that decide when she gives up on an
occurrence and when she gives up on a routine. What each routine may
reach lives in `yuba/routines.md`, one block per routine.

### 🛠️ Projets

The folder whose sub-folders can be handed to Claude Code, the executable,
and how long a job may run (`src/jarvis/projets/projets.spec.md`). The
folder is empty by default, which keeps the feature off: it sends the
user's code to Anthropic, so it belongs where he can see it and switch it
off, not only in `config.json`.

### Piper TTS

Besides the voice's pace and expressiveness, the page holds the two ways to
choose a voice (`tts.spec.md`, "Which voice"): `tts_piper_model_path`, one
voice for every language that is not listed, and `tts_piper_voices`, a
`map` field of language to voice, one row per language, for example
`fr -> fr_FR-siwis-medium`. A voice is a path to a `.onnx` model or a Piper
voice name, fetched on first use. An empty map is the default and is not
written to `config.json`.

### LLM Provider

Selects the runtime that serves the LLM and holds the provider-aware
connection fields: `llm_provider` (Ollama / OpenAI-compatible / Anthropic, the
values of `LLM_PROVIDERS`), `llm_base_url`, `llm_api_key_env` (the name of the
environment variable holding the key, plain text since it is not a secret),
`llm_api_key` (password), `llm_chat_model`, and the four `embedding_*` fields
(`embedding_provider`, `embedding_base_url`, `embedding_api_key`,
`embedding_model`). The model fields are free-text `str` — a remote provider's
model name is not in the Ollama `SUPPORTED_CHAT_MODELS` catalogue. The
Anthropic provider is set here only: the setup wizard does not offer it.

Every connection/credential/model field is nullable: leaving it empty falls
back to the Ollama settings on the "LLM & AI Models" page. A default Ollama
install therefore never needs to open this page, and the minimal-config save
behaviour keeps these keys out of `config.json` until the user sets them.

Unlike the setup wizard's provider page, the settings window does **not**
clear the OpenAI-compatible fields when the user switches `llm_provider` back
to Ollama: it is metadata-driven with no cross-field logic, and a blanket
clear would wipe the supported "Ollama chat + remote embeddings" split
(`llm_provider: ollama` with `embedding_provider: openai_compatible`). Stale
values are harmless because the backend resolves per-provider: the Ollama path
uses `ollama_base_url` / `ollama_chat_model` and `OllamaBackend` ignores any
API key. To drop a leftover value, clear that field and save.

### Speech Recognition (Whisper)

The `whisper_model` dropdown lists every model in `SUPPORTED_WHISPER_MODELS`
(`jarvis/config.py`), the list the setup wizard's Whisper page draws its
slider from, so a size picked in either one is listed by the other. An
English-only id (`.en`) reads as its size name followed by "(English only)".

`whisper_vocabulary` is a list field, one name per row: the names Whisper is
given as a hint alongside `weather_city` (`listening.spec.md`, "The names he
uses").

### Features

The Features category exposes user-facing runtime toggles that do not need a
dedicated page:

- `web_search_enabled`
- `brave_search_api_key`
- `wikipedia_fallback_enabled`
- `low_power_mode`
- `tune_enabled`
- `update_check_enabled`
- `dictation_enabled`
- `dictation_hotkey`
- `dictation_filler_removal`
- `dictation_custom_dictionary`

`dictation_hotkey` is a dropdown of `DICTATION_HOTKEY_PRESETS` (`jarvis/config.py`)
built by `dictation_hotkey_choices()` (`jarvis/dictation/hotkeys.py`), the same builder the setup wizard's
Dictation page uses, so both windows offer the same presets with the platform
default marked `(default)`.

`low_power_mode` is a boolean toggle. When enabled, the voice listener skips
LLM startup warmup and the Ollama keep-alive windows used by warmup and the
intent judge are short. The setting is saved only when it differs from the
default, like every other metadata-managed field.

`update_check_enabled` is a boolean toggle, on by default. Off, the bundled app
sends no update check by itself at startup; the tray's "Check for Updates" still
works when the user asks (see "Update System" in `desktop_app.spec.md`).

## Hardware Device Selection

The Voice Input tab includes a device dropdown populated at window open time via `sounddevice.query_devices()`. It lists each input-capable device name once, without surrounding spaces: Windows lists one microphone under each of its audio systems, and every entry would save the same value. Choosing one stores that name; choosing System Default stores nothing. A name stays true across a replug, where an index can move onto another device.

The listener and hold-to-dictate match a stored name against the input device names (`resolve_input_device`, see `listening.spec.md`), so the config can also name a microphone by part of its name (`"voice_device": "MSI GV60"`). A stored name that is exactly a listed one selects that option. Any other value that does not read as an index is added as its own option, shown as `🎤 <name> (by name)` whether or not a connected device matches it, and selected, so a Save writes it back as it was read. Choosing a listed option, or Reset, replaces it.

A stored index, in any form the listener reads as that index (`"1"`, `1`, `"01"`), selects the microphone it names, and a Save writes that microphone's name in its place. An index that names no listed microphone (a device unplugged when the window opens, or an output-only one) is not kept: the dropdown falls back to System Default and a Save drops the key. A blank value, and `default` or `system` in any letter case, which the config and the listener read as the system default, also show as System Default.

## Save Behaviour

- Only keys that differ from `get_default_config()` are written.
- A `choice` value the field does not list, and a microphone set by name, are written back exactly as they were read: saving another setting never moves one to the first option or to the default. Choosing a listed option, or Reset, still replaces it.
- Existing keys not managed by the UI are preserved (e.g. `mcps`, `active_profiles`, `wake_aliases`, `allowlist_bundles`, `stop_commands`).
- After save, a dialog confirms success and reminds the user to restart.
- If the daemon is running when save completes, the tray app offers to restart it.

## Reset to Defaults

- Prompts for confirmation.
- Resets all widget values to `get_default_config()` values.
- Does NOT immediately save — user must still click Save.

## Integration

- Accessed via "⚙️ Settings" in the system tray menu.
- Opens as a modal QDialog.
- Imported from `show_settings` when the window is first opened. Importing the module loads none of sounddevice, numpy or pynput: the hotkey labels come from `jarvis/dictation/hotkeys.py`. sounddevice is imported when the window is built, to list the input devices for the Voice Input tab.

## MCP Servers Section

The MCP Servers category is **not** metadata-driven — it uses a custom page because `mcps` is a complex dict structure.

### Layout

- Description label explaining what MCP servers are
- List widget showing configured servers (display name from catalogue if recognised, otherwise `🔌 {name}`)
- Buttons: **Add from Catalogue**, **Add Custom**, **Edit**, **Remove**
- Detail panel showing the selected server's name, command, args, and env vars

### Add from Catalogue

Opens `_MCPCatalogueDialog` showing all entries from `mcp_catalogue.CATALOGUE`. Already-configured servers appear checked and disabled. Servers that require an API key show a 🔑 badge. When the user confirms, they're prompted for any needed API keys.

### Add Custom

Opens `_MCPEditDialog` with fields for name, command, args (space-separated), and env vars (KEY=VALUE pairs). Validates that name and command are non-empty.

### Edit

Opens `_MCPEditDialog` pre-filled with the selected server's config. Name is read-only during edit.

### Remove

Prompts for confirmation, then removes the server from the in-memory dict.

### Save Behaviour

On save, the `mcps` dict is written to config.json if non-empty, or removed entirely if empty. On reset, all MCPs are cleared.

## Fields NOT Exposed in UI

These fields are managed elsewhere or are too complex for a simple form:

- `db_path` / `sqlite_vss_path` — internal storage paths
- `active_profiles` — list managed by setup wizard
- `allowlist_bundles` — list of bundle IDs
- `wake_aliases` — list of strings (complex editing)
- `stop_commands` / `stop_command_fuzzy_ratio` — list of strings
- `use_stdin` — developer/CLI flag
- `voice_debug` — environment variable only
- `whisper_min_audio_duration` / `whisper_min_word_length` — rarely changed advanced params
- `vad_frame_ms` / `vad_pre_roll_ms` — low-level VAD timing

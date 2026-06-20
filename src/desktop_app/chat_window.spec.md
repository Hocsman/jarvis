# Chat Window Specification

A text chat interface for Jarvis, alongside the existing voice path. Voice
remains the primary modality; text is a first-class sibling that shares the
same conversation, memory, and tools.

## Core principle: one conversation

Voice and text are two views onto the **same** conversation. Both feed the
single `_global_dialogue_memory` owned by the daemon. A question asked by voice
and a follow-up typed in the chat window are part of one continuous turn
sequence, share the same hot window, and produce a single diary entry at the
end of the session. There is no "text conversation" vs "voice conversation"
split in storage.

## Daemon contract

The core `jarvis` package exposes a text-submission entry point with no
knowledge of the desktop app. It mirrors the diary-callbacks pattern already
used for end-of-session UI updates.

### `submit_text_query(text: str) -> None` (in `jarvis.daemon`)

- Fire-and-forget: spawns a worker thread and returns immediately so the
  caller (the Qt main thread) never blocks. The final reply is delivered via
  the `on_complete` callback / `complete` IPC event, never via the return
  value. This mirrors how `_check_and_update_diary` is invoked from a worker
  QThread in the existing desktop app.
- The worker thread runs
  `run_reply_engine(db, cfg, tts=None, text, _global_dialogue_memory, language=None)`.
- `tts=None` — text chat never speaks. Audio output stays a voice-only concern.
- `language=None` — text input has no Whisper-detected language; tools fall
  back to their own defaults, same as a voice query with no language hint.
- Redaction still applies (`run_reply_engine` calls `redact(text)` before
  anything reaches the model or the diary). This is the privacy boundary and
  it is shared with the voice path. The redacted query is what the `start`
  event carries.

### Concurrency: one query at a time

A single `_chat_query_lock` guards the reply engine. The text path acquires
it non-blocking: if a query is already running (voice or text), a new text
submission is **rejected**, not queued, and the caller is notified via the
`busy` event so the UI can show "Jarvis is busy" rather than silently
dropping the message. The voice path acquires the same lock blocking (via
`jarvis.daemon.query_lock`) so a voice query waits for an in-flight text
query to finish rather than being dropped. Voice and text therefore cannot
run `run_reply_engine` concurrently against the shared dialogue memory.

### Cancellation

`cancel_active_chat_query` sets a per-query `threading.Event`. The chat
worker checks it after `run_reply_engine` returns and, if set, drops the
reply (delivering `complete(None)`). The Stop button calls this, not
`request_stop` (which is the daemon lifecycle shutdown signal and would tear
down the whole voice assistant). Cancellation does not abort the in-flight
LLM compute — `run_reply_engine` has no mid-loop abort hook — it discards the
result so it is not displayed.

### Callbacks (bundled mode, same process)

`submit_text_query` accepts optional per-call callbacks as keyword arguments.
The desktop app wires these to Qt signal emitters so UI updates happen on the
main thread. All are optional and default to `None`.

| Callback | Payload | When |
|----------|---------|------|
| `on_start` | `str` (the redacted query, for display) | Worker thread has picked up the query |
| `on_token` | `str` | Not emitted by the current engine; reserved for future streaming reply support |
| `on_tool_call` | `dict` | Not emitted by the current engine; reserved for future per-tool-call visibility |
| `on_complete` | `Optional[str]` (final reply, or `None` on failure/stop/cancel) | Worker thread is done |
| `on_busy` | `None` | A submission was rejected because a query is already running |

Callbacks fire from the worker thread. The desktop app must marshal them onto
the Qt main thread via signals (same pattern as `DiaryUpdateDialog`).

### IPC protocol (subprocess mode)

When the daemon runs as a subprocess (development mode), callbacks are not
available. The daemon emits newline-delimited JSON events prefixed with
`__CHAT__:` to stdout. The desktop app intercepts these lines (alongside the
existing `__DIARY__:` lines) and forwards them to the chat window.

Event shapes (mirrors the diary IPC):

```json
{"type": "start",  "data": "<redacted query>"}
{"type": "token",  "data": "<chunk>"}        // reserved for future streaming; not emitted today
{"type": "tool",   "data": {"name": "...", "args": "...", "result": "..."}}  // reserved for future per-tool visibility; not emitted today
{"type": "complete", "data": "<final reply or null>"}
{"type": "busy",   "data": null}
```

`__CHAT__:` lines must never contain unredacted user text. The `start` event
carries the already-redacted query (redaction happens before the worker thread
starts, so the IPC payload is safe to log).

### Subprocess query-in channel (desktop → daemon)

In subprocess mode the desktop app and daemon are separate processes, so the
``ChatWindow`` cannot call ``submit_text_query`` directly. The desktop app
writes a single line to the daemon's stdin:

```json
__CHAT_QUERY__:{"text":"<user input>"}
```

The daemon's stdin monitor (extended from the existing ``SHUTDOWN`` handler)
parses these lines and calls ``submit_text_query(text, use_ipc=True)`` so the
reply comes back via the ``__CHAT__:`` event stream above. Lines that don't
match either prefix are ignored (the monitor still treats bare ``SHUTDOWN``
and EOF as shutdown signals, unchanged).

## Desktop window

### `ChatWindow` (in `desktop_app.chat_window`)

A `QMainWindow` with:

- A read-only transcript area (chat bubbles or monospaced log style; theme
  colours from `themes.py`).
- A single-line input box with send button. Enter sends; Shift+Enter inserts a
  newline (multi-line input).
- A "Stop" button that calls `jarvis.daemon.cancel_active_chat_query()`. This
  sets a per-query cancellation flag; the chat worker drops the reply when
  `run_reply_engine` returns (delivering `complete(None)`) and the thinking
  indicator resets immediately. It is distinct from `request_stop` (full
  daemon shutdown) and never tears down the voice listener. Visible only
  while a query is in flight.
- A status indicator label that shows "Jarvis is thinking…" while a query is
  running and is hidden otherwise.

### Tray integration

A `💬 Chat...` entry is added to the tray menu, below the existing
face/logs/memory entries. Clicking it shows (or raises) the `ChatWindow`. The
window is created lazily on first open and kept alive for the session
(same lifecycle as `DictationHistoryWindow`).

### Theme

All styling uses `JARVIS_THEME_STYLESHEET` from `desktop_app.themes`. No
hardcoded colours. The window is dark-themed and consistent with the rest of
the app.

### Lifecycle

- Created lazily on first tray open.
- Hidden windows stay responsive: a `showEvent` reloads nothing (the transcript
  is in-memory and authoritative for the session); the daemon-side callback
  still fires while hidden, so a reply that lands while the window is closed
  appears on next open.
- Closing the window hides it; it does not stop the daemon or end the
  conversation. The conversation ends on the same inactivity timeout as the
  voice path (`cfg.dialogue_memory_timeout`).

## Privacy

- Redaction runs inside `run_reply_engine` before the query reaches the model
  or is written to the dialogue memory. This is the same boundary the voice
  path uses and it is what protects the durable record (diary) and the model
  context.
- The transcript area shows the user's local echo (what they just typed) so
  the conversation reads naturally. The transcript is in-memory only and is
  never persisted to disk; the diary remains the single durable record,
  written through the existing `update_diary_from_dialogue_memory` path at
  session end, and that path sees only the redacted query.
- The `__CHAT__:` IPC lines carry only the redacted query (in the `start`
  event) and event metadata, so the subprocess stdout stream (which the
  desktop app captures for the log viewer) never leaks raw user input.

## What the system does not do

- **No streaming tokens.** The reply is delivered as a single string on
  `on_complete`. The `on_token` / `on_tool_call` callbacks and IPC event
  types are declared but not emitted by the current engine; they are reserved
  for future streaming and per-tool-call visibility work.
- **No external integrations** (Slack, Telegram, Discord). Those would route
  through the same `submit_text_query` entry point but are not wired.
- **No text-input wake word.** Text is always "directed": there is no intent
  judge, no echo detection, no wake word. The user typing is the intent.
- **No TTS.** Text chat is silent. If the user wants spoken replies, they use
  the voice path.

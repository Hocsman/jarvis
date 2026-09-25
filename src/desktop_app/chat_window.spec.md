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
- `tts=None`: text chat never speaks. Audio output stays a voice-only concern.
- `language=None`: text input has no Whisper-detected language; tools fall
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

Stop never calls `request_stop`, which is the daemon lifecycle shutdown
signal and would tear down the whole voice assistant. It cancels the one
query in flight, and does so in three places because no single one of them
is sufficient.

**In the window.** Pressing Stop marks the exchange abandoned and resets the
thinking indicator at once. `_on_complete` then declines the reply for that
exchange and clears the mark, so the next send is unaffected. This is the
part that actually keeps the answer out of the transcript: cancellation
cannot unwind a request already inside the engine, so the reply arrives
regardless.

**In the daemon.** `cancel_active_chat_query` sets a per-query
`threading.Event`; the chat worker checks it after `run_reply_engine` returns
and drops the reply, delivering `complete(None)`.

**Across the process boundary.** In subprocess mode the query runs in the
daemon, whose module globals are a different instance from the desktop app's,
so calling the cancel function locally would set a flag nobody reads. The app
writes `__CHAT_CANCEL__` to the daemon's stdin, the same pipe submissions use,
and `handle_chat_cancel_stdin_line` applies it there. A broken pipe is not
surfaced: the window has already refused the reply, and a dead daemon has no
query to cancel.

Cancellation does not abort the in-flight LLM compute (the engine has no
mid-loop abort hook): it discards the result so it is never shown.

### Confirmed actions

A destructive action approved from the card runs on the daemon side, and its
narration is delivered like a reply: the `complete` event in subprocess
mode, the `on_confirm_reply` callback (wired with the other confirmation
callbacks) in bundled mode. A chat-originated action is not spoken; a
voice-originated one still is.

### Callbacks (bundled mode, same process)

`submit_text_query` accepts optional per-call callbacks as keyword arguments.
The desktop app wires these to Qt signal emitters so UI updates happen on the
main thread. All are optional and default to `None`.

| Callback | Payload | When |
|----------|---------|------|
| `on_start` | `str` (the redacted query, for display) | Worker thread has picked up the query |
| `on_token` | `str` | A content delta, as the model generates it (see [Streaming](#streaming)) |
| `on_stage` | `dict` (`{"stage": str, "detail": Optional[str]}`) | The engine entered a preparation phase (see [Streaming](#streaming)) |
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
{"type": "token",  "data": "<chunk>"}        // content delta while the reply generates
{"type": "stage",  "data": {"stage": "routing|memory|planning|tool|generating", "detail": "<tool name or null>"}}
{"type": "tool",   "data": {"name": "...", "args": "...", "result": "..."}}  // reserved for future per-tool visibility; not emitted today
{"type": "complete", "data": "<final reply or null>"}
{"type": "busy",   "data": null}
{"type": "confirm", "data": {"request_id": "...", "shown": "...", "hazards": []}}
{"type": "confirm_settled", "data": {"request_id": "...", "outcome": "..."}}
{"type": "confirm_nack", "data": {"request_id": "...", "outcome": "..."}}
{"type": "rewound", "data": {"user_index": 2}}      // the daemon's verdict on a __CHAT_REWIND__: line
{"type": "rewind_nack", "data": {"user_index": 2}}
```

`__CHAT__:` lines must never contain unredacted user text. The `start` event
carries the already-redacted query (redaction happens before the worker thread
starts, so the IPC payload is safe to log), and the rewind verdicts carry the
ordinal only, never the message.

### Subprocess query-in channel (desktop -> daemon)

In subprocess mode the desktop app and daemon are separate processes, so the
`ChatWindow` cannot call `submit_text_query` directly. The desktop app
writes a single line to the daemon's stdin:

```json
__CHAT_QUERY__:{"text":"<user input>"}
```

The daemon's stdin monitor (extended from the existing `SHUTDOWN` handler)
parses these lines and calls `submit_text_query(text, use_ipc=True)` so the
reply comes back via the `__CHAT__:` event stream above. A bare
`__CHAT_CANCEL__` line cancels the query in flight, travelling the same pipe
for the same reason: the query runs in this process, so the flag has to be set
in it. Lines that don't match any prefix are ignored (the monitor still treats
bare `SHUTDOWN` and EOF as shutdown signals, unchanged).

Rewind travels the same pipe (the conversation lives in the daemon's memory,
which the desktop process cannot touch in subprocess mode):

```json
__CHAT_REWIND__:{"user_index": 2, "content": "<the message as shown>"}
```

`user_index` is 1-based over the window's transcript and `content` is the
message's text; the daemon anchors on the text (see **Rewind**). It answers
on the `__CHAT__:` bus with `rewound` or `rewind_nack`, carrying the same
ordinal and never the text, and the window acts only on that verdict. A line
with no usable ordinal is consumed and ignored, mirroring `__CHAT_QUERY__:`
handling.

Confirmation decisions travel on stdin via `__CHAT_DECISION__:`:

```json
__CHAT_DECISION__:{"request_id":"cf_123","approved":true}
```

Chat IPC lines are routed to the chat window and never reach the general log
viewer, in subprocess mode (the log reader withholds them) and in bundled
mode alike (the in-process stdout capture drops them). The `complete` event
carries the whole assistant reply, which can echo back whatever the user
typed, and the log window is outside the redaction invariant the chat path
maintains. Diary IPC still reaches the log viewer, which is what it is for.

## Desktop window

### `ChatWindow` (in `desktop_app.chat_window`)

A `QMainWindow` styled as a futuristic phone with a single contact:

- A translucent, frameless window with a rounded graphite chassis, inset
  display and amber accents from the shared theme. Its initial portrait size
  is capped by the available screen, with a 380 x 560 minimum usable size.
- A draggable top bar with accessible minimise and close controls. Double-click
  toggles maximise/restore; a bottom-right resize grip resizes the window.
  The platform close shortcut also hides the chat without ending the conversation.
- A vector amber core emblem, compact chat capsule and home indicator complete
  the phone shell. No remote assets, network requests or decorative animations
  are required.
- An empty conversation shows an introductory panel, never a fabricated message.
  It disappears as soon as a typed, seeded, assistant or local notice arrives,
  whether the window is on screen at that moment or not.
- A contact header (avatar, "Jarvis", and a presence line such as "Online" or
  "Typing..." while a query is in flight).
- A read-only transcript area: a scrollable stack of speech bubbles (theme
  colours from `themes.py`). The user's messages are right-aligned accent
  bubbles, Jarvis's replies are left-aligned dark bubbles, and local notices
  are small centred lines. Each bubble carries a small muted timestamp. Sent
  messages the daemon took additionally carry a subtle `⟲` rewind button
  (see **Rewind** below) to the left of the bubble. Whenever any user,
  assistant, or local notice message is added, the transcript scrolls to the
  bottom after layout so the latest message stays visible; a resize or a
  re-wrap keeps a reader who scrolled up where they were. The transcript
  mirrors the single conversation's message list and is rebuilt atomically
  on rewind.
- Confirmation card: placed inline above the composer. When an action requires
  permission, this card raises with the action details and hazards. Declining
  is the default (Escape key, default button).
- A multi-line input box with send button. Enter sends; Shift+Enter inserts a
  newline (multi-line input). A send while a reply is pending is refused and
  the typed text stays in the box.
- The inset composer uses labelled, keyboard-accessible icon controls and a
  separate keyboard hint. Message bodies and local notices are selectable plain
  text, including strings that resemble HTML. Bubbles resize with the window.
- A "Stop" button. It marks the exchange abandoned locally, routes the
  cancellation to whichever process is running the query (see Cancellation),
  and resets the thinking indicator immediately. It is distinct from `request_stop` (full
  daemon shutdown) and never tears down the voice listener. Visible only
  while a query is in flight.
- A status indicator label that shows "Jarvis is thinking..." while a query is
  running. When the daemon is starting, stopping, stopped, or has exited
  unexpectedly, the same area stays visible as a local lifecycle banner and
  explains whether the user should wait or start listening again.

### One conversation, like an SMS contact

There is exactly one chat: the daemon's single dialogue memory, displayed as a
text-message thread. There is no session list, no "new session" button, and
nothing is written to disk, so a fresh app run starts blank (the transcript is
in-memory and authoritative for the session thereafter; in bundled mode the
daemon's hot window seeds recent voice turns on first show).

### Rewind

Every sent message the daemon took carries a subtle `⟲` button to the left
of its bubble. Clicking it:

1. Asks the daemon to roll the shared memory back to before that user turn
   (`daemon.rewind_chat_to_user(user_index, text)` in bundled mode, the
   `__CHAT_REWIND__:` line in subprocess mode) and shows the exchange as in
   flight meanwhile: no send, no second rewind, Stop available.
2. On the daemon's `rewound` verdict, truncates the transcript to keep the
   message itself (its old reply and everything after it are dropped) and
   re-submits the same text, so the agent generates a fresh reply that lands
   through the normal `complete` path. If Stop was pressed while the verdict
   was pending, the transcript is truncated to match the memory and nothing
   is re-asked.
3. On `rewind_nack`, changes nothing and appends a local notice: the
   transcript and the memory never part ways on a refusal.

The daemon anchors the rewind on the message's text, the ordinal only
telling identical messages apart (`DialogueMemory.rewind_before_user_message`).
The transcript can hold fewer user turns than the memory (a voice exchange
while the window is open, turns older than the seeding window) or more (a
message the daemon never took), so a positional count alone would name the
wrong turn. A turn the memory no longer holds (pruned by a diary pass,
rebuilt by a daemon restart) is refused, never approximated.

A rewind drops the turn itself so the regeneration does not duplicate it,
clears the hot-window caches and tool carryover, and closes a question still
waiting for a decision as `expiré` (the context that produced it is being
rewritten; see `policy.spec.md`). Rewinding rolls back the voice context
too: it is the same memory. Rewind is disabled while a query or another
rewind is in flight and when the daemon is not running.

Rows keep their ordinals in step with the memory: a message the daemon
rejected as busy, or one the engine gave up on before storing, keeps its
bubble and loses its button; a cancelled message keeps both, because the
engine stores the turn and only the answer is declined. When a daemon comes
back after a stop, every earlier row loses its button (its ordinal named a
turn of a memory that is gone) and the new turns count from one again.

### Tray integration

A `💬 Chat` entry is added to the tray menu, below the existing
face/logs/memory entries. Clicking it shows (or raises) the `ChatWindow`. The
window is created lazily on first open and kept alive for the session
(same lifecycle as `DictationHistoryWindow`).

The chat window is usable only while the daemon is running. When the tray
state is starting, stopping, stopped, or has ended unexpectedly, the input and
send button are disabled and a local banner explains the state. On daemon
start/restart, the tray hands the window its submit, cancel and control hooks
together (they share a lifecycle), hides the banner, and re-enables the
controls. On daemon stop, failed start or unexpected subprocess exit, the tray
clears all three hooks and the confirmation decision writer, so neither a
query, a cancel, a rewind nor a decision can be written to a dead pipe.

### Theme

All styling uses `JARVIS_THEME_STYLESHEET`, `CHAT_THEME_STYLESHEET` and the shared
palette from `desktop_app.themes`. No hardcoded colours. The window is
dark-themed and consistent with the rest of the app.

### Lifecycle

- Created lazily on first tray open.
- First show seeds the transcript from the daemon's current hot window
  (`jarvis.daemon.get_hot_window_messages()`), so a user who has been talking
  by voice sees their recent turns instead of a blank panel. Seeding runs once
  per daemon life: re-showing (from the tray or after a hide) never duplicates
  turns, and a daemon that comes back after a stop seeds its new memory on the
  next show. The hot-window content is already redacted (redaction runs before
  a turn is added to the dialogue memory), so seeding never leaks raw
  sensitive input. Only bundled mode has the memory in-process: in subprocess
  mode the accessor returns nothing and the window opens blank until the user
  types.
- Hidden windows stay responsive: the daemon-side callback still fires while
  hidden, so a reply that lands while the window is closed appears on next
  open, with the introductory panel gone and the transcript in its place.
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

## Streaming

The reply is displayed progressively while the model generates it. The path:

1. `OpenAICompatibleBackend.chat(..., on_token=...)` requests an SSE stream
   and forwards each content delta. Tool-call deltas are reassembled by
   `index` and the result is returned in the **same normalised shape as the
   buffered path**, so the reply engine's tool loop is identical either way.
   Without `on_token` the request is not streamed.
2. `run_reply_engine(..., on_token=...)` threads the callback to its
   generation turn.
3. `submit_text_query` forwards each delta as an `on_token` callback and a
   `{"type": "token"}` IPC event.
4. The dashboard fills a provisional bubble (blinking caret) from those
   deltas and discards it when `complete` arrives.

**`complete` remains authoritative.** Streaming is display only: a
tool-calling turn can emit text that is not part of the final answer, so
consumers must settle on the completed reply rather than on accumulated
tokens. Both the engine and the daemon fall back to the buffered path if the
callee predates the parameter (`TypeError`), and a raising `on_token` never
costs the reply.

### Preparation progress

Tokens only cover generation; several seconds pass before it starts (tool
routing, memory lookup, tool execution). `run_reply_engine(..., on_stage=...)`
announces each phase as `(stage_id, detail)`, forwarded as a `stage` IPC
event and rendered in the same pending bubble the tokens will later fill.

Stage ids are **neutral identifiers**, `routing`, `memory`, `planning`,
`tool`, `generating`, never display strings: the assistant is not tied to
one language, so wording belongs to the presenting layer (`_STAGE_VIEW` /
`stage_label` in the dashboard bridge, which falls back to a generic label
for ids it doesn't know). `detail` carries a bare noun, today the tool name.

A phase is announced only when it will actually do work: a hot-cache hit or
a disabled planner emits nothing, since a label that flashes for zero seconds
is noise rather than progress. Reporting is advisory and fail-open
throughout: a raising consumer is logged and ignored.

## What the system does not do

- **No per-tool-call visibility.** The `on_tool_call` callback and the
  `{"type": "tool"}` IPC event are declared but not emitted; they are
  reserved for future work.
- **No external integrations** (Slack, Telegram, Discord). Those would route
  through the same `submit_text_query` entry point but are not wired.
- **No text-input wake word.** Text is always "directed": there is no intent
  judge, no echo detection, no wake word. The user typing is the intent.
- **No TTS.** Text chat is silent. If the user wants spoken replies, they use
  the voice path.
- **No chat sessions.** The window shows one continuous conversation with the
  daemon's shared memory; there is no new-session / session-switching UI, and
  the daemon exposes no way to replace or archive the conversation.

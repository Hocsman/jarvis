# Latency journal

How long a spoken exchange takes, stage by stage, kept on the user's machine so that "is she faster?" can be answered from numbers and not from a feeling.

The journal records **durations and nothing else**. It never holds what was said, what was answered, what a tool returned, or any text at all. That is the property that lets it stay on by default, so it is enforced by the table's shape (no free-text column) and by a test, not by care at the call sites.

## What a turn is

A turn starts when the user stops speaking and ends when the reply text is complete. The clock zero is the end of the **last utterance heard before the query was dispatched**: when the user speaks twice inside the collection window, the second utterance is the one measured, because the silence after it is what the user waited through.

When the end of speech is not known (a query that did not come from a heard utterance, or one whose last heard utterance is more than five minutes old), the clock starts at the dispatch and `transcribed_ms`, `judge_ms`, `judge_outcome` and `dispatch_ms` are empty.

Only voice turns are recorded. A text chat turn has no end of speech and no first sound, so it has nothing to measure that the chat window does not already show.

## The table

`latency_log`, in the assistant's database next to `action_log`. One row per dispatched voice query.

| Column | Meaning |
|---|---|
| `ts_utc` | When the user stopped speaking (clock zero). |
| `origin` | `voice`. |
| `transcribed_ms` | End of speech to transcript ready (speech recognition). |
| `judge_ms` | How long the intent judge call took. `NULL` when it was not asked. |
| `judge_outcome` | `directed`, `not_directed` or `no_verdict` (the judge could not answer). `NULL` when it was not asked. |
| `dispatch_ms` | End of speech to the query being handed to the reply engine: recognition, the judge, and the wait for the user to carry on. |
| `route_ms`, `plan_ms`, `memory_ms`, `tools_ms`, `generate_ms` | Time the reply engine spent in each stage **before the first sound**; all of it when nothing was spoken. `tools_ms` adds every tool stage of the turn. |
| `first_sound_ms` | End of speech to the first audible sound. `NULL` when nothing was spoken. This is the number the user feels. |
| `total_ms` | End of speech to the reply text complete. |
| `router_degraded` | `1` when the tool router gave up and the keyword strategy answered in its place (see `tools/selection.spec.md`). |
| `tool_calls` | How many tool stages the turn went through. |
| `judge_model`, `router_model`, `chat_model` | The model identifiers in use, so a run on a cloud model and a run on a local one can be told apart. |

A free-text column is never added. The only text columns hold an enumerated outcome or a model identifier.

## Who marks what

A single process-wide journal, `get_latency_journal()`, is configured once by the listener with the database and the settings. Every mark is a cheap no-op when the journal is off or no turn is open, and no mark ever raises.

- **Listener, on a transcript:** `note_heard(speech_end)` records the end of the utterance and when its transcript became ready.
- **Listener, around the intent judge:** `note_judge(seconds, outcome)`.
- **Listener, on dispatch:** `begin_turn(models)` opens the turn from those notes; the reply engine's `on_stage` callback is handed to the journal; `end_turn()` runs when the engine returns.
- **Reply engine:** calls `note_router_degraded()` when the router gave up this turn.
- **Speech engines:** call `mark_first_sound()` once their audio stream is running, outside the audio locks so a journal write can never hold playback up. This is not where `SPEAKING` is published: that happens when a chunk begins to be synthesised, before anything is audible, and marking there would understate the wait by the synthesis time of the first chunk. Only the first call of a turn counts.

The stage clock is the engine's own announcements (`routing`, `planning`, `memory`, `tool`, `generating`): a stage lasts until the next one is announced. A stage the engine skips (a router cache hit, for example) simply has no time.

## The first sound

With streaming, the first sound arrives while the engine is still generating, so the stage columns are frozen at that moment. Without streaming, the reply is spoken after the engine returns, so the row is written first with `first_sound_ms` empty and filled in by the first sound that follows within 30 seconds. A sound that arrives later than that, or after the row already holds one, belongs to something else (a reminder, a routine) and is ignored.

## Failure

The journal is a side channel. A database that cannot be written, a mark out of order or a missing turn costs the row and nothing else: the voice pipeline never waits on it, never raises because of it, and logs the reason with `debug_log`.

## Settings

- `latency_journal_enabled` (bool, default `true`): turns recording off entirely. Nothing is written and nothing is held in memory beyond the current turn.
- `latency_journal_days` (int, default `30`, clamped to 1 to 365): rows older than this are dropped when the daemon starts.

The user can erase the journal with `Database.clear_latency()`; nothing else reads it back into a prompt.

## Reading it

`scripts/latency_report.py [--days N]` opens the database read-only and prints, for the voice turns of the last N days (default 7), the count, the median, 90th percentile and maximum of each stage and of `first_sound_ms`, a line per day, and how often the router gave up and the judge said what. It prints durations and counts, never text.

## Out of scope

Text chat turns, a viewer tab, counting utterances that never reached the judge (ambient speech, echo), and alerting. Each can be added without changing the table's rule: durations and enumerations only.

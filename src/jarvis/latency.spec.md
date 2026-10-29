# Latency journal

How long a spoken exchange takes, stage by stage, kept on the user's machine so that "is she faster?" can be answered from numbers and not from a feeling.

The journal records **durations and enumerations and nothing else**. It never holds what was said, what was answered, what a tool returned, or any text at all. That is the property that lets it stay on by default, so it is enforced where the data is written and not left to care at the call sites: the table has no free-text column, its checks refuse any text that is not an enumerated outcome or a short model identifier, the marks never read the text they are handed, and tests pin all three.

## What a turn is

A turn starts when the user stops speaking and ends when the reply text is complete. The clock zero is the end of the **last utterance heard before the query was dispatched**: when the user speaks twice inside the collection window, the second utterance is the one measured, because the silence after it is what the user waited through.

The voice detector closes an utterance only after `endpoint_silence_ms` of silence, so the listener passes the journal the moment it closed less the silence it counted, which is when the user stopped speaking. A closure forced by the maximum utterance length has no such silence and is measured from the closure.

When the end of speech is not known (a query that did not come from a heard utterance, or one whose last heard utterance is more than five minutes old), the clock starts at the dispatch and `transcribed_ms`, `judge_ms`, `judge_outcome` and `dispatch_ms` are empty.

Only voice turns are recorded: the ones that go through the listener's dispatch. A text chat turn has no end of speech and no first sound. A reminder, a routine, a click-approved narration or a spoken refusal that never reaches the dispatch is not a turn, and the sound it makes is ignored.

## The table

`latency_log`, in the assistant's database next to `action_log`. One row per dispatched voice query.

| Column | Meaning |
|---|---|
| `ts_utc` | The end of the user's speech (clock zero). |
| `origin` | `voice`. |
| `transcribed_ms` | End of speech to transcript ready: the endpoint silence and speech recognition. |
| `judge_ms` | How long the intent judge call took. `NULL` when it was not asked. |
| `judge_outcome` | `directed`, `not_directed` or `no_verdict` (the judge could not answer). `NULL` when it was not asked. |
| `dispatch_ms` | End of speech to the query being handed to the reply engine: recognition, the judge, and the wait for the user to carry on. |
| `prepare_ms` | Between the dispatch and the engine's first stage announcement: redaction, the MCP refresh on a new conversation, the approval judge when a spoken answer is pending, and the wait for the query lock. A turn the engine ends before announcing anything (a spoken refusal) is all preparation. |
| `route_ms`, `plan_ms`, `memory_ms`, `tools_ms`, `generate_ms` | Time the reply engine spent in each announced stage **before the first sound**; all of it when nothing was spoken. `tools_ms` adds every tool stage that ran before the first sound. |
| `first_sound_ms` | End of speech to the first audible sound. `NULL` when nothing was spoken. This is the number the user feels. |
| `total_ms` | End of speech to the reply text complete. |
| `router_degraded` | `1` when the tool router gave up and the keyword strategy answered in its place (see `tools/selection.spec.md`). |
| `tool_calls` | How many tool stages the turn went through. |
| `judge_model`, `router_model`, `chat_model` | The model identifiers in use, so a run on a cloud model and a run on a local one can be told apart. They are configuration values, not utterances, and are capped at 120 characters. |

The table refuses what it was not built for: `origin` must be `voice`, `judge_outcome` one of the three outcomes, and a model identifier at most 120 characters, whoever calls `record_latency`. A free-text column is never added.

`generate_ms` has two meanings that depend on the path, because it stops at the first sound. With streaming the first sound arrives while the engine is still generating, so it includes the synthesis of the first chunk. When the reply is spoken after the engine returns (the usual case for a one-sentence reply, see below), it ends when the engine returns and the synthesis shows up as the gap between `total_ms` and `first_sound_ms`, which the report prints as "speech start".

## Who marks what

A single process-wide journal, `get_latency_journal()`, is configured by each listener when it is built, with the database and the settings. Every mark is a cheap no-op when the journal is off or no turn is open, and no mark ever raises: each one is guarded where it is defined, so a failing clock or database costs the row and nothing else.

- **Listener, on a transcript:** `note_heard(speech_end)` records the end of the utterance and when its transcript became ready.
- **Listener, around the intent judge:** `note_judge(seconds, outcome)`.
- **Listener, on dispatch:** `begin_turn(models)` opens the turn from those notes, and the reply engine's `on_stage` callback is handed to the journal. When the engine returns, `finish_turn(will_speak)` freezes the numbers with no I/O; the row is written by `flush()` once the reply has been queued for speech, so the write never sits between "reply ready" and "start speaking". `end_turn()` does both for a caller with nothing to do in between.
- **Reply engine:** announces its stages (`routing`, `planning`, `memory`, `tool`, `generating`), announces `generating` again before a model call that follows a tool result so that call is not booked to the tool, and calls `note_router_degraded()` when the router gave up this turn.
- **Speech engines:** call `mark_first_sound()` once their audio stream is running, outside the audio locks; for a Kyutai item drawn on the talking face, once it is heard, so the first sound then includes the face's delay (`output/tts.spec.md`, "Kyutai"). This is not where `SPEAKING` is published: that happens when a chunk begins to be synthesised, before anything is audible, and marking there would understate the wait by the synthesis time of the first chunk. Only the first call of a turn counts.

The stage clock is the engine's own announcements: a stage lasts until the next one is announced, and the time before the first announcement is `prepare_ms`. A stage the engine skips (a router cache hit, for example) simply has no time.

## The first sound

A reply is streamed to speech sentence by sentence, but a sentence closes only when something follows its punctuation, so the last sentence of a reply, and the only one of a short reply, is spoken as the tail after the engine returns (`output/streaming.spec.md`). The first sound therefore usually arrives after the turn is frozen.

- If it arrives before the turn is frozen (a longer, streamed reply), the stage columns are frozen at that moment.
- If it arrives after, it completes the row: from memory when the row has not been written yet, by an update when it has, within 30 seconds.
- Only a turn whose reply is going to be spoken accepts a late sound (`will_speak`). A dismissal, an empty reply or a failed turn does not, so a reminder or the error apology cannot become its first sound; and each turn's freezing replaces whatever claim an earlier row had, so a later reply's sound never completes an earlier row.
- A sound that arrives later than 30 seconds, or after the row already holds one, belongs to something else and is ignored.

## Failure

The journal is a side channel. A database that cannot be written, a mark out of order, a clock that steps back (every duration is clamped at zero) or a missing turn costs the row and nothing else: the voice pipeline never raises because of it, and it logs the reason with `debug_log`. The row is written from the thread that finishes the turn, after the reply is queued; a sound that completes a written row updates it from the speech thread, which is a single short update.

## Known limits

- A transcript that returns early while a query is being collected (the assistant's own echo, an utterance the judge rejects) also moves the recorded end of speech and drops the judge note, so that row's early columns describe the last utterance heard, not the query.
- A query dispatched while the previous reply is still playing (barging in with the wake word) can take the previous reply's next chunk for its first sound.
- Time is the wall clock the listener already uses; a clock set back during a turn gives zero, not a negative number.
- The retention is applied when the daemon starts, so a daemon that stays up for weeks keeps more than `latency_journal_days` until its next start.

## Settings

- `latency_journal_enabled` (bool, default `true`): turns recording off. Nothing is written and no turn is held. A flag hand-written as text (`"false"`, `"off"`, `"0"`, `"no"`) reads as off. When the daemon starts with the journal off, it erases what the journal had kept.
- `latency_journal_days` (int, default `30`, clamped to 1 to 365): rows older than this are dropped when the daemon starts.

The user can erase the journal at any time with `python scripts/latency_report.py --clear`, which touches nothing else in the database. Nothing reads the journal back into a prompt, and nothing but the report reads it at all.

## Reading it

`scripts/latency_report.py [--days N] [--db PATH]` opens the database read-only, in place, and prints, for the voice turns of the last N days (default 7), the count, the median, 90th percentile and maximum of each stage and of `first_sound_ms`, a line per day, and how often the router gave up and what the judge said. It reads only the journal's columns, so nothing else of the database is copied or read, and it works while the assistant is running. It prints durations and counts, never text. It runs from a source checkout with the project's environment active; the packaged app has no viewer for the journal yet.

## Out of scope

Text chat turns, a viewer tab, counting utterances that never reached the judge (ambient speech, echo), and alerting. Each can be added without changing the table's rule: durations and enumerations only.

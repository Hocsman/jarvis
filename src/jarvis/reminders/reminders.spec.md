# ⏰ Reminders — the first thing Yuba does while nobody is watching

Everything else in the assistant assumes a turn: the user says something, she answers. A reminder has no turn. It is set now, kept across restarts, and spoken later whether or not anyone asked anything.

A reminder is a **promise**, and that single word decides most of what follows. A promise silently broken costs more trust than the feature earns, so every choice here prefers late, clumsy or repeated over silent.

Implemented in [extract.py](extract.py) (reading a time), [scheduler.py](scheduler.py) (keeping it), [set_reminder.py](../tools/builtin/set_reminder.py) (the tool), and the `rappels` table in [db.py](../memory/db.py). Surfaced in the memory viewer's **⏰ Rappels** tab.

## Where a reminder lives

A row in the database, at schema version 3. **This is the one part of the assistant that must reach disk.** A pending confirmation deliberately never does — an approval that outlives the process was given without the context that produced it — and a reminder is the exact opposite: if a restart loses it, it was never a reminder.

The core's own files were the obvious alternative and were rejected on a concrete ground: the core is only safe because no background thread writes to it. `remember` reads and writes without a lock, and the memory viewer PUTs the whole file from a different process. A reminder requires a write on a clock, and a file would concede a window where a hand-typed reminder is silently lost — the exact failure this exists to prevent.

`kind` and `payload` ship with defaults, so recurring routines — the next thing built on this — need no `ALTER`. No migration code was written: every statement in the schema script is `CREATE TABLE IF NOT EXISTS` and the script runs on every open, so the table reaches an existing install by the same path as a new one.

`due_utc` fires it. `due_local` and `tz` are what she reads back, stored alongside so a timezone change between setting and firing cannot silently rewrite either.

**`texte` is not redacted**, unlike everything else in this project. It is read aloud, and redacting it would have her say a placeholder to the user's face. `query` is redacted, like the ledger's, because it is bookkeeping rather than speech.

Pruning drops settled and cancelled rows after 90 days and **never touches anything still owed, however old**. A reminder set for next year is still a promise.

## Reading a time

LLM context #16, in `extract_reminder_time`. A model reads the sentence, because matching on "demain" or "jeudi" would make this a French feature.

**The model computes nothing the caller can compute.** "dans vingt minutes" comes back as `{"minutes": 20}`, never a timestamp — a small model that cannot add twenty minutes to 12:47 can still copy the number 20. The only leap left is the one no code can make without a word list: from a named day to a date.

**An omitted field is the statement.** A `date` with no `time` means a day was named and no hour, so the caller applies `reminder_default_hour` and says so aloud — "jeudi à neuf heures" is only honest if the nine came from somewhere the user can hear.

The prompt names no day, no month, no temporal adverb, in any language, and carries no worked example: an example containing "tomorrow" is a hardcoded language pattern smuggled in by demonstration, and it would make the model generalise worse to Turkish than no example at all.

Guards sit on the parsed object rather than being phrased as rules in the prompt, because a rule in a prompt is a request and a guard is a fact. Unknown kind, unparseable date or time, an instant in the past, one beyond 400 days, an empty subject, a timeout, an exception — each raises with its reason, and the reason is spoken. A redaction placeholder is refused, which bites harder here than on the core: a placeholder stored as a fact is merely useless, one in a reminder is read aloud twenty minutes later.

## Setting one

`setReminder`, named at a distance from `remember` on purpose. The defect this closes is the confusion between them, and `remindMe` would share a prefix with the tool whose traffic it takes over, in a catalogue the router matches on.

One optional property, `rappel`. That is what makes the planner's fast path work: `setReminder rappel='…'` is a concrete `key='value'`, so the step dispatches without the resolver — which would otherwise be asked to hallucinate a timestamp, the exact failure this design avoids. Missing, it falls back to what the user just said.

`risk_for` returns `lecture`, like `remember`: it writes only to Yuba's own store, never the user's machine and never anything outward. Asking permission to write down what she was just asked to write down is noise.

**The invariant: she never claims a reminder she has not read back.** The tool writes the row, reads it back from the database, re-parses it, and speaks that. The sentence the user hears is derived from the artefact that will actually fire — not from the model's JSON, and not from what the tool believed a moment ago. A round trip that cannot be found or cannot be parsed **withdraws the row**, not merely the claim: a reminder going off after she said it was not set is worse than no reminder at all.

Saying the time back is also the whole disambiguation strategy. A model that read "jeudi" as Tuesday is caught by the user's ear, in the same turn, in any language, with no word list — the same move as the confirmation card, where the words the user checks against are written by code.

Failure is **noisy and immediate**, which is neither of the usual two directions. The approval judge fails closed because one extra "no" costs a turn; here both silent directions break a promise, and the user is standing there having spoken two seconds ago, so a clarifying sentence costs them nothing.

## Keeping it

A dedicated thread, ticking on `reminder_tick_sec`, woken early when something closer than a tick is set.

**Wall clock, not monotonic — the opposite of the confirmation TTL, for the mirror reason.** `PendingAction.has_expired` uses `time.monotonic()` because a deadline that can move backwards is a resurrectable approval. A confirmation TTL is an attention span; a reminder is an appointment with the world. Polling against the wall clock is correct across a sleep by construction: the comparison happens at tick time, so a laptop shut from 09:00 to 14:00 finds the row due the moment it wakes.

`advance_rappel` moves a recurring row to its next occurrence, and refuses a due time that is not strictly ahead of now. The guard sits beside the `prévu` one: that keeps a cancelled row cancelled, this keeps a moved row moved. A row placed on an instant already past is owed again on the very next tick and on every one after it, which is a loop rather than a schedule.

Its own thread rather than the daemon's main loop, which calls the diary pass synchronously and can block for up to 45 seconds — a 09:00 reminder would land anywhere inside that, and never during shutdown. It is stopped and joined **before the listener dies**, because the diary pass takes another 45 seconds after that and the database outlives both: a reminder firing in that window would be settled as said with nothing able to say it.

**Defer, never drop.** Past `reminder_late_grace_sec` she still says it — she says how late she is. Silently discarding something owed since Thursday would leave one ledger line in a tab nobody has a reason to open.

**Delivery settles it, never queueing.** The row stays owed until the speech has finished, so a crash between the two costs a repeat rather than a broken promise. When nothing can speak at all — `enqueue_reply` returning False is definitive — the row is closed loudly, with a ledger row and an announcement, rather than retried every tick forever or left promising something impossible. `reminder_max_attempts` bounds the rest.

**Three due at once become one utterance.** The speaker holds a single completion callback, and a second call destroys the first's. Nothing is said while a query is in flight: cutting across a reply the user is waiting for is worse than a few seconds late, and the query lock is only *inspected*, never taken, so it keeps the two meanings it already has.

## Seeing and cancelling

The **⏰ Rappels** tab, which is the price of the database. Every other artefact the user owns here is a text file they open and correct by hand; a scheduled thing you cannot see or call off is a thing you stop creating.

Cancelling reports honestly: an unknown id comes back `cancelled: false` rather than a cheerful success, because claiming a reminder was called off when it will still fire is the worst answer available.

Typing one in **reaches no model at all**. A date, a time and a sentence are already unambiguous, and sending the user's own typed request through the extractor would only add a way to be wrong. Those are marked `fichier`, so the ledger can tell them from one she was asked for aloud.

## The ledger

One row per firing, with `origin` set to `rappel` and `request_id` set to the reminder's id — the column `origin` was built for exactly this. `ok` when it was said, `échec` when nothing could say it. As everywhere else, the ledger records what was done and never what was seen.

## Settings

| Key | Default | What it decides |
|-----|---------|-----------------|
| `reminders_enabled` | `True` | Whether the thread runs at all |
| `reminder_model` | `""` | Who reads the time. Empty = the warm small chain |
| `reminder_timeout_sec` | `8.0` | How long that read may take |
| `reminder_default_hour` | `9` | Where a named day with no hour lands |
| `reminder_tick_sec` | `5.0` | How often the clock is checked |
| `reminder_late_grace_sec` | `900.0` | Past this she says how late she is |
| `reminder_max_attempts` | `60` | When a failing delivery stops being retried |

`reminder_model` is deliberately **not** passed through `_cloud_safe_model`, unlike the tool router and the intent judge. That filter rewrites a pinned local tag to the chat model when the configured endpoint is remote, and announces it, and this prompt carries the user's own sentence about their own life — pinning a local model is the only way to keep it off the network, so rescuing it would silently undo the one thing the setting is for.

**And the pin decides the endpoint, not just the name.** `get_llm_backend` chooses the destination from `llm_provider` alone and never looks at the model, so until `get_private_backend` existed a pinned local tag was sent *to the cloud* — verified, the request reached `https://openrouter.ai/api/v1` carrying his sentence, and the tag would have been rejected there anyway. The setting changed a name and nothing else while its documentation promised privacy. With nothing pinned there is nothing to honour and the ordinary provider applies: a user who never asked for this is not quietly moved off the one he chose. The same helper serves the routine extractor, the goal judge and the journal reader, which make the same promise for the same reason.

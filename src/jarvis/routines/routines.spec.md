# Routines

What Yuba does on her own, at a fixed hour, with nobody in the room.

A routine is a sentence the user said once ("tous les matins à 7h,
résume-moi mes mails") turned into three things: a recurrence rule, an
envelope of tools, and a row in `rappels` that says when it next fires.
Every one of them is readable and correctable without a database client.

The governing asymmetry: an attended turn can be wrong cheaply, because
the user is right there and says so. An unattended one cannot. Every
default here therefore leans shut, and the interesting design work is in
what a routine is *not* allowed to do.

## The rule

`recurrence.py`.

`Regle(kind, hour, minute, weekday)` where `kind` is `daily` or `weekly`.
There is no shape finer than a day: a routine that fires on a tick
empties a rate limit and a wallet overnight, so the vocabulary simply
cannot express it.

`next_occurrence(regle, apres)` is calendar arithmetic in the machine's
local zone, never `apres + 86400`. Adding a fixed number of seconds
drifts an hour twice a year, and a 07:00 routine that starts arriving at
06:00 in November looks like a bug in the alarm rather than in the maths.
When a wall-clock time does not exist on a given date — the spring-
forward hour — `_real_instant` nudges forward to the first instant that
does, rather than raising or silently skipping the day.

`should_run(regle, prevu, maintenant)` returns `RUN` or `SKIP`. A run
that is late by less than `staleness_window` still runs; later than that
and it is skipped, because a daily digest delivered eleven hours late is
not the thing the user asked for. The window is
`min(cap, period)` — bounded by the period, so a daily routine can never
have a staleness window that reaches into tomorrow's occurrence.

## The envelope

`scope.py`, and the file at `yuba/routines.md`.

One block per routine: `phrase`, `quand`, an `outils` list, and an
optional `mémoire`. The tool list is the routine's whole world.

```markdown
## matin
phrase: résume-moi mes mails
quand: tous les jours à 07:00
outils:
- webSearch
- fetchWebPage
```

Editing that file is the control surface. Remove a tool line and the
routine is tightened; delete the block and the routine is suspended.
`load_routines` caches on the file's mtime and re-reads when it changes,
so an edit takes effect on the next run rather than the next restart — a
control surface with a restart delay is one people stop trusting.

Reading the file never raises. A file that cannot be decoded, or that has
a syntax error halfway down, yields the blocks that parse and nothing
else. Raising would stop every routine at once over one bad line, and
falling open would be worse.

Defaults, all of them shut:

| Written in the file | What it means |
|---|---|
| an empty `outils` list | nothing, never "all" |
| no `outils` section | nothing |
| no block for a routine name | suspended, never unrestricted |
| an unreadable file | every routine suspended |
| `mémoire` absent or anything but `oui` | the user's profile stays home |

There are no wildcards, unlike `outils.md`. That file has them because
the user is present to see the result; here nobody is, and an MCP server
that gains a tool overnight would gain it inside the envelope too.

Three names no envelope can hold, whatever the file says
(`JAMAIS_EN_ROUTINE`):

- `toolSearchTool` appends any name in the registry to the running turn's
  allow-list and regenerates the schema. An envelope that can widen
  itself is not an envelope.
- `refreshMCPTools` rediscovers servers mid-run, so the catalogue would
  change underneath the envelope.
- `stop` ends a conversation, and a routine is not one.

## Reading a spoken schedule

`extract.py`, LLM context #17.

The same discipline as the reminder extractor and for the same reason: a
word list would make this a French feature. A model reads the sentence
and returns the smallest thing it can — a kind, an hour, a weekday — and
the code does the rest. It never returns a cron expression, which is a
thing a model can be subtly wrong about in a way nobody notices until a
routine fires at 3am on the 31st.

The weekday convention is stated to the model explicitly. `0 = Monday` is
not universal, and leaving it implicit is how a Monday routine runs on
Sunday.

The utterance is fenced as data. It arrives through Whisper and may carry
anything at all, and unlike a one-off reminder this sentence is replayed
to a model every single morning — which is also why a redaction
placeholder in it is refused outright rather than passed along.

It rides the reminder model chain, including the pin: one privacy
decision rather than two things to get wrong.

Failure is `ExtractionFailed`, never a guess. A routine placed at a
moment nobody meant runs at that moment every day until someone notices.

## The catalogue a routine's turn is offered

`run_reply_engine(..., scope=...)`.

With a `scope`, the envelope **is** the catalogue: the tools it names
that actually exist, and nothing else. The tool router is not consulted —
it exists to narrow forty tools for a small model, and an envelope is
already a handful of names the user wrote by hand, so narrowing it
further would spend an LLM call to maybe drop the tool the routine needs,
at 7am, with nobody watching.

One filter runs last, after every branch that can add a name, so nothing
survives outside the envelope. `stop` and `toolSearchTool` are not
appended, and `RoutineScope.allows` would refuse them anyway.

The warm profile block is withheld unless the routine's block says
`mémoire: oui`. Every line of it is a line of the user's private life
leaving the machine while they sleep, and a routine does not need to know
who they are to summarise their mail.

An attended turn — no `scope` — is untouched by all of this.

## The gate, re-run every morning

`registry.py::_out_of_scope`, at the single tool funnel.

The envelope is decided once, when the routine is created. Four checks
run again on every call, because what could have changed in between is
exactly what matters:

1. the name is in this routine's envelope (and not one of the three that
   no envelope can hold);
2. the user's own `outils.md` still says `libre` for it — the envelope
   was written in July, the file may have been edited in October, and the
   newer decision wins;
3. `resolve_risk` says `lecture` **for this morning's arguments** —
   `localFiles` reading yesterday and deleting today is one name the
   envelope cannot tell apart;
4. the tool does not write Yuba's own state (`writes_own_state`). Its
   risk is `lecture` and that is correct for an attended turn, where a
   wrong entry is correctable in the next breath. At 07:00 nobody
   reopens anything.

A routine can therefore only ever read, and only inside its envelope.

`jamais` outranks all of it and is checked first: a tool the user retired
is reported as refused, not as out-of-scope, because those are different
facts and the user's own retirement is the stronger one.

Every refusal is written to the action ledger with outcome
`hors-périmètre`, so the morning after is answerable.

The refusal text deliberately does **not** point at `## Libre` the way
the attended one does. Unattended, that becomes a paragraph recommending
the policy be loosened, written by a thread running while the user
sleeps.

## The runner

`runner.py`.

Takes one due `routine` row and turns it into a turn of the reply engine.
Almost everything worth stating about it is a thing it refuses to touch.

**It never takes the query lock.** Holding it would mean a routine that
reached a slow server at 07:00 leaves Yuba unresponsive until it
finishes: the user talks and nothing happens. The dispatcher *asks*
whether a query is running and defers; the runner holds nothing the user
needs.

**It never uses the shared dialogue memory.** Each run builds its own.
A routine is not a conversation: its turn must not land in the user's
history, must not move the hot window, and must not disturb a
confirmation card already waiting there. Going to bed with a question
pending and waking to find it silently expired is the worst thing this
feature could do to anyone.

**It advances the row before the work**, which is the exact opposite of
the reminder scheduler. A reminder that fails is still owed and stays
owed. A routine that takes the process down with it and is still owed
comes back on the next tick, does it again, and does that forever. Here a
crash costs one morning.

**It runs one at a time.** A submission while a run is in flight returns
`False` and the row keeps its due time, so the dispatcher finds it again
on the next tick. Refusing is not dropping. Two routines at 07:00 sharing
one small model, one rate limit and one machine is not twice the work; it
is two slower runs and a good chance neither finishes. The slot is
released even when a run raises, or one bad morning would suspend every
routine until the next restart.

A routine whose block has gone is suspended: nothing runs, and the
journal says so rather than staying silent, because silence reads as "it
never fired" and sends the user to look at the schedule instead of at the
block they deleted.

Each run is bracketed in the action ledger under `routine:<nom>` — the
tab lists tool calls and a run is not one, so a bare `matin` would invent
a tool nobody can look up. The opening `démarré` row exists to be found
*unclosed*: a run that took the process down with it leaves that row and
nothing after it, which is the only trace of it that would exist
anywhere.

What the run reached for is read back out of the ledger rather than
reported by the engine, which hands back text and nothing else. That also
means the journal cannot claim a call the ledger never recorded.

`payload["steriles"]` counts consecutive mornings that produced nothing
at all. Not for its own sake: a routine that has failed every day for a
week is broken and something has to notice. A quiet morning is not one of
these — "rien à signaler" is the routine working, and counting it would
switch off exactly the ones doing their job.

## The journal

`journal.py`, and the folder at `yuba/journal/`.

One Markdown page per day, appended. `2026-08-02.md`, so a directory
listing sorts the mornings and the whole history opens in anything.

This is the delivery, not a log of it. There is no speech and no chat
bubble at 07:00, so the write-up itself goes in the page — which is the
one place this diverges from the action ledger, which records what was
*done* and never what was *seen*. The ledger is an audit trail; this is
the letter on the kitchen table. It never leaves the machine, sitting in
the same directory as the user's own profile, so there is nothing here to
redact.

Each entry carries the hour it actually ran (not the hour it was due — a
run deferred two hours because the machine was asleep is a different
fact), what was asked for, the write-up, the tools it reached for, and
anything the gate turned away with the reason. A run that produced
nothing still leaves a line: a silent gap reads as "it never fired",
which sends the user to look at the schedule instead of at the error.

Nothing in it raises. By the time it runs the work is done, and losing
the letter is bad while losing the letter *and* taking the runner down
with it is worse.

`prune_journal` drops pages past 90 days, from the same sweep that prunes
settled reminders — one retention answer for everything Yuba writes down
about herself. It only touches files whose names *are* dates and parse as
one: the folder is in the user's own directory, and a sweep that eats
their notes is a sweep that gets the feature turned off.

## Files

| Path | What it is |
|---|---|
| `yuba/routines.md` | the envelopes, generated once then owned by the user |
| `yuba/journal/AAAA-MM-JJ.md` | one page per day, 90-day window |
| `rappels` rows with `kind='routine'` | when each one next fires |

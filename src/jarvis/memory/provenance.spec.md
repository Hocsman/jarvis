# Provenance of World Facts

## Why this exists

The core files are strict about where a belief came from. `profil.md` and
`regles.md` are written only when the user says something or corrects
something, every line carries its source (`dit` / `corrigé` / `migré` /
`confirmé`), and a hand edit wins. `appris.md` goes further: a proposal
becomes a belief when he ticks it, and by no other route.

The `world` branch of the graph had none of that, and it is the branch
fed by content the assistant did not write and cannot vouch for.

The gap is not that the extractor is careless. Its prompt already bans
user facts, assistant recommendations and transient snapshots. The gap is
that it is asked to *guess* provenance from prose:

> Heuristic: would a different assistant on a different day produce the
> same answer? If yes, it's a lookup → extract.

That heuristic is a proxy for "was this looked up", applied to a summary
where the real answer has already been thrown away. A confident invented
statistic has exactly the shape of a successful lookup and passes it.

Observed on 2026-08-16: after reading one web page ranking models on
cybersecurity, the assistant restated a benchmark figure two turns later
with no tool call at all, added an editorial verdict of its own, and a
claim of the same family reached the graph as a bare fact.

## The principle

**Provenance is a property of the transcript, not a judgement about the
prose.** The system already knows which tools ran and what they returned.
That knowledge is carried to the point of extraction instead of being
inferred there.

Everything below follows from that one sentence.

## What the window knows

`DialogueMemory` keeps `_tool_turns` — a timestamped list of each reply's
tool-related messages — deliberately excluded from `get_pending_chunks`
so raw tool payloads never reach the summariser. That exclusion stays:
payloads are noisy and are the injection surface the diary's own fence
exists for.

What crosses is not the payload but the fact of the call. The pending
chunks come with a snapshot timestamp; intersecting that window with
`_tool_turns` yields the set of tool names that ran while those chunks
were being produced. No LLM, no heuristic, one comparison of timestamps.

It is read in the same breath as the chunks and before the summary's own
LLM call marks them saved. Afterwards the window is gone.

`_tool_turns` is capped at sixteen entries and the engine clears it on
new-conversation entry, so a window whose calls have been evicted reports
none and its facts are not extracted. That loses a real fact rather than
inventing one, which is the direction this has to fail in.

## The rule that does the work

**A window in which nothing looked anything up contains no lookups, so it
yields no world facts.** Extraction is skipped before the LLM call, not
filtered after it.

Running a tool is not the same as having consulted something. `stop` and
`toolSearchTool` steer; `remember`, `forget`, `logMeal`, `setReminder`
and the goal tools write; `fetchMeals`, `listGoals` and `reviewLearnings`
read his own records, which are not the world. Only the rest counts, and
only the rest decides the source label — a window that merely wrote a
reminder looked nothing up.

Observed 2026-08-17: a turn whose only tool was `toolSearchTool` — which
returns a list of tool *names* — satisfied the gate as first written.
Nothing reached the graph that day only because the extractor refused the
content for an unrelated reason. Counting "a tool ran" as "something was
learned" is the error this file exists to prevent, made one level in.

An unrecognised name counts as a lookup. Every MCP server's tools land
there, one usually does consult something, and dropping them all would
lose real facts to guard against a mislabelled source.

This is the whole of the defence against the observed failure, and it
costs nothing: the assistant answering from its own priors or from what
it said three turns ago is exactly the case where no tool ran.

It is deliberately blunt. A window with one weather call and one
confabulated claim still passes the gate, and the per-fact source below
is what carries the distinction there. The rule removes the class of
failure where *nothing at all* was consulted; it does not pretend to
grade truth.

## The source vocabulary

One word per fact, recorded at write time, never inferred later:

| Source | Meaning |
|---|---|
| `web` | A page fetched by `webSearch` or `fetchWebPage` ran in the window. Untrusted by construction. |
| `outil` | Some other tool ran: weather, an MCP server, a builtin. Trusted as far as that tool is. |
| `inconnu` | Written before this spec, or by a path that could not establish a source. |

`inconnu` is what a line with no suffix already means, so facts written
before this need no marking and no rewriting: guessing their source now
would be the same mistake at a different moment.

There is deliberately no `modèle` value. A fact the model produced with
no tool behind it does not get a weaker label — it does not get written.

## Where the source is written

On the line, exactly as the core files do it. `profil.md` writes
`- il habite à Genève · dit`; a world fact writes:

```
Le DGX Spark a 128 Go de mémoire unifiée · web · 2026-08-16
```

A fact is a line inside a node's `data` blob, not a row, and one node
accumulates facts from many windows on many days. A column on the node
would therefore describe the node and not the fact, which is the wrong
granularity. Putting it on the line is also the idiom the reader already
knows from the core, so there is one convention in the project rather
than two.

It follows that there is no migration and no schema change. A line
written before this has no suffix, and a line with no suffix *is*
`inconnu` — not by a rule that marks it, but by construction. Nothing is
rewritten, nothing is guessed, and an existing graph cannot be damaged by
a migration that goes wrong.

The suffix is recognised only when it matches the vocabulary and an ISO
date exactly (`· web|outil|inconnu · YYYY-MM-DD` at end of line). A fact
whose own text happens to contain a middle dot is untouched.

**Dedupe compares the fact, not the line.** The daily summary is
cumulative and re-seeds the same facts on every flush, so
`node_contains_fact` matching whole lines would stop recognising a fact
re-extracted on a later date and append it again, once per flush, for
ever. It compares the fact part.

The merge step matches on the fact part for the same reason. It rewrites
a whole node through an LLM, which will not reproduce a suffix verbatim,
so keying `incorporated_indices` on whole lines would report every
genuinely merged fact as consolidated out: stored but never announced,
which is the failure this work exists to remove.

**The merge is handed bare facts, and the suffixes are put back after
it.** Asking an LLM to reproduce an exact format is asking for the class
of failure this file is about, and it does not: measured in the field,
the first fact written into a populated node came back with no marker at
all. Since a populated node always takes the merge path, that is the
normal case and not the edge — left alone, a suffix would almost never
survive.

So the repair is deterministic. Before the rewrite, every claim in play
is indexed by its text: the lines the node already carried keep whatever
they had, and this flush's facts take the window's source. Any line the
rewrite hands back matching one of those keys gets its suffix again.
Wording is never touched.

A line the rewrite reworded matches nothing and stays bare, which reads
as `inconnu` — the truthful answer, since after a rewrite nobody can say
what that sentence rested on.

## What changes at the far end

A stored fact stops being a bare sentence. What is displayed, and what is
injected into a later system prompt, says what it rests on.

The envelope around injected graph results claims no provenance of its
own. It used to open with "Things you looked up in earlier
conversations", which is a claim made over every line at once, including
the ones nobody recorded a source for.

Instead it says what the block *is* — kept from earlier conversations,
describing the world and not the user, losing to the core on any
conflict — and then explains the markers the lines carry. `web` is named
as read on a page on a date and possibly wrong; `inconnu`, or no marker,
is named as a line the model must not present as something it
established.

Only the markers actually present are explained, so a block of
tool-sourced lines does not spend prompt on `web`.

The reader — the model in a later turn, or the user in the memory
viewer — can then tell "DGX Spark has 128 GB unified memory · outil"
from "Mythos 5 scores 78% on ExploitBench · web" without having to
already know which is which.

## What this does not do

It does not judge truth. A wrong fact from a real page is still stored,
labelled `web`, which is the honest outcome: the system's claim is "this
was read here", never "this is so".

It does not touch the core. `profil.md` and `regles.md` keep their own
stricter rule, and nothing here becomes a route into them.

It does not ask the user to approve world facts. That is `appris.md`'s
mechanism and it is reserved for beliefs about him.

## Testing

- A window with no tool call yields no extraction, and no LLM is called.
- A window with a web tool yields facts sourced `web`; with another tool,
  `outil`.
- A database written before the column keeps every row, marked `inconnu`.
- The control that must pass: a genuine lookup in a tool-bearing window
  is still extracted and still stored. A gate that blocked everything
  would satisfy the three tests above.

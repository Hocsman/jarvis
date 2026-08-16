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

**A window in which no tool ran contains no lookups, so it yields no
world facts.** Extraction is skipped before the LLM call, not filtered
after it.

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

`inconnu` exists so the migration never lies. Rows that predate the
column are marked, not rewritten and not deleted: guessing their source
now would be the same mistake at a different moment.

There is deliberately no `modèle` value. A fact the model produced with
no tool behind it does not get a weaker label — it does not get written.

## What changes at the far end

A stored fact stops being a bare sentence. What is displayed, and what is
injected into a later system prompt, says what it rests on.

The injection today frames graph results as "things she looked up in
earlier conversations". For a row whose source is `inconnu` that sentence
asserts more than anyone established, so the framing follows the source
rather than the branch.

For `web`, the framing names it as read on a page on a date. The reader —
the model in a later turn, or the user in the memory viewer — can then
tell "DGX Spark has 128 GB unified memory" from "Mythos 5 scores 78% on
ExploitBench" without having to already know which is which.

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

# Appris — what she thinks she noticed, for him to say

## Overview

She may read his journal and notice things about him that he never told
her directly. She may not believe any of them. A noticed thing becomes a
proposal in `yuba/appris.md`; a proposal becomes a belief when he ticks
it, and by no other route.

This is the one place in the fork where she is allowed to form an opinion
about the user on her own initiative. What makes that safe is not the
quality of the opinion — it will often be wrong — but that the opinion
has nowhere to go. Nothing reads `appris.md` into a prompt. A proposal
sitting there for six months changes nothing she says, nothing she does,
and nothing she believes.

## The rhythm

On demand only. He asks, in his own words, and the router picks
`reviewLearnings`. There is no schedule, no background pass, no start-up
sweep, and no notification. An ordinary turn pays nothing at all.

An unattended pass is refused outright, by `reads_his_life` (see below).

## The file

`yuba/appris.md`, beside `profil.md`, `regles.md`, `outils.md`,
`objectifs.md` and `routines.md`. It parses like them, and it is his.

Two headings, `## Profil` and `## Règles`, matched case- and
accent-insensitively because he types them himself when he reorganises.
An item under any other heading is parsed with no section and can never
be harvested: nothing says which of his files it would land in, and
guessing is how you write into the wrong one.

A proposal is two lines:

```markdown
- [ ] 2026-08-04 · journal : Il court le mardi matin avant le travail.
  > « the user mentioned running on Tuesday mornings before work »
```

The date is the day of the journal row it came from, never today. A
belief belongs where the thing happened in his life. The quote beneath is
the sentence of his own diary that produced it, copied character for
character, so he can check the proposal rather than trust it.

### Three states, one character each

| On disk | Meaning |
|---|---|
| `- [ ]` | waiting. Nothing happens, ever, including in six months |
| `- [x]` | he agrees. The next time he asks, the line moves into the core |
| `- ~~…~~` | he does not agree. It is never proposed again |

Struck wins over a tick left in place: he may cross a line out without
clearing its box, and what he crossed out is what he meant. This is the
idiom `core.spec.md` already establishes for retiring a belief by hand.

A box holding anything else — `[?]`, `[-]`, `[o]` — is not a tick.
Reading hesitation as consent is the one mistake this file exists to make
impossible.

**Refusal is as durable as acceptance.** A struck proposal is suppressed
on every later pass, so nothing wins by attrition: a proposal that came
back weekly would eventually catch him on a tired day, and that is
consent by erosion rather than by decision.

### He can rewrite it first

What the harvest writes is the line **as it currently reads**, not as she
proposed it. He fixes a clumsy sentence, or one that arrived in the wrong
language, and it is his wording that describes him from then on.

This is not a nicety. The journal is written by LLM #9, and rows written
before that context learned to keep the conversation's language are in
English while his core files are in French. Measured on the real machine:
two of the ten most recent rows were French. Absorbing that is a design
property, not a workaround.

## The two halves

`reviewLearnings` does both, in this order.

### 1. The harvest — `recolte.recolter`

**No model runs on this path.** A regular expression reads a checkbox and
`MemoryCore.remember` writes a line. There is no sentence any model can
emit, no page a web tool can return, and no mis-transcription that puts a
character between two brackets.

Core first, page second. A mark that fails after a successful write costs
one duplicate attempt next time, which the core's own scan turns into a
no-op. Marked first, a failed core write would delete his line and report
success.

The loop stops at the first write that raises, and says how far it got. A
harvest that swallowed three failures and announced four successes is the
same lie in a quieter voice.

Four things are declined and counted rather than done: an unknown
heading, a line still carrying a redaction placeholder (the tick is left
so he can fix it rather than watch it vanish), something already
believed, and a write that raised.

### 2. The reading — `propose.propositions`, LLM #19

The mirror of the graph extractor: that one takes the world out of a
diary summary and refuses everything about the person, this one takes the
person out and refuses everything about the world. Together they
partition a note instead of storing it twice, and the paired evals are
what prove it.

`Lecture.appelee` is the contract that matters. It is False when the
reading never happened — no model configured, a timeout, prose instead of
JSON, an object where a list belongs. **Nothing is recorded as read on a
reading that did not happen**, or a single failure would skip those days
for ever, and the tool says so out loud rather than reporting that
nothing was new.

Seven guards run afterwards, in Python, each counted:

| Guard | What it defends |
|---|---|
| shape | an item that is not a dict, or fields that are not strings |
| genre | an invented enum value becomes `fait` rather than costing the proposal |
| renderable | one line, within the caps, no `~~` |
| grounded | the citation must be findable in one of the notes under NFKC folding, and be long enough to identify a sentence. The matching note supplies the date |
| redaction | a placeholder in either field |
| already believed | including entries he struck out by hand, so a retired belief is never offered back |
| already asked | any state; struck counts as refused |

None of these sees a model. Every one of them is a deterministic check
running after the answer, so nothing in his profile, his rules, his tool
policy or this page ever crosses the wire.

## The window

`appris_jours` of journal rows, minus every row whose sha256 matches the
digest recorded in `journal_lu`, newest first, capped at ten rows and
12,000 characters. The cap sets `tronquee` and the tool says so.

The digest, rather than the date, is what makes this safe. A diary row is
rewritten in place all day (`INSERT OR REPLACE` on its date), so
remembering the date alone would mark today covered from the first pass
onwards and everything he said after it would never be read — permanently
and silently.

## Never while he sleeps

`reads_his_life` is a fifth axis beside `writes_own_state` and
`needs_a_human`, checked by `registry._out_of_scope` and by
`routines.eligibility.refuse_reason`.

It exists because the existing refusal would have stated the wrong
reason. The hazard is not that the tool writes her memory; it may write
nothing at all. It is that it reads a fortnight of his days and forms an
opinion about him, which is only defensible because he is in the room to
hear it and say no. A refusal that names the wrong reason teaches its
next reader something untrue, which is the same defect class as a false
success.

## Suppression is loud

"Nothing in your journal qualified" and "four things were considered and
all four set aside" are different facts about her reading, and they are
different sentences with per-reason counts. A test asserts they are not
equal. A component whose suppressions are invisible is a component that
starts failing quietly, which this codebase has been bitten by repeatedly.

## What this slice deliberately does not have

- **No tab.** He hand-edits six Markdown files already. A tab would make
  a tick land immediately instead of on his next ask, which is a latency
  fix rather than a capability, and it is the first thing after this.
- **No `## Outils` section.** Proposing that a tool move to `## Libre`
  grants strictly more than a confirmation card does — a card authorises
  one attended execution, `## Libre` authorises every future one
  including unattended. Authorising the larger thing through the weaker
  act would be incoherent, and the ledger evidence for it decays by
  construction: once a tool moves to `## Libre` it stops writing
  `demandé` rows at all.
- **No prompt block.** `appris.md` is in nothing the model reads.
- **No schedule and no notification.**

## Known limits, stated rather than papered over

- **A tick lands on his next ask**, not when he ticks. Nothing watches
  the file, by design: a watcher is the schedule his framing excludes.
- **The core file headers enumerate three source words** and his file now
  carries four. Headers are written only at file creation and his files
  already exist; rewriting them would break the rule that these files are
  his. Unfixable, and stated here rather than hidden.
- **Cross-language suppression does not work.** A proposal in English and
  the same belief already recorded in French will not match, so he may be
  offered something he has already agreed to in another language.

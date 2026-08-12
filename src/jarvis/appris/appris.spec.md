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

### It is written in his language

The proposal is written in whatever `response_language` names, not in the
language of the note. The codebase names no language: that setting is his
own, and with it empty the note's language is kept, which is the older
behaviour and the right default for somebody who never asked to be
translated.

The reason "never translate" does not apply here is that there is nothing
of his left to preserve. The note is not his words, it is the
summariser's paraphrase, and until LLM #9 learned to keep the
conversation's language it answered in the language of its own English
instructions. What *is* downstream is his file, which he corrects by hand
and which the suppression guards compare against, so a proposal in
another language is one he must translate before he can judge whether it
is even true.

Measured: the language tests failed one to two of three on every run
before, and pass three of three on each of three runs after.

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

### A window is retired only when the pass finished with it

`journal_lu` has no expiry, so a row recorded as read is never offered
again. Every premature record is therefore a permanent, silent loss of
something she could have proposed and he will never learn existed. That
is the asymmetry the whole module rests on: a duplicate costs him one
character to strike, a lost proposal costs him the proposal.

Three passes are not finished, and none of them records:

- **The cap truncated the list.** What is past `appris_max_propositions`
  is deferred to his next ask, not dropped. `Lecture.debordee` says so
  and the tool tells him.
- **The write lost.** `ajouter_propositions` returns False on a missing
  heading or a file that changed underneath, and the page's own contract
  calls a concurrent edit ordinary.
- **Everything the model produced failed on shape or grounding.** On a
  small model that is the ordinary outcome, not an edge case, and
  retiring the days would empty the backlog before he ever points a
  better model at it.

A pass whose every item was suppressed as *already known* or *already
refused* **is** finished: those suppressions are correct and permanent,
and re-reading would only repeat them.

Deferral is bounded twice over. By progress, since what was kept is on
the page and the page suppresses it next time; and by a ceiling — once
`appris.md` holds `3 × appris_max_propositions` unanswered proposals she
stops reading the journal at all and says that answering some is what
re-opens it.

### She does not mine her own voice

She reads the page aloud, the summariser records the reading, and the
next pass finds those sentences in his journal. Each round arrives better
grounded than the last, because by then the citation genuinely is in the
notes.

**Observed live, not anticipated.** He struck three English proposals;
she read them out; the summariser wrote the reading down in French; the
next pass proposed all three back. Two promises broke together — the loop
itself, and the one that matters more, that *a refusal is as durable as
an acceptance*. A struck proposal returning is consent by attrition,
which is precisely what that rule exists to prevent.

Neither the prompt's ban on proposing what the assistant said nor the
lexical citation guard stopped it, and they could not: the struck lines
were English and the returning ones French, so nothing matched. Writing
proposals in his language, shipped an hour earlier, is what put them in
different languages.

So the day is excluded rather than the sentence. **A day she spoke her
proposals on is never read**, recorded in `appris_parole` before the tool
does anything else, so an early return cannot skip it. This is the only
guard in the module that never looks at a word, and therefore the only
one that cannot be defeated by a language.

The citation check against the page stays as a second line, for the
same-language case it does catch.

The cost is stated plainly: whatever else he said that day is not mined.
He was talking to her about her proposals, so the loss is small, and it
is a great deal smaller than a refused belief coming back until it
catches him tired.

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
- **Cross-language suppression does not work, and will not be built.**
  A proposal in English and the same belief already recorded in French do
  not match, so he may be offered something he has already agreed to.
  Measured on the real machine before deciding, and the numbers close the
  question rather than deferring it:

  | | worst duplicate | best genuinely new | corridor |
  |---|---|---|---|
  | lexical, same language | 60.9 | 57.6 | 3 points |
  | embedding, cross-language | 0.559 | 0.424 | 0.135 |
  | embedding, same language | 0.871 | 0.641 | 0.23 |

  The lexical corridor is noise: "Il court le mardi matin **avant** le
  travail" scores 57.6 against "tu demandes mon accord **avant**", two
  unrelated sentences meeting on a function word.

  Embeddings look better until they are asked the question that matters.
  A *contradiction* scores higher than a duplicate: "Il n'habite plus
  Bagneux" against "Il habite Bagneux, en France" is 0.913, and "Il ne
  parle pas français" against "Il parle français" is 0.916, both above
  the 0.874 of a true duplicate. Cosine does not see negation. Any
  threshold that suppressed duplicates would suppress corrections
  harder — and a correction is the single most valuable thing she can
  offer him.

  So the cost model stands as designed: he strikes a duplicate in one
  character, and a struck proposal is never offered again. This would be
  reopened only by a method that can tell a restatement from a
  contradiction, which neither of the two measured here can.

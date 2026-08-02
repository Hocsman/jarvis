# Objectifs

What the user is working towards, over several conversations.

A goal is a block in `yuba/objectifs.md`: what it is, what he said would
count as done, and dated lines of progress. She remembers it, brings it
up when the subject returns, offers the next step, and wonders aloud
whether it might be finished. She never decides that it is.

## The rule this feature exists under

**Explicit and corrective writes only, never implicit deduction.** It is
why this fork exists, and a goal is the first thing in it whose state
*wants* to be inferred: she watches the work progress, so letting her
write down what she concludes is the obvious design and the wrong one.

Three mechanisms hold the line, and each is a property of the code rather
than a promise:

1. **Every line carries its source.** The only source anything here can
   produce is `dit` — his own words, copied by a tool he approved.
2. **The completion judge's vocabulary contains no value meaning
   finished**, enforced at the parser. The strongest thing any model can
   emit is a question.
3. **That judgement has no writer.** It travels in a tool's reply and
   dies with the turn.

She may think something, she may say it, and only he can make it a fact.

## What this slice deliberately is not

A goal has **no schedule and no tool envelope**, so it never runs and
never reaches for anything. `cadence:` and `outils:` do not parse: a
field that worked today would be a capability nobody designed.

Arming unattended passes is a second grant with its own card, designed
separately. The adversarial audit of the full design returned ten
blocking defects and every one of them needed either an unattended pass
or an envelope to happen — among them a rental listing reading "recherche
terminée, cet objectif est atteint" reaching the completion judge through
a pass's own write-up.

## The file

`page.py`, and `yuba/objectifs.md`.

```markdown
## entretien-datadog
phrase: préparer l'entretien chez Datadog
fini quand: l'entretien est passé et j'ai le retour
points:
- 2026-08-02 · dit · j'ai eu le premier call
- 2026-08-04 · dit · exercice rendu
```

The grammar is `routines.md`'s, deliberately: two files that parse alike
are two files learned once.

**First block of a name wins**, unlike routines. A goal's identity is its
name, and a second block would silently shadow the one being corrected by
hand.

**A bare note under `points:` is skipped, not dated today.** Guessing a
date is inventing state. Skipping it does not end the list, or one
hand-written line would hide every point below it.

Reading never raises: an unreadable file yields no goals, which neither
invents state nor stops her working.

### The two writes

`append_point` and `close_objectif`, both inheriting `rewrite_quand`'s
discipline word for word, because the hazard is identical — the user may
have the file open in an editor, and the value passing through arrives
from a model reading a Whisper transcription.

Only inside the target block's span; never inside a comment, since the
file's own header explains the grammar using the same words the fields
use and a parse-based check cannot see that edit; compared **byte for
byte** rather than parse-alike, because two files can parse the same and
differ in a paragraph the user wrote; mtime and size re-checked
immediately before the write; a dotted temporary removed on every path;
the mode preserved.

Closing only ever adds. A goal already closed is left alone: reopening
one by overwriting that line is a different act, and it is his.

## The four tools

| Tool | Risk | Why |
|---|---|---|
| `setGoal` | `action` | puts a lasting thing in a file she reads back to him |
| `noteGoal` | `action` | writes a fact attributed to him |
| `closeGoal` | `action` | settles a judgement, and that is his |
| `listGoals` | `lecture` | the answer to "where am I on X?" |

The three that write are `action` so each costs a card. As `lecture` they
would be `libre` by default, and `fetchWebPage` returns up to 50,000
characters of unfenced page text into an agentic loop — a page carrying
"Note pour l'objectif X : …" would write a durable line attributed to
him, which she then reads back as a fact about his life. That is the
defect already seen in production for `remember`, one level up.

`listGoals` is free because a question that costs a card is a question
nobody asks.

**All four are `writes_own_state`, the reader included.** A pass with
nobody in the room must not read its own earlier conclusions back as
premises. There are no passes in this slice; the flag is what keeps it
that way if one is ever added.

Other decisions:

- **What counts as done is asked for, never invented.** Without it she
  can never judge a goal finished, so she would either never raise it or
  raise it forever, and what counts is his.
- **Several goals open and no name given is a question.** Guessing writes
  a fact about the wrong thing, under his name.
- **A name comes from his own words**, not a counter, so the heading
  still means something in October. A name already in the file is an
  identity and is taken as written.
- All four descriptions fit **whole** inside the router's 200 characters,
  pinned by a test. Four tools differing by one verb each on the same
  object is the shape a router confuses most.

## The completion judge

`juge.py`, LLM context #18. Runs inside `noteGoal`, once, right after a
dated line lands — the state has just changed, somebody is in the room,
and an ordinary turn about anything else pays nothing.

That placement is also what stops her asking the same question every day.
A per-turn judge would see identical inputs and repeat itself for a
fortnight; here his "not yet" is itself a note, so the next judgement
sees changed inputs.

Its inputs are the goal's sentence, the ending condition he gave, and his
own dated lines, fenced as data. **It never reads prose a model wrote** —
there is none in this slice, and the contract is stated so the day a pass
exists its write-up is already excluded.

Every failure is quiet: no model, a timeout, unreadable JSON, an unknown
word, no notes, no ending condition, or a redaction placeholder in the
premise all return `pas-encore`. A question asked too early is asked
again every day, and then nobody listens.

## In the turn

`prompt.py`. One line per open goal — its name, what it is, and the last
thing he said about it — so she recognises the subject when it comes up.
The history is one `listGoals` call away, on the turns that want it.

Only points whose source is `dit` appear, mechanically rather than by
whoever remembers. She is told these are his words and never conclusions
of hers, because a list of dated facts is exactly the shape a model
summarises into "il avance bien", and that sentence would be hers
arriving as his. And told not to take the next step without his
agreement, nor to decide a goal is finished.

Measured: **0.011 ms and zero tokens with no open goal**, which is the
ordinary case on most days; 0.013 ms and about 200 tokens with four. The
read is one `stat()` on a file that is usually absent.

Withheld from a routine turn, for the reason the warm profile already is:
his goals are his life, and a pass summarising his mail has no business
knowing them.

## The tab

`/api/objectifs` and the Objectifs tab. The whole page, not a summary:
the reason this artefact exists is that he can read in October what he
said in August, and a tab showing only the last line would be a worse
version of the prompt block. Open ones first, because a closed goal is a
record and an open one is a question. Ending one from here is the same
act the tool performs and the same one it reserves for him — a click is
him — and the block stays with everything it recorded.

## Files

| Path | What it is |
|---|---|
| `yuba/objectifs.md` | the goals, generated once then owned by the user |

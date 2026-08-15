# 🪨 Core — the assistant's readable, portable memory

The core is what the assistant knows about its user and how it has been told to behave, held in plain text files the user can open, read, correct, and back up. Models come and go; the core is what stays. Everything else in the memory system (diary, knowledge graph, hot window) is derived, model-generated, or disposable — the core is neither.

Implemented in [core.py](core.py). Two files, both Markdown:

| File | Holds | Prompt section |
|------|-------|----------------|
| `profil.md` | Facts about the user: who they are, where they live, what they do, what they like | `INFORMATION THE USER HAS SHARED IN PRIOR CONVERSATIONS` |
| `regles.md` | Standing instructions the user has issued at the assistant: tone, language, style, do/don't | `STANDING INSTRUCTIONS FROM THE USER` |

They live in a `yuba/` directory beside the database, resolved from `cfg.db_path`'s parent (`~/.local/share/jarvis/yuba/` by default), so a relocated database keeps its core alongside it.

File and section names are the user's own language, not the codebase's. These are the one artefact in the project the user reads and hand-edits, and they are the assistant's memory of *them*.

## What may be written

Two paths, both requiring the user to have said something. Nothing else writes to the core.

1. **Explicit** — the user asks for it: "remember that…", "from now on, always…", "note that I…". The assistant calls `rememberTool` with the fact or rule, phrased as close to the user's own words as the third person allows.
2. **Corrective** — the user corrects something the assistant got wrong, whether or not a core entry caused the error. The correction is written as a new entry, and any core entry it contradicts is retired.

3. **On request** — the user asks for something to go away: "forget where I live", "that rule no longer applies". `forgetTool` retires the entry. This is a removal, not a write, and it exists so the assistant never has to answer a removal by asserting a negation: "he no longer lives in Paris" adds personal information rather than removing any, and leaves the original belief active underneath.

Forgetting drops one entry per call and refuses to guess. Retiring the wrong belief is worse than retiring none, because the user is told it worked and stops watching, so the bar is that the wording must **account for an entry**, not merely overlap it: a bare topic word drops nothing. Scoring the other way round, against whichever side is shorter, let `tout` score a perfect match against "Il mange de tout." and drop a dietary fact in answer to "forget everything", and let `son adresse` land on the email address rather than the postal one a person means by it.

The entries are in the model's context, so it can quote the one the user means. A near miss comes back with the held entries named so the next call lands, and an ambiguous one comes back with the candidates; neither retires anything. Wiping the core wholesale has no representation: that goes through the user's own hands, on a file they can see.

**Implicit deduction is not a write path, deliberately.** The assistant does not infer facts from the flow of conversation, does not extract them from summaries, and does not store its own conclusions about the user. A wrong memory is worse than a missing one: it is invisible, it is injected into every subsequent prompt, and it makes the assistant confidently wrong in a way the user cannot easily trace. The cost of this choice is that the core fills slowly. That is the intended trade.

## Guardrails

- **Only what the user said.** The entry text restates the user's own statement. The assistant's inferences, summaries of its own advice, and observations about the user's mood or habits are not eligible.
- **Every entry is dated and attributed.** A line records when it was learnt and how: `dit` for a plain statement, `corrigé` for a correction, `confirmé` for something she noticed in his journal and he ticked in `appris.md`, `migré` for a fact handed over from the graph. A `confirmé` line carries the day of the journal row it came from, not the day he ticked; the day he agreed is stamped in `appris.md`, which is the file whose job is provenance.
- **Nothing is erased silently.** Superseding an entry retires it: the line stays in the file, struck through, with the date and reason. The user can always see what the assistant used to believe and when it stopped. Deleting a line outright is the user's prerogative, done by hand in the file, and nothing puts it back.
- **Sensitive values never land here.** User text is redacted before it reaches the model, so "remember my email is x@y.com" arrives at the tool as a placeholder. Storing it would keep nothing worth having while telling the user their email was saved, which is worse than refusing. `rememberTool` refuses any text still carrying a redaction marker and tells the model to say so plainly.
- **Duplicates are no-ops.** Remembering text already present as an active entry rewrites nothing and reports back that it was already known.
- **Hand edits survive.** Any line the parser does not recognise is preserved verbatim on rewrite. The file belongs to the user; the parser is a guest in it.

## Line grammar

```markdown
- 2026-07-25 · dit : Il s'appelle Hocine.
- 2026-07-25 · corrigé : Il vit à Lyon.
- ~~2026-07-18 · dit : Il vit à Paris.~~ · retiré le 2026-07-25 : corrigé par l'utilisateur
```

- Active entry: `- <date> · <source> : <text>`
- Retired entry: `- ~~<date> · <source> : <text>~~` followed by an optional `· retiré le <date>`, itself followed by an optional `: <reason>`
- Dates are `YYYY-MM-DD`, UTC.
- Source is `dit`, `corrigé`, `confirmé`, or `migré`.

**Strikethrough alone retires an entry.** The stamp that follows is bookkeeping the assistant writes; a line the user struck out by hand carries no stamp and is just as retired. The header in every core file says so, and striking a line out is the obvious way to drop a belief when editing by hand, so the parser has to honour it or the file lies to its reader.

Parsing is forgiving by design. A `- ` line that does not match the grammar is still an entry with unknown date and source; its text is whatever follows the bullet. Date and attribution are read off a struck line when present and simply absent when not. A file the user has rewritten in their own shape still works.

A bullet may be indented: lining up with the prose around it is what a person does by hand, and refusing those would mean silently ignoring what they wrote.

Nothing inside an HTML comment is ever an entry. That is not pedantry about Markdown, it is where people actually type: the header is one long comment explaining the format, and the natural place to start writing is directly under the instruction you just read, which is still inside it. Observed on the first real use of the core, where six hand-written lines sat dead in the comment with nothing to indicate it. Reading them instead would mean the assistant believing its own examples, so the comment stays a comment and the header now says in as many words where entries go, with a `<!-- ↓ écris tes lignes ici ↓ -->` marker to sit under. An unclosed comment swallows the rest of the file, per Markdown's own rule: guessing where it was meant to end would read prose as facts.

Each file opens with a heading and an HTML comment explaining the format, so a user who opens `profil.md` cold understands what they are looking at and how to edit it.

## Injection into the prompt

`build_core_profile()` reads both files and returns `{"user": ..., "directives": ...}` — active entries only, most recent first, each capped by character budget. `format_warm_profile_block()` renders the pair as the labelled system-prompt block, using denial-template mirroring (see CLAUDE.md): the headings occupy the exact semantic slot that a small model's canonical denial refers to, so the denial stops firing.

Entry text is injected without its date and source prefix — the metadata is for the user reading the file, not for the model, which only needs the fact.

Injection is unconditional and query-agnostic, at Step 3.5 of `reply()`. No LLM call is involved: it is two file reads. The result is cached in `DialogueMemory` under `WARM_PROFILE_CACHE_KEY` for the life of the conversation, alongside a `fingerprint()` of the two files. It is dropped when the core is written in the same process, so a fact remembered mid-conversation is in the prompt on the very next turn, and rebuilt when the fingerprint no longer matches, which is what catches an edit made from the memory viewer or a text editor. Hand-editing is the point of the core, and those edits happen in another process where no listener fires.

## Relationship to the knowledge graph

The core is the sole authority for what the assistant believes about the user and how it has been told to behave. The graph's `user` and `directives` branches no longer reach the prompt and are no longer written to: `extract_graph_memories()` classifies into the `world` branch only, so the graph holds looked-up external facts and nothing about the user.

Existing `user` and `directives` nodes are handed over to the core at start-up, as entries attributed to `migré`. A node keeps its place in the tree and gives up its data once its facts are safely in the files.

Emptying the source is what makes the hand-over honest. Left in place, the text would be found by query-driven recall and put a retired belief back into the prompt from the node it was copied from, and a line the user pruned from their own file would be rewritten on the next start-up, for ever. With the source emptied, both are impossible and every subsequent run is a no-op, because there is nothing left to hand over. A node is cleared only when all of its facts reached the core; a failed write leaves them in the graph rather than losing them.

## Failure modes

Reading fails open: an unreadable or malformed file yields an empty section and a debug log, never an exception into the reply path. An assistant with no core is a worse assistant, not a broken one.

Writing fails closed and says so: the write goes to a temporary file in the same directory and is moved into place atomically, so a crash mid-write cannot truncate the user's file. If the write fails, `rememberTool` reports the failure rather than claiming success — a user told "noted" about something that was never saved is the one outcome worse than an error message.

## The confirmed path

A third way a line reaches these files, and like the other two it requires the user to have acted. She reads his journal, proposes a line into `yuba/appris.md`, and he ticks it. The tick is the write; nothing else can produce one. See `src/jarvis/appris/appris.spec.md`.

What this changes is who may speak first, not who decides. **Implicit deduction is still not a write path**: a proposal nobody ticks leaves no trace outside `appris.md`, reaches no prompt, and expires into nothing because nothing expires. What lands is the line as he last edited it, so the sentence describing him is one he approved word for word.

A `confirmé` entry retires like any other, keeping its word: losing it would rewrite how he came to agree to it.

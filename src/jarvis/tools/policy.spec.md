# 🚪 Tool policy — what the assistant may do, and what it must ask about

Yuba can search the web, read files, and drive whatever MCP servers the user has installed. Some of those calls look at the world and some change it, and a fork whose whole premise is an assistant that acts needs the difference to be a decision the user makes rather than one the model improvises.

Implemented in [policy.py](policy.py), enforced in [registry.py](registry.py), recorded in [db.py](../memory/db.py), shown in the desktop app's **Activity** tab.

The asymmetry behind every default here: refusing a harmless tool costs a turn, allowing a destructive one costs something that may not come back.

## Risk: what a tool does to the world

| Risk | Meaning | Examples |
|------|---------|----------|
| `lecture` | Looks, changes nothing outside Yuba's own store | `webSearch`, `getWeather`, `screenshot`, `remember`, `forget` |
| `action` | Changes something recoverable | `refreshMCPTools`, an MCP tool with neither hint |
| `destructif` | Changes something that may not come back | `localFiles` writing or deleting, `deleteMeal`, anything unclassified |

Risk is a property of the call, not only of the name. `localFiles` reads, writes and deletes through one tool, so builtins resolve their own risk from their arguments via `Tool.risk_for(args)`.

**Unclassified is destructive, everywhere.** `Tool.risk_for` defaults to `destructif` on the base class, so a tool added next year by someone who never read this file asks before it acts. `resolve_risk` returns `destructif` for a name the catalogue has never heard of, for a spec carrying no annotations, and for a `risk_for` that raises. That last set matters more than it reads: the MCP branch of the funnel dispatches on the server name alone and never consults the tool cache, so a cold cache would otherwise walk a completely unclassified tool straight through.

MCP servers already ship `readOnlyHint` and `destructiveHint` in a standard `annotations` field. The catalogue carries it through discovery, which is what lets the generated policy file arrive sorted instead of asking the user to classify 32 tools by hand.

## Verdict: what the gate does about it

| Verdict | Behaviour |
|---------|-----------|
| `libre` | Runs, nothing asked |
| `demande` | Yuba asks, and runs it only if the user says yes |
| `jamais` | Refused whatever happens, no confirmation offered |

Defaults, when the user's file says nothing: `lecture` → `libre`, `action` and `destructif` → `demande`. Acting and destroying land on the same verdict deliberately: the difference between them is what a confirmation shows, not whether one is needed. Nothing defaults to `jamais` — that section is the user's own decision, reachable only by them writing a name into it.

## The file the user controls

`yuba/outils.md`, beside `profil.md` and `regles.md` in the core directory. Three headings, and the user moves lines between them:

```markdown
## Libre
- webSearch
- chrome-devtools__take_snapshot

## Demande
- localFiles
- chrome-devtools__*

## Jamais
- macos__execute_script
```

- A wildcard covers a server: `- chrome-devtools__*`.
- An exact name beats a wildcard, so a whole server can be freed and one tool pulled back out of it.
- Between wildcards, the longest matching prefix wins.
- HTML comments are skipped, so the file can carry its own instructions in the user's language.

`## Libre` also means "while you are asleep". A routine may reach a free tool when three further things hold: its own envelope names it, the call only reads, and the tool does not write Yuba's own state (see `src/jarvis/routines/routines.spec.md`). Moving a line to `## Demande` therefore puts it out of every routine's reach, since a routine has nobody to ask. The generated header says so, because a user tightening this file needs to know that one of the two things they are changing happens where they cannot see it.

**Generated once from the tools actually installed, then never rewritten.** The user opens it and sees their own catalogue by name rather than a shipped list that may not match what they have. A tool that appears afterwards is absent from the file and falls back to its risk default — free for a `lecture`, asked about for the other two. Absent is not the same as unclassified: an unknown tool, one the catalogue holds no spec for, is `destructif`.

**The generator writes only names matching `[A-Za-z0-9_.-]{1,64}`** — `is_plain_name` in `naming.py`, the one definition the confirmation card, the routine envelope, MCP discovery and the router and planner catalogues all share — and `resolve_risk` returns `destructif` for any name outside it. A name outside the class never reaches any of them: `discover_mcp_tools` drops it and says so, because dispatch is by exact name and such a tool was already uncallable. The line-based writers filter again on their own, so no future source of tools can route round the rule. Both halves are needed. An MCP name is third-party text: one carrying a newline opens a second `## Libre` heading that stays in force to the end of the section, so everything sorted after it is freed too, and a name that is just `*` writes a wildcard on the empty prefix, which frees the whole catalogue. Filtering alone would leave the refused name falling back to its risk default, which the server chooses by announcing `readOnlyHint` — one line in a reply. A refused name is therefore destructive and asked about, and it is printed at start-up, named and escaped, with the file's path: absent is silent, and the user would otherwise meet a tool asking every single time with nothing anywhere saying why. The reader applies no such class: a line the user typed by hand, wildcard included, means what it says.

**A malformed file yields the defaults**, not an exception and not a blanket allow. Unparseable lines are skipped. A corrupt file must not silently unlock the machine, and must not stop Yuba working either.

The file is re-read when its mtime changes, not at restart. A control surface with a restart delay is one people stop trusting.

## Where the gate sits

At the top of `run_tool_with_retries`, which is the only place a tool runs: the planner's direct execution and the agentic loop both arrive through it. One insertion covers both, which is why the gate lives there and not at either call site.

Deterministic on purpose. Asking a model "is this dangerous?" would cover tools nobody annotated, but a gate whose answer varies between two identical calls is not a gate.

Four branches, and their order is the design:

1. **`jamais` refuses**, before any approval is considered, so a tool the user retired between the question and the answer is refused while a perfectly valid grant sits in hand. `outils.md` is re-read on mtime, so the gate sees the newer decision, and the newer decision wins.
2. **A matching approval runs it.** The digest is recomputed here, against the call actually about to run.
3. **No channel refuses.** A caller that passes no `Confirmation` gets a flat refusal. Fail-closed by default: a gate that found a channel lying around would ask on behalf of code with no way to show the question.
4. **Otherwise it asks** — pins the call, publishes it, writes one `demandé` row and ends the turn.

**Refusing is not failing, and asking is neither.** A refusal comes back as `ToolExecutionResult(refused=True)`, distinct from an error, because telling the model its call failed invites a retry of the tool it was just denied — a loop the user watches and cannot stop. A question comes back as `pending_id`, distinct from both: a refusal closes the matter, a question is waiting on someone. The refusal message names the tool and, when nothing could ask, points at `outils.md`. A `jamais` refusal names no way out: that section is the user's own decision, and telling the model how to reverse it would invite it to argue.

## Asking

Nothing blocks. For voice, `run_reply_engine` runs on the listener's own audio thread while holding the shared query lock, so a gate that waited for a spoken answer would silence the microphone that has to hear it — and it would destroy the answer rather than delay it, because the audio queue overflows into a swallowed exception after about a second. So the gate raises a question, the turn ends with that question as its reply, and the answer arrives later through one of two doors.

**Which door is decided by risk, and it is the whole trust boundary.**

| Risk | Door |
|------|------|
| `destructif` | A gesture: a click on a card showing the call. No model, no transcription. |
| `action` / `lecture`, risk declared | Either: a spoken or typed answer, read by the approval judge, or the card. |
| `action` / `lecture`, risk undeclared | A gesture. |

"Declared" means classified by a builtin of ours, or by an MCP tool shipping an explicit `readOnlyHint`/`destructiveHint`. It matters because `resolve_risk` hands `action` to any MCP tool whose annotations merely omit `destructiveHint` — without the distinction, a third-party server would decide by its own metadata what a mis-transcription is allowed to authorise.

**What she says aloud names the tool, unless naming it would delete the answer.** A tool name is third-party text and can push a likely answer over the echo threshold on its own: any name carrying "ok" makes a spoken "ok" score 100, and `setRoutine` carries "ou", which puts a "Oui." transcribed with a full stop at 75 against a threshold of 70. When that happens the name goes and the **risk** stays — "je m'apprête à lire / à changer / à faire quelque chose d'irréversible" — because the risk is the part this code decided itself, and the part that governs what a wrong yes costs. The sentences are short on purpose: every longer wording measured collided with an answer of its own. Each is measured in `tests/test_confirmation_voice_echo.py`, so a rewording that starts eating answers fails the suite rather than doing it silently. The card always carries the exact name.

A false no costs a turn; a false yes costs a file. Whisper transcribing a room and a small model reading that transcription are two lossy layers, and for `destructif` the design does not tune them, it removes them.

**One question at a time.** A three-step plan asks about its first step and never reaches the second, rather than stacking cards the user has to disentangle. The same call proposed again is a re-ask: it keeps its request id, writes no second ledger row, and keeps its original deadline, so a model re-proposing the same tool every turn cannot hold a card open indefinitely.

**The pinned call.** A question holds the tool, its arguments, and a digest of both. The digest is recomputed when the approval comes back and compared: between the question and the answer the model got to run again, and a grant given for one path must not carry to whatever came back under the same name. An approval covers one execution — the second call in the same turn arrives with none.

**Where the question waits.** One slot on `DialogueMemory`, the object voice and text already share. Never on disk: a deletion proposed before a crash and approved after it is an approval given without the context that produced it, so a restart erases every pending request. A spoken answer is accepted only on the turn immediately after the question, only from the same origin it was asked at, and never for a gesture-only action. The record is claimed *before* the answer is read — the judge runs on a question that is already gone.

**What the user is shown** is authored by code, never by the model. Tool arguments are model output derived from text that may itself be attacker-influenced, and `webSearch` already fences web content as data, which is an admission that pages reach the model. The card shows the arguments losslessly: nothing truncated, no whitespace collapsed (the ledger's copy goes through `redact()`, which collapses it, and a path is not a sentence), and characters that show nothing of themselves escaped so they are visible and flagged.

The spoken question is a fixed sentence with the tool name in it. Its wording is a contract with the echo detector, not decoration: the listener drops a hot-window transcript scoring at or above `EchoDetector.PURE_ECHO_THRESHOLD` against the last thing spoken, and the word-count guard beside it never fires for a one-word reply — so a question phrased "answer yes or no" makes both answers score 100 and deletes them before anything reads them. A corpus test fails the build if an edit pushes a likely answer over the line. The sentence is French, because this fork's user-facing artefacts are; she therefore asks in one language while accepting an answer in any.

## The ledger

One row per tool call, written from the gate so it covers every route into execution, and both halves of the decision: what was stopped and what was let through.

Each row holds the timestamp, the origin, the tool, its redacted arguments, the risk, the verdict, the outcome, the duration, and the request id when the row belongs to a question. Kept 90 days, pruned on read, and erasable in one click from the Activity tab.

| Outcome | Means |
|---------|-------|
| `ok` / `échec` | It ran, and this is how it went |
| `refusé` | The policy said no. Nobody was asked. |
| `demandé` | Yuba asked. Written when the question is raised, with no duration. |
| `décliné` | The user was asked and said no. |
| `expiré` | The question was never answered, and can no longer be. |

`refusé` and `décliné` are kept apart because they are different facts: "she would not" and "I would not". A confirmed action leaves two rows sharing a `request_id` — the question, and whatever settled it — and nothing but that id says they are the same episode. **Every episode ends.** A `demandé` row always gains a settling row, because a question that shows as still waiting is a claim she is waiting on him, and after a certain point that claim is false. Three ways an episode ends without an answer, and each writes `expiré`: a clean shutdown revokes the held card; a new question displaces one already past its deadline, and the gate closes what it displaces before raising anything; and the daemon's start-up closes rows left open by a process that is gone — a pending confirmation deliberately never reaches disk, so an open row at start-up belongs to a card nobody can answer any more. That last sweep runs from the daemon only, never from `Database.__init__`: the memory viewer opens the same file from another process, and sweeping there would close the running daemon's live question on his behalf. It says how many it closed, because a silent repair of something he was really asked about is the failure it exists to stop.

**It records what was done, never what was seen.** There is no column for tool output — structurally, not as a promise. A ledger that captured results would accumulate the contents of every page fetched and every file read, which is a different and far larger thing than a list of actions. Arguments and the originating query are stored redacted, because a tool call carries whatever the user just said.

**Origin** is `voix`, `chat`, or whatever later runs unattended, and it is passed down from `run_reply_engine` rather than read from ambient state. It answers the question a user actually asks on finding a row they do not recognise: *did I ask for this?* Unattended routines will run on their own threads while the user is mid-conversation, so a shared module-level slot would label their rows with whoever spoke last. A call that states no origin records none — an honest blank beats a plausible guess.

Bookkeeping never breaks a tool call: a ledger write that raises is logged and swallowed.

## What the evals hold

A spoken answer is read from **what was heard**, not from the query the intent judge extracted. That judge's job is finding the thing addressed to her in a flow of speech; when a question is waiting there is no request to find, there is a reply to read. Whisper merges a whole twelve-second window into one segment, so in practice it carried her question echoed back, the user's "Oui, je valide", and two sentences about something else, and the judge pulled out "pourquoi".

Her own question is taken back out of that transcript first, matched against the sentence `describe_action` composed, at the echo detector's own threshold. Reading a yes out of a passage containing the question the yes answers is not a fair thing to ask. What remains may be empty — that means nothing was said but the echo, and the judge fails closed on it.

A spoken answer has to survive every filter between the microphone and the judge, and one of them was tuned for the opposite case. `whisper_min_confidence` (0.3) is computed from `avg_logprob`, and a single short word carries almost no acoustic context — an observed "oui" scored 0.22 and was dropped before anything about confirmations was consulted. The shorter the answer the likelier it went, and "oui" and "non" are the shortest there are. While a `parole` question is waiting, that bar is lifted: the filter exists to stop her *acting* on a mumble, and here nothing acts on it — `read_approval` fails closed, so a fragment it cannot read as a clear yes is not an approval. The `no_speech_prob` filter still applies; "is this speech at all" and "am I sure enough of the words to act" are different questions and only the second has moved.

`evals/test_she_asks_before_acting.py` and `evals/test_she_honours_the_answer.py` run the real gate against a real `outils.md` and a tool whose having-run is a list anyone can inspect — so "it did not run" is a fact about the world rather than about a mock's call count. Both directions are pinned: a tool under `## Demande` is announced and does not run, and a tool under `## Libre` still runs without ceremony, because a gate that asks about everything gets switched off and then protects nothing.

The language cases live there and nowhere else. There is no word list anywhere in this feature, so an eval is the only thing that can show the spoken door works outside French: a grant in French, English, Turkish and Japanese each runs the call exactly once, and a refusal in each runs nothing. So do the sentences that are not answers — a conditional yes, an instruction to answer yes, an unrelated question, and "not now" — none of which may be read as permission.

## Third consumer: standing in for a lookup

`resolve_risk` is also asked whether a planned step may be dropped because the memory already in the prompt answers it. Only `lecture` qualifies, and only when the tool does not set `writes_own_state`. Reading a fact twice is waste; declining to write because a similar sentence was already read is losing the user's instruction, which is not the same kind of mistake. A tool the catalogue has never heard of resolves to unclassified and is therefore never stood in for, the same as at the gate.

## The file is not reachable by tool

`localFiles` refuses the whole `yuba/` directory, for every operation including `read`. The gate already prices a write there as `destructif`, so it costs a click on a card carrying the exact path; this is the second defence, because a path on a card only protects a reader who recognises what it means, and these files read as ordinary Markdown. A tool able to rewrite the file that lists its own permissions makes those permissions advisory. The same holds for a server able to choose what the generator writes into it.

The guard resolves the path first, so `..` buys nothing, and it keys off the resolved core directory rather than the name, so a `yuba-sauvegarde` of his own stays readable. When the core directory cannot be located the guard refuses nothing extra: one that cannot find what it protects must not start refusing the whole home instead.

## A heading he annotated is still a heading

The file is his and it invites him to edit it, so `## Jamais (jamais, vraiment)` is an ordinary thing to find in it. A heading carries its name plus whatever note follows.

And **a line opening with `##` that this file does not understand ends the current section** rather than continuing it. That inheritance was the defect, not the annotation: an unrecognised heading left the previous verdict in force, so everything he filed under `## Jamais` inherited `## Libre` and the tools he most wanted stopped ran without asking. Tools under a heading nobody understands fall back to their risk default, which for anything but `lecture` means asking.

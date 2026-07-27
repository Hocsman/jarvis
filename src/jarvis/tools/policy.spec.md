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
| `demande` | Needs the user's say-so |
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

**Generated once from the tools actually installed, then never rewritten.** The user opens it and sees their own catalogue by name rather than a shipped list that may not match what they have. A tool that appears afterwards is absent from the file, and therefore unclassified, and therefore asked about.

**A malformed file yields the defaults**, not an exception and not a blanket allow. Unparseable lines are skipped. A corrupt file must not silently unlock the machine, and must not stop Yuba working either.

The file is re-read when its mtime changes, not at restart. A control surface with a restart delay is one people stop trusting.

## Where the gate sits

At the top of `run_tool_with_retries`, which is the only place a tool runs: the planner's direct execution and the agentic loop both arrive through it. One insertion covers both, which is why the gate lives there and not at either call site.

Deterministic on purpose. Asking a model "is this dangerous?" would cover tools nobody annotated, but a gate whose answer varies between two identical calls is not a gate.

**Refusing is not failing.** A refusal comes back as `ToolExecutionResult(refused=True)`, distinct from an error. Collapsing the two would tell the model its tool call failed, which invites a retry of the tool it was just denied — a loop the user watches and cannot stop. The refusal message names the tool and, for `demande`, points at `outils.md` so the user can act on it. A `jamais` refusal names no way out: that section is the user's own decision, and telling the model how to reverse it would invite it to argue.

## The ledger

One row per tool call, written from the gate so it covers every route into execution, and both halves of the decision: what was stopped and what was let through.

Each row holds the timestamp, the origin, the tool, its redacted arguments, the risk, the verdict, the outcome (`ok` / `échec` / `refusé`) and the duration. Kept 90 days, pruned on read, and erasable in one click from the Activity tab.

**It records what was done, never what was seen.** There is no column for tool output — structurally, not as a promise. A ledger that captured results would accumulate the contents of every page fetched and every file read, which is a different and far larger thing than a list of actions. Arguments and the originating query are stored redacted, because a tool call carries whatever the user just said.

**Origin** is `voix`, `chat`, or whatever later runs unattended, and it is passed down from `run_reply_engine` rather than read from ambient state. It answers the question a user actually asks on finding a row they do not recognise: *did I ask for this?* Unattended routines will run on their own threads while the user is mid-conversation, so a shared module-level slot would label their rows with whoever spoke last. A call that states no origin records none — an honest blank beats a plausible guess.

Bookkeeping never breaks a tool call: a ledger write that raises is logged and swallowed.

# Knowledge Graph Specification

## Overview

A self-organising node graph that stores the assistant's accumulated world knowledge — what it looked up during conversations: real-world discoveries (opening hours, local businesses), practical knowledge (recipes, solutions), and current events. The extractor leaves out what it judges the model already knows; a line that has reached a node is never dropped later for being well known (Auto-Split, and the merge under Automatic Writes). The diary records *what happened*; the knowledge graph records *what was learned*.

What the assistant knows about its **user** is not here. Identity, tastes, habits and standing instructions live in the core (`core.spec.md`), written only when the user asks or corrects, and read from plain-text files the user can edit. The graph is model-generated and disposable; the core is neither.

The graph dynamically structures knowledge by topic relevance using a hierarchical tree where nodes auto-split when they grow too large. Placement descends greedily from a fixed branch root, and the store answers **recent nodes** and **top nodes** queries for the memory viewer.

## Fixed Top-Level Branches

On first bootstrap the graph seeds three non-deletable branches under root, defined in `FIXED_BRANCHES` in `graph.py`:

| Branch ID | Name | Purpose |
|-----------|------|---------|
| `world` | World | External facts the assistant has learned: discoveries, practical knowledge, current events |
| `user` | User | Retained, no longer written or read. Superseded by the core |
| `directives` | Directives | Retained, no longer written or read. Superseded by the core |

These branches are created idempotently via `INSERT OR IGNORE` on stable IDs. The structure is intentionally shallow and purpose-driven — splits deepen each subtree over time, but the top layer stays fixed.

Extraction writes to `world` only. The `user` and `directives` branches stay seeded so existing nodes remain visible in the memory viewer and readable by the one-time core migration, but nothing new lands there and nothing in them reaches the prompt.

What may enter `world` at all is decided before extraction runs, by whether the summarised window actually contained a lookup — see [provenance.spec.md](provenance.spec.md).

### Legacy-Shape Migration

`GraphMemoryStore.migrate_legacy_shape()` checks the on-disk graph against the expected shape at daemon start-up. The graph is non-conforming if root has any direct child that isn't one of the fixed branches, or if root's own `data` column is non-empty (cold-start writes that landed on root before the taxonomy existed).

What it removes is exactly what traversal can no longer reach: root's own data, and each stray child together with its subtree. Everything filed under `user`, `directives`, or `world` is kept. Root and any missing fixed branch are then re-seeded.

The reason a stray goes is that branch-pinned traversal starts at the fixed branches, so a node hanging directly off root is unreachable forever and carrying it is dead weight. That reasoning covers the stray and nothing else: a correctly-filed fact is reachable, was put there deliberately, and has no business being collateral. Removal is announced on stdout as well as in the debug log, so a graph that shrinks at start-up says so rather than looking like a graph that was never written to.

The diary is untouched either way, so "Import from Diary" in the memory viewer remains the way to re-populate after a genuine loss.

Called **only** from the daemon start-up path in `daemon.main()`. The memory viewer and reply engine instantiate `GraphMemoryStore` without triggering the migration, so a mid-session open never wipes anything.

### Searching Without the Accents

The extraction step works in English and writes "Cafe Rouviere"; he asks
in French, with the accents. `LIKE` compares code points, so without
folding the node is present, the search is correct, and nothing is found
— the quietest kind of miss, since the only consequence is that she looks
it up on the web again.

`name_fold`, `description_fold` and `data_fold` hold the folded text and
are what the search compares. They are filled by trigger rather than by
each writer: there are five write sites and nothing stops a sixth, and a
writer that forgot would leave a node searchable by nothing, which reads
exactly like a node nobody wrote about. A graph written before the
columns existed is migrated and backfilled when it is opened.

Stored rather than computed, and that is the whole point. Wrapping the
columns in the folding function at query time cost one Python call per
column, per keyword, per row, over a full scan: measured on a thousand
nodes, 86 ms at two keywords and 692 ms at sixteen, against 2.2 ms and
13.6 ms once stored. The call sits on the reply path, which is the
listener's own loop — the one draining a 64-frame audio queue at 20 ms a
frame, so 1.28 s before microphone frames start being dropped. A search
must not be able to eat what he is saying.

### Branch-Pinned Traversal

`find_best_node(..., branch_root_id=...)` takes a branch root and descends from it only: recent and top nodes are never consulted, and the branch is not optional. Extraction pins every fact to the World root, so a looked-up fact cannot drift into the retained user subtrees.

## Handing User Knowledge to the Core

`migrate_graph_branches_into_core(store, core)` walks the `user` and `directives` subtrees and writes every stored fact into the core's `profil.md` and `regles.md`, attributed as `migré`. It runs at daemon start-up, before the legacy-shape check that could wipe the table.

A hand-over, not a copy: a node keeps its place in the tree and gives up its data once its facts are in the core. Text left behind would be found by query-driven recall and put a retired belief back into the prompt, and would be rewritten into the core on every subsequent start-up even after the user pruned it. Emptying the source makes the migration idempotent by construction. A node is cleared only when all of its facts reached the core, so a failed write leaves them in the graph rather than losing them.

## Data Model

### MemoryNode

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID string | Unique identifier (root node has id `"root"`) |
| `name` | string | Human-readable label |
| `description` | string | 1-2 sentences used by traversal to decide which branch to explore |
| `data` | string | The actual memories held at this node |
| `parent_id` | UUID or null | Back-reference (null for root) |
| `access_count` | int | Total accesses (for top-nodes ranking) |
| `last_accessed` | ISO 8601 | For recent-nodes ranking |
| `created_at` | ISO 8601 | When the node was created |
| `updated_at` | ISO 8601 | Last modification time |
| `data_token_count` | int | Cached token estimate (len/4 heuristic) |

### Storage

SQLite table `memory_nodes` in the same database as the diary system. Schema is initialised automatically on first access. The root node is created if absent.

### Ranked Queries

| Query | Selects | Used by |
|-------|---------|---------|
| Recent nodes | Last N accessed (excl. root) | The memory viewer's recent list |
| Top nodes | Highest decayed access score (excl. root) | The memory viewer's top list |

## Core Operations

### Create

New nodes are created with a name, description, optional data, and a parent_id (defaults to root). Token count is computed on creation.

### Read

Nodes can be fetched individually, as children of a parent, as a subtree (nested dict), or as graph data (flat nodes + edges for visualisation).

### Update

Any combination of name, description, and data can be updated. Token count is recomputed when data changes. `updated_at` is always refreshed.

### Delete

Any node except root can be deleted. Children are orphaned (parent_id set to NULL via FK). The UI should warn before deleting nodes with children.

### Touch

Increments `access_count` and updates `last_accessed`. Called automatically when a node is viewed in the UI or retrieved during query traversal.

### Mutation Listeners

The graph module exposes a small observer registry, `register_graph_mutation_listener(cb)` / `unregister_graph_mutation_listener(cb)`, invoked after every successful `create_node`, `update_node`, `delete_node`, and (transitively) `append_to_node`. Callbacks receive `action`, `node_id`, and `branch` (the FIXED_BRANCH ancestor id, or `None` for root-level mutations and unresolvable nodes). Listener exceptions are logged via `debug_log` and swallowed so they cannot break a write.

Touch is intentionally NOT a mutation event: it changes access metadata only, not the warm-profile-relevant fields, so it does not need to invalidate caches.

Nothing in the reply layer listens to this hook: the profile injected into the system prompt comes from the core, which has its own write listener (`register_core_mutation_listener` in `core.py`).

### Access Decay

All ordering by access frequency uses a **time-decayed score** computed at query time: `access_count / (1 + age_days / half_life)`. This is hyperbolic decay — a node's effective score halves every `DECAY_HALF_LIFE_DAYS` (default 14) since its last access. The raw `access_count` is never modified, so changing the half-life retroactively reweights all nodes. This applies to `get_top_nodes`, `get_children`, `get_all_nodes`, and `search_nodes` tie-breaking.

### Search

- **search_nodes(query, limit)** — Keyword search across name, description, and data fields. Nodes matching more keywords rank higher. Excludes root. Touches matched nodes for access tracking. Both sides of the LIKE are passed through `fold_for_search`, so matching ignores case and diacritics in both directions: a query typed "rouvière" finds a node the extractor wrote as "Rouviere", and the reverse. The folding is NFKD decomposition with combining marks dropped, so it is a property of Unicode rather than a table of one language's letters. Without it the miss is silent, since the caller simply falls back to searching the web.
- **find_node_by_name(name, parent_id)** — Exact name match (case-insensitive), optionally scoped to a parent node. Excludes root when no parent specified.

## Tree & Graph Queries

- **get_subtree(node_id, max_depth)** — Nested dict for tree sidebar
- **get_ancestors(node_id)** — Path from root to node (breadcrumb)
- **get_graph_data(root_id, max_depth)** — Flat {nodes, edges} for canvas rendering. Each node includes depth and has_children flags.

## Auto-Split (Natural Reduction)

Triggered automatically when `data_token_count > SPLIT_THRESHOLD` (1500 tokens) after a write. Auto-split is the system's primary consolidation mechanism — it's where temporal events get distilled into patterns, filler nobody asked for gets dropped, and the tree structure deepens organically.

1. LLM analyses the node's data and proposes 2-5 child categories
2. Each fact it keeps is assigned to exactly one child
3. **Consolidation**: duplicate facts are merged, and repeated similar activities across different dates are consolidated into patterns (e.g. "ate sushi on Mon, ate sushi on Thu" → "regularly eats sushi"). Date context is preserved only for significant events.
4. **Pruning is narrow, for the reason the merge's is** (LLM Integration > Automatic Writes, step 5, "Pruning is narrow on purpose"). Generic filler nobody asked for goes: stock advice, encyclopaedic padding, how-to steps for everyday tasks, and advice that would fit anyone and names no particular thing. So does a line with the assistant as its subject, checked as each fact is written, in the merge's register. Being in the model's training data is not a reason to drop a line: a fact is on a node because the user asked about it, so a name, a date, a place or a figure about one particular thing stays however well known. A split rewrites every fact of an oversized node at once, so a wider rule here would take back, on the largest rewrite a node ever gets, the answers the merge keeps. The rule is the merge's rule 5 with two examples the merge does not carry, one of advice addressed to anyone and one of a figure that stays: without them a split kept 22 of 150 health-advice lines and lost a nutrition figure in 9 runs in 25.

   Measured with `evals/test_split_pruning.py` on `google/gemini-2.5-flash`. The fixed World branch holds 101 lines: 13 well-known answers (three of them in no class the examples illustrate), 2 practical answers, one sourced recommendation, 72 niche lookups, 10 lines of filler and padding (six unlike the examples) and 3 narration lines (one in French). They are shuffled in five orders, 25 splits a rule, the rules interleaved call by call. Each figure below reads wide rule first, then narrow, the wide rule being one that dropped whatever the model already knew:
   - runs keeping every well-known and practical answer: 4 and 19 of 25 (p = 4e-5);
   - well-known lines kept: 158 and 319 of 325; practical lines kept: 36 and 49 of 50;
   - runs writing no narration: 14 and 25 of 25 (p = 2e-4);
   - filler lines written: 37 and 27 of 250;
   - niche lines kept: 1789 and 1793 of 1800.

   Two cases stay open under both rules:
   - The NHS recommendation of 150 minutes of exercise a week is an answer that reads like the health advice beside it, and is dropped in 23 and 21 runs in 25. A clause keeping "a recommendation that names its source" changed nothing (22 in 25), so it is not in the prompt.
   - The how-to for a cup of tea and the line on what museums display survive in about a third of runs. Listing "a recipe or a technique" among the answers made the tea how-to survive in 13 runs in 25, so that is not in the prompt either.

   On `gemma4:e2b` the split does not hold under either rule (20 a rule). On a node this size the 4096-token window cuts most answers off before the JSON closes, and the narrow rule's longer prompt leaves less room: 12 and 2 answers in 20 parse, and 15 of the narrow rule's 18 failures are cut off. The ones that parse lose about half the niche lines and keep about half the filler. A cut-off split fails safe, since the node keeps its data for the next flush, whereas a split that parses loses what it dropped for good. A wider window without a guard on what a split may drop would turn safe failures into losses.
5. Child nodes are created under the split node
6. Parent data is cleared; parent description updated to a summary

This means the tree depth itself encodes a raw→refined spectrum: surface-level nodes hold recently ingested knowledge, deeper nodes hold knowledge that survived several split cycles, its repetitions folded into patterns. A model that knows more does not shrink the graph: what a model already knows is not what the user asked.

Split quality safeguards:
- Minimum 2 categories required (abort if LLM proposes fewer)
- Each category must have a name, given as text, and at least one fact, and its facts must be a list of strings. A `facts` given as one string, or an item that is not a string, aborts the split: one string would be written one character per line. An item that packs several lines is read as those lines, one fact each, as the node holds them. A category description or a summary that is not text is not written: the child gets `Memories about: <name>` and the parent keeps its description.
- **Retention guard**: the children must carry at least three quarters of the node's claims (`_SPLIT_MIN_RETENTION`, 0.75). Clearing the parent is the one step of a split that cannot be undone, so a claim the children leave out is gone for good. Claims are compared the way the merge compares them: the provenance suffix, case, whitespace and a trailing full stop do not count. A reworded line counts as carried when it scores at least `_REWRITE_MATCH` (85) against a line of the children and that line holds every figure it held: lines worded alike that differ only in a date, a time or a price (one per performance, one per day) are not carried by whichever of them survives. Two lines that differ only by a name do score above the bar against each other, so dropping one of them goes unseen. A claim the node holds twice in the same words counts once, and one line of the children may carry several, so a duplicate folded into a line that scores the bar against both wordings costs nothing; a looser rewording counts as lost. The quarter left over is shared by everything a split may drop: what item 4 prunes, filler, narration, duplicates and repeated activities folded into a pattern. A split that prunes more than that is refused. Calibrated by replaying recorded answers for two World nodes of 90 and 101 lines (8 and 13 of them filler or narration) through the guard. Under the split prompt, whose pruning is as narrow as the merge's, `google/gemini-2.5-flash` passed 75 of 75, carrying at least 84 % of the claims each time. Under a wider prompt that also told it to drop what it already knew, it passed 62 of 75: the 13 refused had dropped most of the well-known answers along with the filler. Every `gemma4:e2b` answer that parsed was refused, 79 of 79, carrying at most 69 %: that tier loses about half the node's niche lookups. Widening its context window from 4096 to 8192 tokens does not change that: interleaved, 25 answers a window, more of them parse (12 against 3) but none carries more than 64 %. On that tier an oversized node therefore does not split. It keeps its data and goes on growing, and is asked again only as it grows (below).
- **Growth guard**: as in the merge, children holding more lines than the node did plus `_MERGE_GROWTH_SLACK` (2) abort the split. A split redistributes and consolidates; lines it adds were invented.
- **Numeric guard**: as in the merge, a digit run in the children that appears nowhere in the node's claims aborts the split; the provenance suffixes' dates do not count. A figure may be dropped, never invented. None of the 229 recorded split answers that parsed tripped it, nor any of the 28 recorded merge answers that parsed.
- **Redaction guard**: as in the merge, a marker a remote model hands back for a value it saw hidden is replaced by the line of the node it stands for, and more redaction markers in the children than the node held after that aborts the split.
- **Hidden values are carried**: every value the redacting backend hides (`hidden_values`: an address, a card-like number, a key) that the node holds must appear in the children, however small a share of the node its line is. A remote model saw it only as a marker, so it could not judge the line, and once the parent is cleared nothing brings the value back.
- If the split fails, the node retains its data. An answer that came back and was refused (no JSON, unparseable JSON, a malformed category, a guard) is remembered with the chat model that gave it and the node's size. A strong model refuses now and then at random (`google/gemini-2.5-flash`, 13 of 75 under the current split prompt), and its next answer usually passes, so one refusal is asked again at the next flush. After `_SPLIT_REFUSALS_BEFORE_BACKOFF` (two) in a row from the same model, that model is not asked to split the node again until the node has grown by `_SPLIT_RETRY_GROWTH` (a quarter) since the last one. On the small tier every split is refused, and asking at every flush would pay each time for a rewrite of the whole node, thrown away; growing by a quarter each time, a node is asked a handful of times per process. Another chat model is asked at once, and so is a node smaller than when it was refused (only an edit shrinks a node past the threshold). A split that got no answer (the deadline turned it away, a timeout, an error, an empty reply) was not refused, and the next flush that finds the node too large asks again. A split that succeeds forgets the refusals. The memory lasts as long as the process, so a restart asks once more.
- The splits of a flush are asked after all of its merges. A split can wait for the next flush, while a merge the deadline turns away drops its facts to plain append, so a split never spends the deadline a merge of the same flush needed.
- The split call runs on the chat model, like the rest of the graph, inside the pass's deadline (see [The Pass's Deadline](#the-passs-deadline) and `docs/llm_contexts.md`, context 11c); one the deadline turns away, or that gets no answer, leaves the node whole for the next flush to retry

## Auto-Merge (Future — requires LLM integration)

When all children collectively hold fewer than 200 tokens:

1. Collapse children's data back into parent
2. Delete child nodes
3. Update parent description
4. Cascade summaries upward

## Housekeeping (Future)

Periodic process that:
- Promotes buried-but-hot nodes (high access, depth > 3)
- Compresses cold branches (no access in > Y days)
- Merges sparse subtrees
- Validates parent summaries

## LLM Integration

The graph memory system is fully automatic — no tool calls required. It integrates at two points in the existing pipeline.

### Automatic Writes (via `graph_ops.py`)

Piggybacks on the existing diary update flow in `conversation.py`:

1. After a successful diary update, the conversation summary is passed to `update_graph_from_dialogue()`
2. **Extract**: LLM pulls external facts the assistant looked up out of the summary. Output is a JSON array of strings. The prompt bans anything about the user in as many words — their identity, tastes, and standing instructions are the user's to state, and a guessed belief about someone would otherwise ride in every future prompt with no visible provenance. Requests are reframed as knowledge ("user asked about CEX hours" → "CEX Kensington closes at 6pm on Sundays"). Patterns and consolidation emerge through auto-split.
3. **Traverse**: Each fact is placed in the best-fitting node using branch-pinned descent from the World branch root. Recent and top nodes are never consulted, so cross-branch contamination is impossible:
   - **Branch descent** — greedy top-down; the LLM picks the best child at each level, or stops at the current node if none fit or it has no children
   - **Model**: the picker, like every LLM call in the graph, runs on the chat model the caller hands to `update_graph_from_dialogue` (`cfg.llm_chat_model`, from the daemon's diary flush and from the memory viewer); there is no separate picker model to configure. Placement looks like a small classification, but a wrong answer is costly and invisible: `NONE` files the fact at the branch root, where no later crawl looks for it, and a small router model answered `NONE` for a cinema fact in front of a node named Cinema. The merge (step 5) is the stronger reason: it rewrites a node in full, and its guards catch growth and invented figures, not a dropped line, so a fact the rewrite loses is lost for good. The picker and the merge never ask their model to think: both run inside the pass's one deadline (see [The Pass's Deadline](#the-passs-deadline)), the picker once per level of every fact and the merge once per node, and time spent thinking would leave facts at the branch root and merges on the plain-append fallback. A missing answer is logged on the `memory` debug channel, apart from a `NONE`.
4. **Dedupe (fast-path)**: Before any fact is placed, it is compared against every line the World branch already holds (`fact_key`: the claim without its provenance suffix, under Unicode-aware folding, `unicodedata.NFKC` + `str.casefold` + whitespace collapse), so ASCII casing, locale quirks (Turkish `İ`/`ı`, German `ß`/`ss`), and incidental whitespace don't cause false negatives. The branch is read once per flush; no model is asked. Exact matches are skipped, **not** reported as newly learned, and do **not** touch any node's access score (a re-extraction isn't fresh reinforcement). Two extracted facts that fold to the same key are one fact. The check runs before the descent, not after it, because the daily summary is cumulative: each flush extracts again what earlier flushes filed, and sent down the picker first those facts would spend the pass's deadline ([The Pass's Deadline](#the-passs-deadline)) and leave the window's new facts to land at the branch root, where an already-known one would also be appended again and reported as learned. Covering the whole branch rather than the node a descent would choose also means a fact filed under one node is not copied under another. The merge step below still collapses near-duplicates, which differ in wording. Skips are counted: `update_graph_from_dialogue` returns a `GraphUpdateResult(stored, skipped, unread)` so the CLI can log "nothing new (N duplicates skipped)" on all-duplicate flushes; silencing that line would make the memory pipeline look broken.
5. **Merge** (batched per node): `merge_node_data(store, node_id, new_facts: list[str], ...)` sends the existing node data + **all** new facts routed to that node in this flush to the chat model and asks it to produce a clean, consolidated, contradiction-free fact list, which is written back as the node's full `data`. The orchestrator groups the flush by `node_id` first so a 5-fact flush against one World node fires **one** rewrite that incorporates all five facts, not five separate rewrites of the same `data`. The call returns a `MergeResult(success: bool, incorporated_indices: list[int])` so the orchestrator can report only the facts that actually survived as new lines (consolidated-out facts aren't claimed as "newly stored"). One LLM call subsumes four behaviours: (a) **supersession** — contradictions, negations, and same-attribute updates drop the old line, **including when both lines carry dates**: rule 1 shows a dated example, because a date is what the diary puts on every entry and what made rule 4's historical carve-out look like it applied. Measured on `gemma4:e2b`, interleaved, 24 draws an arm: the same contradiction superseded 25/25 undated and 8/25 dated, and the only dated case that worked was the one rule 1 already illustrated. Adding a sentence to rule 4 saying a date does not make a line historical changed nothing at all (8/24 either way, p = 1); adding a dated *example* to rule 1 took it to 24/24 (p = 6.5e-7), with two genuinely independent dated events still both kept, 8/8 in both arms. This model follows examples and ignores abstract clauses, which is worth knowing before writing the next rule. The defect belongs to that tier alone: the same four cases run 32/32 on `openai/gpt-oss-120b` both with and without the example, so a large model already supersedes a dated contradiction unaided and the added example costs it nothing. That is what makes the example safe to ship — it repairs the small tier without touching the large one, rather than trading one for the other ("user does not need a daily check-in" replaces both "user has a need for a daily check-in" and the same need framed as an interest); (b) **near-duplicate dedupe** — different wordings of the same fact collapse to one canonical phrasing; (c) **consolidation** — repeated daily activities fold into patterns ("ate sushi on Monday", "ate sushi on Thursday" → "regularly eats sushi"); (d) **meta-narrative pruning** — lines that narrate the assistant's own behaviour, capabilities, or denials ("The assistant is unable to navigate to a web page", "The assistant suggested grilled salmon") are extractor artefacts from earlier prompt versions and get dropped. This one is not a numbered rule but a self-check the prompt asks for while writing each output line, in the register the summariser's equivalent uses: if the line about to be written has the assistant as its subject, do not write it. As the sixth of seven numbered rules it lost 18 runs in 25 on a self-consolidation, measured with the arms interleaved so they shared provider conditions; as a write-time check it won 43 of 43 with no real fact ever dropped. Sequential measurement had reported it healthy, which it is not — arms run one after another against a remote provider are not comparable. Counterpart to the extractor's BANNED FACT FORMS list: the extractor blocks them at write-time, the merge prompt scrubs the historical leftovers that a `consolidate-all` sweep can then surface. Genuine user-issued imperatives ("Always reply in British English") are not meta-narrative and survive. Independent facts coexist (a "user ate a Big Mac" line does not silently drop "user is vegetarian"; the contradiction stays visible). **Pruning is narrow on purpose.** The merge prompt drops generic filler nobody asked for — stock advice, encyclopaedic padding, how-to steps — and explicitly does *not* drop a line for being common knowledge. A fact is on a node because the user asked about it, so an answer to a real question stays even when the model already knew it. The earlier rule dropped "facts that are common knowledge already in your training data", which is the graph's own contents described as a defect. Measured before changing it: common knowledge he had asked about survived under both wordings, 15/15 either way, so the theory that this rule was competing with the subject check is **not** supported. What is supported is that the narrower rule never loses the meta line (45/45 against 41/45, interleaved) and regresses nothing, and that the wider one asked for a behaviour this spec never listed among the four. Because the latest prompt rewrites the whole node (any node under `SPLIT_THRESHOLD`), updated conventions propagate to old data without a separate migration. **Hallucination guard**: the rewrite is rejected if it returns more lines than `len(existing) + len(new) + 2` — a runaway model can't quietly inflate the node. **Numeric guard**: the rewrite is also rejected if it carries a digit run that appears in neither the node nor the new facts. Figures are read from each line's claim, not from its provenance suffix: a node whose lines carry ` · web · 2026-10-16` would otherwise vouch for an invented 16, 10 or 2026 anywhere in the rewrite. The line guard counts lines, so a figure changed inside one passes it by construction — measured on his graph 2026-08-17, a battery life of 19 hours came back out of a merge as 26, with three repeats of the same merge correct, so it is occasional rather than systematic and nothing saw it happen. Dropping a figure stays allowed, since consolidation is meant to shrink; inventing one does not. Numbers are where a rewrite does the most damage per character: a reworded sentence is still roughly the fact, a wrong figure is a different fact wearing the same words, and it is the shape a later answer quotes with confidence. **Redaction guard**: a remote chat model sees the node through the redacting backend, so an address or a long number on the node reaches it as a marker (`[REDACTED_EMAIL]`, `[REDACTED_CARD]`, …) and comes back that way. Before the guards run, each line of the rewrite that carries a marker and makes exactly the scrubbed claim of an input line (the same key) is replaced by that input line, value and all; lines that differ only by their provenance suffix count as one. A line the model reworded is left as it came, since putting the old line back would undo its edit and drop any fact it folded in; so is a marker that matches no input line, or input lines that hid different values. The rewrite is then rejected if it carries more markers than the node and the new facts did: written over the node, the value it hid would be gone. A marker the inputs already held covers no new one this way: an input line copied back makes its own claim, and a redacted copy of a line whose value the node held gets the value back. Markers are counted rather than detected, so one the node already held (diary text is redacted before it is summarised) neither blocks the rewrite nor covers a second one. **A node past `SPLIT_THRESHOLD` is not merged**: its new facts are appended, and reorganising it is the split's job (Auto-Split). The merge rewrites the whole node, and past the threshold that rewrite has nothing to stand on. Measured on `gemma4:e2b` at Ollama's 4096-token window, interleaved, 20 merges a size: on a 101-line node (a 3,300-token prompt) none came back whole, and on a 152-line node (4,400 tokens, past the window) the start of the prompt was cut away with the instruction to answer in JSON, and none parsed either (about 8 s and 16 s a call, every one wasted). Given an 8192-token window, 7 of 20 were accepted, keeping as little as 62 % of the node and as few as 43 of its 72 niche lookups, and the merge has no guard that counts lost lines. `google/gemini-2.5-flash` merged the 101-line node well (19 of 19, at least 67 of 72 niche lookups), but there the split follows in the same flush, and it consolidates as it distributes. The 🧹 sweep leaves such a node as it is. Fail-open: empty/cold node, LLM error, parse failure, oversized rewrite, unexplained marker, or an empty rewrite all fall back to plain `append_to_node` for each new fact so they still land — a contradiction is recoverable, a silent wipe or hallucinated bloat is not.
6. **Split**: Once every merge of the flush is done, each node the merge or fallback append left past `SPLIT_THRESHOLD` is split, unless the same chat model has refused its split twice in a row and it has not grown enough since (Auto-Split, Split quality safeguards)

Cold start: each fact lands directly on the World branch root until enough data accumulates there for the first auto-split. The tree structure emerges organically beneath it.

LLM failure at any step is non-fatal — the diary update still succeeds, and the graph simply misses that cycle.

### The Pass's Deadline

One graph pass has one deadline, a `GraphDeadline`, and every call in it shares it: the extraction, each level of each placement, each merge, each split. A call is handed what the deadline has left. A slow chat model (a large local one, a provider that takes 12 s an answer) therefore gets as long as the pass can give, rather than a figure sized for a small router model, which would file every fact at the branch root and merge nothing. The pass waits on each call itself instead of trusting the HTTP timeout, which bounds the gap between two bytes rather than the call: a provider that keeps the connection open cannot hold the pass past its deadline. A call left behind runs out on its own thread and nothing it brings back is read.

Once the deadline is spent, nothing more is asked, and the pass finishes on the paths that need no model, the ones a call that got no answer already takes: a fact stays where its descent got to (the World root, for one not yet placed), a node takes a plain append, a split waits for the next flush that finds the node still too large. A fact once extracted is filed less well, never lost. The extraction is the exception: a pass with no time left for it has read nothing, and says so (`GraphUpdateResult.unread`) rather than reporting an empty answer.

| Caller | Deadline |
|--------|----------|
| Daemon diary flush, periodic | `llm_chat_timeout_sec` from the start of the diary pass, shared with the summary, which is asked first; ends early on a stop request |
| Daemon diary flush, at shutdown | `SHUTDOWN_DIARY_TIMEOUT_SEC` (45 s) from the start of the diary pass, shared with the summary the same way |
| Memory viewer, Import from Diary | one `llm_chat_timeout_sec` per summary |
| Memory viewer, 🧹 | one `llm_chat_timeout_sec` per node; the sweep as a whole is unbounded, since it is asked for and streams its progress. It shares the chat model with replies all the same |

**Shutdown.** The desktop app gives a daemon subprocess 60 s to stop and then kills it, and a kill skips everything the daemon does after the diary pass: stopping the speaker, closing the MCP sessions, closing the database. So the shutdown pass gives the graph what the summary leaves of `SHUTDOWN_DIARY_TIMEOUT_SEC`, and the graph never makes that pass outlast it. It is not skipped: the next flush re-extracts the day's cumulative summary only when its own window held a lookup ([provenance.spec.md](provenance.spec.md)), so a skipped pass would lose what the last window looked up for good, where a short one usually files it. The stop request is why that pass runs, so it does not end it.

A periodic pass still running when a stop is requested (`request_stop`: the desktop app's stop in bundled mode, the daemon's stdin closing in subprocess mode) gives way: it asks nothing more, leaves the call in flight behind within a fifth of a second, and files what it already extracted without the model, so quitting never waits for the graph. The daemon's diary checks run on its main loop, the same thread that runs shutdown, which is why a pass that kept going would hold it. A pass the stop reached before its extraction answered (a stop during the summary, or during the extraction itself) has read nothing, and its window is already marked saved, so the shutdown pass would find nothing pending. The window is therefore recorded on the dialogue memory (`DialogueMemory.owe_graph_pass`, with its date, source and the tools that vouched for its lookups) and the shutdown pass reads that day's summary with what is left of its budget (`settle_owed_graph_pass`).

**What it does not bound.** The summary and its embedding keep their own timeouts, and the stop request does not interrupt them. At shutdown the summary always streams (to the diary dialog or the IPC events), so its timeout bounds the gap between two tokens rather than the call, and the embedding after it has 15 s of its own: a summary that overruns leaves the graph no time, and the pass as a whole can run past `SHUTDOWN_DIARY_TIMEOUT_SEC` by that much. A signal (SIGINT, which the desktop app sends a daemon subprocess off Windows) does not go through the stop request: it raises `KeyboardInterrupt` in the main loop and ends the pass where it is, without the paths above. A call that hangs spends the rest of the pass, and what comes after it is filed without the model.

### Automatic Reads (via enrichment in `engine.py`)

At the start of each reply cycle, the reply engine enriches the system prompt with graph context:

1. **Subject-driven**: the crawl keys off whatever names the subject of the query. Two sources feed it. The extractor's `keywords`, when the extractor ran. And the concrete arguments the planner composed into its own tool steps (`lookup_terms_of`), which name the subject with pronouns already resolved to literal entities, and which exist on a turn where no memory was planned at all. Extracted `questions` join the search text when present but cannot gate it: they are defined as implicit questions *about the user*, and the user lives in the core files, not here.
2. **Not gated on the extractor.** The graph is a SQLite `LIKE` scan, no LLM and no embedding. Reaching it through an LLM call would trade a network round trip for a local one and call it a saving, so a plan that named a lookup opens the graph directly. The recall gate shuts the extractor and the diary; it does not shut the graph.
3. **Search**: keywords, questions and plan terms are joined, stop-worded, and used to find matching nodes (up to 5 results with data previews). Each node reaches the prompt within `_GRAPH_PREVIEW_CHARS` (300) characters. A node that fits is shown whole; a larger one is shown by its lines that name the most search words, most first, as many as fit (`node_preview`). The search matches a node's whole text, and a node holds up to `SPLIT_THRESHOLD` tokens (some 6,000 characters, twenty times the budget) before it splits, more when its split keeps failing, so the line that answers can sit anywhere and its first lines are no more likely than any others to be it. Lines are matched the way the search matches them (folded for case and diacritics, a word anywhere in the line). When the best line alone is longer than the budget, it is cut to the budget rather than passed over for a weaker one. A node found by its name or description alone, with no line holding a word, is shown by its first `_GRAPH_PREVIEW_CHARS` characters. Fewer than two content words is treated as too thin to search — one generic term against a LIKE match surfaces noise, and noise in the prompt costs more than a search she pays for again.
4. **Skipped entirely** when none of the three produced anything. A utility query (the time, a sum) has nothing to look up.
5. Results are injected as what she has kept from earlier conversations, explicitly framed as describing the world rather than the user, and explicitly losing to the core sections above on any conflict. The envelope claims no provenance of its own: each line carries its own (`· web` / `· outil` / `· inconnu`, see [provenance.spec.md](provenance.spec.md)) and the envelope explains only the markers actually present in the block. A line with no marker is named as one nobody recorded a source for, and the model is told not to present it as something it established.

No tool calls needed. The LLM sees relevant graph memories as part of its system context.

Controlled by `memory_enrichment_source` config:
- `"all"` — both diary and graph enrich replies
- `"diary"` — only diary (conversation summaries) used for enrichment
- `"graph"` — only graph (structured knowledge) used for enrichment

Default is `"all"` — both channels enrich replies. Both systems always receive writes regardless of this setting.

Note: the always-on core profile, injected on every turn, is separate from query-driven enrichment. The core covers "who the user is"; enrichment covers "what the user has said or seen about this specific topic". Only enrichment reads the graph.

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `SPLIT_THRESHOLD` | 1500 | Tokens before auto-split |
| `RECENT_NODES_COUNT` | 10 | Recent nodes to surface |
| `TOP_NODES_COUNT` | 15 | Top nodes to surface |
| `DECAY_HALF_LIFE_DAYS` | 14 | Days until a node's access score halves |
| `MAX_TRAVERSAL_DEPTH` | 8 | Safety limit on graph traversal |
| `SUMMARY_MAX_LENGTH` | 300 | Max chars for node description |
| `memory_enrichment_source` | `"all"` | Which system enriches replies: `"all"`, `"diary"`, or `"graph"` |

## UI: Memory Viewer Integration

The graph explorer appears as the **Knowledge** tab in the memory viewer, positioned between the Diary and Meals tabs.

### Three-Panel Layout

1. **Left sidebar — Tree navigator**: Collapsible tree showing the full hierarchy. Clicking a node selects it in both the tree and the graph canvas. Shows child count badges.

2. **Centre — Graph canvas**: Interactive HTML5 Canvas with radial tree layout. Supports pan (drag), zoom (scroll wheel), and click-to-select. Toolbar provides zoom in/out, fit-to-view, add-node, import-from-diary and consolidate (🧹) actions. Node size reflects access count. Selected node is highlighted with accent glow.

3. **Right sidebar — Node detail**: Shows breadcrumb path, name, description, metadata (accesses, tokens, last seen, children count), stored data, children list, and action buttons (edit, add child, delete).

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/graph/nodes` | Graph data (nodes + edges) for canvas |
| GET | `/api/graph/tree` | Nested tree structure for sidebar |
| GET | `/api/graph/node/<id>` | Single node + children + ancestors |
| POST | `/api/graph/node` | Create new node |
| PUT | `/api/graph/node/<id>` | Update node fields |
| DELETE | `/api/graph/node/<id>` | Delete node (not root) |
| GET | `/api/graph/recent` | Recently accessed nodes |
| GET | `/api/graph/top` | Most frequently accessed nodes |
| GET | `/api/graph/stats` | Node count and total data tokens (`total_tokens = 0` means the graph holds no knowledge) |
| POST | `/api/graph/import-diary` | Import all diary summaries into graph (streaming NDJSON) |
| POST | `/api/graph/consolidate-all` | Self-consolidate every populated node of the World branch under `SPLIT_THRESHOLD` (streaming NDJSON) — runs the merge LLM with no new facts on each such node so updated conventions and supersession rules apply to historical data |

### Import from Diary

The graph toolbar includes an "Import from Diary" button (📥) that bootstraps the graph with existing diary data. This is a one-time migration path so users don't lose their accumulated memories when switching from diary-only to graph enrichment.

The endpoint streams NDJSON progress events (`start`, `progress`, `complete`, `error`) so the UI shows real-time feedback. Each diary summary is processed through the standard `update_graph_from_dialogue()` pipeline (extract → traverse → append → split), with one `llm_chat_timeout_sec` of its own as its deadline. Failures on individual summaries are non-fatal — the import continues with the remaining entries.

### Consolidate All (🧹)

The toolbar's 🧹 button walks every populated node of the World branch, its root included, and calls `merge_node_data` with an empty `new_facts` list (a node past `SPLIT_THRESHOLD` is left as it is, without a call: it is the split's to reorganise, see Merge), prompting the chat model to re-apply the latest supersession/dedupe/consolidation rules to data that landed before those rules existed (or before the prompt was tightened). With the chat model on a remote provider, one sweep sends every one of those nodes under the threshold there, and the modal says so. Most of it is world facts the chat model already saw when it extracted them, but not all: nodes written while the chat model was local, and anything typed into a node by hand in the viewer, leave the machine for the first time. Like Import from Diary, it streams NDJSON progress events. Each node's rewrite gets one `llm_chat_timeout_sec`, so a chat model slower than a small router model still gets to consolidate. Per-node failures are non-fatal so a single bad node can't abort the sweep. The UI confirms before starting and reports the total line-count delta on completion.

**World only.** The User and Directives subtrees belong to the core (`core.spec.md`), which is written only when the user asks or corrects. A node still holding data there is one whose hand-over has not finished: rewriting it would hand the core a model's wording of the user, attributed `migré`, and would first send the user's profile and rules to whichever provider runs the chat model. The walk never crosses into another fixed branch, so a core branch re-parented under World is still not swept, and a node cut off from World by the deletion of its parent is not reached. `populated_world_nodes()` is the one definition of the walk, so the viewer's up-front node count, which drives the progress bar, is the set the sweep visits. Each node's data is read again when the sweep reaches it rather than taken from that snapshot: a sweep runs for minutes, and the daemon flushes into the same file meanwhile, so a rewrite from the snapshot would overwrite what it filed since.

**Each line keeps its source.** The merge puts every suffix back from the line's claim before it writes, the same repair a flush gets, and ignores whatever suffix the model wrote (see [provenance.spec.md](provenance.spec.md)). A line the sweep kept or only retypeset keeps its source and date; a line it reworded, or a sentence it folded two claims into, matches nothing and reads as `inconnu`, which the modal says.

**The modal describes the prompt.** The button's title and the confirmation modal list what the merge prompt does: merge near-duplicates, replace an outdated value, fold repeated activities into patterns, drop filler nobody asked for and lines narrating the assistant. They do not promise to prune common knowledge, because the prompt keeps an answer to a real question however well known (see **Pruning is narrow on purpose** under Merge).

## Relationship to Existing Systems

The graph memory system lives alongside the existing diary system (conversation_summaries + FTS + vector search). It shares the same SQLite database but uses its own table. The diary system remains the primary memory system for now; the graph is a v2 system being built in parallel.

Users can import existing diary data into the graph via the "Import from Diary" button in the Memory Viewer. This processes all historical summaries through the extract-and-place pipeline, building the graph structure organically.

### Diary Summariser Hygiene

Graph extraction ingests diary summaries, so the graph inherits whatever corruption the summary contains. Summariser hygiene rules (no deflection narration, attribution preservation, topic separation) are documented in [`summariser.spec.md`](summariser.spec.md).

## Privacy

All data is stored locally in the user's SQLite database, and the graph store has no network dependencies. The graph's LLM calls (extraction, placement, merge, split, and the 🧹 sweep) go to whichever provider runs the chat model, so with a remote chat model the facts they carry leave the device; see `docs/llm_contexts.md` #10 to #11c.

The store is closed after every enrichment read. A read that fails is reported to the user as well as logged: a graph that has silently stopped answering must not be indistinguishable from an empty one.

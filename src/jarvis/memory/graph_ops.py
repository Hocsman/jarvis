"""
🧠 Knowledge Graph Operations — LLM-dependent graph logic.

Keeps graph.py as a pure data store (SQLite only). This module handles:
- Knowledge extraction from conversation summaries
- Best-node traversal (greedy descent via recent → top → root entry points)
- Auto-split when a node exceeds the token threshold

All LLM calls go through the configured backend (factory dispatch on
``cfg.llm_provider``); the local ``call_llm_direct`` wrapper is the
single intercept point tests patch.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Iterator, NamedTuple, Optional, Sequence

from ..debug import debug_log
from ..llm import get_llm_backend
from .provenance import fact_line, fact_text, source_for_tools
from .core import (
    SECTION_PROFILE,
    SECTION_RULES,
    SOURCE_MIGRATED,
    MemoryCore,
)
from .graph import (
    BRANCH_DIRECTIVES,
    BRANCH_USER,
    BRANCH_WORLD,
    FIXED_BRANCHES,
    GraphMemoryStore,
    MAX_TRAVERSAL_DEPTH,
    MemoryNode,
    SPLIT_THRESHOLD,
    normalise_fact,
)


def call_llm_direct(*, cfg, chat_model, system_prompt, user_content,
                    timeout_sec=10.0, thinking=False, num_ctx=4096,
                    temperature=None):
    """Local indirection: route graph-ops LLM calls through the backend
    configured by ``cfg.llm_provider``. Tests patch this single symbol
    to intercept every LLM round-trip in this module."""
    return get_llm_backend(cfg).direct(
        chat_model, system_prompt, user_content,
        timeout_sec=timeout_sec, thinking=thinking,
        num_ctx=num_ctx, temperature=temperature,
    )


# ── Memory extraction from dialogue ───────────────────────────────────


def extract_graph_memories(
    summary: str,
    cfg,
    chat_model: str,
    tools_used: Optional[Sequence[str]],
    timeout_sec: float = 30.0,
    thinking: bool = False,
    date_utc: Optional[str] = None,
) -> list[str]:
    """Extract external facts the assistant looked up during a conversation.

    Everything returned lands in the World branch. Facts about the user
    and standing instructions are not extracted here at all: they belong
    to the core, which is written only when the user asks for it or
    corrects something (see ``core.spec.md``). Inferring them from a
    summary is how a wrong belief gets into every future prompt without
    the user ever being able to see where it came from.

    Returns an empty list if nothing novel was found.

    Args:
        tools_used: The tools that ran in the window this summary covers,
            or ``None`` when the caller cannot establish them. An empty
            sequence means no tool ran, and a window with no tool holds no
            lookups, so nothing is extracted and the model is not asked —
            filtering afterwards would still pay for the call and would
            leave the decision to the same judgement that fails on a
            confidently invented statistic. See ``provenance.spec.md``.
        date_utc: Optional date string (YYYY-MM-DD) for the diary entry.
            Included as a date prefix on each fact for temporal context.
    """
    if tools_used is not None and not tools_used:
        debug_log("graph memory extraction: no tool ran in this window, "
                  "nothing was looked up", "memory")
        return []
    system_prompt = (
        "You extract EXTERNAL FACTS the assistant looked up during a "
        "conversation. Each fact must be a self-contained statement worth "
        "recalling in a future conversation: films, books, businesses, "
        "recipes, techniques, named entities, post-cutoff events, "
        "corrections to wrong assumptions. Write each as a direct factual "
        "statement. Examples: 'Trenches Boxing Club in Hackney offers "
        "evening classes', 'Possessor (2020) is a sci-fi horror film "
        "directed by Brandon Cronenberg', 'A soy-oyster-teriyaki glaze "
        "works well for air-fried chicken breast'.\n\n"
        "NEVER EXTRACT ANYTHING ABOUT THE USER. Not who they are, where "
        "they live, their relationships, tastes, preferences, habits, "
        "plans, or opinions. Not instructions they gave the assistant "
        "about how to behave, answer, or write. The user states those "
        "themselves and they are stored elsewhere; guessing at them from "
        "a conversation is how a wrong belief about someone becomes "
        "permanent. If a summary says the user is vegetarian, wants "
        "shorter replies, or lives in Lyon, extract NONE of it.\n\n"
        "DO NOT EXTRACT — these are NEVER knowledge, no exceptions:\n"
        "- ASSISTANT-GENERATED RECOMMENDATIONS, ADVICE, OR SUGGESTIONS. "
        "If the assistant 'recommended X', 'suggested Y', 'advised Z' "
        "from its own priors, NONE of X / Y / Z is a fact — they are "
        "the assistant's own opinions and will be regenerated next "
        "time. Distinct from this: an EXTERNAL LOOKUP the assistant "
        "performed (a film's release year, a restaurant's address, a "
        "post-cutoff event) IS a fact, because the assistant looked it "
        "up rather than generating it. Heuristic: would a different "
        "assistant on a different day produce the same answer? If yes, "
        "it's a lookup → extract. If no, it's a recommendation → drop.\n"
        "- TRANSIENT SNAPSHOTS that go stale within hours: the current "
        "weather, the current temperature, today's wind / cloud / "
        "humidity readings, the current time of day, what day of the "
        "week it is right now. Even if the conversation contains them, "
        "they are NOT knowledge — they describe a moment, not a fact. "
        "(A persistent climate fact like 'London has mild winters' is "
        "fine; '20°C and partly cloudy in London right now' is not.)\n"
        "- Common knowledge you already have.\n"
        "- Vague, content-free statements ('user explored options').\n"
        "- Pure meta-interaction (greetings, thank-yous, requests for "
        "a recap).\n\n"
        "MIXED SUMMARIES: a summary may interleave lookups with user "
        "facts, assistant recommendations and current weather. Drop the "
        "banned content and keep ALL the lookups in the same summary — "
        "never emit `[]` just because part of the summary was banned. "
        "Example: 'It is 22°C in Hackney right now. The user adopted a "
        "cat named Miso. Miso is a Japanese word for fermented soybean "
        "paste.' → extract only the meaning of the word, drop the "
        "weather and drop the user's cat.\n\n"
        "BANNED FACT FORMS — never emit a fact whose text matches any "
        "of these:\n"
        "- ANY sentence that starts with 'The assistant ...' or 'I ...' "
        "(the assistant). This includes every verb: said, told, "
        "suggested, recommended, advised, proposed, provided, offered, "
        "answered, replied, mentioned, noted, explained, gave, etc. "
        "Meta-narrative about what the assistant did is never a fact — "
        "the underlying lookup, if any, is the fact, not the act of "
        "saying it.\n"
        "- ANY sentence about the user: 'The user is / lives / likes / "
        "wants / asked / prefers ...'. If the subject is the user, it "
        "does not belong here.\n"
        "- ANY fact about current weather, temperature, sky condition, "
        "wind, cloud cover, humidity, time of day, or day of the week. "
        "This applies whether the place is named or not, and whether "
        "the temperature is 5°C or 30°C: 'The weather in Hackney is "
        "22 degrees and sunny', 'It is 20°C in London', 'The temperature "
        "is 22 degrees', 'It is partly cloudy', 'Wind is from the "
        "southwest at 15 km/h', 'It is currently 3:45 PM on a Sunday'. "
        "These describe a moment, not knowledge — they are stale within "
        "hours and must NEVER be extracted, even when the surrounding "
        "summary contains other novel facts.\n"
        "If the underlying lookup was a real external fact, rephrase "
        "without attribution: 'Possessor (2020) is directed by Brandon "
        "Cronenberg', not 'the assistant said Possessor is...'.\n\n"
        "Write facts as KNOWLEDGE, not as interaction descriptions:\n"
        "Wrong: 'User asked about boxing gyms'\n"
        "Right: 'Trenches Boxing Club in Hackney has evening classes'\n\n"
        "Respond with ONLY a JSON array of strings. If nothing novel was "
        "learned, respond with `[]`.\n"
        "Example:\n"
        '["Trenches Boxing Club in Hackney offers evening classes",\n'
        ' "Possessor (2020) is directed by Brandon Cronenberg"]'
    )

    # Include date so each fact carries temporal context
    date_prefix = f"(Date: {date_utc}) " if date_utc else ""
    user_content = (
        f"Extract novel external knowledge from this conversation "
        f"summary:\n{date_prefix}{summary}"
    )

    debug_log(f"graph memory extraction: sending {len(summary)} chars to {chat_model}", "memory")

    # Knowledge extraction is a rule-following classification task —
    # determinism beats creativity here. Ollama's default ~0.8 makes
    # small models flake on the banned-form list (sometimes obeying,
    # sometimes drifting back into meta-narrative or stale-snapshot
    # extraction); temperature=0 lets the prompt do its job consistently.
    response = call_llm_direct(
        cfg=cfg,
        chat_model=chat_model,
        system_prompt=system_prompt,
        user_content=user_content,
        timeout_sec=timeout_sec,
        thinking=thinking,
        temperature=0.0,
    )

    if not response:
        debug_log("graph memory extraction: LLM returned no response", "memory")
        return []

    debug_log(f"graph memory extraction: got response ({len(response)} chars)", "memory")

    # Parse JSON array from the response
    json_match = re.search(r'\[.*\]', response, re.DOTALL)
    if not json_match:
        debug_log(f"graph memory extraction: no JSON array found in response: {response[:200]}", "memory")
        return []

    try:
        parsed = json.loads(json_match.group())
        if not isinstance(parsed, list):
            debug_log(f"graph memory extraction: parsed JSON is not a list: {type(parsed)}", "memory")
            return []
    except (json.JSONDecodeError, ValueError) as e:
        debug_log(f"graph memory extraction: JSON parse failed — {e}", "memory")
        return []

    facts: list[str] = []
    for item in parsed:
        # Small models drift towards the object shape they have seen in
        # other extraction prompts, so a `{"fact": "..."}` wrapper is read
        # rather than discarded. Model output is a boundary we don't own.
        if isinstance(item, dict):
            item = item.get("fact") or item.get("text") or ""
        if not isinstance(item, str):
            continue
        fact_text = item.strip()
        if fact_text:
            facts.append(fact_text)

    debug_log(f"graph memory extraction: got {len(facts)} facts", "memory")
    return facts


# ── Best-node traversal ───────────────────────────────────────────────


def _llm_pick_best_child(
    fragment: str,
    children: list[MemoryNode],
    cfg,
    chat_model: str,
    timeout_sec: float = 15.0,
    thinking: bool = False,
    picker_model: Optional[str] = None,
) -> Optional[str]:
    """Ask the LLM which child node best fits a memory fragment.

    Returns the chosen child's id, or None if none fit well.
    """
    if not children:
        return None

    options = []
    for i, child in enumerate(children, 1):
        options.append(f"{i}. {child.name}: {child.description}")
    options_text = "\n".join(options)

    system_prompt = (
        "You are a memory organiser. Given a fact to store and a list of "
        "category nodes, pick the single best-fitting category.\n"
        "If NONE of the categories fit well, respond with NONE.\n"
        "Respond with ONLY the number (1, 2, ...) or NONE. Nothing else."
    )
    user_content = (
        f"Fact to store: {fragment}\n\n"
        f"Categories:\n{options_text}"
    )

    # Picker is a one-digit classification — reuse the small picker_model
    # when the caller provides one (resolved from intent_judge_model → chat_model).
    # Falls back to the chat model when no small model is configured.
    effective_model = picker_model or chat_model
    response = call_llm_direct(
        cfg=cfg,
        chat_model=effective_model,
        system_prompt=system_prompt,
        user_content=user_content,
        timeout_sec=timeout_sec,
        thinking=thinking,
    )

    if not response:
        return None

    response = response.strip().upper()
    if "NONE" in response:
        return None

    # Extract a number
    num_match = re.search(r'(\d+)', response)
    if num_match:
        idx = int(num_match.group(1)) - 1
        if 0 <= idx < len(children):
            return children[idx].id

    return None


def find_best_node(
    store: GraphMemoryStore,
    fragment: str,
    cfg,
    chat_model: str,
    timeout_sec: float = 15.0,
    thinking: bool = False,
    picker_model: Optional[str] = None,
    branch_root_id: Optional[str] = None,
) -> str:
    """Find the best node to store a memory fragment.

    When ``branch_root_id`` is provided (one of the fixed taxonomy
    branches — User / Directives / World), the shortcut entry points
    (recent / top) are skipped entirely and traversal descends only
    through that branch's subtree. This guarantees the purpose-shaped
    top-level taxonomy is respected — a User fact can never end up in
    the World subtree just because a World node happened to be
    recently accessed.

    When ``branch_root_id`` is None (legacy callers), the old three-
    entry-point heuristic is used:

    1. Recent nodes — check if fragment fits a recently accessed node
    2. Top nodes — check frequently accessed domains
    3. Root traversal — greedy top-down descent from root

    Returns the id of the best node.
    """
    debug_log(
        f"graph traversal: placing '{fragment[:60]}...' "
        f"(branch={branch_root_id or 'any'})",
        "memory",
    )

    if branch_root_id is None:
        # Entry point 1: Check recent nodes
        recent = store.get_recent_nodes(limit=5)
        if recent:
            debug_log(f"graph traversal: trying {len(recent)} recent nodes: {[n.name for n in recent]}", "memory")
            best = _llm_pick_best_child(
                fragment, recent, cfg, chat_model,
                timeout_sec=timeout_sec, thinking=thinking, picker_model=picker_model,
            )
            if best is not None:
                matched = store.get_node(best)
                name = matched.name if matched else best[:8]
                debug_log(f"graph traversal: matched recent node '{name}'", "memory")
                return best

        # Entry point 2: Check top nodes (excluding any already checked as recent)
        recent_ids = {n.id for n in recent} if recent else set()
        top = [n for n in store.get_top_nodes(limit=10) if n.id not in recent_ids]
        if top:
            debug_log(f"graph traversal: trying {len(top)} top nodes: {[n.name for n in top]}", "memory")
            best = _llm_pick_best_child(
                fragment, top, cfg, chat_model,
                timeout_sec=timeout_sec, thinking=thinking, picker_model=picker_model,
            )
            if best is not None:
                matched = store.get_node(best)
                name = matched.name if matched else best[:8]
                debug_log(f"graph traversal: matched top node '{name}'", "memory")
                return best

    # Entry point 3 (or sole entry point when branch is pinned):
    # greedy descent from the branch root (or root when no branch).
    start_id = branch_root_id or "root"
    debug_log(f"graph traversal: descending from '{start_id}'", "memory")
    current_id = start_id
    depth = 0
    for depth in range(MAX_TRAVERSAL_DEPTH):
        children = store.get_children(current_id)
        if not children:
            debug_log(f"graph traversal: leaf node at depth {depth}", "memory")
            break  # Leaf node — write here

        debug_log(f"graph traversal: depth {depth}, choosing from {[c.name for c in children]}", "memory")
        best = _llm_pick_best_child(
            fragment, children, cfg, chat_model,
            timeout_sec=timeout_sec, thinking=thinking, picker_model=picker_model,
        )
        if best is None:
            debug_log(f"graph traversal: no children fit at depth {depth}, stopping", "memory")
            break  # None of the children fit — write to current node
        matched = store.get_node(best)
        name = matched.name if matched else best[:8]
        debug_log(f"graph traversal: descended into '{name}'", "memory")
        current_id = best

    final = store.get_node(current_id)
    final_name = final.name if final else current_id[:8]
    debug_log(f"graph traversal: writing to '{final_name}' (depth {depth})", "memory")
    return current_id


# ── Merge (combine existing node data + new fact via LLM rewrite) ─────


_MERGE_SYSTEM_PROMPT = (
    "You curate a knowledge store. You are given the CURRENT facts on a node, and usually a NEW fact to incorporate. Write out the REVISED set of facts that replaces the node's contents, deciding line by line what you write. When no new fact is given, the job is the same rewrite with nothing to incorporate.\n"
    '\n'
    'Apply these rules in order:\n'
    "1. CONTRADICTION / REVERSAL: if the new fact contradicts, negates, or updates the current value of the same attribute as an existing fact, drop the old version. Examples: 'User dislikes coffee' replaces 'User likes coffee'. 'User lives in Berlin' replaces 'User lives in Hackney'. '[2026-03-15] User's phone is a Pixel 9' replaces '[2025-12-01] User's phone is a Pixel 7' — the two dates do not make two facts, they make one fact and its earlier value. 'User does not need a daily check-in' replaces 'User has a need for a daily check-in' AND any line that lists that same need as an interest.\n"
    '2. DUPLICATION: drop existing lines that say the same thing as the new fact, even with different wording, casing, or punctuation. Keep one canonical phrasing.\n'
    "3. CONSOLIDATION: when several existing facts describe the same repeated activity across different days (e.g. 'ate sushi on Monday', 'ate sushi on Thursday'), merge them into a pattern ('regularly eats sushi'). Preserve dates only for significant one-off events (a job change, a move).\n"
    "4. INDEPENDENCE: keep existing facts that describe a different attribute, even if related. 'User ate a Big Mac' does NOT replace 'User is vegetarian' — leave both visible so the inconsistency stays inspectable. Past-tense historical events ('Visited Paris in 2023') coexist with current-state facts.\n"
    "5. PRUNING: drop generic filler nobody asked for — stock advice, encyclopaedic padding, how-to steps for everyday tasks. Being already in your training data is not by itself a reason to drop a line: these facts are on this node because the user asked about them, so an answer to a real question stays even when you already knew it ('Dune (2021) is directed by Denis Villeneuve' stays). This rule and the subject check below never point at the same line: narration about the assistant goes however novel it looks, an answer the user asked for stays however obvious it looks. If the two lines left in front of you are one of each, drop the narration and keep the answer.\n"
    '6. ORDER: keep the most enduring, identity-defining facts near the top; transient / specific facts below.\n'
    '\n'
    "Then, as you write each line of the revised set, check it as you write it — whose sentence is this? If the line you are about to write has the assistant as its SUBJECT ('The assistant suggested grilled salmon', 'The assistant is unable to navigate to a web page', 'The assistant said / recommended / advised ...', 'I (the assistant) ...', or that same shape in any other language), do not write it. Just omit it. The revised set is shorter — that is correct. The verb does not matter; the subject is the tell.\n"
    '\n'
    "The check fires on the subject and nothing else: a line whose subject is the user, a person, a place, a film, a product or anything else in the world gets written, even if the assistant once looked it up; a line addressed AT the assistant in the imperative ('Always reply in British English') is a directive, not narration, and gets written.\n"
    '\n'
    'Respond with ONLY a JSON object of shape `{"facts": ["fact 1", "fact 2", ...]}`. Each fact is a self-contained sentence. No prose outside the JSON, no explanations, no markdown fences.'
)


def _split_data_lines(data: Optional[str]) -> list[str]:
    """Split node data into non-empty, stripped lines.

    Single source of truth for how the merge pipeline tokenises a
    node's `data` blob into facts — the merge body, the
    consolidate-all walk, and the boundary test all use this so a
    future change to the parsing rule (e.g. `splitlines()`,
    blank-line handling) propagates atomically.
    """
    return [l for l in (data or "").split("\n") if l.strip()]


def is_populated_node(node: MemoryNode) -> bool:
    """A node worth visiting in a consolidate-all sweep.

    Shared predicate so the streaming endpoint can pre-count nodes
    using the same definition the generator walks with — drift here
    would silently desynchronise the UI's progress bar from reality.
    """
    return node.id != "root" and bool((node.data or "").strip())


# Slack added to the hallucination-guard cap. Consolidation should
# only ever shrink or hold, but we allow a small overrun (e.g. the
# model splitting a clumsy two-clause fact into two cleaner lines)
# before treating the rewrite as runaway invention.
_MERGE_GROWTH_SLACK = 2


@dataclass
class MergeResult:
    """Outcome of a `merge_node_data` call.

    `success` — True when the rewrite passed all guards and was
    persisted. False means the caller should fall back to plain
    append for any non-incorporated facts.

    `incorporated_indices` — for each input position in `new_facts`,
    True if the cleaned output contains that fact under
    `normalise_fact` folding (so it's safe to consider it landed in
    the node). A fact whose index is missing was either consolidated
    out, dropped as common knowledge, or silently lost — caller
    decides whether to append it as a fallback or skip.
    """

    success: bool
    incorporated_indices: list[int] = field(default_factory=list)


_JSON_DECODER = json.JSONDecoder()


def _extract_facts_object(response: str) -> Optional[dict]:
    """Pull a `{"facts": [...]}` object out of an LLM response.

    Tries direct `json.loads` first (the strict prompt + T=0 should
    produce clean JSON in the common case). Otherwise scans every `{`
    and uses ``json.JSONDecoder.raw_decode`` to consume a balanced
    object starting there. ``raw_decode`` handles nested braces, so a
    fact string containing ``{`` or ``}`` parses correctly — unlike a
    `[^{}]`-scoped regex which would refuse to match the outer
    object. Returns the first parsed object that has a list-valued
    ``facts`` key.
    """
    stripped = response.strip()
    if stripped.startswith("{"):
        try:
            parsed = json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            parsed = None
        if isinstance(parsed, dict) and isinstance(parsed.get("facts"), list):
            return parsed
    # O(n) over the response: at most one `{` per character. Picker
    # responses are bounded (single rewrite, T=0), so this stays cheap.
    for match in re.finditer(r"\{", response):
        try:
            parsed, _ = _JSON_DECODER.raw_decode(response, match.start())
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict) and isinstance(parsed.get("facts"), list):
            return parsed
    return None


def merge_node_data(
    store: GraphMemoryStore,
    node_id: str,
    new_facts: list[str],
    cfg,
    chat_model: str,
    timeout_sec: float = 20.0,
    thinking: bool = False,
    picker_model: Optional[str] = None,
    node: Optional[MemoryNode] = None,
) -> MergeResult:
    """Merge ``new_facts`` into ``node_id``'s data via one LLM rewrite.

    Combines the existing node data and the queued new facts, asks the
    model to produce a clean, consolidated, contradiction-free fact
    list, and writes that back as the node's full data. This subsumes
    dedupe, supersession, and per-write consolidation in a single
    pass — the latest prompt always rewrites the node, so updated
    conventions propagate to existing data without a separate
    migration step.

    Pass an empty ``new_facts`` list to run a self-consolidation pass
    on the node's existing data alone (dedupe / consolidate / prune
    only — no fact incorporation). The merge prompt's rules apply
    equally to the existing data, so the same LLM call serves both
    "incorporate new facts" and "tidy existing facts".

    Hallucination guard: the cleaned rewrite is rejected if it grows
    beyond ``len(existing_lines) + len(new_facts) + 2`` entries.
    Consolidation should only ever shrink or hold; runaway growth
    means the model invented content.

    Fail-open on any error (LLM failure, parse failure, empty
    rewrite, oversized rewrite). Caller's append path then writes the
    fact directly. We never let a flaky LLM erase data — a
    contradiction is recoverable, a silent wipe is not.

    Pass ``node`` if the caller has already fetched it; saves a
    redundant SQLite read on the orchestrator's hot path.
    """
    if node is None:
        node = store.get_node(node_id)
    if node is None:
        return MergeResult(success=False)

    existing_lines = _split_data_lines(node.data)
    # Re-join from the parsed lines so the prompt body and the line
    # count agree byte-for-byte — `existing` was previously a separate
    # `.strip()` of the raw blob, which could disagree with the parsed
    # list on edge whitespace.
    existing = "\n".join(existing_lines)
    sanitised_new: list[str] = [f.strip() for f in new_facts if f and f.strip()]

    if not existing_lines and not sanitised_new:
        # Nothing to do.
        return MergeResult(success=False)

    if not existing_lines:
        # Cold start: no existing data to merge against. Caller's
        # append path will write each new fact verbatim. Skipping the
        # LLM call keeps cold-start writes cheap.
        return MergeResult(success=False)

    if sanitised_new:
        new_facts_block = "\n".join(f"- {f}" for f in sanitised_new)
        user_content = (
            f"CURRENT facts on the node:\n{existing}\n\n"
            f"NEW facts to incorporate:\n{new_facts_block}"
        )
    else:
        user_content = (
            f"CURRENT facts on the node (no new facts to add — "
            f"consolidate / dedupe / prune only):\n{existing}"
        )

    effective_model = picker_model or chat_model
    response = call_llm_direct(
        cfg=cfg,
        chat_model=effective_model,
        system_prompt=_MERGE_SYSTEM_PROMPT,
        user_content=user_content,
        timeout_sec=timeout_sec,
        thinking=thinking,
        temperature=0.0,
    )

    if not response:
        return MergeResult(success=False)

    parsed = _extract_facts_object(response)
    if parsed is None:
        return MergeResult(success=False)

    cleaned: list[str] = []
    for item in parsed["facts"]:
        if not isinstance(item, str):
            continue
        line = item.strip()
        if line:
            cleaned.append(line)

    # Empty rewrite is suspicious — a non-empty `existing` plus
    # (optional) new facts should never collapse to nothing. Treat as
    # failure and let the caller's append path run.
    if not cleaned:
        return MergeResult(success=False)

    # Hallucination guard: bound the output relative to the input.
    # Consolidation rules can shrink or hold but should never grow
    # beyond `existing + new + small slack` — anything larger means
    # the model invented content not present in either input.
    max_kept = len(existing_lines) + len(sanitised_new) + _MERGE_GROWTH_SLACK
    if len(cleaned) > max_kept:
        debug_log(
            f"merge: rejected rewrite — {len(cleaned)} lines exceeds "
            f"guard cap of {max_kept}",
            "memory",
        )
        return MergeResult(success=False)

    # Identify which of the new facts actually survived the rewrite.
    # Uses the dedupe primitive's Unicode folding plus a tolerant
    # trailing-punctuation strip — picker models routinely rephrase
    # facts by dropping the trailing full stop ("X." → "X"), and a
    # strict `normalise_fact` match would then under-report every
    # batched flush as "0 stored". A new fact missing from the
    # cleaned set was consolidated out, treated as a duplicate, or
    # silently dropped — caller can then decide whether to skip
    # reporting or append-fallback.
    # Keyed on the claim, never on the whole line. The rewrite is an LLM
    # and will not reproduce a `· web · 2026-08-16` suffix verbatim, so
    # matching whole lines would report every genuinely-merged fact as
    # consolidated out — a fact stored but never announced, which is the
    # failure this file's provenance work exists to remove.
    def _match_key(text: str) -> str:
        return normalise_fact(fact_text(text)).rstrip(".,;:!?")

    cleaned_keys = {_match_key(line) for line in cleaned if line.strip()}
    incorporated_indices: list[int] = []
    for idx, fact in enumerate(new_facts):
        if not fact or not fact.strip():
            continue
        key = _match_key(fact)
        if key and key in cleaned_keys:
            incorporated_indices.append(idx)

    new_data = "\n".join(cleaned)
    store.update_node(node_id, data=new_data)
    return MergeResult(success=True, incorporated_indices=incorporated_indices)


# ── Auto-split ─────────────────────────────────────────────────────────


def auto_split_node(
    store: GraphMemoryStore,
    node_id: str,
    cfg,
    chat_model: str,
    timeout_sec: float = 45.0,
    thinking: bool = False,
) -> bool:
    """Split a node whose data exceeds SPLIT_THRESHOLD into child nodes.

    The LLM proposes 2-5 categories and distributes the facts among them.
    The parent node's data is cleared and its description updated to a summary.

    Returns True if the split succeeded.
    """
    node = store.get_node(node_id)
    if node is None or node.data_token_count <= SPLIT_THRESHOLD:
        return False

    debug_log(f"auto-split: node '{node.name}' ({node_id[:8]}) has {node.data_token_count} tokens", "memory")

    system_prompt = (
        "You are a knowledge organiser. A collection of facts has grown too "
        "large for a single node. Organise them into 2-5 categories.\n\n"
        "Rules:\n"
        "- Each fact must be assigned to exactly one category\n"
        "- Category names should be concise (2-4 words)\n"
        "- Descriptions should be 1-2 sentences explaining what the category covers\n\n"
        "Consolidation — apply while distributing:\n"
        "- Merge duplicate or near-duplicate facts into one\n"
        "- If repeated similar activities appear across different dates "
        "(e.g. ate X on Monday, ate X on Thursday), consolidate into a pattern "
        '(e.g. "Regularly eats X") — drop individual occurrences\n'
        "- Preserve date context only for significant events "
        "(e.g. started new job on 2025-03-01)\n\n"
        "Pruning — DROP facts that are common knowledge:\n"
        "- Remove anything you already know from your training data "
        "(e.g. general nutrition facts, well-known places, public figures' "
        "basic info, how-to steps for common tasks)\n"
        "- Only keep knowledge that is NOVEL to you: user-specific details, "
        "local/niche information, personal circumstances, recent events "
        "after your training cutoff, or corrections to what you'd assume\n"
        "- When in doubt, keep it — but actively look for things to prune\n\n"
        "Respond with ONLY valid JSON in this format:\n"
        '{"categories": [{"name": "Category Name", "description": "What this covers", '
        '"facts": ["fact 1", "fact 2"]}], "summary": "1-2 sentence summary of everything"}'
    )

    user_content = (
        f"Current node: {node.name}\n"
        f"Current description: {node.description}\n\n"
        f"Facts to organise:\n{node.data}"
    )

    response = call_llm_direct(
        cfg=cfg,
        chat_model=chat_model,
        system_prompt=system_prompt,
        user_content=user_content,
        timeout_sec=timeout_sec,
        thinking=thinking,
    )

    if not response:
        debug_log("auto-split: LLM returned no response", "memory")
        return False

    # Parse JSON from response
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if not json_match:
        debug_log("auto-split: no JSON found in response", "memory")
        return False

    try:
        result = json.loads(json_match.group())
    except (json.JSONDecodeError, ValueError) as e:
        debug_log(f"auto-split: JSON parse failed — {e}", "memory")
        return False

    categories = result.get("categories", [])
    summary = result.get("summary", node.description)

    # Validate: need at least 2 categories
    if len(categories) < 2:
        debug_log("auto-split: fewer than 2 categories proposed, aborting", "memory")
        return False

    # Validate: each category needs a name and at least one fact
    for cat in categories:
        if not cat.get("name") or not cat.get("facts"):
            debug_log(f"auto-split: invalid category {cat.get('name', '?')}, aborting", "memory")
            return False

    # Create child nodes
    for cat in categories:
        child_data = "\n".join(str(f) for f in cat["facts"])
        store.create_node(
            name=str(cat["name"]),
            description=str(cat.get("description", f"Memories about: {cat['name']}")),
            data=child_data,
            parent_id=node_id,
        )
        debug_log(f"  auto-split: created child '{cat['name']}' with {len(cat['facts'])} facts", "memory")

    # Clear parent data and update description to summary
    store.update_node(node_id, data="", description=str(summary))

    debug_log(f"auto-split: node '{node.name}' split into {len(categories)} children", "memory")
    return True


# ── Orchestrator ───────────────────────────────────────────────────────


class GraphUpdateResult(NamedTuple):
    """Result of a graph update pass.

    ``stored`` lists newly-appended facts so the CLI can show *what* was
    learned. ``skipped`` counts facts the picker routed to a node that
    already contained them — surfacing this lets callers print a status
    line on every flush, even when the cumulative diary re-extraction
    produces only duplicates (#282 dedupe would otherwise silence the
    "knowledge graph: learned N facts" log).
    """

    stored: "list[tuple[str, str]]"
    skipped: int


def update_graph_from_dialogue(
    store: GraphMemoryStore,
    summary: str,
    cfg,
    chat_model: str,
    tools_used: Optional[Sequence[str]],
    timeout_sec: float = 30.0,
    thinking: bool = False,
    date_utc: Optional[str] = None,
    picker_model: Optional[str] = None,
) -> GraphUpdateResult:
    """End-to-end: extract memories from a summary, place each in the best
    node, and trigger auto-split if needed.

    Args:
        date_utc: Optional date string (YYYY-MM-DD) for the diary entry.
            Passed to extraction to help distinguish daily events from enduring facts.

    Returns a ``GraphUpdateResult`` with a ``stored`` list of
    ``(fact, node_name)`` tuples for each newly-appended fact and a
    ``skipped`` count of duplicates the picker landed on. Callers must
    unpack via ``result.stored`` / ``result.skipped`` (or tuple
    destructuring) — the NamedTuple does not masquerade as the old list.
    """
    # Step 1: Extract discrete branch-tagged facts from the summary
    facts = extract_graph_memories(
        summary=summary,
        cfg=cfg,
        chat_model=chat_model,
        tools_used=tools_used,
        timeout_sec=timeout_sec,
        thinking=thinking,
        date_utc=date_utc,
    )

    if not facts:
        debug_log("graph update: no facts extracted from summary", "memory")
        return GraphUpdateResult(stored=[], skipped=0)

    debug_log(f"graph update: placing {len(facts)} facts into knowledge graph", "memory")

    # Step 2: Place — resolve the destination node for every fact up
    # front, applying the cheap exact-match dedupe fast-path along the
    # way. Then group surviving facts by node so the merge step below
    # rewrites each node at most once per flush instead of once per
    # fact. Without batching, a 5-fact flush against a populated User
    # node fires 5 small-model rewrites of the same `data`; with
    # batching, it's one rewrite that incorporates all five.
    pending: list[tuple[str, str]] = []  # (fact, node_id)
    seen_keys_per_node: dict[str, set[str]] = {}
    skipped = 0
    for fact in facts:
        try:
            node_id = find_best_node(
                store=store,
                fragment=fact,
                cfg=cfg,
                chat_model=chat_model,
                timeout_sec=15.0,
                thinking=thinking,
                picker_model=picker_model,
                branch_root_id=BRANCH_WORLD,
            )
        except Exception as e:
            debug_log(f"graph update: traversal failed for '{fact[:50]}...' — {e}", "memory")
            continue

        # Exact-match dedupe (fast-path, no LLM): skip facts already
        # stored verbatim on the chosen node. Cumulative daily summaries
        # re-extract the same facts on every flush; the SQL-only check
        # short-circuits the merge LLM call for the most common no-op
        # case. Re-extractions are not fresh learning — we don't report
        # them as newly stored and we don't touch the access score.
        # Skips are still counted so callers can log "nothing new (N
        # duplicates skipped)" on all-duplicate flushes.
        if store.node_contains_fact(node_id, fact):
            target = store.get_node(node_id)
            target_name = target.name if target else node_id[:8]
            skipped += 1
            debug_log(
                f"graph update: skipped duplicate '{fact[:50]}...' → "
                f"'{target_name}'",
                "memory",
            )
            continue

        # Within a single flush, two extractor outputs that fold to the
        # same key should also dedupe against each other before reaching
        # the merge step.
        key = normalise_fact(fact)
        node_keys = seen_keys_per_node.setdefault(node_id, set())
        if key and key in node_keys:
            debug_log(
                f"graph update: skipped intra-flush duplicate '{fact[:50]}...'",
                "memory",
            )
            continue
        if key:
            node_keys.add(key)

        pending.append((fact, node_id))

    if not pending:
        debug_log("graph update: nothing to merge after dedupe", "memory")
        return GraphUpdateResult(stored=[], skipped=skipped)

    # What the window rests on, decided once for the whole flush: every
    # fact here came out of the same summary of the same window.
    source = source_for_tools(tools_used)

    # Group by destination node so each node gets a single merge call.
    # The provenance suffix is composed here so both write paths below —
    # the merge rewrite and the plain-append fallback — carry it.
    by_node: dict[str, list[str]] = {}
    for fact, node_id in pending:
        by_node.setdefault(node_id, []).append(
            fact_line(fact, source, date_utc))

    stored: "list[tuple[str, str]]" = []
    for node_id, node_facts in by_node.items():
        node = store.get_node(node_id)
        node_name = node.name if node else node_id[:8]

        # Step 3: Merge — combine the existing node data with all
        # queued new facts in a single LLM rewrite. Subsumes
        # supersession (contradictions drop the old line),
        # near-duplicate dedupe (different wordings collapse), and
        # ongoing consolidation (repeated activities fold into
        # patterns). The latest prompt always rewrites the whole
        # node, so updated conventions propagate to old data without
        # a separate migration step.
        #
        # Fail-open: if the merge returns success=False (empty node,
        # LLM failure, parse failure, empty rewrite, or rewrite that
        # tripped the hallucination guard), each fact falls back to
        # plain append below. We never let a flaky LLM erase data —
        # a contradiction is recoverable, a silent wipe is not.
        merge_result = MergeResult(success=False)
        try:
            merge_result = merge_node_data(
                store=store,
                node_id=node_id,
                new_facts=node_facts,
                cfg=cfg,
                chat_model=chat_model,
                timeout_sec=20.0,
                thinking=thinking,
                picker_model=picker_model,
                node=node,
            )
        except Exception as e:
            debug_log(f"graph update: merge failed for node '{node_name}' — {e}", "memory")

        if merge_result.success:
            # Merge wrote the consolidated data. Only the facts the
            # rewrite actually retained get reported as stored — a
            # fact that was consolidated out (e.g. folded into a
            # pattern, or treated as a near-duplicate) was not
            # newly learned and shouldn't be claimed as such.
            incorporated = set(merge_result.incorporated_indices)
            for idx, fact in enumerate(node_facts):
                if idx in incorporated:
                    stored.append((fact, node_name))
                    debug_log(
                        f"graph update: merged '{fact[:50]}...' → "
                        f"'{node_name}'",
                        "memory",
                    )
                else:
                    debug_log(
                        f"graph update: '{fact[:50]}...' consolidated "
                        f"out by merge on '{node_name}' — not reported",
                        "memory",
                    )
        else:
            # Cold start, merge failure, or guard rejection — fall
            # back to plain append for every queued fact so nothing
            # is lost.
            for fact in node_facts:
                store.append_to_node(node_id, fact)
                stored.append((fact, node_name))
                debug_log(
                    f"graph update: appended '{fact[:50]}...' → "
                    f"'{node_name}' (merge skipped)",
                    "memory",
                )

        store.touch_node(node_id)

        # Step 4: Auto-split if the node has grown too large.
        refreshed = store.get_node(node_id)
        if refreshed is not None and refreshed.data_token_count > SPLIT_THRESHOLD:
            debug_log(
                f"graph update: node '{node_name}' exceeded threshold, splitting",
                "memory",
            )
            try:
                auto_split_node(
                    store=store,
                    node_id=node_id,
                    cfg=cfg,
                    chat_model=chat_model,
                    timeout_sec=45.0,
                    thinking=thinking,
                )
            except Exception as e:
                debug_log(f"graph update: auto-split failed for '{node_name}' — {e}", "memory")

    debug_log(
        f"graph update: stored {len(stored)}/{len(facts)} facts "
        f"({skipped} duplicate{'' if skipped == 1 else 's'} skipped)",
        "memory",
    )
    return GraphUpdateResult(stored=stored, skipped=skipped)


def consolidate_all_populated_nodes(
    store: GraphMemoryStore,
    cfg,
    chat_model: str,
    timeout_sec: float = 20.0,
    thinking: bool = False,
    picker_model: Optional[str] = None,
) -> "Iterator[tuple[str, int, int]]":
    """One-shot self-consolidation across every populated node.

    Walks every node with non-empty `data` and runs `merge_node_data`
    with an empty new-facts list, so the merge prompt's rules
    (contradiction handling, near-duplicate collapse, consolidation,
    pruning) tidy the existing data in place. This is the migration
    path for nodes that accumulated contradictions before the
    merge-on-write step landed: under merge-on-write, a node only
    gets cleaned when a new related fact arrives, so backlog stays
    dirty until something nudges it. Calling this op nudges
    everything at once.

    Yields ``(node_name, lines_before, lines_after)`` per node as the
    walk progresses, so a streaming caller (e.g. an NDJSON endpoint)
    can surface per-node feedback in real time on graphs with many
    nodes. Fail-open: a node that fails to merge is left untouched
    and reported with ``lines_after == lines_before``.
    """
    # Snapshot all nodes up front so a rewrite mid-walk doesn't
    # cause us to revisit or skip nodes.
    all_nodes = store.get_all_nodes()
    for node in all_nodes:
        if not is_populated_node(node):
            continue
        before = len(_split_data_lines(node.data))
        try:
            result = merge_node_data(
                store=store,
                node_id=node.id,
                new_facts=[],
                cfg=cfg,
                chat_model=chat_model,
                timeout_sec=timeout_sec,
                thinking=thinking,
                picker_model=picker_model,
                node=node,
            )
        except Exception as e:
            debug_log(f"consolidate-all: failed for '{node.name}' — {e}", "memory")
            result = MergeResult(success=False)

        refreshed = store.get_node(node.id)
        after = len(_split_data_lines(refreshed.data)) if refreshed else before
        debug_log(
            f"consolidate-all: '{node.name}' {before} → {after} lines "
            f"(success={result.success})",
            "memory",
        )
        yield (node.name, before, after)


# ── Handing user knowledge over to the core ──────────────────────────


# The graph branches whose contents belong in the core, and the core
# section each one lands in.
_BRANCH_TO_CORE_SECTION = {
    BRANCH_USER: SECTION_PROFILE,
    BRANCH_DIRECTIVES: SECTION_RULES,
}


def migrate_graph_branches_into_core(store: GraphMemoryStore, core: MemoryCore) -> int:
    """Hand the user and directives branches over to the core files.

    The core is the authority for what the assistant believes about its
    user, so knowledge the graph accumulated under those two branches has
    to reach the files the user can read and correct. Returns the number
    of entries actually written.

    A hand-over, not a copy: once a node's facts are safely in the core,
    the node keeps its place in the tree but gives up its data. Leaving
    the text behind would break the core's two central promises. Recall
    still searches the graph, so a retired belief would be pulled back
    into the prompt from the node it was copied from; and a line the user
    pruned from their own file would be rewritten on the next start-up,
    for ever. Emptying the source makes both impossible and makes the
    migration a no-op on every subsequent run, since there is nothing
    left to hand over.

    A node is cleared only when every one of its facts reached the core.
    Text the core already holds counts as arrived, so a partial run
    finishes cleanly on the next boot; a write that fails leaves the
    facts in the graph rather than losing them.
    """
    written = 0
    for branch_id, section in _BRANCH_TO_CORE_SECTION.items():
        for node_id, facts in _walk_branch_nodes(store, branch_id):
            handed_over = True
            for text in facts:
                if core.knows(section, text):
                    continue
                try:
                    core.remember(section, text, source=SOURCE_MIGRATED)
                    written += 1
                except (OSError, ValueError) as e:
                    handed_over = False
                    debug_log(f"core migration skipped '{text[:40]}': {e}", "memory")
            if handed_over:
                store.update_node(node_id, data="")
    if written:
        debug_log(f"core migration: {written} entries moved from the graph", "memory")
    return written


def _walk_branch_nodes(
    store: GraphMemoryStore, branch_root_id: str,
) -> Iterator[tuple[str, list[str]]]:
    """Yield ``(node_id, facts)`` for every populated node in a subtree."""
    if store.get_node(branch_root_id) is None:
        return

    queue: list[str] = [branch_root_id]
    visited: set[str] = set()
    while queue:
        node_id = queue.pop(0)
        if node_id in visited:
            continue
        visited.add(node_id)
        node = store.get_node(node_id)
        if node is None:
            continue
        facts = [line.strip() for line in _split_data_lines(node.data)]
        if facts:
            yield node_id, facts
        for child in store.get_children(node_id):
            queue.append(child.id)

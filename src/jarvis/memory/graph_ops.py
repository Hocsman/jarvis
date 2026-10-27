"""
🧠 Knowledge Graph Operations — LLM-dependent graph logic.

Keeps graph.py as a pure data store (SQLite only). This module handles:
- Knowledge extraction from conversation summaries
- Best-node traversal (greedy descent from a fixed branch root)
- Auto-split when a node exceeds the token threshold

All LLM calls run on the chat model the caller hands in and go through the
auxiliary dispatcher (a bare model tag on a remote provider runs on local
Ollama); the local ``call_llm_direct`` wrapper is the single intercept point
tests patch. Every call is made through ``_ask``, inside the pass's
``GraphDeadline``.
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Iterator, NamedTuple, Optional, Sequence

from rapidfuzz import fuzz, process

from ..config import get_default_config
from ..debug import debug_log
from ..llm import get_auxiliary_backend
from ..utils.redact import redaction_placeholders
from .provenance import (
    SOURCE_UNKNOWN,
    fact_line,
    fact_source,
    fact_text,
    lookup_tools,
    source_for_tools,
)
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
    fact_key,
    normalise_fact,
)


def call_llm_direct(*, cfg, chat_model, system_prompt, user_content,
                    timeout_sec=10.0, thinking=False, num_ctx=4096,
                    temperature=None):
    """Local indirection: route graph-ops LLM calls through the auxiliary
    dispatcher, so a bare model tag on a remote provider runs on local
    Ollama instead of dying on its HTTP 400. Tests patch this single
    symbol to intercept every LLM round-trip in this module."""
    return get_auxiliary_backend(cfg, chat_model).direct(
        chat_model, system_prompt, user_content,
        timeout_sec=timeout_sec, thinking=thinking,
        num_ctx=num_ctx, temperature=temperature,
    )


# ── The pass's deadline ───────────────────────────────────────────────


def _chat_timeout_sec(cfg) -> float:
    """How long one answer from the chat model may take: the user's own
    figure, the one the reply is held to."""
    timeout = getattr(cfg, "llm_chat_timeout_sec", None)
    return float(timeout or get_default_config()["llm_chat_timeout_sec"])


class GraphDeadline:
    """When one graph pass has to be over.

    The calls of a pass share it: each is handed what is left, so a slow
    chat model gets as long as the pass can give rather than a figure sized
    for a faster one, and none is waited on past it. Once it is spent, or
    once ``stop`` says the daemon is going down, nothing more is asked and
    the pass finishes on the paths that need no model: a fact stays where
    the descent got to, a node takes a plain append, a split waits for the
    next flush. A fact once extracted is filed less well, never lost; a
    summary the extraction never read is reported ``unread`` so the caller
    can give it another pass.
    """

    def __init__(self, seconds: float, *, stop: Optional[Callable[[], bool]] = None,
                 clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._end = clock() + max(0.0, float(seconds))
        self._stop = stop
        self._turned_away = False

    @property
    def turned_away(self) -> bool:
        """Whether a call has gone unasked or been left behind for want of
        time. Never reset: a deadline does not come back."""
        return self._turned_away

    def _stopped(self) -> bool:
        return self._stop is not None and bool(self._stop())

    def remaining(self) -> float:
        if self._stopped():
            return 0.0
        return max(0.0, self._end - self._clock())

    def spent(self) -> bool:
        """True once nothing more may be asked. Said once per pass, not once
        per call it turns away."""
        if self.remaining() > 0:
            return False
        if not self._turned_away:
            self._turned_away = True
            why = "a stop was requested" if self._stopped() else "its deadline passed"
            debug_log(f"graph pass: {why}, nothing more is asked of the model", "memory")
        return True


def _deadline_or_one_chat_timeout(deadline: Optional[GraphDeadline], cfg) -> GraphDeadline:
    return deadline if deadline is not None else GraphDeadline(_chat_timeout_sec(cfg))


# How often a call in flight is checked against the deadline and the stop
# request: the most a stop waits before the call is left behind.
_WAIT_SLICE_SEC = 0.2


def _ask(deadline: GraphDeadline, what: str, **call) -> Optional[str]:
    """One model call inside the pass, handed what the deadline has left.

    Waited on here rather than trusted to the HTTP timeout, which bounds
    the gap between two bytes, not the call: a provider that keeps the
    connection open would otherwise hold the pass, and shutdown with it. A
    call left behind runs out on its own thread, and nothing it brings back
    is read. ``None`` when not asked, left behind or unanswered, which every
    caller already reads as no answer.
    """
    if deadline.spent():
        return None
    budget = deadline.remaining()
    outcome: dict = {}

    def run():
        try:
            outcome["answer"] = call_llm_direct(timeout_sec=budget, **call)
        except Exception as e:
            outcome["error"] = e

    worker = threading.Thread(target=run, name=f"graph {what}", daemon=True)
    worker.start()
    while worker.is_alive():
        worker.join(min(_WAIT_SLICE_SEC, deadline.remaining()))
        if worker.is_alive() and deadline.spent():
            debug_log(f"{what}: left behind, {call['chat_model']} had not answered", "memory")
            return None
    if "error" in outcome:
        raise outcome["error"]
    answer = outcome.get("answer")
    if not answer:
        # Logged apart from what a caller parses as a refusal (the picker's
        # NONE): a timeout, an error or an empty reply all land here.
        debug_log(f"{what}: no answer from {call['chat_model']} "
                  f"(handed {budget:.0f}s)", "memory")
    return answer


# How close a rewritten line must be to a claim we know the source of.
# Measured on his graph after one real merge: the same line scored 94
# across the rewrite, and the closest unrelated fact in the same node
# scored 34. Set high in that gap — a line the rewrite consolidated
# beyond recognition should fall to `inconnu` rather than borrow a source
# it cannot claim.
_REWRITE_MATCH = 85


_DIGITS = re.compile(r"\d+")


def _redaction_markers(lines) -> Counter:
    """How many of each redaction marker ``lines`` carry.

    Counted rather than merely detected: a node that already held one
    marker must not cover a second one the rewrite put where a real value
    stood.
    """
    return Counter(m for line in lines for m in redaction_placeholders(line))


def _numbers_are_grounded(cleaned, inputs) -> bool:
    """True when every digit run in the rewrite already went in.

    The rewrite may drop a figure — consolidation is meant to shrink — but
    it may not invent one. Measured on his graph, 2026-08-17: a battery
    life of 19 hours came back out of the merge as 26. Three repeats were
    correct, so it is occasional rather than systematic, and nothing saw
    it happen.

    The growth guard cannot: it counts *lines* against the lines that
    went in, so a digit changed inside a line passes by construction.
    Numbers are where a rewrite does the most damage per character — a
    reworded sentence is still roughly the fact, a wrong figure is a
    different fact wearing the same words, and it is the shape a later
    answer quotes with confidence.

    Refusing is safe: the caller falls back to plain append, so a false
    positive costs consolidation, never a fact.
    """
    connus = set()
    for line in inputs:
        connus.update(_DIGITS.findall(line or ""))
    for line in cleaned:
        for n in _DIGITS.findall(line or ""):
            if n not in connus:
                return False
    return True


def _match_key(text: str) -> str:
    """The claim a line makes, folded for comparison across a rewrite.

    Keyed on the claim, never on the whole line: a rewrite is an LLM and
    will not reproduce a `· web · 2026-08-16` suffix verbatim. Uses the
    dedupe primitive's Unicode folding plus a tolerant trailing-punctuation
    strip, since models routinely drop the full stop ("X." → "X").
    """
    return normalise_fact(fact_text(text)).rstrip(".,;:!?")


# The share of a node's claims a split must carry into its children. A
# split may leave out what its prompt tells it to prune, and it folds
# duplicates and repeated activities into one line, so it cannot be held
# to every claim. Calibrated on recorded split answers for two World nodes
# of 90 and 101 lines, 8 and 13 of them filler or narration.
# `google/gemini-2.5-flash` carried at least 84 % of the claims in each of
# 75 splits under pruning as narrow as the merge's, and 68 % to 100 % in
# 75 more under the split prompt's wider pruning, which also drops what
# the model already knows (62 of those pass). Every `gemma4:e2b` answer
# that parsed (79 of 200) carried at most 69 %, having lost about half the
# node's niche lookups. The floor sits in that gap and leaves a quarter of
# the node for everything a split may drop.
_SPLIT_MIN_RETENTION = 0.75


def _claims_kept(before, after) -> tuple[int, int]:
    """How many of the claims in ``before`` a rewrite carried into ``after``.

    Returns ``(kept, total)``. A claim is carried when a line of ``after``
    makes it: the same key, or a rewording that scores ``_REWRITE_MATCH``,
    the bar a reworded merge line has to clear to get its source back, and
    holds every figure the claim held. Without the figures, lines worded
    alike that differ only in a date, a time or a price (one per
    performance, one per day) would all be carried by whichever of them
    survived. A claim the node holds twice in the same words counts once,
    and one line of ``after`` may carry several claims, so a duplicate
    folded into a line that scores ``_REWRITE_MATCH`` against both
    wordings costs nothing. Two lines that differ only by a name still
    score above the bar against each other, so dropping one of them goes
    unseen.
    """
    claims = {_match_key(line) for line in before} - {""}
    figures = {line: set(_DIGITS.findall(line))
               for line in {_match_key(line) for line in after} - {""}}
    choices = list(figures)
    kept = 0
    for claim in claims:
        if claim in figures:
            kept += 1
            continue
        held = set(_DIGITS.findall(claim))
        if any(held <= figures[line] for line, _, _ in process.extract(
                claim, choices, scorer=fuzz.ratio,
                score_cutoff=_REWRITE_MATCH, limit=None)):
            kept += 1
    return kept, len(claims)


def _source_index(lines) -> dict:
    """Map each line's claim to the provenance suffix it carried.

    Built from what a node held before a merge plus what this flush is
    adding, so the repair afterwards can recognise either.
    """
    index: dict = {}
    for line in lines:
        if not line or not line.strip():
            continue
        source = fact_source(line)
        if source == SOURCE_UNKNOWN:
            continue
        index[normalise_fact(fact_text(line))] = line[len(fact_text(line)):]
    return index


def _reattach_sources(lines, index: dict) -> list:
    """Give a rewritten node's lines their provenance back.

    The merge is an LLM rewriting a whole node, and it does not reproduce
    a suffix it was never asked to keep — which made the marker almost
    never survive, since a populated node always takes the merge path.
    Asking the rewrite to preserve an exact format would be asking an LLM
    for the guarantee this area exists because LLMs do not give, so the
    repair is deterministic and happens after it.

    Wording is never touched. A line the rewrite reworded matches nothing
    and stays bare, which reads as `inconnu` — the truthful answer, since
    after a rewrite nobody can say what that sentence rested on.
    """
    repare = []
    for line in lines:
        if not line or not line.strip() or fact_source(line) != SOURCE_UNKNOWN:
            repare.append(line)
            continue
        cle = normalise_fact(fact_text(line))
        suffixe = index.get(cle)
        if suffixe is None and index:
            # The rewrite retypesets and rewords, so an exact key matches
            # almost nothing: measured on his graph, one line came back
            # with the currency symbol moved, narrow spaces inserted, a
            # hyphen swapped for U+2011 and "prior" changed to "earlier".
            # Similarity separates that cleanly — 94 for the same line
            # across the rewrite, 34 for the closest unrelated fact in the
            # same node — so the bar sits high in that gap.
            meilleur, score = None, 0.0
            for autre, suf in index.items():
                s = fuzz.ratio(cle, autre)
                if s > score:
                    meilleur, score = suf, s
            if score >= _REWRITE_MATCH:
                suffixe = meilleur
        repare.append(line + suffixe if suffixe else line)
    return repare


# ── Memory extraction from dialogue ───────────────────────────────────


def extract_graph_memories(
    summary: str,
    cfg,
    chat_model: str,
    tools_used: Optional[Sequence[str]],
    deadline: Optional[GraphDeadline] = None,
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
        deadline: The pass this extraction opens; one chat-model timeout
            from now when it runs on its own.
        date_utc: Optional date string (YYYY-MM-DD) for the diary entry.
            Included as a date prefix on each fact for temporal context.
    """
    if tools_used is not None and not lookup_tools(tools_used):
        debug_log("graph memory extraction: nothing in this window looked "
                  f"anything up (tools: {list(tools_used) or 'none'})",
                  "memory")
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
    response = _ask(
        _deadline_or_one_chat_timeout(deadline, cfg),
        "graph memory extraction",
        cfg=cfg,
        chat_model=chat_model,
        system_prompt=system_prompt,
        user_content=user_content,
        thinking=thinking,
        temperature=0.0,
    )

    if not response:
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
    deadline: Optional[GraphDeadline] = None,
) -> Optional[str]:
    """Ask the chat model which child node best fits a memory fragment.

    Returns the chosen child's id, or None if none fit well. The model is
    never asked to think: the answer is one number, asked again at every
    level of every fact inside the pass's one deadline, and time spent
    thinking leaves facts at the branch root.
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

    response = _ask(
        _deadline_or_one_chat_timeout(deadline, cfg),
        "graph picker",
        cfg=cfg,
        chat_model=chat_model,
        system_prompt=system_prompt,
        user_content=user_content,
    )

    if not response:
        # The fact stays at this level, as on NONE.
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
    branch_root_id: str,
    deadline: Optional[GraphDeadline] = None,
) -> str:
    """Find the best node, inside one fixed branch, to store a memory fragment.

    Traversal descends greedily from ``branch_root_id`` (one of the fixed
    taxonomy branches: User / Directives / World): at each level the picker
    chooses the child that fits the fragment, and the walk stops at the
    first node that has no children or none that fit. Recent and top nodes
    are never consulted, so the purpose-shaped top-level taxonomy is
    respected: a World fact can never end up in another branch because a
    node there happened to be accessed recently.

    The whole descent shares one deadline (one chat-model timeout from now
    when none is given): once it is spent, the walk stops where it is.

    Returns the id of the best node.
    """
    deadline = _deadline_or_one_chat_timeout(deadline, cfg)
    debug_log(
        f"graph traversal: placing '{fragment[:60]}...' "
        f"(branch={branch_root_id})",
        "memory",
    )

    current_id = branch_root_id
    depth = 0
    for depth in range(MAX_TRAVERSAL_DEPTH):
        children = store.get_children(current_id)
        if not children:
            debug_log(f"graph traversal: leaf node at depth {depth}", "memory")
            break  # Leaf node — write here

        debug_log(f"graph traversal: depth {depth}, choosing from {[c.name for c in children]}", "memory")
        best = _llm_pick_best_child(
            fragment, children, cfg, chat_model,
            deadline=deadline,
        )
        if best is None:
            if deadline.turned_away:
                debug_log(f"graph traversal: out of time at depth {depth}, stopping", "memory")
            else:
                debug_log(f"graph traversal: no children fit at depth {depth}, stopping", "memory")
            break  # Write to the current node
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
    # O(n) over the response: at most one `{` per character. Merge
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
    deadline: Optional[GraphDeadline] = None,
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

    Like the picker, the rewrite runs on the chat model and never asks it
    to think: the whole node has to come back inside the pass's deadline
    (one chat-model timeout from now when none is given), and a rewrite
    that runs out of time drops every new fact to plain append.
    """
    if node is None:
        node = store.get_node(node_id)
    if node is None:
        return MergeResult(success=False)

    existing_lines = _split_data_lines(node.data)
    # Re-join from the parsed lines so the prompt body and the line
    # count agree byte-for-byte; a separate `.strip()` of the raw blob
    # could disagree with the parsed list on edge whitespace.
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

    response = _ask(
        _deadline_or_one_chat_timeout(deadline, cfg),
        "merge",
        cfg=cfg,
        chat_model=chat_model,
        system_prompt=_MERGE_SYSTEM_PROMPT,
        user_content=user_content,
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

    # The guard above counts lines, so a digit changed inside one passes
    # it. Numbers are where a rewrite does the most damage per character,
    # and this one was observed changing 19 hours into 26.
    if not _numbers_are_grounded(cleaned, list(existing_lines) + list(sanitised_new)):
        debug_log(
            "merge: rejected rewrite — it carries a figure that was in "
            "neither the node nor the new facts",
            "memory",
        )
        return MergeResult(success=False)

    # A remote model sees the node through the redacting backend, so an
    # address or a long number on it comes back as a marker. Written
    # over the node, the value it hid would be gone for good. A marker
    # the inputs already carried (redacted diary text) may stay; one
    # more than they carried stands for a value.
    if _redaction_markers(cleaned) - _redaction_markers(
        list(existing_lines) + list(sanitised_new)
    ):
        debug_log(
            "merge: rejected rewrite — it carries a redaction marker where "
            "the node held a real value",
            "memory",
        )
        return MergeResult(success=False)

    # Identify which of the new facts actually survived the rewrite.
    # A strict `normalise_fact` match would under-report every batched
    # flush as "0 stored" once the model drops a full stop, and matching
    # whole lines would report every genuinely-merged fact as
    # consolidated out, since the suffix does not survive — a fact
    # stored but never announced, which is the failure this file's
    # provenance work exists to remove. A new fact missing from the
    # cleaned set was consolidated out, treated as a duplicate, or
    # silently dropped — caller can then decide whether to skip
    # reporting or append-fallback.
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


# How much a node must grow, as a share of its size when its split was
# refused, before the chat model that refused it is asked again. A refused
# split leaves the node whole, and on the small tier every split is
# refused, so without it each flush that touches the node would pay for a
# full rewrite of the node that is thrown away. Geometric, so a node is
# asked a handful of times over its life rather than once per flush. A
# different chat model is asked at once.
_SPLIT_RETRY_GROWTH = 0.25

# The last refused split of each node, keyed by the graph's database and
# the node: the chat model that gave the answer and the node's size then.
# Held for the life of the process, so a restart asks once more.
_SPLIT_REFUSALS: dict[tuple[str, str], tuple[str, int]] = {}


def _split_worth_asking(store: GraphMemoryStore, node: MemoryNode, chat_model: str) -> bool:
    """Whether to ask ``chat_model`` to split ``node`` now.

    Only an answer that came back and was refused counts against the node:
    one the deadline turned away, or that timed out, says nothing about the
    next attempt.
    """
    refused = _SPLIT_REFUSALS.get((store.db_path, node.id))
    if refused is None or refused[0] != chat_model:
        return True
    return node.data_token_count >= refused[1] * (1 + _SPLIT_RETRY_GROWTH)


def auto_split_node(
    store: GraphMemoryStore,
    node_id: str,
    cfg,
    chat_model: str,
    deadline: Optional[GraphDeadline] = None,
    thinking: bool = False,
) -> bool:
    """Split a node whose data exceeds SPLIT_THRESHOLD into child nodes.

    The LLM proposes 2-5 categories and distributes the facts among them.
    The parent node's data is cleared and its description updated to a summary.
    Asked inside the pass's deadline (one chat-model timeout from now when
    none is given); a split that gets no answer is retried by the next
    flush that finds the node still too large.

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

    response = _ask(
        _deadline_or_one_chat_timeout(deadline, cfg),
        "auto-split",
        cfg=cfg,
        chat_model=chat_model,
        system_prompt=system_prompt,
        user_content=user_content,
        thinking=thinking,
    )

    if not response:
        return False

    plan = _read_split(node, response)
    if plan is None:
        # An answer came back and was refused: remembered, so the same
        # model is not asked again until the node has grown.
        _SPLIT_REFUSALS[(store.db_path, node_id)] = (chat_model, node.data_token_count)
        return False
    children, summary, kept, total = plan
    _SPLIT_REFUSALS.pop((store.db_path, node_id), None)

    # Create child nodes
    for name, description, lines in children:
        store.create_node(
            name=name,
            description=description,
            data="\n".join(lines),
            parent_id=node_id,
        )
        debug_log(f"  auto-split: created child '{name}' with {len(lines)} facts", "memory")

    # Clear parent data and update description to summary; a summary that
    # is not text leaves the description as it was.
    if summary:
        store.update_node(node_id, data="", description=summary)
    else:
        store.update_node(node_id, data="")

    debug_log(
        f"auto-split: node '{node.name}' split into {len(children)} children, "
        f"carrying {kept} of its {total} claims",
        "memory",
    )
    return True


def _read_split(node: MemoryNode, response: str):
    """The children a split answer proposes, or ``None`` when it is refused.

    Returns ``(children, summary, kept, total)``: each child as
    ``(name, description, lines)``, the summary when it is text (``None``
    otherwise), and the claims the children carry out of the node's. Every
    refusal is logged with its reason.
    """
    # Parse JSON from response
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if not json_match:
        debug_log("auto-split: no JSON found in response", "memory")
        return None

    try:
        result = json.loads(json_match.group())
    except (json.JSONDecodeError, ValueError) as e:
        debug_log(f"auto-split: JSON parse failed — {e}", "memory")
        return None

    categories = result.get("categories", [])
    summary = result.get("summary")
    summary = summary.strip() if isinstance(summary, str) and summary.strip() else None

    # Validate: need a list of at least 2 categories
    if not isinstance(categories, list):
        debug_log("auto-split: categories are not a list, aborting", "memory")
        return None
    if len(categories) < 2:
        debug_log("auto-split: fewer than 2 categories proposed, aborting", "memory")
        return None

    # Validate: each category needs a name and at least one fact, and its
    # facts must be a list of strings. A `facts` given as one string would
    # be written one character per line, and an item that is not a string
    # has no sentence to write.
    children: list[tuple[str, str, list[str]]] = []
    for cat in categories:
        name = cat.get("name") if isinstance(cat, dict) else None
        if not isinstance(name, str) or not name.strip():
            debug_log("auto-split: a category has no name, aborting", "memory")
            return None
        name = name.strip()
        facts = cat.get("facts")
        if not isinstance(facts, list) or not all(isinstance(f, str) for f in facts):
            debug_log(
                f"auto-split: category '{name}' does not give its facts "
                "as a list of strings, aborting",
                "memory",
            )
            return None
        # One fact per line, as a node holds them: an item that packs
        # several lines is measured and written as those lines.
        lines = [part.strip() for f in facts for part in f.split("\n") if part.strip()]
        if not lines:
            debug_log(f"auto-split: category '{name}' has no facts, aborting", "memory")
            return None
        description = cat.get("description")
        if not isinstance(description, str) or not description.strip():
            description = f"Memories about: {name}"
        children.append((name, description.strip(), lines))

    written = [line for _, _, lines in children for line in lines]
    before = _split_data_lines(node.data)

    # Clearing the parent is the one step here that cannot be undone, so
    # what the children leave out is gone for good. A split may drop what
    # its prompt prunes and fold duplicates and repeated activities, but
    # one that leaves out more than the floor allows lost answers he asked
    # for.
    kept, total = _claims_kept(before, written)
    if kept < _SPLIT_MIN_RETENTION * total:
        debug_log(
            f"auto-split: rejected — the children carry {kept} of the node's "
            f"{total} claims, under the {_SPLIT_MIN_RETENTION:.0%} floor",
            "memory",
        )
        return None

    # The merge's guards, for the same reasons: a split redistributes and
    # consolidates, so it may shrink a node but not grow it; a figure may
    # be dropped but not invented; and a remote model sees the node through
    # the redacting backend, so a marker it hands back stands for a value
    # the node held.
    if len(written) > len(before) + _MERGE_GROWTH_SLACK:
        debug_log(
            f"auto-split: rejected — the children hold {len(written)} lines "
            f"against the node's {len(before)}",
            "memory",
        )
        return None
    if not _numbers_are_grounded(written, before):
        debug_log(
            "auto-split: rejected — the children carry a figure the node did not hold",
            "memory",
        )
        return None
    if _redaction_markers(written) - _redaction_markers(before):
        debug_log(
            "auto-split: rejected — the children carry a redaction marker where "
            "the node held a real value",
            "memory",
        )
        return None

    return children, summary, kept, total


# ── Orchestrator ───────────────────────────────────────────────────────


class GraphUpdateResult(NamedTuple):
    """Result of a graph update pass.

    ``stored`` lists newly-appended facts so the CLI can show *what* was
    learned. ``skipped`` counts facts the World branch already held. It
    lets callers print a status line on every flush, even when the
    cumulative diary re-extraction produces only duplicates, so the
    "knowledge graph: learned N facts" log does not go silent.
    ``unread`` says the deadline was spent before the extraction
    answered, so the summary was never read: nothing was learned, and
    not because there was nothing to learn.
    """

    stored: "list[tuple[str, str]]"
    skipped: int
    unread: bool = False


def _fact_homes(store: GraphMemoryStore, branch_root_id: str) -> dict[str, str]:
    """Every fact a branch holds, by ``fact_key``, with the name of the node
    it is on. One read of the subtree, no model."""
    homes: dict[str, str] = {}
    for node_id, lines in _walk_branch_nodes(store, branch_root_id):
        node = store.get_node(node_id)
        name = node.name if node else node_id[:8]
        for line in lines:
            key = fact_key(line)
            if key:
                homes.setdefault(key, name)
    return homes


def update_graph_from_dialogue(
    store: GraphMemoryStore,
    summary: str,
    cfg,
    chat_model: str,
    tools_used: Optional[Sequence[str]],
    deadline: Optional[GraphDeadline] = None,
    thinking: bool = False,
    date_utc: Optional[str] = None,
) -> GraphUpdateResult:
    """End-to-end: extract memories from a summary, place each in the best
    node, and trigger auto-split if needed.

    Args:
        deadline: When the whole pass has to be over, shared by every call
            in it (see ``GraphDeadline``). One chat-model timeout from now
            when none is given.
        date_utc: Optional date string (YYYY-MM-DD) for the diary entry.
            Passed to extraction to help distinguish daily events from enduring facts.

    Returns a ``GraphUpdateResult`` with a ``stored`` list of
    ``(fact, node_name)`` tuples for each newly-appended fact, a
    ``skipped`` count of facts the World branch already held, and
    ``unread`` when the deadline was spent before the summary was read.
    Callers read the fields by name; the result is not a list.
    """
    deadline = _deadline_or_one_chat_timeout(deadline, cfg)

    # Step 1: Extract discrete branch-tagged facts from the summary
    facts = extract_graph_memories(
        summary=summary,
        cfg=cfg,
        chat_model=chat_model,
        tools_used=tools_used,
        deadline=deadline,
        thinking=thinking,
        date_utc=date_utc,
    )

    if not facts:
        if deadline.turned_away:
            # Not an empty answer: the summary was never read. The caller
            # can tell the two apart through ``unread``.
            debug_log("graph update: the summary was not read, the pass had no time left", "memory")
            return GraphUpdateResult(stored=[], skipped=0, unread=True)
        debug_log("graph update: no facts extracted from summary", "memory")
        return GraphUpdateResult(stored=[], skipped=0)

    debug_log(f"graph update: placing {len(facts)} facts into knowledge graph", "memory")

    # Step 2: Exact-match dedupe (fast-path, no LLM) against every line the
    # World branch already holds, before anything is placed. The daily
    # summary is cumulative, so each flush extracts again what earlier
    # flushes filed; sent down the picker first, those facts would spend
    # the pass's deadline and leave the new ones to land at the root.
    # Re-extractions are not fresh learning: they are not reported as
    # stored and do not touch the access score, but they are counted so
    # callers can log "nothing new (N duplicates skipped)".
    known = _fact_homes(store, BRANCH_WORLD)
    fresh: list[str] = []
    seen: set[str] = set()
    skipped = 0
    for fact in facts:
        key = fact_key(fact)
        if key and key in known:
            skipped += 1
            debug_log(
                f"graph update: skipped duplicate '{fact[:50]}...' → '{known[key]}'",
                "memory",
            )
            continue
        # Two extractor outputs that fold to the same key are one fact.
        if key and key in seen:
            debug_log(
                f"graph update: skipped intra-flush duplicate '{fact[:50]}...'",
                "memory",
            )
            continue
        if key:
            seen.add(key)
        fresh.append(fact)

    # Step 3: Place — resolve the destination node for every new fact up
    # front, then group them by node so the merge step below rewrites
    # each node at most once per flush instead of once per fact. Without
    # batching, a 5-fact flush against a populated node fires 5 rewrites
    # of the same `data`; with batching, it's one rewrite that
    # incorporates all five.
    pending: list[tuple[str, str]] = []  # (fact, node_id)
    for fact in fresh:
        try:
            node_id = find_best_node(
                store=store,
                fragment=fact,
                cfg=cfg,
                chat_model=chat_model,
                branch_root_id=BRANCH_WORLD,
                deadline=deadline,
            )
        except Exception as e:
            debug_log(f"graph update: traversal failed for '{fact[:50]}...' — {e}", "memory")
            continue
        pending.append((fact, node_id))

    if not pending:
        debug_log("graph update: nothing to merge after dedupe", "memory")
        return GraphUpdateResult(stored=[], skipped=skipped)

    # What the window rests on, decided once for the whole flush: every
    # fact here came out of the same summary of the same window.
    # Only what actually consulted something decides the label: a window
    # that merely wrote a reminder looked nothing up.
    source = source_for_tools(lookup_tools(tools_used) if tools_used else tools_used)

    # Grouped bare, so the merge is never handed a format to preserve.
    # The suffix is written by the plain-append path, and put back by
    # `_reattach_sources` after a rewrite.
    by_node: dict[str, list[str]] = {}
    for fact, node_id in pending:
        by_node.setdefault(node_id, []).append(fact)

    stored: "list[tuple[str, str]]" = []
    merged: "list[tuple[str, str]]" = []  # (node_id, node_name)
    for node_id, node_facts in by_node.items():
        node = store.get_node(node_id)
        node_name = node.name if node else node_id[:8]

        # What every claim in play rests on, read before the rewrite can
        # lose it: the node's existing lines keep whatever they carried,
        # and this flush's facts get the window's source.
        sources = _source_index((node.data or "").split("\n") if node else [])
        sources.update(_source_index(
            fact_line(f, source, date_utc) for f in node_facts))

        # Step 4: Merge — combine the existing node data with all
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
                deadline=deadline,
                node=node,
            )
        except Exception as e:
            debug_log(f"graph update: merge failed for node '{node_name}' — {e}", "memory")

        if merge_result.success:
            # The rewrite hands lines back bare. Give them their
            # provenance again before anything else reads the node: a
            # line it reworded matches nothing and stays `inconnu`,
            # which is the truthful reading of a sentence nobody can
            # trace any more.
            rewritten = store.get_node(node_id)
            if rewritten and rewritten.data:
                repare = "\n".join(
                    _reattach_sources(rewritten.data.split("\n"), sources))
                if repare != rewritten.data:
                    store.update_node(node_id, data=repare)

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
                store.append_to_node(node_id, fact_line(fact, source, date_utc))
                stored.append((fact, node_name))
                debug_log(
                    f"graph update: appended '{fact[:50]}...' → "
                    f"'{node_name}' (merge skipped)",
                    "memory",
                )

        store.touch_node(node_id)
        merged.append((node_id, node_name))

    # Step 5: Auto-split the nodes that have grown too large, once every
    # merge of the flush is done. A split can wait for the next flush; a
    # merge the deadline turns away drops its facts to plain append.
    for node_id, node_name in merged:
        refreshed = store.get_node(node_id)
        if refreshed is None or refreshed.data_token_count <= SPLIT_THRESHOLD:
            continue
        if not _split_worth_asking(store, refreshed, chat_model):
            _, refused_at = _SPLIT_REFUSALS[(store.db_path, node_id)]
            debug_log(
                f"graph update: node '{node_name}' is over the threshold, but "
                f"{chat_model} refused its split at {refused_at} tokens; asked "
                f"again from {math.ceil(refused_at * (1 + _SPLIT_RETRY_GROWTH))}",
                "memory",
            )
            continue
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
                deadline=deadline,
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

    Each node gets one chat-model timeout of its own. The sweep is asked
    for and streams its progress, so it is bounded per node rather than as
    a whole.
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

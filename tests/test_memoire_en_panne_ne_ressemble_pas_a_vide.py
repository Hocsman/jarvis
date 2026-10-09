"""A memory pass that could not run, against one that ran and found nothing.

Two questions arrive at the same place. "What time is it?" needs no
memory, and the extractor is trained to say so. A backend that timed out
also produces no keywords. Today both hand back ``{}``, both print
nothing, and both leave the turn memory-blind — one correctly, one
because the machinery broke.

The digest below it inverts the stakes. There, "nothing relevant" is an
instruction to throw the raw diary and graph blocks away. A distil call
that never happened says the same word, so a failed LLM call deletes
what she had already found.

The engine already knows this rule in three other places: the graph
prints when it cannot be read, the core profile prints when it cannot be
read, and the tool router refuses to cache its own fall-open. These are
the same rule applied to the two passes that were missing it.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ── The extractor: a failed read is not an empty answer ────────────────


def _cfg_plat():
    from types import SimpleNamespace

    return SimpleNamespace(
        llm_chat_model="m",
        ollama_base_url="http://x",
        ollama_chat_model="m",
        llm_provider="ollama",
    )


def test_a_dead_backend_is_not_a_query_that_needed_no_memory():
    """The two states the caller has to tell apart.

    `{"keywords": []}` is the extractor reading the question and deciding
    it needs no memory search — the prompt trains it to answer exactly
    that for "what time is it?". A backend that returned nothing did not
    read anything.
    """
    from src.jarvis.reply.enrichment import extract_search_params_for_memory

    with patch("src.jarvis.reply.enrichment.call_llm_direct", return_value=None):
        en_panne = extract_search_params_for_memory("q", _cfg_plat(), "m", timeout_sec=0.1)

    with patch("src.jarvis.reply.enrichment.call_llm_direct",
               return_value='{"keywords": []}'):
        rien_a_chercher = extract_search_params_for_memory("q", _cfg_plat(), "m", timeout_sec=0.1)

    assert en_panne != rien_a_chercher
    assert rien_a_chercher == {"keywords": []}
    assert en_panne is None


def test_prose_instead_of_json_is_also_a_failed_read():
    """Two attempts, both unusable. Nothing was extracted, so nothing is
    claimed to have been extracted."""
    from src.jarvis.reply.enrichment import extract_search_params_for_memory

    with patch("src.jarvis.reply.enrichment.call_llm_direct",
               return_value="Sure! Here are some search parameters."):
        assert extract_search_params_for_memory(
            "q", _cfg_plat(), "m", timeout_sec=0.1) is None


def test_no_model_configured_is_a_read_that_did_not_happen():
    """The short-circuit still burns no LLM call, and still says plainly
    that no extraction took place."""
    from src.jarvis.reply.enrichment import extract_search_params_for_memory

    with patch("src.jarvis.reply.enrichment.call_llm_direct") as appel:
        assert extract_search_params_for_memory(
            "q", _cfg_plat(), "", timeout_sec=0.1) is None
    appel.assert_not_called()


# ── The digest: a failed distil is not a verdict of irrelevance ────────


def _digest_kwargs():
    return dict(
        query="what does he think of it?",
        cfg=_cfg_plat(),
        chat_model="m",
        timeout_sec=0.1,
    )


def test_a_broken_distil_is_not_an_empty_digest():
    """Empty means the distil read the snippets and judged none relevant,
    which licenses the caller to drop them. A call that never completed
    licenses nothing."""
    from src.jarvis.reply.enrichment import (
        MemoryDigestError,
        digest_memory_for_query,
    )

    entree = "[2026-04-20] " + ("il a parlé du film pendant une heure " * 20)

    with patch("src.jarvis.reply.enrichment.call_llm_direct",
               side_effect=RuntimeError("boom")):
        with pytest.raises(MemoryDigestError):
            digest_memory_for_query(diary_entries=[entree], graph_parts=[],
                                    **_digest_kwargs())

    with patch("src.jarvis.reply.enrichment.call_llm_direct", return_value="NONE"):
        assert digest_memory_for_query(diary_entries=[entree], graph_parts=[],
                                       **_digest_kwargs()) == ""


def test_a_timed_out_distil_is_not_an_empty_digest():
    """The shape a timeout actually takes: the backends return None rather
    than raising, so the guard that only watches for exceptions misses the
    most common failure of all."""
    from src.jarvis.reply.enrichment import (
        MemoryDigestError,
        digest_memory_for_query,
    )

    entree = "[2026-04-20] " + ("il a parlé du film pendant une heure " * 20)

    with patch("src.jarvis.reply.enrichment.call_llm_direct", return_value=None):
        with pytest.raises(MemoryDigestError):
            digest_memory_for_query(diary_entries=[entree], graph_parts=[],
                                    **_digest_kwargs())


# ── And what the engine does with each ─────────────────────────────────


class _Noeud:
    def __init__(self, name, data):
        self.id, self.name, self.data = name, name, data
        self.data_token_count = len(data) // 4


# A single node's data is cut at 300 characters before it reaches the
# digest, and the digest leaves any block under 400 alone. Two nodes clear
# the bar, so the distil call actually happens.
_DEUX_NOEUDS = [
    _Noeud("World", "Le film de Guzmán s'appelle Nostalgie de la lumière. " * 5),
    _Noeud("Cinema", "Il l'a vu au cinéma de la rue Pasteur un dimanche. " * 5),
]


def _tour(tmp_path, *, modele="test-large", digest=False, extracteur=None,
          direct=None, noeuds=None, memoire=None):
    """One engine turn, handing back what reached the system prompt."""
    from src.jarvis.memory.conversation import DialogueMemory
    from src.jarvis.reply.engine import run_reply_engine

    cfg = MagicMock()
    cfg.db_path = str(tmp_path / "t.db")
    cfg.llm_chat_model = modele
    cfg.mcps = {}
    cfg.voice_debug = False
    cfg.memory_enrichment_source = "all"
    cfg.memory_enrichment_max_results = 3
    cfg.memory_digest_enabled = digest
    cfg.tool_result_digest_enabled = False
    cfg.location_enabled = False
    cfg.llm_thinking_enabled = False
    cfg.agentic_max_turns = 8
    cfg.tool_selection_strategy = "all"
    cfg.tool_carryover_max_turns = 2
    cfg.tool_carryover_per_entry_chars = 1200
    cfg.tool_search_max_calls = 3
    cfg.llm_chat_timeout_sec = 45.0
    cfg.llm_tools_timeout_sec = 8.0

    store = MagicMock()
    store.search_nodes.return_value = noeuds if noeuds is not None else []
    store.get_ancestors.return_value = [_Noeud("World", "")]

    vu = {}

    def _capture(**kw):
        messages = kw.get("messages") or []
        vu.setdefault("system", messages[0].get("content", "") if messages else "")
        return {"message": {"content": "ok"}}

    piles = [
        patch("src.jarvis.reply.engine.plan_query",
              return_value=["searchMemory topic='x'", "Reply."]),
        patch("src.jarvis.memory.graph.GraphMemoryStore", return_value=store),
        patch("src.jarvis.reply.engine.chat_with_messages", side_effect=_capture),
        patch("src.jarvis.reply.engine.extract_text_from_response", return_value="ok"),
    ]
    if extracteur is not None:
        piles.append(patch("src.jarvis.reply.engine.extract_search_params_for_memory",
                           return_value=extracteur))
    if direct is not None:
        piles.append(patch("src.jarvis.reply.enrichment.call_llm_direct", **direct))

    from contextlib import ExitStack

    with ExitStack() as pile:
        for p in piles:
            pile.enter_context(p)
        run_reply_engine(
            db=MagicMock(), cfg=cfg, tts=None,
            text="qu'est-ce qu'il pense du film de Guzmán ?",
            dialogue_memory=memoire if memoire is not None else DialogueMemory(),
            origin="chat",
        )
    return store, vu.get("system", "")


def test_the_console_says_when_the_keyword_pass_could_not_run(tmp_path, capsys):
    """Same turn, same silence in the prompt, two different reasons. Only
    one of them is her working correctly, and he has to be able to see
    which."""
    _tour(tmp_path, direct={"return_value": None})
    en_panne = capsys.readouterr().out

    _tour(tmp_path, direct={"return_value": '{"keywords": []}'})
    rien_a_chercher = capsys.readouterr().out

    assert "⚠️" in en_panne
    assert "⚠️" not in rien_a_chercher


def test_one_bad_minute_is_not_pinned_to_the_whole_conversation(tmp_path):
    """The hot cache has no age-based expiry. Caching a failed extraction
    keeps answering the same question memory-blind long after the backend
    came back — the reason the tool router refuses to cache its own
    fall-open."""
    from src.jarvis.memory.conversation import DialogueMemory

    memoire = DialogueMemory()
    faits = [_Noeud("World", "Le film de Guzmán s'appelle Nostalgie de la lumière.")]

    _tour(tmp_path, direct={"return_value": None}, noeuds=faits, memoire=memoire)

    store, prompt = _tour(
        tmp_path,
        direct={"return_value": '{"keywords": ["guzman", "film"]}'},
        noeuds=faits, memoire=memoire,
    )

    assert store.search_nodes.called
    assert "Nostalgie de la lumière" in prompt


def test_a_digest_that_could_not_run_keeps_the_memory_it_could_not_read(tmp_path):
    """The engine's own `except` around this step is a fail-open: when the
    step raises, the clearing below it never happens. It was unreachable
    because the failure was swallowed one call earlier."""
    _, prompt = _tour(
        tmp_path, modele="gemma4:e2b", digest=True,
        extracteur={"keywords": ["guzman", "film"]},
        direct={"return_value": None},
        noeuds=_DEUX_NOEUDS,
    )

    assert "Nostalgie de la lumière" in prompt


def test_a_broken_digest_and_an_empty_one_do_not_look_alike(tmp_path, capsys):
    """The guard that keeps the fix honest: a distil that ran and found
    nothing relevant must still clear the raw blocks, or the digest stops
    doing its job for every small model."""
    commun = dict(modele="gemma4:e2b", digest=True,
                  extracteur={"keywords": ["guzman", "film"]}, noeuds=_DEUX_NOEUDS)

    _, prompt_panne = _tour(tmp_path, direct={"return_value": None}, **commun)
    sortie_panne = capsys.readouterr().out

    _, prompt_vide = _tour(tmp_path, direct={"return_value": "NONE"}, **commun)
    sortie_vide = capsys.readouterr().out

    assert "Nostalgie de la lumière" in prompt_panne
    assert "Nostalgie de la lumière" not in prompt_vide
    assert "⚠️" in sortie_panne
    assert "no directly-relevant past memory" in sortie_vide

"""When she cannot read his profile, she says so.

The core is the one thing that makes her *his* assistant: his name, where
he lives, the rules he gave her. It is loaded on every attended turn from
two files he owns and edits by hand — so a permission change, a half-
written line, a disk hiccup are all ordinary.

The load sat inside a `try` whose `except` only wrote a debug line.
`warm_profile_block` stayed empty, the prompt went out without him in it,
and she answered a stranger. Nothing on screen, nothing in the reply,
nothing anywhere except a channel nobody reads — and "she seems to have
forgotten me" is indistinguishable from a model having a bad day.

The turn must still happen: an unreadable file is not a reason to stop
answering. But it is a reason to say so, once, where he will see it.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _turn(tmp_path, *, casse=False, capsys=None):
    from src.jarvis.memory.conversation import DialogueMemory
    from src.jarvis.reply.engine import run_reply_engine

    cfg = MagicMock()
    cfg.db_path = str(tmp_path / "t.db")
    cfg.llm_chat_model = "test-large-model"
    cfg.mcps = {}
    cfg.voice_debug = False
    cfg.memory_enrichment_source = "diary"
    cfg.location_enabled = False
    cfg.llm_thinking_enabled = False
    cfg.agentic_max_turns = 4
    cfg.tool_selection_strategy = "all"
    cfg.planner_enabled = False

    from src.jarvis.memory.core import SECTION_PROFILE, MemoryCore
    MemoryCore.for_config(cfg).remember(SECTION_PROFILE, "Il s'appelle Hocine.")

    vu = {}

    def _chat(**kw):
        messages = kw.get("messages") or []
        vu.setdefault("system", messages[0].get("content", "") if messages else "")
        return {"message": {"content": "ok"}}

    piles = [
        patch("src.jarvis.reply.engine.chat_with_messages", side_effect=_chat),
        patch("src.jarvis.reply.engine.extract_text_from_response", return_value="ok"),
        patch("src.jarvis.reply.engine.plan_query", return_value=[]),
    ]
    if casse:
        # Imported inside the function, so it is patched at its source.
        piles.append(patch("src.jarvis.memory.core.build_core_profile",
                           side_effect=OSError("permission refusée")))
    for p in piles:
        p.start()
    try:
        run_reply_engine(db=MagicMock(), cfg=cfg, tts=None, text="qui suis-je ?",
                         dialogue_memory=DialogueMemory(), origin="chat")
    finally:
        for p in reversed(piles):
            p.stop()
    return vu.get("system", "")


def test_the_profile_reaches_the_prompt_when_it_can_be_read(tmp_path):
    assert "Hocine" in _turn(tmp_path)


def test_an_unreadable_core_still_answers(tmp_path):
    """Refusing to reply because a file would not open would be worse
    than replying without it."""
    prompt = _turn(tmp_path, casse=True)

    assert prompt


def test_an_unreadable_core_is_said_out_loud(tmp_path, capsys):
    """The whole defect: it went to a debug channel nobody reads, and
    "she has forgotten me" looks exactly like a model having a bad day."""
    _turn(tmp_path, casse=True)

    sortie = capsys.readouterr().out
    assert "🪨" in sortie or "profil" in sortie.lower() or "core" in sortie.lower()
    assert any(m in sortie.lower() for m in ("pas pu", "could not", "⚠️"))


def test_the_model_is_told_it_is_working_blind(tmp_path):
    """So it can say "I cannot read your profile right now" instead of
    confidently answering as if he had never told it anything."""
    prompt = _turn(tmp_path, casse=True)

    assert any(m in prompt.lower() for m in
               ("could not be read", "not available", "unavailable")), prompt[:400]

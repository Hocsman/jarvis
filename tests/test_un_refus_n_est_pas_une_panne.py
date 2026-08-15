"""« Elle n'a pas le droit » and « ça n'a pas marché » are different facts.

`ToolExecutionResult.refused` exists so a caller can tell them apart, and
its own docstring says callers deciding whether to try again must check
it first. The engine never reads it. A refusal comes back with
`success=False`, so it collects the banner written for a *failure*:
"no data was returned, tell the user you could not retrieve it and ask
for what is missing" — over a refusal text that says, verbatim, not to
offer to try again and not to suggest a way around it.

Then the message is stamped `tool_failed`, which the carry-over guard
reads to re-widen next turn's allow-list with the tool he forbade. He
says no, she asks him for the missing piece, calls it again, and the
ledger gains a `refusé` line every turn.

The refusal text is written by code and is already addressed to the
model. It needs no banner and no second chance.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.jarvis.memory.conversation import DialogueMemory
from src.jarvis.reply.engine import run_reply_engine
from src.jarvis.tools.types import ToolExecutionResult

from tests.test_engine_tool_carryover_guard import _mock_cfg


def _appel(nom, call_id="a1"):
    return {"message": {"content": "", "tool_calls": [{
        "id": call_id, "type": "function",
        "function": {"name": nom, "arguments": {}},
    }]}}


def _tour(resultat, capsys=None):
    """One turn whose single tool call comes back with `resultat`."""
    vu = {}

    def _capture(**kw):
        msgs = kw.get("messages") or []
        for m in msgs:
            if m.get("tool_name") or m.get("role") == "tool":
                vu.setdefault("messages", []).append(m)
        if "appelé" not in vu:
            vu["appelé"] = True
            return _appel("deleteMeal")
        return {"message": {"content": "compris."}}

    with patch("src.jarvis.memory.core.format_warm_profile_block", return_value=""), \
         patch("src.jarvis.memory.core.build_core_profile",
               return_value={"user": "", "directives": ""}), \
         patch("src.jarvis.memory.graph.GraphMemoryStore"), \
         patch("src.jarvis.reply.engine.plan_query", return_value=[]), \
         patch("src.jarvis.reply.engine.extract_search_params_for_memory", return_value={}), \
         patch("src.jarvis.reply.engine.run_tool_with_retries", return_value=resultat), \
         patch("src.jarvis.reply.engine.extract_text_from_response",
               side_effect=["", "compris."]), \
         patch("src.jarvis.reply.engine.chat_with_messages", side_effect=_capture), \
         patch("src.jarvis.reply.engine.select_tools", return_value=["deleteMeal"]):
        run_reply_engine(db=MagicMock(), cfg=_mock_cfg(), tts=None,
                         text="supprime le repas de midi",
                         dialogue_memory=DialogueMemory())
    return vu.get("messages", [])


REFUS = ToolExecutionResult(
    success=False, refused=True,
    reply_text=('The tool "deleteMeal" was not run: the user set it to jamais. '
                "Tell them that, and do not offer to try again or suggest a way "
                "around it."),
)

# A failure that came back with text, which is the branch the banner
# lives on — the same branch a refusal takes.
PANNE = ToolExecutionResult(
    success=False, reply_text="the weather API returned 503",
)


def test_a_refusal_is_not_dressed_up_as_a_failure():
    """The banner tells the model to ask for what is missing. Nothing is
    missing: he said no."""
    messages = _tour(REFUS)

    assert messages, "le résultat doit atteindre le modèle"
    texte = "\n".join(m.get("content", "") for m in messages)
    assert "FAILED" not in texte
    assert "ask for what is missing" not in texte
    assert "was not run" in texte


def test_a_refusal_does_not_buy_the_tool_another_turn():
    """`tool_failed` is what re-widens the allow-list next turn. A tool he
    forbade must not come back through the door he closed."""
    messages = _tour(REFUS)

    assert not any(m.get("tool_failed") for m in messages)


def test_a_real_failure_still_gets_its_banner():
    """The control, and it matters: the banner exists because a failed
    tool is where models invent a plausible figure."""
    messages = _tour(PANNE)

    texte = "\n".join(m.get("content", "") for m in messages)
    assert "FAILED" in texte
    assert any(m.get("tool_failed") for m in messages)

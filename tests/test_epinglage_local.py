"""Pinning a model for these four contexts means running it here.

Four prompts carry the user's own sentences about their own life: the
time they said out loud, the routine they described, the goal they are
working towards, and the fortnight of diary the learning step reads. All
four resolve their model through the same chain, and all four specs say
the same thing — pinning a model is how you keep that sentence off the
network.

It was not true. `get_llm_backend` picks the endpoint from
`llm_provider` alone and never looks at the model, so on a cloud
provider a pinned local tag was sent *to the cloud*: verified, the
request went to `https://openrouter.ai/api/v1` carrying his sentence,
and the tag would have 400'd there anyway. The setting changed a name
and nothing else, while its documentation promised privacy.

So the pin now decides the destination as well. With nothing pinned
there is nothing to honour, and the ordinary provider applies — a user
who never asked for this is not quietly moved.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def _cfg(**kw):
    base = dict(
        llm_provider="openai_compatible",
        llm_base_url="https://openrouter.ai/api/v1",
        ollama_base_url="http://127.0.0.1:11434",
        llm_chat_model="deepseek/deepseek-v4-flash",
        llm_api_key="",
        auto_redact_before_cloud=True,
        reminder_model="",
        tool_router_model="",
        intent_judge_model="",
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _destination(backend):
    """Where a request would actually go, through any wrapper."""
    vu = backend
    for _ in range(4):
        url = getattr(vu, "base_url", None) or getattr(vu, "_base_url", None)
        if url:
            return url
        vu = (getattr(vu, "_inner", None) or getattr(vu, "backend", None)
              or getattr(vu, "_backend", None))
        if vu is None:
            break
    return None


# ── What the pin now buys ─────────────────────────────────────────────


def test_a_pinned_model_runs_where_it_lives(): 
    from src.jarvis.llm import get_private_backend

    b = get_private_backend(_cfg(), "gemma4:e2b")

    assert "127.0.0.1" in (_destination(b) or ""), _destination(b)


def test_without_a_pin_nothing_moves():
    """Somebody who never asked for this keeps the provider they chose."""
    from src.jarvis.llm import get_llm_backend, get_private_backend

    cfg = _cfg()

    assert _destination(get_private_backend(cfg, "")) == _destination(
        get_llm_backend(cfg))


def test_an_ollama_user_is_unaffected():
    from src.jarvis.llm import get_private_backend

    b = get_private_backend(_cfg(llm_provider="ollama"), "gemma4:e2b")

    assert "127.0.0.1" in (_destination(b) or "")


def test_it_does_not_touch_the_ordinary_backend():
    """The chat model, the router and the judges are unchanged: they do
    not carry a sentence about his life and moving them would be a
    performance decision nobody asked for."""
    from src.jarvis.llm import get_llm_backend

    assert "openrouter" in (_destination(get_llm_backend(_cfg())) or "")


# ── The four contexts that promised it ────────────────────────────────


@pytest.mark.parametrize("module,fonction", [
    ("src.jarvis.reminders.extract", "extract_reminder_time"),
    ("src.jarvis.routines.extract", "extract_routine_rule"),
])
def test_the_contexts_that_read_his_life_use_it(module, fonction):
    """Pinned in, and the request must not leave the machine."""
    import importlib
    from unittest.mock import patch

    mod = importlib.import_module(module)
    cfg = _cfg(reminder_model="gemma4:e2b")

    # The provider-only resolver must not be reachable from here at all:
    # its presence is what made the pin cosmetic.
    assert not hasattr(mod, "get_llm_backend"), (
        f"{module} can still resolve its endpoint by provider alone")

    with patch(f"{module}.get_private_backend") as prive:
        prive.return_value.chat.return_value = None
        try:
            getattr(mod, fonction)(cfg, "dans vingt minutes")
        except Exception:
            pass

    assert prive.called, f"{fonction} did not route through the pin"
    _, epingle = prive.call_args[0]
    assert epingle == "gemma4:e2b", (
        f"the pin did not reach the backend choice: {epingle!r}")


def test_the_promise_is_the_same_for_all_four():
    """One helper, so the four cannot drift apart. Three of them were
    written by copying the fourth."""
    import src.jarvis.appris.propose as appris
    import src.jarvis.objectifs.juge as juge
    import src.jarvis.reminders.extract as rappels
    import src.jarvis.routines.extract as routines

    for mod in (rappels, routines, juge, appris):
        assert hasattr(mod, "get_private_backend"), mod.__name__

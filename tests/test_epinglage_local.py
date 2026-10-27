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

Then the same leak one link further down. The destination followed the
context's own pin while the name followed the whole chain, so with
`appris_model` empty and `reminder_model: qwen2.5:3b` the journal reader
sent ten diary summaries to OpenRouter under a local tag. The two are
now one decision: the request goes where the model it names lives.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import requests


@pytest.fixture(autouse=True)
def _backends_neufs():
    """Backends are cached by endpoint; each test starts from none."""
    from src.jarvis.llm import clear_backend_cache

    clear_backend_cache()
    yield
    clear_backend_cache()


def _cfg(**kw):
    base = dict(
        llm_provider="openai_compatible",
        llm_base_url="https://openrouter.ai/api/v1",
        ollama_base_url="http://127.0.0.1:11434",
        llm_chat_model="deepseek/deepseek-v4-flash",
        llm_api_key="",
        auto_redact_before_cloud=True,
        appris_model="",
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


# ── The four contexts, driven from their public entry points ──────────


def _lire_le_journal(cfg):
    from src.jarvis.appris.propose import propositions

    class _Journal:
        def get_recent_conversation_summaries(self, days):
            return [{"date_utc": "2026-10-20",
                     "summary": "Il habite à Lyon depuis dix ans."}]

        def journal_deja_lu(self):
            return {}

        def jours_ou_elle_a_parle(self):
            return set()

    propositions(cfg, _Journal(), core=None, deja=())


def _lire_un_rappel(cfg):
    from src.jarvis.reminders.extract import extract_reminder_time

    extract_reminder_time(cfg, "dans vingt minutes")


def _lire_une_routine(cfg):
    from src.jarvis.routines.extract import extract_routine_rule

    extract_routine_rule(cfg, "tous les lundis matin")


def _juger_un_objectif(cfg):
    from src.jarvis.objectifs.juge import peut_etre_fini
    from src.jarvis.objectifs.page import Objectif, Point

    peut_etre_fini(cfg, Objectif(
        nom="semi", phrase="courir un semi-marathon",
        fini_quand="le semi est couru",
        points=[Point(date="2026-10-20", source="dit", texte="couru 15 km")],
    ))


_CONTEXTES = {
    "#16 rappels": _lire_un_rappel,
    "#17 routines": _lire_une_routine,
    "#18 objectifs": _juger_un_objectif,
    "#19 journal": _lire_le_journal,
}


def _requetes(lire, cfg):
    """Every request the context sends, as (url, model), sent nowhere."""
    parties = []

    def _capter(url, **kw):
        parties.append((url, (kw.get("json") or {}).get("model")))
        raise requests.exceptions.Timeout("intercepted")

    def _session_post(self, url, **kw):
        return _capter(url, **kw)

    with patch("requests.post", _capter), \
         patch.object(requests.Session, "post", _session_post):
        try:
            lire(cfg)
        except Exception:
            # Every one of them reports a timeout its own way: raising
            # `ExtractionFailed`, or answering quietly. Where the request
            # went is all this file asks.
            pass
    return parties


def _ici(url):
    return url.startswith("http://127.0.0.1:11434")


# ── What the pin buys ─────────────────────────────────────────────────


def test_a_pinned_model_runs_where_it_lives():
    from src.jarvis.llm import get_private_backend

    b = get_private_backend(_cfg(), "gemma4:e2b", "gemma4:e2b")

    assert "127.0.0.1" in (_destination(b) or ""), _destination(b)


def test_a_pin_keeps_a_namespaced_local_tag_here_too():
    """Stricter than the auxiliary route: an `hf.co/...` Ollama tag has a
    slash, and a pin still means this machine."""
    from src.jarvis.llm import get_private_backend

    tag = "hf.co/bartowski/Qwen2.5-3B-Instruct-GGUF:Q4_K_M"
    b = get_private_backend(_cfg(), tag, tag)

    assert "127.0.0.1" in (_destination(b) or ""), _destination(b)


def test_without_a_pin_nothing_moves():
    """Somebody who never asked for this keeps the provider they chose."""
    from src.jarvis.llm import get_llm_backend, get_private_backend

    cfg = _cfg()

    assert _destination(
        get_private_backend(cfg, "", cfg.llm_chat_model)
    ) == _destination(get_llm_backend(cfg))


def test_without_a_pin_a_bare_tag_still_goes_where_it_lives():
    """A local tag reached through a fallback names a model the remote
    endpoint does not have. Sending it there leaks the sentence and fails."""
    from src.jarvis.llm import get_private_backend

    b = get_private_backend(_cfg(), "", "qwen2.5:3b")

    assert "127.0.0.1" in (_destination(b) or ""), _destination(b)


def test_an_ollama_user_is_unaffected():
    from src.jarvis.llm import get_private_backend

    b = get_private_backend(_cfg(llm_provider="ollama"), "gemma4:e2b",
                            "gemma4:e2b")

    assert "127.0.0.1" in (_destination(b) or "")


def test_it_does_not_touch_the_ordinary_backend():
    """The chat model, the router and the judges are unchanged: they do
    not carry a sentence about his life and moving them would be a
    performance decision nobody asked for."""
    from src.jarvis.llm import get_llm_backend

    assert "openrouter" in (_destination(get_llm_backend(_cfg())) or "")


# ── The four contexts that promised it ────────────────────────────────


@pytest.mark.parametrize("lire", _CONTEXTES.values(), ids=_CONTEXTES.keys())
def test_a_pinned_context_does_not_leave_the_machine(lire):
    cfg = _cfg(reminder_model="gemma4:e2b", appris_model="gemma4:e2b")

    parties = _requetes(lire, cfg)

    assert parties, "the context sent nothing at all"
    assert all(_ici(url) and model == "gemma4:e2b"
               for url, model in parties), parties


@pytest.mark.parametrize("lire", _CONTEXTES.values(), ids=_CONTEXTES.keys())
def test_a_local_tag_reached_by_falling_back_stays_here(lire):
    """No pin of its own; the chain lands on the router's bare tag."""
    cfg = _cfg(tool_router_model="qwen2.5:3b")

    parties = _requetes(lire, cfg)

    assert parties, "the context sent nothing at all"
    assert all(_ici(url) and model == "qwen2.5:3b"
               for url, model in parties), parties


@pytest.mark.parametrize("lire", _CONTEXTES.values(), ids=_CONTEXTES.keys())
def test_nothing_local_anywhere_keeps_the_chosen_provider(lire):
    """Pinning nothing changes nothing: the chat model, where it runs."""
    cfg = _cfg()

    parties = _requetes(lire, cfg)

    assert parties, "the context sent nothing at all"
    assert all(url.startswith("https://openrouter.ai/api/v1")
               and model == cfg.llm_chat_model
               for url, model in parties), parties


def test_the_journal_honours_the_reminder_pin_it_falls_back_to():
    """The user's configuration on 2026-10-27: `appris_model` empty,
    `reminder_model` a local tag, OpenRouter as the provider. The diary
    went to OpenRouter under the local tag."""
    cfg = _cfg(reminder_model="qwen2.5:3b")

    parties = _requetes(_lire_le_journal, cfg)

    assert parties, "the journal was not read at all"
    assert all(_ici(url) and model == "qwen2.5:3b"
               for url, model in parties), parties


def test_the_reminder_pin_keeps_the_journal_here_whatever_its_shape():
    """A pin is a pin for every context that falls back to it, slash or
    not: the reminder context runs this tag here, so the journal must."""
    tag = "hf.co/bartowski/Qwen2.5-3B-Instruct-GGUF:Q4_K_M"
    cfg = _cfg(reminder_model=tag)

    parties = _requetes(_lire_le_journal, cfg)

    assert parties, "the journal was not read at all"
    assert all(_ici(url) and model == tag for url, model in parties), parties


def test_the_promise_is_the_same_for_all_four():
    """One helper, so the four cannot drift apart. Three of them were
    written by copying the fourth."""
    import src.jarvis.appris.propose as appris
    import src.jarvis.objectifs.juge as juge
    import src.jarvis.reminders.extract as rappels
    import src.jarvis.routines.extract as routines

    for mod in (rappels, routines, juge, appris):
        assert hasattr(mod, "get_private_backend"), mod.__name__
        assert not hasattr(mod, "get_llm_backend"), (
            f"{mod.__name__} can still resolve its endpoint by provider alone")

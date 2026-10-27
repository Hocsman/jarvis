"""Where a prompt carrying the user's own life is read.

Four prompts carry the user's own sentences about their own life: the
time they said out loud (#16), the routine they described (#17), the
goal they are working towards (#18), and the fortnight of diary the
learning step reads (#19).

The first two read a sentence the turn sends to the chat model anyway:
`setReminder` and `setRoutine` are tools of the reply loop, and every
reply ends on the chat model. So they are read where that model runs.
With the chat off this machine, the chat model reads it on the reply's
own backend; with the chat at home, the reminder chain and its pin do,
by the rule below.

The other two read what no single turn sends, and keep the chain and its
pin wherever the chat runs. `get_llm_backend` alone picks the endpoint
from `llm_provider` and never looks at the model, so on a cloud provider
it would carry a local tag, and the sentence with it, to the cloud, where
the tag does not even exist. Hence the rule these tests hold, at the
wire:

- a pin means this machine, whatever the tag looks like;
- with no pin, the model the chain resolved decides, as on the auxiliary
  route: a bare tag on a remote provider runs on local Ollama, and a
  namespaced model stays on the provider the user chose;
- the name sent and the destination always come from the same model,
  however far down the chain it was found.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import urlparse

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


# The sentence of the turn: the chat model receives it whatever happens.
_DU_TOUR = {
    "#16 rappels": _lire_un_rappel,
    "#17 routines": _lire_une_routine,
}

# What no single turn sends: the pin holds wherever the chat runs.
_HORS_DU_TOUR = {
    "#18 objectifs": _juger_un_objectif,
    "#19 journal": _lire_le_journal,
}

_CONTEXTES = {**_DU_TOUR, **_HORS_DU_TOUR}


@contextmanager
def _intercepte():
    """Every request sent inside, as (url, body), sent nowhere: each one
    times out."""
    parties = []

    def _capter(url, **kw):
        parties.append((url, kw.get("json") or {}))
        raise requests.exceptions.Timeout("intercepted")

    def _session_post(self, url, **kw):
        return _capter(url, **kw)

    with patch("requests.post", _capter), \
         patch.object(requests.Session, "post", _session_post):
        yield parties


def _envois(lire, cfg):
    """Every request the context sends, as (url, body), sent nowhere.

    A context that raised before anything reached the wire fails here
    with what it raised, rather than later with only "sent nothing".
    """
    with _intercepte() as parties:
        try:
            lire(cfg)
        except Exception as erreur:
            # Every one of them reports a timeout its own way: raising
            # `ExtractionFailed`, or answering quietly. Where the request
            # went is all this file asks, unless it never left.
            if not parties:
                raise AssertionError(
                    f"the context sent nothing at all: {erreur!r}") from erreur
    return parties


def _requetes(lire, cfg):
    """Every request the context sends, as (url, model)."""
    return [(url, corps.get("model")) for url, corps in _envois(lire, cfg)]


def _ici(url):
    return url.startswith("http://127.0.0.1:11434")


def _chat_dans_le_cloud(**kw):
    """The chat and the intent judge on OpenRouter, the router and the
    pin on this machine."""
    base = dict(
        llm_chat_model="google/gemini-2.5-flash",
        intent_judge_model="google/gemini-3.1-flash-lite",
        tool_router_model="qwen2.5:3b",
        reminder_model="qwen2.5:3b",
    )
    base.update(kw)
    return _cfg(**base)


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


def test_without_a_pin_a_namespaced_model_keeps_the_provider():
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


# ── #18 and #19: the pin, wherever the chat runs ──────────────────────


@pytest.mark.parametrize("lire", _HORS_DU_TOUR.values(), ids=_HORS_DU_TOUR.keys())
def test_a_pinned_context_does_not_leave_the_machine(lire):
    cfg = _cfg(reminder_model="gemma4:e2b", appris_model="gemma4:e2b")

    parties = _requetes(lire, cfg)

    assert parties, "the context sent nothing at all"
    assert all(_ici(url) and model == "gemma4:e2b"
               for url, model in parties), parties


@pytest.mark.parametrize("lire", _HORS_DU_TOUR.values(), ids=_HORS_DU_TOUR.keys())
def test_a_local_tag_reached_by_falling_back_stays_here(lire):
    """No pin of its own; the chain lands on the router's bare tag."""
    cfg = _cfg(tool_router_model="qwen2.5:3b")

    parties = _requetes(lire, cfg)

    assert parties, "the context sent nothing at all"
    assert all(_ici(url) and model == "qwen2.5:3b"
               for url, model in parties), parties


def test_the_journal_honours_the_reminder_pin_it_falls_back_to():
    """`appris_model` empty, `reminder_model` a local tag, OpenRouter as
    the provider: the fortnight of diary stays here."""
    cfg = _cfg(reminder_model="qwen2.5:3b")

    parties = _requetes(_lire_le_journal, cfg)

    assert parties, "the journal was not read at all"
    assert all(_ici(url) and model == "qwen2.5:3b"
               for url, model in parties), parties


def test_the_reminder_pin_keeps_the_journal_here_whatever_its_shape():
    """A pin is a pin for every context that falls back to it, slash or
    not: the goal judge runs this tag here, so the journal must."""
    tag = "hf.co/bartowski/Qwen2.5-3B-Instruct-GGUF:Q4_K_M"
    cfg = _cfg(reminder_model=tag)

    parties = _requetes(_lire_le_journal, cfg)

    assert parties, "the journal was not read at all"
    assert all(_ici(url) and model == tag for url, model in parties), parties


@pytest.mark.parametrize("lire", _HORS_DU_TOUR.values(), ids=_HORS_DU_TOUR.keys())
def test_a_namespaced_router_reached_by_falling_back_keeps_the_provider(lire):
    """No pin, and the chain lands on a cloud ID: that model lives on the
    provider, and so does the request. Nothing is moved that was not
    asked to be."""
    cfg = _cfg(tool_router_model="openai/gpt-4o-mini")

    parties = _requetes(lire, cfg)

    assert parties, "the context sent nothing at all"
    assert all(url.startswith("https://openrouter.ai/api/v1")
               and model == "openai/gpt-4o-mini"
               for url, model in parties), parties


def test_the_journal_s_own_pin_wins_over_the_reminder_pin():
    cfg = _cfg(appris_model="gemma4:e2b", reminder_model="qwen2.5:3b")

    parties = _requetes(_lire_le_journal, cfg)

    assert parties, "the journal was not read at all"
    assert all(_ici(url) and model == "gemma4:e2b"
               for url, model in parties), parties


def test_the_journal_s_own_pin_keeps_it_here_whatever_its_shape():
    tag = "hf.co/bartowski/Qwen2.5-3B-Instruct-GGUF:Q4_K_M"
    cfg = _cfg(appris_model=tag)

    parties = _requetes(_lire_le_journal, cfg)

    assert parties, "the journal was not read at all"
    assert all(_ici(url) and model == tag for url, model in parties), parties


@pytest.mark.parametrize("lire", _HORS_DU_TOUR.values(), ids=_HORS_DU_TOUR.keys())
def test_with_the_chat_in_the_cloud_the_pin_still_holds(lire):
    """The shape that moves #16 and #17 to the chat model leaves these
    two where they were: on this machine, with the pinned model."""
    parties = _requetes(lire, _chat_dans_le_cloud())

    assert parties, "the context sent nothing at all"
    assert all(_ici(url) and model == "qwen2.5:3b"
               for url, model in parties), parties


# ── #16 and #17: read where the sentence already is ───────────────────


@pytest.mark.parametrize("lire", _DU_TOUR.values(), ids=_DU_TOUR.keys())
def test_with_the_chat_in_the_cloud_the_chat_model_reads_it(lire):
    """The turn that calls `setReminder` or `setRoutine` sends this
    sentence to the chat model anyway. Read here as well, it would be
    kept no more private and read worse, so the pin does not apply."""
    cfg = _chat_dans_le_cloud()

    parties = _requetes(lire, cfg)

    assert parties, "the context sent nothing at all"
    assert all(url.startswith("https://openrouter.ai/api/v1")
               and model == cfg.llm_chat_model
               for url, model in parties), parties


@pytest.mark.parametrize("lire", _DU_TOUR.values(), ids=_DU_TOUR.keys())
def test_a_cloud_reading_that_fails_is_said_and_not_tried_again(lire):
    """The chat model times out: one request, and a refusal she speaks.
    A second attempt on the small local model would be a reading wrong
    one time in two, not a safety net."""
    from src.jarvis.reminders.extract import ExtractionFailed

    with _intercepte() as parties, pytest.raises(ExtractionFailed):
        lire(_chat_dans_le_cloud())

    assert len(parties) == 1, [url for url, _ in parties]
    assert not _ici(parties[0][0]), parties[0][0]


def test_with_the_chat_remote_and_no_chat_model_nothing_reads_the_time():
    """The pin is not a fallback: with no chat model to read it, the
    channel says it is shut and `setReminder` refuses out loud."""
    from src.jarvis.reminders.extract import reminder_channel_available

    assert reminder_channel_available(
        _chat_dans_le_cloud(llm_chat_model="")) is False


@pytest.mark.parametrize("lire", _DU_TOUR.values(), ids=_DU_TOUR.keys())
def test_a_cloud_chat_named_without_a_vendor_reads_where_it_replies(lire):
    """`gpt-4o-mini` on api.openai.com has no slash. The reply goes there
    by provider, and the reading follows the reply, not the tag's shape."""
    cfg = _cfg(llm_base_url="https://api.openai.com/v1",
               llm_chat_model="gpt-4o-mini", reminder_model="qwen2.5:3b")

    parties = _requetes(lire, cfg)

    assert parties, "the context sent nothing at all"
    assert all(url.startswith("https://api.openai.com/v1")
               and model == "gpt-4o-mini"
               for url, model in parties), parties


def test_the_cloud_reading_is_scrubbed_like_the_reply():
    """Same backend as the reply, so the same redaction: an address the
    chat model never saw does not reach it through the reminder."""
    from src.jarvis.reminders.extract import extract_reminder_time

    adresse = "jean.dupont@exemple.fr"

    envoyes = _envois(
        lambda cfg: extract_reminder_time(
            cfg, f"rappelle-moi demain d'écrire à {adresse}"),
        _chat_dans_le_cloud(),
    )

    assert envoyes, "nothing was sent at all"
    assert adresse not in json.dumps(envoyes, ensure_ascii=False)


_SERVEURS_A_LA_MAISON = {
    "LM Studio on this machine": "http://localhost:1234/v1",
    "a box on the home network": "http://192.168.1.42:8000/v1",
}


@pytest.mark.parametrize("serveur", _SERVEURS_A_LA_MAISON.values(),
                         ids=_SERVEURS_A_LA_MAISON.keys())
@pytest.mark.parametrize("lire", _DU_TOUR.values(), ids=_DU_TOUR.keys())
def test_with_an_openai_compatible_server_at_home_nothing_leaves_home(lire, serveur):
    """A server on this machine or this network counts as home: the
    chain and its pin, and no request to any other host."""
    cfg = _cfg(llm_base_url=serveur, llm_chat_model="qwen2.5-7b-instruct",
               reminder_model="qwen2.5:3b")

    parties = _requetes(lire, cfg)

    assert parties, "the context sent nothing at all"
    assert all(urlparse(url).hostname in ("localhost", "127.0.0.1",
                                          urlparse(serveur).hostname)
               and model == "qwen2.5:3b"
               for url, model in parties), parties


# ── Where the reply goes ──────────────────────────────────────────────

_COTE_DE_LA_REPONSE = {
    "OpenRouter": (dict(), True),
    "a cloud provider spelt loosely": (dict(llm_provider=" OpenAI_Compatible "), True),
    "LM Studio on this machine": (dict(llm_base_url="http://localhost:1234/v1"), False),
    "a box on the home network": (dict(llm_base_url="http://192.168.1.42:8000/v1"), False),
    "no URL of its own, Ollama's then": (dict(llm_base_url=""), False),
}


@pytest.mark.parametrize("reglages,distant", _COTE_DE_LA_REPONSE.values(),
                         ids=_COTE_DE_LA_REPONSE.keys())
def test_the_side_is_where_the_reply_goes(reglages, distant):
    """`chat_is_remote` and the reply's own backend agree, so a reading
    that follows the chat lands where the reply does."""
    from src.jarvis.llm import chat_is_remote, get_llm_backend

    cfg = _cfg(**reglages)
    hote = urlparse(_destination(get_llm_backend(cfg)) or "").hostname

    assert chat_is_remote(cfg) is distant
    assert (hote not in ("localhost", "127.0.0.1", "192.168.1.42")) is distant, hote


def test_the_ollama_provider_counts_as_home_whatever_its_url():
    """The convention `get_private_backend` already keeps: with Ollama as
    the provider, the pinned chain goes to that same Ollama."""
    from src.jarvis.llm import chat_is_remote

    assert chat_is_remote(_cfg(llm_provider="ollama",
                               ollama_base_url="https://ollama.example.com")) is False


# ── All four ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("lire", _CONTEXTES.values(), ids=_CONTEXTES.keys())
def test_nothing_local_anywhere_keeps_the_chosen_provider(lire):
    """Pinning nothing changes nothing: the chat model, where it runs."""
    cfg = _cfg()

    parties = _requetes(lire, cfg)

    assert parties, "the context sent nothing at all"
    assert all(url.startswith("https://openrouter.ai/api/v1")
               and model == cfg.llm_chat_model
               for url, model in parties), parties


@pytest.mark.parametrize("lire", _CONTEXTES.values(), ids=_CONTEXTES.keys())
def test_with_the_chat_at_home_the_pin_reads_it_here(lire):
    """An install with nothing in the cloud: all four on this machine,
    with the pinned model."""
    cfg = _cfg(llm_provider="ollama", llm_chat_model="gemma4:e2b",
               reminder_model="qwen2.5:3b")

    parties = _requetes(lire, cfg)

    assert parties, "the context sent nothing at all"
    assert all(_ici(url) and model == "qwen2.5:3b"
               for url, model in parties), parties

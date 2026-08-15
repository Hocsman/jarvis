"""Dictation is the largest thing he says, and it stays on the machine.

`dictation.spec.md` promises the cleanup pass "sends the text to the
local Ollama instance". The pass now dispatches through
`get_llm_backend(cfg)`, which picks its endpoint from `llm_provider`
alone — so on a machine configured for a remote provider, every dictated
sentence leaves, in whatever application he happened to be typing into.

The four contexts that read his own sentences use `get_private_backend`,
where a pinned model decides the destination. Dictation is neither short
nor occasional: it is everything he dictates, all day, into apps that
have nothing to do with the assistant. So it does not merely honour a
pin — it stays local, and the pass silently falls back to the raw
transcript when nothing local answers, exactly as it already does for
every other failure.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest


def _cfg_distant():
    return SimpleNamespace(
        llm_provider="openai_compatible",
        llm_base_url="https://openrouter.ai/api/v1",
        llm_api_key="sk-fake",
        ollama_base_url="http://localhost:11434",
        llm_chat_model="deepseek/deepseek-v4-flash",
        auto_redact_before_cloud=True,
    )


def _url_appelee(cfg) -> str:
    """Where the cleanup pass would actually send the sentence."""
    from src.jarvis.dictation import dictation_engine

    vu = {}

    class _Backend:
        def __init__(self, base_url):
            vu["url"] = base_url

        def direct(self, *a, **kw):
            return None

    def _capture(settings):
        from src.jarvis.llm.factory import _resolve_provider

        provider = _resolve_provider(getattr(settings, "llm_provider", None))
        if provider == "openai_compatible":
            return _Backend(getattr(settings, "llm_base_url", "") or "")
        return _Backend(getattr(settings, "ollama_base_url", "") or "")

    with patch.object(dictation_engine, "get_llm_backend", _capture, create=True), \
         patch("src.jarvis.llm.get_llm_backend", _capture):
        dictation_engine._llm_clean_dictation("euh… rappelle-moi jeudi", cfg)

    return vu.get("url", "")


def test_a_remote_provider_does_not_take_the_dictation_with_it():
    url = _url_appelee(_cfg_distant())

    assert "openrouter" not in url, f"la dictée est partie vers {url}"
    assert "localhost" in url or "127.0.0.1" in url


def test_nothing_local_means_the_raw_transcript_comes_back():
    """The failure mode stays what it already was: no cleanup, and the
    words he dictated, rather than an error or a silence."""
    from src.jarvis.dictation import dictation_engine

    brut = "euh… rappelle-moi jeudi"

    def _mort(settings):
        raise ConnectionError("rien n'écoute")

    with patch("src.jarvis.llm.get_llm_backend", _mort):
        assert dictation_engine._llm_clean_dictation(brut, _cfg_distant()) == brut


def test_an_ordinary_local_setup_still_gets_its_cleanup():
    """The control. A change that routed everything into the void would
    pass the two tests above."""
    from src.jarvis.dictation import dictation_engine

    class _Backend:
        def direct(self, *a, **kw):
            return "rappelle-moi jeudi"

    with patch("src.jarvis.llm.get_llm_backend", lambda s: _Backend()):
        propre = dictation_engine._llm_clean_dictation(
            "euh… rappelle-moi jeudi",
            SimpleNamespace(llm_provider="ollama",
                            ollama_base_url="http://localhost:11434",
                            llm_chat_model="gemma4:e2b"),
        )

    assert propre == "rappelle-moi jeudi"

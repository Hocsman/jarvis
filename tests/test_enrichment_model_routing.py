"""Which model reads a question for the memory lookup (context 3).

`memory_extractor_model` names it. Empty, the lookup rides the tool
router's chain (`tool_router_model → intent_judge_model → llm_chat_model`),
already warm for the router. Named, it wins over that
chain, and the name is dispatched like every auxiliary model: one with a
vendor slash goes to the configured provider, a bare tag to local Ollama.

The key is worth having because the lookup's hard part is the time window:
on 32 dated sentences × 3, a small local router model put a wrong window on
38 % of them (44 % in French), and a wrong window filters the right memory
out without anyone seeing it.

See src/jarvis/reply/reply.spec.md, step 4, and docs/llm_contexts.md, #3.
"""

from __future__ import annotations

import json
from contextlib import ExitStack
from unittest.mock import patch

import pytest


ROUTER = "qwen2.5:3b"
EXTRACTOR = "google/gemini-3.1-flash-lite"
CHAT = "google/gemini-2.5-flash"

_EXTRACTOR_PROMPT = "Extract search parameters"


def _answer_as_extractor(model, system_prompt, asked, where):
    if system_prompt.startswith(_EXTRACTOR_PROMPT):
        asked.append((where, model))
    return json.dumps({"keywords": ["films", "cinema"]})


class _RecordingBackend:
    """Whatever backend the lookup is handed, answering every model."""

    def __init__(self, asked):
        self.asked = asked

    def direct(self, model, system_prompt, user_content, **kwargs):
        return _answer_as_extractor(model, system_prompt, self.asked, "any")


def _run_memory_turn(cfg, db, dialogue_memory, extra_patches=()):
    """One turn whose plan asks for memory; the reply itself is stubbed."""
    from jarvis.reply import engine as engine_mod

    with ExitStack() as stack:
        stack.enter_context(patch(
            "jarvis.memory.conversation.search_conversation_memory_by_keywords",
            return_value=[]))
        stack.enter_context(patch.object(
            engine_mod, "chat_with_messages",
            return_value={"message": {"role": "assistant", "content": "Voilà."}}))
        stack.enter_context(patch.object(engine_mod, "select_tools", return_value=["stop"]))
        stack.enter_context(patch.object(
            engine_mod, "plan_query",
            return_value=["searchMemory topic='films'", "Reply to the user."]))
        for extra in extra_patches:
            stack.enter_context(extra)
        engine_mod.run_reply_engine(
            db=db, cfg=cfg, tts=None,
            text="quels films j'ai regardés hier soir ?",
            dialogue_memory=dialogue_memory,
        )


def _models_the_lookup_asked(cfg, db, dialogue_memory) -> list[str]:
    from jarvis.reply import enrichment

    asked: list[tuple[str, str]] = []
    _run_memory_turn(cfg, db, dialogue_memory, extra_patches=[
        patch.object(enrichment, "get_auxiliary_backend",
                     return_value=_RecordingBackend(asked)),
    ])
    return [model for _, model in asked]


# ── Which model ──────────────────────────────────────────────────────


@pytest.mark.unit
def test_a_named_extractor_model_reads_the_question(mock_config, db, dialogue_memory):
    mock_config.llm_chat_model = CHAT
    mock_config.intent_judge_model = ROUTER
    mock_config.tool_router_model = ROUTER
    mock_config.memory_extractor_model = EXTRACTOR

    assert _models_the_lookup_asked(mock_config, db, dialogue_memory) == [EXTRACTOR]


@pytest.mark.unit
@pytest.mark.parametrize("router,judge,expected", [
    (ROUTER, "judge-m", ROUTER),
    ("", "judge-m", "judge-m"),
    ("", "", CHAT),
])
def test_an_empty_key_leaves_the_lookup_on_the_router_chain(
    mock_config, db, dialogue_memory, router, judge, expected
):
    mock_config.llm_chat_model = CHAT
    mock_config.intent_judge_model = judge
    mock_config.tool_router_model = router
    mock_config.memory_extractor_model = ""

    assert _models_the_lookup_asked(mock_config, db, dialogue_memory) == [expected]


# ── Where it goes ────────────────────────────────────────────────────


class _Endpoint:
    """Stands in for the backend the factory would build for one provider."""

    def __init__(self, where, asked):
        self.where = where
        self.asked = asked

    def direct(self, model, system_prompt, user_content, **kwargs):
        return _answer_as_extractor(model, system_prompt, self.asked, self.where)


def _backends_the_lookup_reached(cfg, db, dialogue_memory) -> list[tuple[str, str]]:
    """Run the turn through the real dispatch; record which provider's
    backend carried the lookup and with which model name."""
    from jarvis.llm import factory

    asked: list[tuple[str, str]] = []

    def _build(provider, base_url, api_key, redact=False, extra_body=None):
        where = "provider" if provider == "openai_compatible" else "ollama"
        return _Endpoint(where, asked)

    _run_memory_turn(cfg, db, dialogue_memory, extra_patches=[
        patch.object(factory, "_build", side_effect=_build),
    ])
    return asked


def _on_a_remote_provider(cfg):
    cfg.llm_provider = "openai_compatible"
    cfg.llm_base_url = "https://openrouter.ai/api/v1"
    cfg.llm_chat_model = CHAT
    cfg.intent_judge_model = ROUTER
    cfg.tool_router_model = ROUTER


@pytest.mark.unit
def test_a_namespaced_extractor_model_goes_to_the_provider(mock_config, db, dialogue_memory):
    _on_a_remote_provider(mock_config)
    mock_config.memory_extractor_model = EXTRACTOR

    assert _backends_the_lookup_reached(mock_config, db, dialogue_memory) == [
        ("provider", EXTRACTOR),
    ]


@pytest.mark.unit
def test_without_the_key_the_lookup_stays_on_the_local_router_model(
    mock_config, db, dialogue_memory
):
    _on_a_remote_provider(mock_config)
    mock_config.memory_extractor_model = ""

    assert _backends_the_lookup_reached(mock_config, db, dialogue_memory) == [
        ("ollama", ROUTER),
    ]


# ── The key itself ───────────────────────────────────────────────────


def _settings_from(tmp_path, monkeypatch, values: dict):
    from jarvis.config import load_settings

    path = tmp_path / "config.json"
    path.write_text(json.dumps(values), encoding="utf-8")
    monkeypatch.setenv("JARVIS_CONFIG_PATH", str(path))
    return load_settings()


@pytest.mark.unit
def test_the_key_is_empty_by_default():
    from jarvis.config import get_default_config

    assert get_default_config()["memory_extractor_model"] == ""


@pytest.mark.unit
def test_a_configured_name_is_read(tmp_path, monkeypatch):
    cfg = _settings_from(tmp_path, monkeypatch, {"memory_extractor_model": f"  {EXTRACTOR} "})

    assert cfg.memory_extractor_model == EXTRACTOR


@pytest.mark.unit
def test_a_bare_tag_that_cannot_run_anywhere_gives_way_to_the_chat_model(tmp_path, monkeypatch):
    """Same rescue as the router's own pin: on a remote endpoint with no
    local Ollama to route a bare tag to, it would fail every turn."""
    from jarvis import config as config_mod

    monkeypatch.setattr(config_mod, "is_ollama_reachable", lambda *a, **k: False)
    cfg = _settings_from(tmp_path, monkeypatch, {
        "llm_provider": "openai_compatible",
        "llm_base_url": "https://openrouter.ai/api/v1",
        "llm_chat_model": CHAT,
        "memory_extractor_model": ROUTER,
    })

    assert cfg.memory_extractor_model == CHAT


@pytest.mark.unit
def test_the_settings_window_offers_the_key_as_free_text():
    """A provider's model name is not in the Ollama catalogue, so the
    field cannot be a fixed list."""
    from desktop_app.settings_window import FIELD_METADATA

    field = next((fm for fm in FIELD_METADATA if fm.key == "memory_extractor_model"), None)
    assert field is not None
    assert field.category == "memory"
    assert field.field_type == "str"


@pytest.mark.unit
def test_resolve_tool_router_model_is_public():
    """The listener imports it to warm the model the router will use, so
    it must stay part of the public API."""
    from jarvis.reply import engine

    assert hasattr(engine, "resolve_tool_router_model")
    assert not hasattr(engine, "_resolve_tool_router_model"), (
        "callers use the public name"
    )

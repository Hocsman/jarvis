"""Pinning a small model to a small task, on a server on his own machine.

`openai_compatible` is not a synonym for "cloud". The repo documents it
as the way to reach LM Studio, oMLX, `llama-server`, vLLM and LocalAI —
all of them local, all of them taking bare model names. The rewrite that
rescues a stale Ollama tag from a remote endpoint's HTTP 400 read the
provider name and concluded "remote", so it threw away every pin on a
local server too, and quietly ran the intent judge, the router, the
planner and the evaluator on the big chat model.

What separates the two is the host, which the function was never given.
And when the pin genuinely has to go, it is now said out loud: a setting
that is ignored with nothing anywhere showing the effective value is a
setting the user cannot debug.
"""

from __future__ import annotations

import json

import pytest

from src.jarvis.config import load_settings


def _settings_from(tmp_path, monkeypatch, values: dict):
    """Settings built from a real config file, the path production uses."""
    path = tmp_path / "config.json"
    path.write_text(json.dumps(values), encoding="utf-8")
    monkeypatch.setenv("JARVIS_CONFIG_PATH", str(path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    import src.jarvis.config as config

    monkeypatch.setattr(config, "_discarded_pins", set(), raising=False)
    return load_settings()


_PINS = ("intent_judge_model", "tool_router_model", "evaluator_model", "planner_model")


def _config_locale(url: str) -> dict:
    valeurs = {
        "llm_provider": "openai_compatible",
        "llm_base_url": url,
        "llm_chat_model": "qwen3-32b-instruct",
    }
    valeurs.update({champ: "qwen3-1.7b-instruct" for champ in _PINS})
    return valeurs


def test_a_server_on_his_own_machine_keeps_every_pin(tmp_path, monkeypatch):
    """The whole point of pinning a 1.7B model to the intent judge is that
    it is not the 32B one."""
    got = _settings_from(tmp_path, monkeypatch,
                         _config_locale("http://localhost:1234/v1"))

    for champ in _PINS:
        assert getattr(got, champ) == "qwen3-1.7b-instruct"


def test_a_box_on_his_network_counts_as_local(tmp_path, monkeypatch):
    """The host decides, not the spelling of the word localhost."""
    got = _settings_from(tmp_path, monkeypatch,
                         _config_locale("http://192.168.1.42:8000/v1"))

    assert got.tool_router_model == "qwen3-1.7b-instruct"


def test_a_remote_endpoint_still_rescues_a_stale_local_tag(tmp_path, monkeypatch):
    """The rewrite is not wrong, it was only wrong about which endpoints
    it applies to. A bare Ollama tag sent to a remote endpoint 400s and
    the auxiliary task dies for no gain."""
    got = _settings_from(tmp_path, monkeypatch, {
        "llm_provider": "openai_compatible",
        "llm_base_url": "https://openrouter.ai/api/v1",
        "llm_chat_model": "deepseek/deepseek-v4-flash",
        "tool_router_model": "gemma4:e2b",
    })

    assert got.tool_router_model == "deepseek/deepseek-v4-flash"


def test_a_discarded_pin_is_announced(tmp_path, monkeypatch, capsys):
    """Nothing downstream shows the effective value: the settings window
    reads the raw JSON off disk, and three of these four fields have no
    field there at all."""
    _settings_from(tmp_path, monkeypatch, {
        "llm_provider": "openai_compatible",
        "llm_base_url": "https://openrouter.ai/api/v1",
        "llm_chat_model": "deepseek/deepseek-v4-flash",
        "tool_router_model": "gemma4:e2b",
    })

    sortie = capsys.readouterr().out
    assert "tool_router_model" in sortie
    assert "gemma4:e2b" in sortie
    assert "deepseek/deepseek-v4-flash" in sortie


def test_the_announcement_does_not_repeat_on_every_reload(tmp_path, monkeypatch, capsys):
    """`debug_log` reloads the settings every couple of seconds. A line
    printed on each reload is a line he learns to scroll past."""
    valeurs = {
        "llm_provider": "openai_compatible",
        "llm_base_url": "https://openrouter.ai/api/v1",
        "llm_chat_model": "deepseek/deepseek-v4-flash",
        "tool_router_model": "gemma4:e2b",
    }
    _settings_from(tmp_path, monkeypatch, valeurs)
    capsys.readouterr()

    load_settings()

    assert "gemma4:e2b" not in capsys.readouterr().out

"""The settings a reminder needs, and one privacy property worth naming.

`_cloud_safe_model` rewrites any pinned model name without a slash to the
cloud chat model whenever the provider is `openai_compatible`. It exists
for a good reason: a stale local Ollama tag sent to a remote endpoint
fails with a 400, and the auxiliary task dies for no gain.

But it is applied per model, and two of them are deliberately left out —
`confirmation_model` and now `reminder_model`. Those two carry the user's
own sentence about their own life: "rappelle-moi d'appeler l'oncologue
jeudi". On a machine whose provider is a remote endpoint, pinning a local
model is the only way to keep that sentence off the network, and running
it through the filter would silently undo exactly that.

The asymmetry is the point, so it is asserted rather than assumed. A
future tidy-up that makes every model name consistent would otherwise
remove the property without a single test going red.
"""

from __future__ import annotations

import pytest

from src.jarvis.config import load_settings


@pytest.fixture
def cfg():
    return load_settings()


def _settings_from(tmp_path, monkeypatch, values: dict):
    """Settings built from a real config file.

    `load_settings` reads the file itself rather than going through
    `load_config`, so this drives the path production uses.
    """
    import json

    path = tmp_path / "config.json"
    path.write_text(json.dumps(values), encoding="utf-8")
    monkeypatch.setenv("JARVIS_CONFIG_PATH", str(path))
    return load_settings()


# ── Defaults ──────────────────────────────────────────────────────────


def test_reminders_are_on_by_default(cfg):
    assert cfg.reminders_enabled is True


@pytest.mark.parametrize("key,default", [
    ("reminder_model", ""),
    ("reminder_timeout_sec", 8.0),
    ("reminder_default_hour", 9),
    ("reminder_tick_sec", 5.0),
    ("reminder_late_grace_sec", 900.0),
    ("reminder_max_attempts", 60),
])
def test_every_setting_has_a_default(cfg, key, default):
    assert hasattr(cfg, key), f"{key} is missing from Settings"
    if not (key == "reminder_model" and getattr(cfg, key)):
        assert getattr(cfg, key) == default


# ── Bounds, because an out-of-range value breaks the promise quietly ──


@pytest.mark.parametrize("key,value,low,high", [
    ("reminder_timeout_sec", 0.0, 2.0, 30.0),
    ("reminder_timeout_sec", 999.0, 2.0, 30.0),
    ("reminder_default_hour", -3, 0, 23),
    ("reminder_default_hour", 47, 0, 23),
    ("reminder_tick_sec", 0.0, 1.0, 60.0),
    ("reminder_tick_sec", 3600.0, 1.0, 60.0),
    ("reminder_late_grace_sec", -1.0, 0.0, 86400.0),
    ("reminder_max_attempts", 0, 1, 600),
])
def test_an_out_of_range_setting_is_clamped(tmp_path, monkeypatch, key, value, low, high):
    """A tick of 0 spins a core; a timeout of 0 makes every reminder
    unreadable. Clamped rather than honoured, and the bound is read from
    the assertion rather than the code so a widened range is a decision
    someone makes on purpose."""
    got = getattr(_settings_from(tmp_path, monkeypatch, {key: value}), key)

    assert low <= got <= high


# ── The privacy asymmetry ─────────────────────────────────────────────


def test_a_pinned_local_model_survives_a_cloud_provider(tmp_path, monkeypatch):
    """The whole reason `reminder_model` exists.

    The user's sentence about their own life goes to whichever model
    reads it. On a machine pointed at a remote endpoint, pinning a local
    tag is the only way to keep it here — and running that pin through
    `_cloud_safe_model` would rewrite it to the cloud model and send the
    sentence anyway.
    """
    got = _settings_from(tmp_path, monkeypatch, {
        "llm_provider": "openai_compatible",
        "llm_chat_model": "deepseek/deepseek-v4-flash",
        "reminder_model": "gemma4:e2b",
    })

    assert got.reminder_model == "gemma4:e2b"


def test_the_same_holds_for_the_approval_judge(tmp_path, monkeypatch):
    """Its sibling, for the same reason, asserted here so the pair is
    protected by one file rather than by a coincidence."""
    got = _settings_from(tmp_path, monkeypatch, {
        "llm_provider": "openai_compatible",
        "llm_chat_model": "deepseek/deepseek-v4-flash",
        "confirmation_model": "gemma4:e2b",
    })

    assert got.confirmation_model == "gemma4:e2b"


def test_an_auxiliary_model_is_still_made_cloud_safe(tmp_path, monkeypatch):
    """The filter is not wrong, it is just wrong for these two. A stale
    local tag on the tool router still gets rescued, or the router
    400s and the whole turn degrades.

    The endpoint is spelled out because it is what decides: the provider
    name alone covers local servers too, and on those a bare name is the
    only shape that works."""
    got = _settings_from(tmp_path, monkeypatch, {
        "llm_provider": "openai_compatible",
        "llm_base_url": "https://openrouter.ai/api/v1",
        "llm_chat_model": "deepseek/deepseek-v4-flash",
        "tool_router_model": "gemma4:e2b",
    })

    assert got.tool_router_model == "deepseek/deepseek-v4-flash"


def test_a_local_provider_leaves_every_pin_alone(tmp_path, monkeypatch):
    got = _settings_from(tmp_path, monkeypatch, {
        "llm_provider": "ollama",
        "reminder_model": "gemma4:e2b",
        "tool_router_model": "gemma4:e2b",
    })

    assert got.reminder_model == "gemma4:e2b"
    assert got.tool_router_model == "gemma4:e2b"

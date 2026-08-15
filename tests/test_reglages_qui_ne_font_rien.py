"""A setting he can change must change something.

Five keys shipped in `get_default_config()`, exported by
`export_example_config()`, and written by the Settings window — and none
of them reached `Settings`. Every consumer reads them with
`getattr(cfg, key, <hardcoded default>)`, and since the field did not
exist the hardcoded default always won.

So he could open the settings, turn thinking on, save, restart, and
nothing whatsoever would happen. Nothing errors, nothing warns, and the
value sits in his `config.json` looking obeyed.

`stop_command_fuzzy_ratio` was worse than dead: it never reached the
comparison at all, which kept its own default in the signature.

The last test pins the class rather than the five, so the next key added
to the defaults and forgotten in the dataclass fails here instead of
silently doing nothing for a year.
"""

from __future__ import annotations

import dataclasses
import json

import pytest


def _charge(tmp_path, monkeypatch, valeurs):
    """Load settings from a config.json holding exactly these overrides.

    Through `JARVIS_CONFIG_PATH`, which is how the app itself is pointed
    at a file, so this exercises the real path rather than a patched one.
    """
    from src.jarvis import config as mod

    chemin = tmp_path / "config.json"
    chemin.write_text(json.dumps(valeurs), encoding="utf-8")
    monkeypatch.setenv("JARVIS_CONFIG_PATH", str(chemin))
    return mod.load_settings()


CINQ = {
    "llm_thinking_enabled": True,
    "intent_judge_thinking_enabled": True,
    "dictation_thinking_enabled": True,
    "stop_command_fuzzy_ratio": 0.55,
    "stop_commands": ["stop", "tais-toi", "chut"],
}


@pytest.mark.parametrize("cle,valeur", list(CINQ.items()))
def test_what_he_sets_is_what_he_gets(tmp_path, monkeypatch, cle, valeur):
    cfg = _charge(tmp_path, monkeypatch, {cle: valeur})

    assert getattr(cfg, cle) == valeur, (
        f"{cle} was set in his config and did not reach Settings, so every "
        f"consumer's `getattr(cfg, {cle!r}, <default>)` keeps the default and "
        f"his choice does nothing.")


def test_the_defaults_are_still_the_defaults(tmp_path, monkeypatch):
    cfg = _charge(tmp_path, monkeypatch, {})

    assert cfg.llm_thinking_enabled is False
    assert cfg.intent_judge_thinking_enabled is False
    assert cfg.dictation_thinking_enabled is False
    assert cfg.stop_command_fuzzy_ratio == 0.8
    assert "stop" in cfg.stop_commands


def test_the_fuzzy_ratio_reaches_the_comparison(tmp_path, monkeypatch):
    """It was not merely absent from Settings — nothing passed it to
    `is_stop_command`, which kept the default in its own signature. A
    field nobody hands over is a field nobody can change."""
    from src.jarvis.listening.wake_detection import is_stop_command

    # "sopp" is a near miss at 0.75 and, unlike "stopp", does not contain
    # "stop" — so the exact-match branch cannot answer for the fuzzy one.
    assert is_stop_command("sopp", ["stop"], fuzzy_ratio=0.7)
    assert not is_stop_command("sopp", ["stop"], fuzzy_ratio=0.8)


def test_his_own_stop_words_are_the_ones_used(tmp_path, monkeypatch):
    from src.jarvis.listening.wake_detection import is_stop_command

    cfg = _charge(tmp_path, monkeypatch, {"stop_commands": ["tais-toi"]})

    assert is_stop_command("tais-toi", cfg.stop_commands)
    assert not is_stop_command("shush", cfg.stop_commands)


def test_every_default_key_reaches_settings():
    """The class, not the five. A key added to the defaults and forgotten
    in the dataclass does nothing at all, in silence, and this is the
    only place that would ever say so."""
    from src.jarvis.config import Settings, get_default_config

    champs = {f.name for f in dataclasses.fields(Settings)}
    orphelines = sorted(set(get_default_config()) - champs)

    assert not orphelines, (
        f"{orphelines} are offered in the default config and exported in the "
        f"example file, but never reach Settings. Anyone setting them changes "
        f"nothing and is told nothing.")

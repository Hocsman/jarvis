"""Behaviour tests for the ``ui.face`` config field (Phase 2A).

We pin three properties:

1. Fresh install (no ``ui`` key in config) defaults to ``"orb"``. This
   makes the reactive orb the visible default for new users without
   forcing them to edit config.json.

2. Explicit ``"lowpoly"`` is honoured — users who prefer the original
   face must be able to opt back in cleanly.

3. Invalid values (typos, future renames, anything outside the allowed
   set) fall back to ``"orb"`` with a logged warning. A typo in config
   must not crash the desktop app or silently flip behaviour in a
   surprising way.
"""

from __future__ import annotations

import json

import pytest


class TestUiFaceConfig:
    """The parser must produce a typed ``UISettings`` regardless of
    what shape the user wrote (missing key, bad type, bad value)."""

    @pytest.mark.unit
    def test_default_face_is_orb_when_ui_key_missing(self, tmp_path, monkeypatch) -> None:
        """Empty config (no ``ui`` block at all) -> cfg.ui.face == 'orb'.
        This is the path every fresh install takes."""
        cfg_path = tmp_path / "jarvis.json"
        cfg_path.write_text(json.dumps({"whisper_model": "medium"}))
        monkeypatch.setenv("JARVIS_CONFIG_PATH", str(cfg_path))

        from jarvis.config import load_settings
        settings = load_settings()

        assert settings.ui.face == "orb"

    @pytest.mark.unit
    def test_explicit_orb(self, tmp_path, monkeypatch) -> None:
        """Redundant but explicit: 'ui.face' = 'orb' resolves to 'orb'.
        Pins that the default-passes-through path also works when the
        user writes it down."""
        cfg_path = tmp_path / "jarvis.json"
        cfg_path.write_text(json.dumps({"ui": {"face": "orb"}}))
        monkeypatch.setenv("JARVIS_CONFIG_PATH", str(cfg_path))

        from jarvis.config import load_settings
        settings = load_settings()

        assert settings.ui.face == "orb"

    @pytest.mark.unit
    def test_face_lowpoly(self, tmp_path, monkeypatch) -> None:
        """Opt-back-in path: users who shipped on the low-poly face
        and prefer it must be able to keep it via config."""
        cfg_path = tmp_path / "jarvis.json"
        cfg_path.write_text(json.dumps({"ui": {"face": "lowpoly"}}))
        monkeypatch.setenv("JARVIS_CONFIG_PATH", str(cfg_path))

        from jarvis.config import load_settings
        settings = load_settings()

        assert settings.ui.face == "lowpoly"

    @pytest.mark.unit
    def test_face_invalid_value_falls_back_to_orb(self, tmp_path, monkeypatch) -> None:
        """Typo or unknown value -> fallback 'orb'. The point is that
        a config mistake never crashes the desktop app at launch."""
        cfg_path = tmp_path / "jarvis.json"
        cfg_path.write_text(json.dumps({"ui": {"face": "holographic"}}))
        monkeypatch.setenv("JARVIS_CONFIG_PATH", str(cfg_path))

        from jarvis.config import load_settings
        settings = load_settings()

        assert settings.ui.face == "orb"

    @pytest.mark.unit
    def test_face_invalid_value_emits_debug_log(self, tmp_path, monkeypatch) -> None:
        """The fallback must surface a breadcrumb via ``debug_log`` so
        the user notices the typo. We patch ``jarvis.debug.debug_log``
        and assert it was called with a message containing the
        offending value — that's the observable side-effect, and it
        survives whether ``voice_debug`` happens to be on or off at
        the moment the test runs."""
        cfg_path = tmp_path / "jarvis.json"
        cfg_path.write_text(json.dumps({"ui": {"face": "holographic"}}))
        monkeypatch.setenv("JARVIS_CONFIG_PATH", str(cfg_path))

        captured_calls: list[tuple[str, str]] = []

        def fake_debug_log(message: str, category: str = "debug") -> None:
            captured_calls.append((message, category))

        # Patch on the source module — config.py imports lazily as
        # ``from .debug import debug_log as _debug_log`` inside the
        # parser, so patching here catches the call site.
        monkeypatch.setattr("jarvis.debug.debug_log", fake_debug_log)

        from jarvis.config import load_settings
        load_settings()

        config_warnings = [
            (msg, cat) for msg, cat in captured_calls if cat == "config" and "ui.face" in msg
        ]
        assert config_warnings, (
            f"Expected at least one debug_log call about ui.face; got: "
            f"{captured_calls}"
        )
        msg = config_warnings[0][0]
        assert "holographic" in msg, (
            f"Warning must mention the offending value 'holographic'; got: {msg!r}"
        )
        assert "orb" in msg, (
            f"Warning must mention the fallback value 'orb'; got: {msg!r}"
        )

    @pytest.mark.unit
    def test_face_value_case_insensitive(self, tmp_path, monkeypatch) -> None:
        """Users mix case in JSON config all the time ('ORB', 'LowPoly').
        The parser normalises to lowercase before validating."""
        cfg_path = tmp_path / "jarvis.json"
        cfg_path.write_text(json.dumps({"ui": {"face": "LowPoly"}}))
        monkeypatch.setenv("JARVIS_CONFIG_PATH", str(cfg_path))

        from jarvis.config import load_settings
        settings = load_settings()

        assert settings.ui.face == "lowpoly"

    @pytest.mark.unit
    def test_ui_is_not_a_dict_falls_back_to_orb(self, tmp_path, monkeypatch) -> None:
        """Hostile config: someone writes 'ui': 'broken' (string instead
        of dict). The parser must coerce silently to 'orb' rather than
        raise — config load is on the daemon's hot path and an
        exception here would brick startup."""
        cfg_path = tmp_path / "jarvis.json"
        cfg_path.write_text(json.dumps({"ui": "broken"}))
        monkeypatch.setenv("JARVIS_CONFIG_PATH", str(cfg_path))

        from jarvis.config import load_settings
        settings = load_settings()

        assert settings.ui.face == "orb"


class TestUiSettingsDataclass:
    """The dataclass itself: shape + immutability."""

    @pytest.mark.unit
    def test_is_frozen(self) -> None:
        """``UISettings`` is frozen (immutable). Settings are a load-once
        object; mutating them after load would mask the source of any
        runtime divergence."""
        from dataclasses import FrozenInstanceError
        from jarvis.config import UISettings

        ui = UISettings(face="orb")
        with pytest.raises(FrozenInstanceError):
            ui.face = "lowpoly"  # type: ignore[misc]

    @pytest.mark.unit
    def test_valid_face_values_constant_exposed(self) -> None:
        """The set of allowed face values is a module-level constant
        so app.py / select_face / tests share one source of truth."""
        from jarvis.config import _VALID_FACE_VALUES

        assert "orb" in _VALID_FACE_VALUES
        assert "lowpoly" in _VALID_FACE_VALUES
        assert "garbage" not in _VALID_FACE_VALUES

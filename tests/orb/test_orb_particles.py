"""Particle layer and config flag tests.

- ``cfg.ui.orb_particles_enabled`` toggles the particle layer in
  config (default True).
- The particles orbit on the clock alone: the orb has no audio input.

Tests pin the config field behaviour and that the particles are drawn
and drift with time.
"""

from __future__ import annotations

import json
import math
import os
from unittest.mock import MagicMock

import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def _qapp():
    from PyQt6.QtWidgets import QApplication
    import sys

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


def _make_snapshot():
    from desktop_app.orb.state_controller import OrbState, StateSnapshot

    return StateSnapshot(
        state=OrbState.IDLE,
        color=(0.5, 0.7, 1.0),
        intensity=0.8,
        displacement_scale=1.0,
        pulse_period_s=3.0,
        transitioning=False,
        time_seconds=0.0,
    )


# ── Config flag ──────────────────────────────────────────────────────────


class TestOrbParticlesConfigFlag:
    """The config knob lives at ``cfg.ui.orb_particles_enabled`` and
    defaults to True. ``False`` must short-circuit particle construction
    so even a misbehaving paintEvent can't accidentally render them."""

    @pytest.mark.unit
    def test_default_is_true(self, tmp_path, monkeypatch) -> None:
        """Fresh install: config doesn't set the key -> True."""
        cfg_path = tmp_path / "jarvis.json"
        cfg_path.write_text(json.dumps({}), encoding="utf-8")
        monkeypatch.setenv("JARVIS_CONFIG_PATH", str(cfg_path))

        from jarvis.config import load_settings
        settings = load_settings()
        assert settings.ui.orb_particles_enabled is True

    @pytest.mark.unit
    def test_explicit_false(self, tmp_path, monkeypatch) -> None:
        """Users who want a quieter orb set the key to False."""
        cfg_path = tmp_path / "jarvis.json"
        cfg_path.write_text(json.dumps({"ui": {"orb_particles_enabled": False}}), encoding="utf-8")
        monkeypatch.setenv("JARVIS_CONFIG_PATH", str(cfg_path))

        from jarvis.config import load_settings
        settings = load_settings()
        assert settings.ui.orb_particles_enabled is False

    @pytest.mark.unit
    def test_string_false_coerced(self, tmp_path, monkeypatch) -> None:
        """JSON has bool literals but users sometimes write 'false'
        as a string by mistake. We coerce common false-y strings."""
        cfg_path = tmp_path / "jarvis.json"
        cfg_path.write_text(json.dumps({"ui": {"orb_particles_enabled": "false"}}), encoding="utf-8")
        monkeypatch.setenv("JARVIS_CONFIG_PATH", str(cfg_path))

        from jarvis.config import load_settings
        settings = load_settings()
        assert settings.ui.orb_particles_enabled is False

    @pytest.mark.unit
    def test_integer_0_coerced_false(self, tmp_path, monkeypatch) -> None:
        """``"orb_particles_enabled": 0`` is honest typo territory.
        Coerced to False via Python's bool() rules."""
        cfg_path = tmp_path / "jarvis.json"
        cfg_path.write_text(json.dumps({"ui": {"orb_particles_enabled": 0}}), encoding="utf-8")
        monkeypatch.setenv("JARVIS_CONFIG_PATH", str(cfg_path))

        from jarvis.config import load_settings
        settings = load_settings()
        assert settings.ui.orb_particles_enabled is False


class TestOrbWidgetParticlesEnabledFlag:
    """Pass-through: ``OrbWidget(particles_enabled=False)`` must not
    construct any particle structures, so even a buggy paintEvent
    can't surface particles by accident."""

    @pytest.mark.unit
    def test_particles_disabled_skips_construction(self, _qapp) -> None:
        from desktop_app.orb.orb_widget import OrbWidget

        widget = OrbWidget(particles_enabled=False)
        try:
            assert widget._particles is None, (
                "particles_enabled=False must skip build_particles "
                "entirely so the particle structures are absent"
            )
        finally:
            widget.deleteLater()

    @pytest.mark.unit
    def test_particles_enabled_constructs(self, _qapp) -> None:
        """Sanity check: default True still builds particles."""
        from desktop_app.orb.orb_widget import OrbWidget

        widget = OrbWidget(particles_enabled=True)
        try:
            assert widget._particles is not None
            assert widget._particles.orbits.shape[0] > 0
        finally:
            widget.deleteLater()


# ── Orbit on the clock ───────────────────────────────────────────────────


class TestParticlesOrbitOnTheClock:
    """The particles are drawn and their orbit drifts with time; there
    is no other input."""

    @pytest.mark.unit
    def test_particles_are_drawn(self, _qapp) -> None:
        from desktop_app.orb.orb_widget import OrbWidget

        widget = OrbWidget()
        try:
            painter = MagicMock()
            widget._draw_particles(
                painter, cx=200, cy=200, orb_r=100, glow_r=180, snap=_make_snapshot(),
            )
            assert painter.drawEllipse.call_count > 0, "no particles drawn"
        finally:
            widget.deleteLater()

    @pytest.mark.unit
    def test_the_orbit_drifts_with_time(self, _qapp) -> None:
        """Two frames at different instants place the particles at
        different longitudes."""
        from desktop_app.orb.orb_widget import OrbWidget
        from desktop_app.orb.state_controller import OrbState, StateSnapshot

        def _snapshot_at(t: float) -> StateSnapshot:
            return StateSnapshot(
                state=OrbState.IDLE, color=(0.5, 0.7, 1.0), intensity=0.8,
                displacement_scale=1.0, pulse_period_s=3.0, transitioning=False,
                time_seconds=t,
            )

        def _xs(painter):
            return [args[0].x() for args, _kw in painter.drawEllipse.call_args_list]

        widget = OrbWidget()
        try:
            painter_then = MagicMock()
            painter_later = MagicMock()
            widget._draw_particles(painter_then, cx=200, cy=200, orb_r=100, glow_r=180,
                                   snap=_snapshot_at(0.0))
            widget._draw_particles(painter_later, cx=200, cy=200, orb_r=100, glow_r=180,
                                   snap=_snapshot_at(2.5))
            differing = sum(1 for a, b in zip(_xs(painter_then), _xs(painter_later))
                            if abs(a - b) > 0.01)
            assert differing > 0, "the particles did not move between two instants"
        finally:
            widget.deleteLater()

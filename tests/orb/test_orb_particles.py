"""Phase 2D particle polish + config flag tests.

Phase 2D adds:
- ``cfg.ui.orb_particles_enabled`` — toggle the particle layer in
  config (default True).
- Enriched audio coupling: particle size scales with high band,
  orbital speed scales with mid band, alpha keeps the Phase 1
  high-band coupling.

Tests pin the config field behaviour and the audio coupling contract
(particles get bigger / faster with louder audio).
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


def _make_bands(*, rms=0.0, bass=0.0, mid=0.0, high=0.0):
    from jarvis.utils.audio_bands import BandReading
    return BandReading(rms=rms, bass=bass, mid=mid, high=high)


# ── Config flag ──────────────────────────────────────────────────────────


class TestOrbParticlesConfigFlag:
    """The config knob lives at ``cfg.ui.orb_particles_enabled`` and
    defaults to True. ``False`` must short-circuit particle construction
    so even a misbehaving paintEvent can't accidentally render them."""

    @pytest.mark.unit
    def test_default_is_true(self, tmp_path, monkeypatch) -> None:
        """Fresh install: config doesn't set the key -> True."""
        cfg_path = tmp_path / "jarvis.json"
        cfg_path.write_text(json.dumps({}))
        monkeypatch.setenv("JARVIS_CONFIG_PATH", str(cfg_path))

        from jarvis.config import load_settings
        settings = load_settings()
        assert settings.ui.orb_particles_enabled is True

    @pytest.mark.unit
    def test_explicit_false(self, tmp_path, monkeypatch) -> None:
        """Users who want a quieter orb set the key to False."""
        cfg_path = tmp_path / "jarvis.json"
        cfg_path.write_text(json.dumps({"ui": {"orb_particles_enabled": False}}))
        monkeypatch.setenv("JARVIS_CONFIG_PATH", str(cfg_path))

        from jarvis.config import load_settings
        settings = load_settings()
        assert settings.ui.orb_particles_enabled is False

    @pytest.mark.unit
    def test_string_false_coerced(self, tmp_path, monkeypatch) -> None:
        """JSON has bool literals but users sometimes write 'false'
        as a string by mistake. We coerce common false-y strings."""
        cfg_path = tmp_path / "jarvis.json"
        cfg_path.write_text(json.dumps({"ui": {"orb_particles_enabled": "false"}}))
        monkeypatch.setenv("JARVIS_CONFIG_PATH", str(cfg_path))

        from jarvis.config import load_settings
        settings = load_settings()
        assert settings.ui.orb_particles_enabled is False

    @pytest.mark.unit
    def test_integer_0_coerced_false(self, tmp_path, monkeypatch) -> None:
        """``"orb_particles_enabled": 0`` is honest typo territory.
        Coerced to False via Python's bool() rules."""
        cfg_path = tmp_path / "jarvis.json"
        cfg_path.write_text(json.dumps({"ui": {"orb_particles_enabled": 0}}))
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


# ── Audio coupling ───────────────────────────────────────────────────────


class TestParticleAudioCoupling:
    """High-band drives size and alpha; mid-band drives orbital
    speed. We verify by drawing a single particle's frame to a mock
    painter and checking that the QRadialGradient's radius (which
    encodes the particle size) is larger with louder audio."""

    @pytest.mark.unit
    def test_high_band_increases_particle_size(self, _qapp) -> None:
        """Same orb radius, same time — louder high band must produce
        a larger drawn ellipse for the same particle."""
        from desktop_app.orb.orb_widget import OrbWidget

        widget = OrbWidget()
        try:
            painter_quiet = MagicMock()
            painter_loud = MagicMock()
            snap = _make_snapshot()

            widget._draw_particles(
                painter_quiet, cx=200, cy=200, orb_r=100, glow_r=180,
                snap=snap, bands=_make_bands(high=0.0),
            )
            widget._draw_particles(
                painter_loud, cx=200, cy=200, orb_r=100, glow_r=180,
                snap=snap, bands=_make_bands(high=1.0),
            )

            # Compare the radii argument of drawEllipse calls. For
            # the same particle index, the loud-frame radius must be
            # strictly larger than the quiet-frame radius.
            quiet_radii = [args[1] for args, _kw in painter_quiet.drawEllipse.call_args_list]
            loud_radii = [args[1] for args, _kw in painter_loud.drawEllipse.call_args_list]

            assert quiet_radii and loud_radii, "no particles drawn in either frame"
            # We can't guarantee 1-to-1 correspondence (some particles
            # are behind the orb), so we compare medians.
            quiet_med = sorted(quiet_radii)[len(quiet_radii) // 2]
            loud_med = sorted(loud_radii)[len(loud_radii) // 2]
            assert loud_med > quiet_med, (
                f"high-band coupling not active: quiet median radius "
                f"{quiet_med}, loud median {loud_med} (loud should be "
                f"strictly larger)"
            )
        finally:
            widget.deleteLater()

    @pytest.mark.unit
    def test_mid_band_speeds_orbit(self, _qapp) -> None:
        """At the same time t, louder mid band must place particles
        at a *different* longitude than quiet mid band (the speed
        multiplier rotates them faster around the orbit).

        We catch this by comparing particle x positions between two
        frames at the same time but with different mid values. They
        must differ for at least some particles."""
        from desktop_app.orb.orb_widget import OrbWidget
        from desktop_app.orb.state_controller import OrbState, StateSnapshot

        widget = OrbWidget()
        try:
            painter_quiet = MagicMock()
            painter_loud = MagicMock()
            # Use a non-zero time so the longitude term ``t * row[3]
            # * speed_mult`` actually varies between the two frames.
            snap = StateSnapshot(
                state=OrbState.IDLE,
                color=(0.5, 0.7, 1.0),
                intensity=0.8,
                displacement_scale=1.0,
                pulse_period_s=3.0,
                transitioning=False,
                time_seconds=2.5,
            )

            widget._draw_particles(
                painter_quiet, cx=200, cy=200, orb_r=100, glow_r=180,
                snap=snap, bands=_make_bands(mid=0.0),
            )
            widget._draw_particles(
                painter_loud, cx=200, cy=200, orb_r=100, glow_r=180,
                snap=snap, bands=_make_bands(mid=1.0),
            )

            # Collect particle centre x positions.
            def _xs(painter):
                return [args[0].x() for args, _kw in painter.drawEllipse.call_args_list]

            xs_quiet = _xs(painter_quiet)
            xs_loud = _xs(painter_loud)
            # At least one position must differ between the two.
            differing = sum(1 for a, b in zip(xs_quiet, xs_loud) if abs(a - b) > 0.01)
            assert differing > 0, (
                "mid-band orbital speed coupling has no effect: every "
                "particle x matched between mid=0 and mid=1 frames"
            )
        finally:
            widget.deleteLater()

"""Phase 2D chromatic-aberration rim tests.

The chromatic-aberration helper draws two extra coloured ellipses
(red + blue) at small horizontal offsets when the orb is in an
"active" state. Tests pin:

1. State-gating: THINKING, SPEAKING, ERROR -> the helper draws.
   IDLE, LISTENING -> the helper short-circuits (no extra ellipses).

2. Two extra ellipses are drawn (one red rim, one blue rim).

3. The painter's previous composition mode is restored after the
   helper exits — important because subsequent ``paintEvent`` draws
   (particles, inner highlight) assume CompositionMode_SourceOver.

4. The offset magnitude grows with the audio high band (a small
   sanity check on the audio coupling).
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, call

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


def _make_snapshot(state, *, time_seconds: float = 0.0):
    """Build a minimal StateSnapshot for the requested state."""
    from desktop_app.orb.state_controller import StateSnapshot

    return StateSnapshot(
        state=state,
        color=(0.5, 0.7, 1.0),
        intensity=0.8,
        displacement_scale=1.0,
        pulse_period_s=3.0,
        transitioning=False,
        time_seconds=time_seconds,
    )


def _make_bands(*, high: float = 0.0):
    from jarvis.utils.audio_bands import BandReading
    return BandReading(rms=0.0, bass=0.0, mid=0.0, high=high)


class TestChromaticAberrationStateGate:
    """Active states fire; calm states skip."""

    @pytest.mark.unit
    @pytest.mark.parametrize("state_name", ["THINKING", "SPEAKING", "ERROR"])
    def test_active_state_draws_aberration(self, _qapp, state_name) -> None:
        """The 3 "active" states must trigger 2 extra ellipse draws."""
        from desktop_app.orb.orb_widget import OrbWidget
        from desktop_app.orb.state_controller import OrbState

        widget = OrbWidget()
        try:
            painter = MagicMock()
            painter.compositionMode.return_value = MagicMock()
            state = getattr(OrbState, state_name)
            widget._draw_chromatic_aberration(
                painter, cx=160.0, cy=160.0, r=80.0,
                snap=_make_snapshot(state), bands=_make_bands(),
            )
            # Two extra ellipses: red rim + blue rim.
            assert painter.drawEllipse.call_count == 2, (
                f"State={state_name!r}: expected 2 ellipse draws "
                f"(red + blue rim), got {painter.drawEllipse.call_count}"
            )
        finally:
            widget.deleteLater()

    @pytest.mark.unit
    @pytest.mark.parametrize("state_name", ["IDLE", "LISTENING"])
    def test_calm_state_skips_aberration(self, _qapp, state_name) -> None:
        """IDLE and LISTENING must produce zero extra ellipses — the
        rim would compete with the soft halo bloom and looks wrong."""
        from desktop_app.orb.orb_widget import OrbWidget
        from desktop_app.orb.state_controller import OrbState

        widget = OrbWidget()
        try:
            painter = MagicMock()
            painter.compositionMode.return_value = MagicMock()
            state = getattr(OrbState, state_name)
            widget._draw_chromatic_aberration(
                painter, cx=160.0, cy=160.0, r=80.0,
                snap=_make_snapshot(state), bands=_make_bands(),
            )
            assert painter.drawEllipse.call_count == 0, (
                f"State={state_name!r}: expected no extra draws (calm "
                f"state), got {painter.drawEllipse.call_count}"
            )
        finally:
            widget.deleteLater()


class TestChromaticAberrationCompositionMode:
    """The helper temporarily flips to CompositionMode_Plus. It must
    restore the previous mode on exit — otherwise downstream draws
    (particles, inner highlight) blend additively when they shouldn't."""

    @pytest.mark.unit
    def test_composition_mode_restored_on_exit(self, _qapp) -> None:
        from desktop_app.orb.orb_widget import OrbWidget
        from desktop_app.orb.state_controller import OrbState

        widget = OrbWidget()
        try:
            painter = MagicMock()
            sentinel_mode = MagicMock(name="sentinel-composition-mode")
            painter.compositionMode.return_value = sentinel_mode

            widget._draw_chromatic_aberration(
                painter, cx=160.0, cy=160.0, r=80.0,
                snap=_make_snapshot(OrbState.THINKING), bands=_make_bands(),
            )

            # The last setCompositionMode call must restore the
            # sentinel (whatever it was before the helper ran).
            mode_calls = painter.setCompositionMode.call_args_list
            assert mode_calls, "Helper should have set composition mode at least once"
            assert mode_calls[-1] == call(sentinel_mode), (
                f"Last setCompositionMode was {mode_calls[-1]}, expected "
                f"restoration to the sentinel mode captured at entry."
            )
        finally:
            widget.deleteLater()


class TestChromaticAberrationAudioCoupling:
    """The horizontal offset grows with the audio high band. Higher
    high-band energy => louder visual aberration. Sanity check on
    the formula ``shift = 2 + 2 * bands.high``."""

    @pytest.mark.unit
    def test_silence_uses_minimum_shift(self, _qapp) -> None:
        """At silence (high=0), shift should be at the floor (2 px).
        The red ellipse centre is at cx-2 and blue at cx+2."""
        from PyQt6.QtCore import QPointF
        from desktop_app.orb.orb_widget import OrbWidget
        from desktop_app.orb.state_controller import OrbState

        widget = OrbWidget()
        try:
            painter = MagicMock()
            painter.compositionMode.return_value = MagicMock()
            cx = 160.0
            widget._draw_chromatic_aberration(
                painter, cx=cx, cy=160.0, r=80.0,
                snap=_make_snapshot(OrbState.SPEAKING),
                bands=_make_bands(high=0.0),
            )
            # Two drawEllipse calls; first arg is a QPointF.
            centres = [args[0] for args, _kwargs in painter.drawEllipse.call_args_list]
            assert len(centres) == 2
            # Red rim shifted left (cx - 2), blue rim shifted right (cx + 2).
            xs = sorted([c.x() for c in centres])
            assert xs[0] == pytest.approx(cx - 2.0)
            assert xs[1] == pytest.approx(cx + 2.0)
        finally:
            widget.deleteLater()

    @pytest.mark.unit
    def test_high_band_doubles_shift(self, _qapp) -> None:
        """At full high-band saturation (high=1), shift = 4 px."""
        from desktop_app.orb.orb_widget import OrbWidget
        from desktop_app.orb.state_controller import OrbState

        widget = OrbWidget()
        try:
            painter = MagicMock()
            painter.compositionMode.return_value = MagicMock()
            cx = 160.0
            widget._draw_chromatic_aberration(
                painter, cx=cx, cy=160.0, r=80.0,
                snap=_make_snapshot(OrbState.SPEAKING),
                bands=_make_bands(high=1.0),
            )
            centres = [args[0] for args, _kwargs in painter.drawEllipse.call_args_list]
            xs = sorted([c.x() for c in centres])
            # 2 + 2 * 1.0 = 4 px shift.
            assert xs[0] == pytest.approx(cx - 4.0)
            assert xs[1] == pytest.approx(cx + 4.0)
        finally:
            widget.deleteLater()

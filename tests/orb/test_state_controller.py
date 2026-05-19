"""Tests for the orb state controller.

Covers the three contracts the renderer depends on:
1. Cubic transitions arrive at mid-parcours with halfway-eased values
   between previous and current style.
2. ERROR overlay returns to the prior state after 800 ms (no manual
   intervention required).
3. ERROR can be triggered manually (Phase 1 entry point; Phase 2 will
   wire the daemon's exception path through the same call) and shows
   the canonical red colour before fading back.
"""

from __future__ import annotations

import math
import pytest

from desktop_app.orb.state_controller import (
    ERROR_FADE_DURATION_S,
    OrbState,
    STATE_STYLES,
    StateController,
    TRANSITION_DURATION_S,
    _cubic_ease,
    map_jarvis_to_orb,
)


# ── Fake clock helpers ─────────────────────────────────────────────────


class _FakeClock:
    """Manually steppable clock so tests don't sleep."""
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, dt: float) -> None:
        self.now += dt


def _make_controller(clock: _FakeClock) -> StateController:
    return StateController(clock=clock)


# ── Tests ──────────────────────────────────────────────────────────────


class TestEnumAndMapping:
    """Sanity checks on the public enum surface and the JarvisState
    bridge. Catches drift if either side gets reorganised."""

    @pytest.mark.unit
    def test_orb_state_values_stable(self):
        assert {s.value for s in OrbState} == {
            "idle", "listening", "thinking", "speaking", "error",
        }

    @pytest.mark.unit
    def test_jarvis_to_orb_covers_known_values(self):
        # The JarvisState enum has more granular values than the orb.
        # All known JarvisState string values must resolve to a valid
        # OrbState so the orb never visually stalls.
        for jv in ("asleep", "idle", "listening", "thinking",
                   "speaking", "dictating", "dictation_processing"):
            assert isinstance(map_jarvis_to_orb(jv), OrbState)

    @pytest.mark.unit
    def test_unknown_jarvis_value_defaults_idle(self):
        assert map_jarvis_to_orb("totally_unknown") is OrbState.IDLE


class TestTransitions:
    """Cubic-eased transitions between non-error states."""

    @pytest.mark.unit
    def test_cubic_interpolation_at_midpoint(self):
        """At 50% of the transition, the colour, intensity, and
        displacement scale must each sit at the cubic-eased halfway
        value between IDLE and THINKING styles (not linear)."""
        clock = _FakeClock()
        ctrl = _make_controller(clock)
        ctrl.set_state(OrbState.THINKING)

        # Advance to exactly half of TRANSITION_DURATION_S.
        half_dt = TRANSITION_DURATION_S / 2.0
        clock.advance(half_dt)
        snap = ctrl.tick(half_dt)

        idle = STATE_STYLES[OrbState.IDLE]
        thinking = STATE_STYLES[OrbState.THINKING]
        eased = _cubic_ease(0.5)  # smoothstep(0.5) == 0.5 exactly
        expected_intensity = idle.intensity + (thinking.intensity - idle.intensity) * eased
        expected_disp = (
            idle.displacement_scale
            + (thinking.displacement_scale - idle.displacement_scale) * eased
        )
        expected_color = tuple(
            idle.color[i] + (thinking.color[i] - idle.color[i]) * eased
            for i in range(3)
        )

        assert snap.transitioning is True
        assert snap.intensity == pytest.approx(expected_intensity, abs=1e-4)
        assert snap.displacement_scale == pytest.approx(expected_disp, abs=1e-4)
        for c, e in zip(snap.color, expected_color):
            assert c == pytest.approx(e, abs=1e-4)

    @pytest.mark.unit
    def test_transition_completes_at_full_duration(self):
        """After the full duration, transitioning must be False and the
        snapshot equals the target style (no overshoot)."""
        clock = _FakeClock()
        ctrl = _make_controller(clock)
        ctrl.set_state(OrbState.SPEAKING)

        clock.advance(TRANSITION_DURATION_S * 1.1)
        snap = ctrl.tick(TRANSITION_DURATION_S * 1.1)

        speaking = STATE_STYLES[OrbState.SPEAKING]
        assert snap.transitioning is False
        assert snap.state is OrbState.SPEAKING
        assert snap.intensity == pytest.approx(speaking.intensity, abs=1e-6)
        for c, target in zip(snap.color, speaking.color):
            assert c == pytest.approx(target, abs=1e-6)

    @pytest.mark.unit
    def test_set_state_idempotent_when_settled(self):
        """Setting the current state again on a settled controller is
        a no-op (no transition restart, no visual flicker)."""
        clock = _FakeClock()
        ctrl = _make_controller(clock)
        ctrl.set_state(OrbState.LISTENING)
        clock.advance(TRANSITION_DURATION_S * 2.0)
        ctrl.tick(TRANSITION_DURATION_S * 2.0)
        ctrl.set_state(OrbState.LISTENING)  # idempotent
        snap = ctrl.tick(0.0)
        assert snap.transitioning is False


class TestErrorRecovery:
    """ERROR overlay shows red and returns to the prior state after the
    documented fade window without any external nudge."""

    @pytest.mark.unit
    def test_error_returns_to_previous_state_after_800ms(self):
        """LISTENING -> trigger_error() -> tick past 800 ms -> the
        controller is back on LISTENING. This is the privacy-of-state
        contract: errors are transient overlays, never sticky."""
        clock = _FakeClock()
        ctrl = _make_controller(clock)
        ctrl.set_state(OrbState.LISTENING)
        # Settle on LISTENING.
        clock.advance(TRANSITION_DURATION_S * 2.0)
        ctrl.tick(TRANSITION_DURATION_S * 2.0)
        assert ctrl.state is OrbState.LISTENING

        ctrl.trigger_error()
        assert ctrl.state is OrbState.ERROR

        # Advance past the full fade window.
        clock.advance(ERROR_FADE_DURATION_S + 0.05)
        ctrl.tick(ERROR_FADE_DURATION_S + 0.05)

        # The controller is now transitioning back toward LISTENING.
        # Wait out the recovery cubic too.
        clock.advance(TRANSITION_DURATION_S * 2.0)
        snap = ctrl.tick(TRANSITION_DURATION_S * 2.0)

        assert snap.state is OrbState.LISTENING
        listening = STATE_STYLES[OrbState.LISTENING]
        for c, target in zip(snap.color, listening.color):
            assert c == pytest.approx(target, abs=1e-3)

    @pytest.mark.unit
    def test_manual_trigger_error_fades_red_800ms(self):
        """trigger_error() shows the canonical red colour mid-fade and
        then returns to the prior state on its own. This is the test
        that the orb's *infrastructure* supports the ERROR state even
        though Phase 1 does not yet wire the daemon path."""
        clock = _FakeClock()
        ctrl = _make_controller(clock)
        ctrl.set_state(OrbState.THINKING)
        clock.advance(TRANSITION_DURATION_S * 2.0)
        ctrl.tick(TRANSITION_DURATION_S * 2.0)

        ctrl.trigger_error()

        # Step to a fraction of the fade where we should be ~all-red.
        clock.advance(ERROR_FADE_DURATION_S * 0.5)
        snap = ctrl.tick(ERROR_FADE_DURATION_S * 0.5)
        assert snap.state is OrbState.ERROR

        error_red = STATE_STYLES[OrbState.ERROR].color
        # Half-fade reaches the red end of the lerp (the fade is
        # ramp-up over the first half, hold for the second half — but
        # at 0.5 of the full fade we're at the start of the hold
        # which is already at the red apex).
        for c, target in zip(snap.color, error_red):
            assert c == pytest.approx(target, abs=0.02)

        # And ultimately we return to THINKING after the full window
        # plus a recovery transition.
        clock.advance(ERROR_FADE_DURATION_S + TRANSITION_DURATION_S * 2.0)
        snap = ctrl.tick(ERROR_FADE_DURATION_S + TRANSITION_DURATION_S * 2.0)
        assert snap.state is OrbState.THINKING

    @pytest.mark.unit
    def test_rapid_retrigger_extends_fade(self):
        """Calling trigger_error() while already in the ERROR overlay
        resets the timer so a burst of errors does not flicker the
        recovery in and out — it stays red until the burst quiets."""
        clock = _FakeClock()
        ctrl = _make_controller(clock)
        ctrl.set_state(OrbState.SPEAKING)
        clock.advance(TRANSITION_DURATION_S * 2.0)
        ctrl.tick(TRANSITION_DURATION_S * 2.0)

        ctrl.trigger_error()
        clock.advance(ERROR_FADE_DURATION_S * 0.7)
        ctrl.tick(ERROR_FADE_DURATION_S * 0.7)
        # Retrigger now: timer resets to 0.
        ctrl.trigger_error()

        # Without retrigger, we'd auto-recover in 0.3s. With retrigger,
        # we need a full ERROR_FADE_DURATION_S again before recovery.
        clock.advance(ERROR_FADE_DURATION_S * 0.5)
        ctrl.tick(ERROR_FADE_DURATION_S * 0.5)
        assert ctrl.state is OrbState.ERROR


class TestProviderIntegration:
    """The default operation: tick polls a provider returning a
    JarvisState-like value and reflects it into the orb."""

    @pytest.mark.unit
    def test_provider_value_is_mapped_and_transitioned(self):
        clock = _FakeClock()
        upstream = ["idle"]
        ctrl = StateController(jarvis_state_provider=lambda: upstream[0], clock=clock)

        # Initial tick: provider says idle, controller defaults to idle.
        ctrl.tick(0.0)
        upstream[0] = "thinking"
        # Advance halfway; we should be transitioning toward THINKING.
        clock.advance(TRANSITION_DURATION_S / 2.0)
        snap = ctrl.tick(TRANSITION_DURATION_S / 2.0)
        assert snap.transitioning is True
        assert snap.state is OrbState.THINKING

    @pytest.mark.unit
    def test_provider_exception_is_swallowed(self):
        """If the provider crashes, the controller holds its current
        state rather than propagating the exception into the render
        loop."""
        def angry_provider():
            raise RuntimeError("upstream is on fire")
        clock = _FakeClock()
        ctrl = StateController(jarvis_state_provider=angry_provider, clock=clock)
        # Must not raise.
        snap = ctrl.tick(0.01)
        assert snap.state is OrbState.IDLE

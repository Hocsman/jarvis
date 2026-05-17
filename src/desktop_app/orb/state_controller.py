"""Orb state controller: maps Jarvis state to orb visuals.

The orb has its own enum (``OrbState``) that is a *strict subset of
five* visual modes plus ERROR. The shared ``JarvisState`` from
``desktop_app.face_widget`` has more granularity (DICTATING /
DICTATION_PROCESSING / ASLEEP) than the orb cares to express
visually, so we collapse those onto the closest orb-state.

This module is the seam between the daemon-owned bus and the
visual pipeline. It also owns:

- Cubic-eased transitions on colour, intensity, displacement amplitude
  (250 ms minimum per spec).
- The ERROR overlay: a transient state with a fixed 800 ms fade that
  automatically returns to whatever state was active just before the
  error fired. Phase 1 only exposes the trigger via
  ``trigger_error()`` (Phase 2 will plumb the daemon's exception path
  through it).
- A monotonic ``time_seconds`` accessor for shaders that need a clock.

All purely numeric / numpy: no Qt, no GL. Headlessly testable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Tuple


# ── State enum ──────────────────────────────────────────────────────────


class OrbState(str, Enum):
    """Five visual modes the orb can render.

    String values are stable so they can be persisted (state file,
    telemetry) without breaking on enum reordering.
    """
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    ERROR = "error"


# Mapping from the shared ``JarvisState`` (face_widget) onto the orb's
# five modes. Resolved by *value* (string) so the orb does not have to
# import the JarvisState enum class at module load time, which would
# pull face_widget and therefore the entire Qt stack.
_JARVIS_TO_ORB: dict[str, OrbState] = {
    "asleep": OrbState.IDLE,                    # asleep collapses to idle visually
    "idle": OrbState.IDLE,
    "listening": OrbState.LISTENING,
    "thinking": OrbState.THINKING,
    "speaking": OrbState.SPEAKING,
    "dictating": OrbState.LISTENING,            # same family as listening
    "dictation_processing": OrbState.THINKING,  # post-record transcribe = thinking
}


def map_jarvis_to_orb(jarvis_value: str) -> OrbState:
    """Translate a ``JarvisState.value`` to the corresponding ``OrbState``.

    Unknown values default to ``IDLE`` so the orb is never visually
    stranded by a state we have not yet mapped.
    """
    return _JARVIS_TO_ORB.get(jarvis_value, OrbState.IDLE)


# ── Visual style per state ──────────────────────────────────────────────


@dataclass(frozen=True)
class StateStyle:
    """The shader-bound parameters for one state.

    ``color`` is RGB in 0..1. ``intensity`` is a multiplier on the
    fragment shader's final colour. ``displacement_scale`` gates the
    vertex shader's audio-driven bumps. ``pulse_period_s`` is how
    long one cycle of the breathing pulse lasts (0 means no pulse,
    just a steady amplitude).
    """
    color: Tuple[float, float, float]
    intensity: float
    displacement_scale: float
    pulse_period_s: float


# Canonical colours from the spec, converted to 0..1 RGB:
# - IDLE      #1a3a5c (deep night blue)
# - LISTENING #00d4ff (vivid cyan)
# - THINKING  #ff9500 (amber)
# - SPEAKING  #e8f4ff (warm white-blue)
# - ERROR     #ff3838 (red)
def _hex_to_rgb(h: str) -> Tuple[float, float, float]:
    """Helper kept in-module so the colour table reads close to the spec."""
    h = h.lstrip("#")
    return (
        int(h[0:2], 16) / 255.0,
        int(h[2:4], 16) / 255.0,
        int(h[4:6], 16) / 255.0,
    )


STATE_STYLES: dict[OrbState, StateStyle] = {
    OrbState.IDLE: StateStyle(
        color=_hex_to_rgb("#1a3a5c"),
        intensity=0.55,
        displacement_scale=0.4,
        pulse_period_s=2.0,
    ),
    OrbState.LISTENING: StateStyle(
        color=_hex_to_rgb("#00d4ff"),
        intensity=0.95,
        displacement_scale=1.0,
        pulse_period_s=0.0,
    ),
    OrbState.THINKING: StateStyle(
        color=_hex_to_rgb("#ff9500"),
        intensity=0.85,
        displacement_scale=0.6,
        pulse_period_s=0.9,
    ),
    OrbState.SPEAKING: StateStyle(
        color=_hex_to_rgb("#e8f4ff"),
        intensity=1.00,
        displacement_scale=1.2,
        pulse_period_s=0.0,
    ),
    OrbState.ERROR: StateStyle(
        color=_hex_to_rgb("#ff3838"),
        intensity=1.10,
        displacement_scale=0.8,
        pulse_period_s=0.0,
    ),
}

# Minimum transition duration on colour/intensity/displacement between
# any two non-error states. Phase 1 spec.
TRANSITION_DURATION_S = 0.250

# ERROR overlay fade duration. The auto-recovery to the previous state
# fires at the same time the colour finishes fading from red back to
# the recovered state.
ERROR_FADE_DURATION_S = 0.800


# ── Snapshot returned to renderer each frame ────────────────────────────


@dataclass(frozen=True)
class StateSnapshot:
    """What the orb widget reads from the controller each render frame.

    ``time_seconds`` is monotonic since controller construction; the
    shader uses it for breathing motion that is independent of audio.
    """
    state: OrbState                              # the *target* state at this instant
    color: Tuple[float, float, float]            # interpolated RGB
    intensity: float                             # interpolated
    displacement_scale: float                    # interpolated
    pulse_period_s: float                        # current style's pulse period
    transitioning: bool                          # True while interpolation is in progress
    time_seconds: float                          # monotonic clock for shaders


# ── Controller ──────────────────────────────────────────────────────────


def _cubic_ease(t: float) -> float:
    """Cubic ease-in-out on a normalised parameter ``t`` in [0, 1].

    Matches the spec's "interpolation cubique 250ms min" and gives the
    transitions a deliberate, settled feel rather than linear blending.
    """
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    return t * t * (3.0 - 2.0 * t)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _lerp_rgb(
    a: Tuple[float, float, float],
    b: Tuple[float, float, float],
    t: float,
) -> Tuple[float, float, float]:
    return (
        _lerp(a[0], b[0], t),
        _lerp(a[1], b[1], t),
        _lerp(a[2], b[2], t),
    )


class StateController:
    """Holds the orb's current visual state and drives the transitions.

    Construction takes an optional ``jarvis_state_provider`` (any
    callable returning a ``JarvisState`` enum or a str-coercible
    object). The default provider goes through ``face_widget``'s
    singleton — kept indirect so this module imports cleanly in
    headless test contexts (no Qt event loop needed).

    Typical lifecycle::

        ctrl = StateController()
        ...
        # Per render frame at 60 Hz:
        snap = ctrl.tick(dt_seconds=1/60)
        # snap.color, snap.intensity, snap.displacement_scale -> uniforms
    """

    def __init__(
        self,
        jarvis_state_provider: Optional[Any] = None,
        transition_duration_s: float = TRANSITION_DURATION_S,
        error_fade_duration_s: float = ERROR_FADE_DURATION_S,
        clock: Any = None,
    ) -> None:
        # Allow a custom clock so tests can step time deterministically.
        self._clock = clock or time.monotonic
        self._t0 = self._clock()

        # Provider that returns the current Jarvis state. None means
        # "no upstream provider, hold the controller's current state".
        self._provider = jarvis_state_provider

        # Configurable durations so tests can shorten them.
        self._transition_duration_s = max(0.001, transition_duration_s)
        self._error_fade_duration_s = max(0.001, error_fade_duration_s)

        # Active and target states. ``_progress`` runs from 0 to 1
        # over ``_transition_duration_s`` (or _error_fade_duration_s
        # when entering / leaving the ERROR overlay).
        self._current: OrbState = OrbState.IDLE
        self._previous: OrbState = OrbState.IDLE
        self._progress: float = 1.0
        self._active_duration_s: float = self._transition_duration_s

        # ERROR overlay bookkeeping. When ERROR is triggered we save
        # the *non-error* state to return to and run a one-shot fade.
        self._error_active: bool = False
        self._error_recover_to: OrbState = OrbState.IDLE
        self._error_elapsed: float = 0.0

    # ── Public API ─────────────────────────────────────────────────────

    @property
    def state(self) -> OrbState:
        """The state the orb is currently transitioning toward, or
        ERROR while the overlay is active."""
        if self._error_active:
            return OrbState.ERROR
        return self._current

    @property
    def previous_state(self) -> OrbState:
        return self._previous

    def trigger_error(self) -> None:
        """Switch to the ERROR overlay and schedule auto-recovery.

        ``previous_state`` saved at trigger time becomes the recover
        target so a burst of errors does not anchor the recovery to
        an ERROR origin.
        """
        if self._error_active:
            # Already in ERROR; reset the timer (rapid retriggers extend the fade).
            self._error_elapsed = 0.0
            return
        self._error_active = True
        # Recover to the state we were transitioning toward, even mid-transition.
        self._error_recover_to = self._current
        self._error_elapsed = 0.0

    def set_state(self, new_state: OrbState) -> None:
        """Explicit setter used by the default provider and by tests.

        Triggers a fresh cubic transition unless ``new_state`` matches
        the current target (idempotent)."""
        if self._error_active:
            # Setting a non-error state during an error overlay updates
            # the recovery target but does not interrupt the fade.
            if new_state is not OrbState.ERROR:
                self._error_recover_to = new_state
            return
        if new_state is OrbState.ERROR:
            # Setting ERROR via this entry point still routes through
            # the overlay machinery.
            self.trigger_error()
            return
        if new_state is self._current and self._progress >= 1.0:
            return
        self._previous = self._current
        self._current = new_state
        self._progress = 0.0
        self._active_duration_s = self._transition_duration_s

    def tick(self, dt_seconds: float) -> StateSnapshot:
        """Advance the controller by ``dt_seconds`` and return the
        snapshot the renderer should consume this frame."""
        # 1. Pull upstream state, if any provider configured.
        upstream = self._read_provider()
        if upstream is not None and upstream is not self._current and not self._error_active:
            self.set_state(upstream)

        # 2. Advance the error overlay clock.
        if self._error_active:
            self._error_elapsed += max(0.0, dt_seconds)
            if self._error_elapsed >= self._error_fade_duration_s:
                # Recovery: drop the overlay, snap back to the saved state
                # via a fresh cubic transition so the colour eases instead
                # of jumping.
                self._error_active = False
                self._previous = OrbState.ERROR
                self._current = self._error_recover_to
                self._progress = 0.0
                self._active_duration_s = self._transition_duration_s

        # 3. Advance the regular cubic transition.
        if self._progress < 1.0:
            self._progress = min(
                1.0,
                self._progress + max(0.0, dt_seconds) / self._active_duration_s,
            )

        # 4. Compose the snapshot.
        return self._snapshot()

    # ── Internals ──────────────────────────────────────────────────────

    def _read_provider(self) -> Optional[OrbState]:
        if self._provider is None:
            return None
        try:
            value = self._provider()
        except Exception:
            return None
        if value is None:
            return None
        # Accept enum-with-.value, plain str, or duck-typed.
        raw = getattr(value, "value", value)
        if not isinstance(raw, str):
            raw = str(raw)
        return map_jarvis_to_orb(raw)

    def _snapshot(self) -> StateSnapshot:
        t_now = self._clock() - self._t0

        if self._error_active:
            # Crossfade from the saved current style to the ERROR style
            # over the same duration, then hold ERROR for the rest of
            # the fade, then the recovery transition (handled at next
            # tick) eases us back out.
            err_style = STATE_STYLES[OrbState.ERROR]
            from_style = STATE_STYLES[self._error_recover_to]
            # Half the fade goes "into" red, half stays red. Gives a
            # noticeable flash + the recovery cubic eases out smoothly.
            ramp = min(1.0, self._error_elapsed / (self._error_fade_duration_s * 0.5))
            eased = _cubic_ease(ramp)
            return StateSnapshot(
                state=OrbState.ERROR,
                color=_lerp_rgb(from_style.color, err_style.color, eased),
                intensity=_lerp(from_style.intensity, err_style.intensity, eased),
                displacement_scale=_lerp(
                    from_style.displacement_scale, err_style.displacement_scale, eased
                ),
                pulse_period_s=err_style.pulse_period_s,
                transitioning=True,
                time_seconds=t_now,
            )

        # Regular cubic transition between previous and current.
        prev_style = STATE_STYLES[self._previous]
        curr_style = STATE_STYLES[self._current]
        eased = _cubic_ease(self._progress)
        return StateSnapshot(
            state=self._current,
            color=_lerp_rgb(prev_style.color, curr_style.color, eased),
            intensity=_lerp(prev_style.intensity, curr_style.intensity, eased),
            displacement_scale=_lerp(
                prev_style.displacement_scale, curr_style.displacement_scale, eased
            ),
            pulse_period_s=curr_style.pulse_period_s,
            transitioning=(self._progress < 1.0),
            time_seconds=t_now,
        )

"""The floating orb tracks the shared cross-process JarvisState.

The orb's StateController polls a ``jarvis_state_provider`` each frame;
wired to ``get_jarvis_state().state`` it follows the voice pipeline
(idle/listening/thinking/speaking) with no extra signalling. These
tests pin the JarvisState -> OrbState mapping and the provider-driven
tracking.
"""

from __future__ import annotations

import os

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


class TestJarvisToOrbMapping:

    @pytest.mark.unit
    def test_every_jarvis_state_maps(self, _qapp) -> None:
        """No JarvisState value may fall through unmapped — each must
        resolve to a defined OrbState so the orb is never visually
        stranded by a state the pipeline actually emits."""
        from desktop_app.face_widget import JarvisState
        from desktop_app.orb.state_controller import _JARVIS_TO_ORB, map_jarvis_to_orb, OrbState

        for js in JarvisState:
            mapped = map_jarvis_to_orb(js.value)
            assert isinstance(mapped, OrbState)
            # Every state the pipeline emits should be explicitly mapped
            # (not silently defaulted), except none — assert membership.
            assert js.value in _JARVIS_TO_ORB, (
                f"JarvisState.{js.name} ({js.value!r}) is not explicitly "
                f"mapped in _JARVIS_TO_ORB; it would default to IDLE"
            )

    @pytest.mark.unit
    @pytest.mark.parametrize("jarvis_value,orb_name", [
        ("listening", "LISTENING"),
        ("thinking", "THINKING"),
        ("speaking", "SPEAKING"),
        ("idle", "IDLE"),
        ("asleep", "IDLE"),
        ("dictating", "LISTENING"),
        ("dictation_processing", "THINKING"),
    ])
    def test_specific_mappings(self, _qapp, jarvis_value, orb_name) -> None:
        from desktop_app.orb.state_controller import map_jarvis_to_orb, OrbState

        assert map_jarvis_to_orb(jarvis_value) == getattr(OrbState, orb_name)


class TestProviderDrivenTracking:

    @pytest.mark.unit
    def test_controller_follows_jarvis_state(self, _qapp) -> None:
        from desktop_app.face_widget import get_jarvis_state, JarvisState
        from desktop_app.orb.state_controller import StateController, OrbState

        ctrl = StateController(jarvis_state_provider=lambda: get_jarvis_state().state)
        mgr = get_jarvis_state()

        for js, expected in [
            (JarvisState.LISTENING, OrbState.LISTENING),
            (JarvisState.THINKING, OrbState.THINKING),
            (JarvisState.SPEAKING, OrbState.SPEAKING),
            (JarvisState.IDLE, OrbState.IDLE),
        ]:
            mgr.set_state(js)
            snap = ctrl.tick(0.0)  # tick reads the provider
            assert snap.state == expected, (
                f"voice={js.value} should drive orb to {expected.value}, "
                f"got {snap.state.value}"
            )

    @pytest.mark.unit
    def test_provider_exception_is_safe(self, _qapp) -> None:
        """A provider that raises must not crash the render tick — the
        orb just holds its current state."""
        from desktop_app.orb.state_controller import StateController, OrbState

        def boom():
            raise RuntimeError("state file vanished")

        ctrl = StateController(jarvis_state_provider=boom)
        snap = ctrl.tick(0.0)
        assert snap.state == OrbState.IDLE  # held, not crashed

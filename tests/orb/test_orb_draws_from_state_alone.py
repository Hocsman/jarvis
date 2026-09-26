"""The orb is driven by its state and the clock, nothing else.

It has no audio input: no bus to read, no argument to hand one in, no
audio names on the package. Every state renders to a visible frame
with nothing attached, which is what a test, a subprocess run and the
bundled app all give it.
"""

from __future__ import annotations

import importlib.util
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


def _body_patch(widget, half: int = 3):
    """Render one frame to an image and average a small patch inside the
    body, off the pole dot and the key light: (r, g, b, alpha)."""
    from PyQt6.QtGui import QImage

    image = QImage(widget.width(), widget.height(), QImage.Format.Format_ARGB32)
    image.fill(0)
    widget.render(image)
    x0 = widget.width() // 2 + widget.width() // 6
    y0 = widget.height() // 2 + widget.height() // 6
    total = [0, 0, 0, 0]
    count = 0
    for x in range(x0 - half, x0 + half + 1):
        for y in range(y0 - half, y0 + half + 1):
            c = image.pixelColor(x, y)
            for i, v in enumerate((c.red(), c.green(), c.blue(), c.alpha())):
                total[i] += v
            count += 1
    return tuple(v / count for v in total)


def _distance(a, b) -> float:
    return sum((x - y) ** 2 for x, y in zip(a[:3], b[:3])) ** 0.5


class TestTheOrbDrawsFromStateAlone:

    @pytest.mark.unit
    def test_every_state_renders_its_own_frame_with_nothing_attached(self, _qapp) -> None:
        """Each state, once its transition has run, paints the body in a
        colour of its own: the frame follows the state and the clock, and
        nothing else is attached."""
        from desktop_app.orb.orb_widget import OrbWidget
        from desktop_app.orb.state_controller import (
            ERROR_FADE_DURATION_S, TRANSITION_DURATION_S, OrbState,
        )

        def _rendered(state):
            widget = OrbWidget(particles_enabled=False)
            try:
                widget.resize(320, 320)
                controller = widget.state_controller()
                if state is OrbState.ERROR:
                    controller.trigger_error()
                    controller.tick(ERROR_FADE_DURATION_S / 2)
                elif state is not OrbState.IDLE:
                    controller.set_state(state)
                    controller.tick(TRANSITION_DURATION_S)
                return _body_patch(widget)
            finally:
                widget.deleteLater()

        idle = _rendered(OrbState.IDLE)
        assert idle[3] > 0, "IDLE: nothing was drawn"
        for state in OrbState:
            if state is OrbState.IDLE:
                continue
            patch = _rendered(state)
            assert patch[3] > 0, f"{state.name}: nothing was drawn"
            assert _distance(patch, idle) > 10, (
                f"{state.name}: the body still wears IDLE's colour {idle[:3]} -> {patch[:3]}"
            )

    @pytest.mark.unit
    def test_the_orb_takes_no_audio_source(self, _qapp) -> None:
        from desktop_app.orb.orb_widget import OrbWidget
        from desktop_app.orb.orb_window import OrbWindow

        with pytest.raises(TypeError):
            OrbWidget(audio_bus=object())
        with pytest.raises(TypeError):
            OrbWindow(audio_bus=object())

    @pytest.mark.unit
    def test_the_package_has_no_audio_surface(self) -> None:
        import desktop_app.orb as orb

        for name in ("AudioBus", "FFTAnalyser", "register_audio_observer",
                     "unregister_audio_observer"):
            assert name not in orb.__all__
            with pytest.raises(AttributeError):
                getattr(orb, name)
        assert importlib.util.find_spec("desktop_app.orb.audio_bus") is None
        assert importlib.util.find_spec("jarvis.utils.audio_bands") is None

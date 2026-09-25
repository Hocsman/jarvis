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


def _centre_alpha(widget) -> int:
    """Render one frame to an image and read the alpha at its centre."""
    from PyQt6.QtGui import QImage

    image = QImage(widget.width(), widget.height(), QImage.Format.Format_ARGB32)
    image.fill(0)
    widget.render(image)
    return image.pixelColor(widget.width() // 2, widget.height() // 2).alpha()


class TestTheOrbDrawsFromStateAlone:

    @pytest.mark.unit
    def test_every_state_renders_a_frame_with_nothing_attached(self, _qapp) -> None:
        from desktop_app.orb.orb_widget import OrbWidget
        from desktop_app.orb.state_controller import OrbState

        for state in OrbState:
            widget = OrbWidget()
            try:
                widget.resize(320, 320)
                if state is OrbState.ERROR:
                    widget.trigger_error()
                else:
                    widget.state_controller().set_state(state)
                assert _centre_alpha(widget) > 0, f"{state.name}: nothing was drawn"
            finally:
                widget.deleteLater()

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

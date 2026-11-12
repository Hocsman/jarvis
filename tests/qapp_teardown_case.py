"""A case ``test_qapp_tears_down_what_only_a_cycle_holds.py`` runs in a pytest
session of its own, because what it guards against ends the process.

The name keeps it out of the suite's collection; a session given the file by
path still collects it, with the suite's ``conftest.py`` and its ``qapp``.
"""

import gc
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtGui import QPainter  # noqa: E402
from PyQt6.QtWidgets import QWidget  # noqa: E402


class _Child(QWidget):
    """Collects garbage while it paints, as any Python code that allocates
    may, the orb's glow halo among it."""

    def paintEvent(self, event):
        gc.collect()
        painter = QPainter(self)
        painter.drawEllipse(0, 0, self.width(), self.height())
        painter.end()


class _Window(QWidget):
    def __init__(self):
        super().__init__()
        self.itself = self
        self.child = _Child(self)
        self.child.resize(80, 80)
        self.resize(100, 100)


def test_a_shown_window_that_only_a_reference_cycle_holds(qapp):
    # Only the collections the case names may free the window: an automatic
    # one in the stretch pytest runs between this body and qapp's teardown
    # would free it outside a paint and pass the case with or without the
    # teardown under test. Explicit collections still run.
    gc.disable()
    window = _Window()
    window.show()
    window.child.update()
    del window

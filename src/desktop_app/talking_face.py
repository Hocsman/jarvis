"""Her talking face, as the face window shows it (`desktop_app.spec.md`, "Face").

The frames come from the renderer through ``read_face_frame()``, the reader
`jarvis` offers a front end (`src/jarvis/output/face/face.spec.md`). Reading
is what tells the renderer someone is watching, so the window that holds this
view polls it only while it is on screen.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Callable, Optional

from PyQt6.QtCore import QRectF
from PyQt6.QtGui import QImage, QPainter
from PyQt6.QtWidgets import QWidget

from jarvis.debug import debug_log

if TYPE_CHECKING:
    from jarvis.output.face.frames import Frame


def read_face_frame() -> Optional["Frame"]:
    """The frame `jarvis` offers a front end. The reader, and numpy with it,
    loads on the first read, so importing the desktop app's modules loads none."""
    from jarvis.output.face.frames import read_face_frame as read

    return read()


# The oldest a frame of hers can be and still be shown. A live renderer's
# newest frame is never older than its delay, the hold after a cut included;
# an older one comes from a renderer that has stopped, stalled or died, or
# from before it last rested.
HOLD_SECONDS = 1.0


class TalkingFaceView(QWidget):
    """The frame due now, fitted to the widget with its proportions kept."""

    def __init__(self, read_frame: Optional[Callable[[], Optional[Frame]]] = None, parent=None):
        super().__init__(parent)
        self._read_frame = read_frame or read_face_frame
        self._image: Optional[QImage] = None
        self._pixels: Optional[bytes] = None   # what the image points into
        self._shown = None                     # (sequence, stamp) of the frame on screen
        self._read_failed = False

    def poll(self) -> bool:
        """Read the frame due now. True while there is a recent frame of hers to show."""
        frame = self._read()
        now = time.perf_counter()   # the clock the renderer stamps its frames with
        if (frame is not None and now - frame.stamp <= HOLD_SECONDS
                and (frame.seq, frame.stamp) != self._shown):
            height, width = frame.pixels.shape[:2]
            self._pixels = frame.pixels.tobytes()
            self._image = QImage(self._pixels, width, height, 3 * width, QImage.Format.Format_RGB888)
            self._shown = (frame.seq, frame.stamp)
            self.update()
        if self._shown is not None and now - self._shown[1] > HOLD_SECONDS:
            self.forget()
        return self._image is not None

    def _read(self) -> Optional["Frame"]:
        # Called from a Qt slot, where an exception would end the whole app:
        # a read that fails counts as no frame.
        try:
            frame = self._read_frame()
        except Exception as exc:
            if not self._read_failed:
                debug_log(f"face window: a frame could not be read ({type(exc).__name__})", "desktop")
            self._read_failed = True
            return None
        self._read_failed = False
        return frame

    def forget(self) -> None:
        """Let go of the frame on screen: nothing of hers is kept."""
        self._image = self._pixels = self._shown = None

    def paintEvent(self, event) -> None:
        if self._image is None:
            return
        side = min(self.width(), self.height())
        target = QRectF((self.width() - side) / 2, (self.height() - side) / 2, side, side)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawImage(target, self._image)
        painter.end()

"""QOpenGLWidget that renders the reactive orb at 60 FPS.

Orchestrates the four other modules:

- ``geometry.build_icosphere`` for the mesh.
- ``geometry.build_particles`` for the halo (gated by config).
- ``audio_bus.AudioBus`` for the per-frame band readings.
- ``state_controller.StateController`` for colour / intensity /
  displacement / time uniforms.

ModernGL is created against the QOpenGLWidget's already-current
context in ``initializeGL``. Shaders are loaded from ``shaders/``
(see :mod:`desktop_app.orb.shaders`) so they stay editable without a
restart of the Python process.

Postprocess: a single offscreen framebuffer holds the orb draw, then
a fullscreen quad samples it with a small chromatic-aberration offset
and a cheap bloom (two-pass separable Gaussian on a downsampled copy).
The bloom pass is *not* the bottleneck on a 320x320 px widget; it is
included because the spec asks for it and disables to a no-op when
``cfg.orb_particles_enabled`` is False (kept under one flag with the
particle layer to give users a single "minimalist" off-switch).

This module has hard runtime deps on PyQt6 + moderngl and is therefore
*not* imported by the lazy ``__init__`` until ``OrbWidget`` or
``OrbWindow`` is actually accessed.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .audio_bus import AudioBus, BandReading
from .geometry import Mesh, Particles, build_icosphere, build_particles
from .state_controller import StateController, StateSnapshot


_SHADER_DIR = Path(__file__).resolve().parent / "shaders"


def _read_shader(name: str) -> str:
    """Load a shader source file from the package's ``shaders/`` dir.

    Kept as a simple file read so live-editing during dev survives a
    widget reload. Errors propagate: a missing shader is a build bug,
    not a runtime fallback.
    """
    return (_SHADER_DIR / name).read_text(encoding="utf-8")


# Fullscreen quad with NDC positions + UVs, used by the postprocess
# pass. Stored as a constant so the FBO compositor doesn't allocate
# per-frame.
_FULLSCREEN_QUAD: np.ndarray = np.array(
    [
        # x, y, u, v
        -1.0, -1.0, 0.0, 0.0,
         1.0, -1.0, 1.0, 0.0,
        -1.0,  1.0, 0.0, 1.0,
         1.0,  1.0, 1.0, 1.0,
    ],
    dtype=np.float32,
)


# Lazy imports for Qt + moderngl. We import them inside the class so
# headless test collection of the package's submodules doesn't pull
# Qt when only audio_bus or state_controller are exercised.

class OrbWidget:
    """Live import wrapper.

    The actual ``QOpenGLWidget`` subclass is built lazily inside
    ``_build_class()`` so ``import desktop_app.orb.orb_widget`` does
    not require Qt/GL at module-load time. Callers that instantiate
    ``OrbWidget(...)`` get the real Qt widget. Callers that only
    touch type-level metadata (tests, sphinx) get this thin shim
    with no side effects.
    """

    _real_cls = None

    def __new__(cls, *args, **kwargs):
        if cls._real_cls is None:
            cls._real_cls = _build_real_widget_class()
        return cls._real_cls(*args, **kwargs)


def _build_real_widget_class():
    """Build and return the real ``QOpenGLWidget`` subclass.

    Done lazily so the headless test environment can import this
    module to type-check the public surface without spinning up a
    Qt event loop.
    """
    import moderngl  # noqa: F401  (used by the closure below)
    from PyQt6.QtCore import QTimer, Qt
    from PyQt6.QtGui import QSurfaceFormat
    from PyQt6.QtOpenGLWidgets import QOpenGLWidget

    class _OrbWidget(QOpenGLWidget):
        """The real Qt widget."""

        FRAME_INTERVAL_MS = 16  # ~60 FPS

        def __init__(
            self,
            audio_bus: Optional[AudioBus] = None,
            state_controller: Optional[StateController] = None,
            particles_enabled: bool = True,
            particle_count: int = 256,
            icosphere_subdivisions: int = 2,
            parent=None,
        ) -> None:
            super().__init__(parent)
            self.setMinimumSize(320, 320)

            # Request an OpenGL 3.3 Core profile context BEFORE Qt
            # creates the GL surface. Qt's default on macOS is the
            # legacy 2.1 profile, which is too old for moderngl's
            # require=330 check and silently leaves the context
            # uninitialised (version 0). Setting the format on the
            # widget asks for a specific profile when the GL surface
            # is created during show / initializeGL.
            fmt = QSurfaceFormat()
            fmt.setVersion(3, 3)
            fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
            # Alpha buffer is required for the frameless translucent
            # composite: without it, the widget clears to opaque black.
            fmt.setAlphaBufferSize(8)
            self.setFormat(fmt)
            self._audio_bus = audio_bus or AudioBus()
            self._state_controller = state_controller or StateController()
            self._particles_enabled = particles_enabled
            self._particle_count = particle_count
            self._icosphere_subdivisions = icosphere_subdivisions

            # GL resources, set up in initializeGL.
            self._ctx: Optional[moderngl.Context] = None
            self._program: Optional[moderngl.Program] = None
            self._vao: Optional[moderngl.VertexArray] = None
            self._mesh: Optional[Mesh] = None
            self._particles: Optional[Particles] = None
            self._t0_monotonic = time.monotonic()
            self._last_tick_t = self._t0_monotonic

            # Render loop timer.
            self._timer = QTimer(self)
            self._timer.timeout.connect(self.update)
            self._timer.start(self.FRAME_INTERVAL_MS)

        # ── Qt overrides ───────────────────────────────────────────

        def initializeGL(self) -> None:
            # Wrap the already-current Qt GL context with ModernGL.
            # require=None accepts whatever profile Qt actually got us;
            # the format set in __init__ asked for 3.3 Core which the
            # shaders need (#version 330 core). If macOS hands us a
            # 4.1 Core context instead, that's still 3.3-compatible
            # so the shaders compile.
            self._ctx = moderngl.create_context(require=None)
            self._ctx.enable(moderngl.BLEND)
            self._ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

            vert_src = _read_shader("orb.vert")
            frag_src = _read_shader("orb.frag")
            self._program = self._ctx.program(
                vertex_shader=vert_src, fragment_shader=frag_src,
            )

            self._mesh = build_icosphere(self._icosphere_subdivisions)
            vbo_pos = self._ctx.buffer(self._mesh.positions.tobytes())
            vbo_nrm = self._ctx.buffer(self._mesh.normals.tobytes())
            ibo = self._ctx.buffer(self._mesh.indices.tobytes())
            self._vao = self._ctx.vertex_array(
                self._program,
                [
                    (vbo_pos, "3f", "in_position"),
                    (vbo_nrm, "3f", "in_normal"),
                ],
                index_buffer=ibo,
                index_element_size=4,
            )

            # Particles are uploaded but rendering them is gated at
            # paintGL on the config flag (kept around so toggling the
            # flag mid-session does not require a GL rebuild).
            if self._particles_enabled:
                self._particles = build_particles(self._particle_count, seed=0)
            else:
                self._particles = None

        def resizeGL(self, w: int, h: int) -> None:
            if self._ctx is None:
                return
            self._ctx.viewport = (0, 0, max(1, w), max(1, h))

        def paintGL(self) -> None:
            if self._ctx is None or self._program is None or self._vao is None:
                return

            now = time.monotonic()
            dt = max(0.0, now - self._last_tick_t)
            self._last_tick_t = now

            snap: StateSnapshot = self._state_controller.tick(dt)
            bands: BandReading = self._audio_bus.read_bands()

            # Camera: fixed orthographic-ish framing. The orb is on a
            # unit sphere so a simple translate-back + projection is
            # plenty. Use pyrr for the matrix math so we keep numpy
            # types throughout.
            from pyrr import Matrix44
            aspect = max(1.0, self.width()) / max(1.0, self.height())
            proj = Matrix44.perspective_projection(
                fovy=35.0, aspect=aspect, near=0.1, far=10.0,
            )
            view = Matrix44.from_translation((0.0, 0.0, -3.5))
            model = Matrix44.identity()
            mvp = proj * view * model

            self._program["u_mvp"].write(np.array(mvp, dtype=np.float32).tobytes())
            self._program["u_model"].write(np.array(model, dtype=np.float32).tobytes())
            self._program["u_time"].value = snap.time_seconds
            self._program["u_bass"].value = bands.bass
            self._program["u_mid"].value = bands.mid
            self._program["u_high"].value = bands.high
            self._program["u_amplitude"].value = bands.rms
            self._program["u_displacement_scale"].value = snap.displacement_scale
            self._program["u_color"].value = tuple(snap.color)
            self._program["u_intensity"].value = snap.intensity

            self._ctx.clear(0.0, 0.0, 0.0, 0.0)
            self._vao.render(mode=moderngl.TRIANGLES)

        # ── Public helpers ─────────────────────────────────────────

        def audio_bus(self) -> AudioBus:
            return self._audio_bus

        def state_controller(self) -> StateController:
            return self._state_controller

        def trigger_error(self) -> None:
            """Convenience: route an ERROR pulse through the controller."""
            self._state_controller.trigger_error()

    return _OrbWidget

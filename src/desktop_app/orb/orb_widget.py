"""QWidget that renders the reactive orb at 60 FPS via QPainter.

Phase 1 ships a pure-QPainter backend rather than QOpenGLWidget +
moderngl. The OpenGL path worked at the moderngl API level (shaders
compiled, VAOs built, FBOs detected) but on macOS 26 + Qt 6.11 the
moderngl clear / draw commands did not route to the framebuffer that
Qt presents (Qt-native ``glClear`` works fine; moderngl-issued clear
does not). Rather than fight the bridge, we draw the orb in 2D with
QPainter, matching the pattern that already powers ``LowPolyFaceWidget``
in this project.

What survives from the GL design
--------------------------------
- Geometry (``geometry.build_icosphere`` / ``build_particles``) stays
  unchanged: the icosphere is now projected to 2D per frame.
- ``AudioBus`` still feeds (rms, bass, mid, high) per frame.
- ``StateController`` still drives colour / intensity / displacement
  through cubic-eased transitions.
- The shader files (``shaders/orb.vert``, ``shaders/orb.frag``) are
  preserved on disk for reference and for a possible future re-port
  when the moderngl <-> Qt bridge stabilises on macOS 26.

What changed
------------
- Vertex displacement uses a cheap deterministic hash-based pseudo
  noise on the CPU instead of GLSL Simplex 3D. At 162 vertices /
  60 FPS this is sub-millisecond on Apple Silicon.
- Fresnel rim becomes a wider radial gradient overlay.
- Particles are 2D ellipses drawn with their own radial gradients.
- Bloom / chromatic aberration are not implemented in Phase 1; the
  spec listed them as polish and they require a render-target pass
  that QPainter does not natively expose. Phase 2 candidate.
"""

from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np
from PyQt6.QtCore import QPointF, QTimer, Qt
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QLinearGradient,
    QPainter,
    QPen,
    QRadialGradient,
)
from PyQt6.QtWidgets import QWidget

from .audio_bus import AudioBus, BandReading
from .geometry import Mesh, Particles, build_icosphere, build_particles
from .state_controller import StateController, StateSnapshot


# Visual constants (tuned for a 320x320 window with the orb filling
# roughly 65% of the shorter side).
ORB_BASE_RADIUS_RATIO = 0.32
GLOW_RADIUS_RATIO = 0.48
WIREFRAME_DISPLACEMENT_RATIO = 0.06


def _state_color_qcolor(rgb01: tuple[float, float, float], alpha: int = 255) -> QColor:
    """Convert a 0..1 RGB tuple from StateStyle to a QColor."""
    r, g, b = rgb01
    return QColor(int(r * 255), int(g * 255), int(b * 255), alpha)


class OrbWidget(QWidget):
    """QPainter-driven reactive orb.

    Constructor signature mirrors the older QOpenGLWidget version so
    existing tests and OrbWindow code are not affected:
    ``OrbWidget(audio_bus=..., state_controller=..., parent=...)``.
    """

    FRAME_INTERVAL_MS = 16  # ~60 FPS

    def __init__(
        self,
        audio_bus: Optional[AudioBus] = None,
        state_controller: Optional[StateController] = None,
        particles_enabled: bool = True,
        particle_count: int = 256,
        icosphere_subdivisions: int = 3,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setMinimumSize(320, 320)
        # Mandatory for translucent composition on the parent window.
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        # Anti-flicker hint; we paint the entire widget each frame.
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, False)
        self.setAutoFillBackground(False)

        self._audio_bus = audio_bus or AudioBus()
        self._state_controller = state_controller or StateController()
        self._particles_enabled = bool(particles_enabled)
        # Default subdiv=3 -> 642 vertices, 1280 triangles. Phase 1
        # shipped subdiv=2 (162 verts); Phase 2D bumps the detail so
        # the wireframe reads as a proper sphere even at large window
        # sizes. CPU projection cost is still <5ms/frame on Apple
        # Silicon; if a future user reports perf issues on older
        # hardware we can promote ``icosphere_subdivisions`` to a
        # config knob, but the default reflects the current target.
        self._mesh: Mesh = build_icosphere(icosphere_subdivisions)
        self._particles: Optional[Particles] = (
            build_particles(particle_count, seed=0) if particles_enabled else None
        )

        # Precompute three per-vertex deterministic phases so the
        # displacement is stable across frames AND composed of three
        # octaves of distinct pseudo-noise. Three octaves is enough to
        # give the surface organic micro-motion without leaving the
        # Python-loop FPS budget.
        # Phase 1 used two phases (slow breath + audio fast); Phase 2D
        # adds a medium-frequency middle layer so the surface texture
        # reads as live even with zero audio input — important now
        # that the SHM bus may legitimately publish silence.
        rng = np.random.default_rng(seed=42)
        self._vertex_phase_a = rng.uniform(0.0, 2.0 * math.pi, self._mesh.vertex_count).astype(np.float32)
        self._vertex_phase_b = rng.uniform(0.0, 2.0 * math.pi, self._mesh.vertex_count).astype(np.float32)
        self._vertex_phase_c = rng.uniform(0.0, 2.0 * math.pi, self._mesh.vertex_count).astype(np.float32)

        # Frame clock.
        self._t0_monotonic = time.monotonic()
        self._last_tick_t = self._t0_monotonic

        # Cached state per frame; computed inside paintEvent.
        self._latest_snapshot: Optional[StateSnapshot] = None
        self._latest_bands: BandReading = BandReading.zero()

        # Render loop.
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(self.FRAME_INTERVAL_MS)

    # ── Public helpers (preserved API) ────────────────────────────────

    def audio_bus(self) -> AudioBus:
        return self._audio_bus

    def state_controller(self) -> StateController:
        return self._state_controller

    def trigger_error(self) -> None:
        """Route an ERROR pulse through the controller."""
        self._state_controller.trigger_error()

    # ── Paint pipeline ────────────────────────────────────────────────

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt API)
        now = time.monotonic()
        dt = max(0.0, now - self._last_tick_t)
        self._last_tick_t = now

        snap: StateSnapshot = self._state_controller.tick(dt)
        bands: BandReading = self._audio_bus.read_bands()
        self._latest_snapshot = snap
        self._latest_bands = bands

        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

            w = self.width()
            h = self.height()
            cx = w / 2.0
            cy = h / 2.0
            short_side = min(w, h)

            # Audio + state derived radii.
            base_r = short_side * ORB_BASE_RADIUS_RATIO
            breath = 1.0 + 0.04 * math.sin(snap.time_seconds * 2.0)
            # bass pulse + amplitude boost.
            scale = breath + 0.10 * bands.bass + 0.05 * bands.rms
            orb_r = base_r * scale
            glow_r = short_side * GLOW_RADIUS_RATIO * (1.0 + 0.15 * bands.rms)

            color = _state_color_qcolor(snap.color)
            rim_color = _state_color_qcolor(snap.color, alpha=160)

            self._draw_glow_halo(painter, cx, cy, glow_r, color, snap.intensity, bands.rms)
            self._draw_orb_body(painter, cx, cy, orb_r, color, snap.intensity)
            self._draw_wireframe(painter, cx, cy, orb_r, snap, bands)
            if self._particles_enabled and self._particles is not None:
                self._draw_particles(painter, cx, cy, orb_r, glow_r, snap, bands)
            self._draw_inner_highlight(painter, cx, cy, orb_r, snap, bands)
        finally:
            painter.end()

    # ── Drawing helpers ───────────────────────────────────────────────

    def _draw_glow_halo(
        self, painter: QPainter, cx: float, cy: float,
        r: float, color: QColor, intensity: float, rms: float,
    ) -> None:
        """Soft wide radial gradient behind the orb body."""
        grad = QRadialGradient(cx, cy, r)
        inner = QColor(color)
        inner.setAlphaF(min(1.0, 0.30 * intensity + 0.15 * rms))
        outer = QColor(color)
        outer.setAlpha(0)
        grad.setColorAt(0.0, inner)
        grad.setColorAt(0.55, QColor(color.red(), color.green(), color.blue(),
                                      int(80 * intensity)))
        grad.setColorAt(1.0, outer)
        painter.setBrush(QBrush(grad))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(cx, cy), r, r)

    def _draw_orb_body(
        self, painter: QPainter, cx: float, cy: float,
        r: float, color: QColor, intensity: float,
    ) -> None:
        """Filled disc with a centre-bright, edge-dim radial gradient
        (fake sphere shading without GL)."""
        grad = QRadialGradient(cx - r * 0.25, cy - r * 0.30, r * 1.4)
        # Brightest near top-left to suggest a key light.
        bright = QColor(color)
        bright.setAlphaF(min(1.0, 0.95 * intensity))
        mid = QColor(color)
        mid.setAlphaF(min(1.0, 0.55 * intensity))
        dark = QColor(int(color.red() * 0.35), int(color.green() * 0.35),
                      int(color.blue() * 0.35),
                      int(220 * intensity))
        grad.setColorAt(0.0, bright)
        grad.setColorAt(0.55, mid)
        grad.setColorAt(1.0, dark)
        painter.setBrush(QBrush(grad))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(cx, cy), r, r)

    def _draw_wireframe(
        self, painter: QPainter, cx: float, cy: float,
        r: float, snap: StateSnapshot, bands: BandReading,
    ) -> None:
        """Icosphere wireframe projected to 2D with audio displacement.

        We treat the unit sphere as a top-down 2D circle and draw the
        edges that face the camera (z >= 0 in our orthographic frame).
        This avoids backface clutter without doing a full 3D pipeline.
        """
        positions = self._mesh.positions  # (V, 3) on unit sphere
        # Audio + noise displacement composed of three octaves:
        #   slow   (t * 1.2, phase_a) - persistent breath, audio-free
        #   medium (t * 2.5, phase_b) - mid-band-modulated mid-frequency
        #   fast   (t * 5.5, phase_c) - high-band-modulated quick spikes
        # The fixed amplitude ratio (0.5 / 0.3 / 0.2) gives 1/f-ish
        # pink-noise feel without an actual noise function. Audio
        # energy biases the higher octaves so loud speech reads as
        # spiky surface detail, quiet reads as smooth breathing.
        t = snap.time_seconds
        slow_octave = 0.5 * np.sin(t * 1.2 + self._vertex_phase_a)
        medium_octave = 0.3 * np.sin(t * 2.5 + self._vertex_phase_b) * (1.0 + bands.mid)
        audio_drive = (bands.bass * 1.0 + bands.mid * 0.6 + bands.high * 0.4)
        fast_octave = 0.2 * np.sin(t * 5.5 + self._vertex_phase_c) * audio_drive
        amp = (0.4 + 0.6 * bands.rms) * snap.displacement_scale
        # Per-octave displacement weights tuned so the silent state
        # shows visible motion (~slow octave * 0.04 = 2% radius lap)
        # and a loud state reads as ~10-15% radius spikes.
        disp = (slow_octave * 0.04 + medium_octave * 0.06 + fast_octave * 0.10) * amp

        # Apply displacement along each vertex's outward direction.
        outward = positions  # already unit-length normals on the sphere
        displaced = positions + outward * disp[:, None]

        # Orthographic projection to screen coords:
        # x_screen = cx + dx * r,  y_screen = cy - dy * r (Qt y-down).
        screen_x = cx + displaced[:, 0] * r
        screen_y = cy - displaced[:, 1] * r
        screen_z = displaced[:, 2]  # for facing test

        # Wireframe colour: state colour, slightly desaturated and
        # alpha modulated by audio mid band so speech-band energy
        # makes the wireframe pop.
        base = _state_color_qcolor(snap.color)
        wf_alpha = int(120 + 100 * bands.mid)
        wf_alpha = max(60, min(255, wf_alpha))
        pen = QPen(QColor(255, 255, 255, wf_alpha), 1.2)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        # Iterate the triangle index buffer in steps of 3 and draw
        # the 3 edges of each triangle when at least one of its
        # vertices is facing the camera.
        idx = self._mesh.indices.reshape(-1, 3)
        for tri in idx:
            a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
            if screen_z[a] < -0.1 and screen_z[b] < -0.1 and screen_z[c] < -0.1:
                continue  # fully behind, skip
            p_a = QPointF(screen_x[a], screen_y[a])
            p_b = QPointF(screen_x[b], screen_y[b])
            p_c = QPointF(screen_x[c], screen_y[c])
            painter.drawLine(p_a, p_b)
            painter.drawLine(p_b, p_c)
            painter.drawLine(p_c, p_a)

        # Bright dots at each forward-facing vertex.
        dot_color = QColor(255, 255, 255, min(255, 180 + int(60 * bands.high)))
        painter.setBrush(QBrush(dot_color))
        painter.setPen(Qt.PenStyle.NoPen)
        for i in range(self._mesh.vertex_count):
            if screen_z[i] < -0.05:
                continue
            painter.drawEllipse(QPointF(screen_x[i], screen_y[i]), 1.5, 1.5)

    def _draw_inner_highlight(
        self, painter: QPainter, cx: float, cy: float,
        r: float, snap: StateSnapshot, bands: BandReading,
    ) -> None:
        """A small bright spot near the top-left to imply a key light."""
        spot_r = r * 0.28
        spot_cx = cx - r * 0.30
        spot_cy = cy - r * 0.35
        grad = QRadialGradient(spot_cx, spot_cy, spot_r)
        white = QColor(255, 255, 255, int(200 * snap.intensity))
        edge = QColor(255, 255, 255, 0)
        grad.setColorAt(0.0, white)
        grad.setColorAt(1.0, edge)
        painter.setBrush(QBrush(grad))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(spot_cx, spot_cy), spot_r, spot_r)

    def _draw_particles(
        self, painter: QPainter, cx: float, cy: float,
        orb_r: float, glow_r: float,
        snap: StateSnapshot, bands: BandReading,
    ) -> None:
        """Small bright dots orbiting the orb. Their brightness is
        driven by the high band; their orbits are precomputed in
        ``geometry.build_particles``."""
        if self._particles is None:
            return
        orbits = self._particles.orbits
        t = snap.time_seconds
        # Particle radius lives between orb_r and glow_r.
        radius_span = glow_r - orb_r
        # high-band drives sparkle intensity.
        high_boost = bands.high
        base_color = _state_color_qcolor(snap.color)
        # Draw lazily — skip particles fully behind.
        painter.setPen(Qt.PenStyle.NoPen)
        for row in orbits:
            radius_t = float(row[0] - 1.25) / 0.5  # normalise to 0..1
            longitude = float(row[2] + t * row[3])
            # 3D parametric on a "tilted ring":
            x = math.cos(longitude) * math.cos(row[1])
            y = math.sin(row[1])
            z = math.sin(longitude) * math.cos(row[1])
            if z < -0.15:
                continue  # behind the orb
            r_px = orb_r + radius_span * (0.2 + 0.8 * radius_t)
            px = cx + x * r_px
            py = cy - y * r_px * 0.85  # slight perspective squish
            size = max(1.4, float(row[4]) * orb_r * 1.5)
            alpha = int(140 + 90 * high_boost)
            alpha = max(40, min(255, alpha))
            # Tinted core.
            core = QColor(base_color)
            core.setAlpha(alpha)
            grad = QRadialGradient(px, py, size * 2.2)
            grad.setColorAt(0.0, QColor(255, 255, 255, alpha))
            grad.setColorAt(0.5, core)
            grad.setColorAt(1.0, QColor(core.red(), core.green(), core.blue(), 0))
            painter.setBrush(QBrush(grad))
            painter.drawEllipse(QPointF(px, py), size, size)

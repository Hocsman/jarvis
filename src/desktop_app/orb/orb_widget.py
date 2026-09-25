"""QWidget that draws the orb at 60 FPS with QPainter.

The orb is driven by two things: the state it is in (``StateController``,
which eases colour, intensity and displacement between states) and the
clock. It has no audio input. Every frame is built in 2D from the
icosphere geometry projected onto the widget:

- a stack of soft halos behind the body (the bloom),
- the body, a disc shaded by a radial gradient,
- the wireframe of the icosphere, its vertices displaced by two
  octaves of deterministic pseudo-noise so the surface breathes,
- a red and a blue rim at a small horizontal offset in the active
  states (the chromatic aberration),
- particles orbiting between the body and the glow,
- a highlight near the top-left to suggest a key light.

Vertex displacement is two octaves of sine waves with fixed random
per-vertex phases, computed on the CPU; at 642 vertices and 60 FPS this
stays well under the frame budget.
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
    QPainter,
    QPen,
    QRadialGradient,
)
from PyQt6.QtWidgets import QWidget

from .geometry import Mesh, Particles, build_icosphere, build_particles
from .state_controller import OrbState, StateController, StateSnapshot


# Visual constants (tuned for a 320x320 window with the orb filling
# roughly 65% of the shorter side).
ORB_BASE_RADIUS_RATIO = 0.32
GLOW_RADIUS_RATIO = 0.48


def _state_color_qcolor(rgb01: tuple[float, float, float], alpha: int = 255) -> QColor:
    """Convert a 0..1 RGB tuple from StateStyle to a QColor."""
    r, g, b = rgb01
    return QColor(int(r * 255), int(g * 255), int(b * 255), alpha)


class OrbWidget(QWidget):
    """QPainter-driven orb, drawn from its state and the clock."""

    FRAME_INTERVAL_MS = 16  # ~60 FPS

    def __init__(
        self,
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

        self._state_controller = state_controller or StateController()
        self._particles_enabled = bool(particles_enabled)
        # Default subdiv=3: 642 vertices, 1280 triangles, so the
        # wireframe reads as a proper sphere even at large window
        # sizes. The CPU projection costs under 5 ms a frame on Apple
        # Silicon. ``icosphere_subdivisions`` is a constructor argument
        # for tests and scripts; the app always uses the default.
        self._mesh: Mesh = build_icosphere(icosphere_subdivisions)
        self._particles: Optional[Particles] = (
            build_particles(particle_count, seed=0) if particles_enabled else None
        )

        # Precompute two per-vertex deterministic phases so the
        # displacement is stable across frames and composed of two
        # octaves of distinct pseudo-noise: a slow breath and a medium
        # layer of organic micro-motion, within the Python-loop FPS
        # budget.
        rng = np.random.default_rng(seed=42)
        self._vertex_phase_a = rng.uniform(0.0, 2.0 * math.pi, self._mesh.vertex_count).astype(np.float32)
        self._vertex_phase_b = rng.uniform(0.0, 2.0 * math.pi, self._mesh.vertex_count).astype(np.float32)

        # Frame clock.
        self._last_tick_t = time.monotonic()

        # Render loop.
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(self.FRAME_INTERVAL_MS)

    # ── Public helpers ────────────────────────────────────────────────

    def state_controller(self) -> StateController:
        return self._state_controller

    def trigger_error(self) -> None:
        """Route an ERROR pulse through the controller."""
        self._state_controller.trigger_error()

    def pause_rendering(self) -> None:
        """Stop the 60 Hz repaint loop.

        Call when the orb is off-screen (host window hidden/minimised)
        so it stops burning a render frame budget — and the GPU/CPU it
        implies — on pixels nobody can see. Idempotent.
        """
        if self._timer.isActive():
            self._timer.stop()

    def resume_rendering(self) -> None:
        """Restart the repaint loop after :meth:`pause_rendering`.

        Resets the frame clock so the first frame back doesn't apply a
        large ``dt`` (which would make the state transition jump).
        Idempotent.
        """
        if not self._timer.isActive():
            self._last_tick_t = time.monotonic()
            self._timer.start(self.FRAME_INTERVAL_MS)

    # ── Paint pipeline ────────────────────────────────────────────────

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt API)
        now = time.monotonic()
        dt = max(0.0, now - self._last_tick_t)
        self._last_tick_t = now

        snap: StateSnapshot = self._state_controller.tick(dt)

        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

            w = self.width()
            h = self.height()
            cx = w / 2.0
            cy = h / 2.0
            short_side = min(w, h)

            # Radii: the base size with a slow breath on top.
            base_r = short_side * ORB_BASE_RADIUS_RATIO
            breath = 1.0 + 0.04 * math.sin(snap.time_seconds * 2.0)
            orb_r = base_r * breath
            glow_r = short_side * GLOW_RADIUS_RATIO

            color = _state_color_qcolor(snap.color)

            self._draw_glow_halo(painter, cx, cy, glow_r, color, snap.intensity)
            self._draw_orb_body(painter, cx, cy, orb_r, color, snap.intensity)
            self._draw_wireframe(painter, cx, cy, orb_r, snap)
            self._draw_chromatic_aberration(painter, cx, cy, orb_r, snap)
            if self._particles_enabled and self._particles is not None:
                self._draw_particles(painter, cx, cy, orb_r, glow_r, snap)
            self._draw_inner_highlight(painter, cx, cy, orb_r, snap)
        finally:
            painter.end()

    # ── Drawing helpers ───────────────────────────────────────────────

    # Multi-halo bloom stack. Each tuple = (radius_multiplier,
    # inner_alpha_base, outer_alpha_base). The innermost halo is the
    # brightest and tightest; subsequent halos grow in radius and
    # fall in opacity. Stacking 4 halos with these ratios produces a
    # convincing "soft bloom" effect: four stacked radial gradients
    # that the eye reads as light scatter.
    _BLOOM_HALOS = (
        # (radius_x, inner_alpha_mult, mid_alpha_mult)
        (1.0, 0.60, 0.30),  # core halo, follows the orb closely
        (1.3, 0.35, 0.18),  # primary glow
        (1.7, 0.20, 0.10),  # outer wash
        (2.2, 0.10, 0.05),  # far rim, barely visible but adds depth
    )

    def _draw_glow_halo(
        self, painter: QPainter, cx: float, cy: float,
        r: float, color: QColor, intensity: float,
    ) -> None:
        """Soft multi-halo bloom behind the orb body.

        Four concentric halos at growing radii with decreasing
        opacities: the "fake bloom" technique, cheap, additive in the
        alpha channel, and convincing because the eye cannot tell a
        true Gaussian blur from four stacked radial gradients.

        Colour depth: each halo uses three stops rather than two so the
        gradient has a perceptible mid-tint (close to the state colour
        but desaturated) rather than fading the inner colour straight
        to transparent.
        """
        # The state's intensity sets how bright the halo stack is.
        alpha_boost = min(1.0, 0.30 * intensity)
        painter.setPen(Qt.PenStyle.NoPen)

        for radius_mult, inner_mult, mid_mult in self._BLOOM_HALOS:
            halo_r = r * radius_mult
            grad = QRadialGradient(cx, cy, halo_r)

            inner = QColor(color)
            inner.setAlphaF(min(1.0, inner_mult * alpha_boost))
            mid_color = QColor(color)
            mid_color.setAlphaF(min(1.0, mid_mult * alpha_boost))
            # Third colour stop: very desaturated cousin of the state
            # colour, sitting at ~75 % radius to give the gradient a
            # softer roll-off than a straight inner->transparent.
            tint = QColor(
                int(color.red() * 0.7 + 80),
                int(color.green() * 0.7 + 80),
                int(color.blue() * 0.7 + 80),
                int(255 * mid_mult * alpha_boost * 0.4),
            )
            outer = QColor(color)
            outer.setAlpha(0)

            grad.setColorAt(0.0, inner)
            grad.setColorAt(0.45, mid_color)
            grad.setColorAt(0.75, tint)
            grad.setColorAt(1.0, outer)

            painter.setBrush(QBrush(grad))
            painter.drawEllipse(QPointF(cx, cy), halo_r, halo_r)

    def _draw_orb_body(
        self, painter: QPainter, cx: float, cy: float,
        r: float, color: QColor, intensity: float,
    ) -> None:
        """Filled disc with a centre-bright, edge-dim radial gradient
        that reads as sphere shading."""
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
        r: float, snap: StateSnapshot,
    ) -> None:
        """Icosphere wireframe projected to 2D with a breathing displacement.

        We treat the unit sphere as a top-down 2D circle and draw the
        edges that face the camera (z >= 0 in our orthographic frame).
        This avoids backface clutter without doing a full 3D pipeline.
        """
        positions = self._mesh.positions  # (V, 3) on unit sphere
        # Displacement composed of two octaves of pseudo-noise:
        #   slow   (t * 1.2, phase_a) - persistent breath
        #   medium (t * 2.5, phase_b) - organic micro-motion
        # The fixed amplitude ratio (0.5 / 0.3) gives a 1/f-ish
        # pink-noise feel without an actual noise function.
        t = snap.time_seconds
        slow_octave = 0.5 * np.sin(t * 1.2 + self._vertex_phase_a)
        medium_octave = 0.3 * np.sin(t * 2.5 + self._vertex_phase_b)
        amp = 0.4 * snap.displacement_scale
        # Per-octave displacement weights tuned so the surface shows
        # visible motion (~slow octave * 0.04 = 2% radius lap).
        disp = (slow_octave * 0.04 + medium_octave * 0.06) * amp

        # Apply displacement along each vertex's outward direction.
        outward = positions  # already unit-length normals on the sphere
        displaced = positions + outward * disp[:, None]

        # Orthographic projection to screen coords:
        # x_screen = cx + dx * r,  y_screen = cy - dy * r (Qt y-down).
        screen_x = cx + displaced[:, 0] * r
        screen_y = cy - displaced[:, 1] * r
        screen_z = displaced[:, 2]  # for facing test

        # Wireframe: white lines at a fixed, restrained opacity.
        pen = QPen(QColor(255, 255, 255, 120), 1.2)
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
        dot_color = QColor(255, 255, 255, 180)
        painter.setBrush(QBrush(dot_color))
        painter.setPen(Qt.PenStyle.NoPen)
        for i in range(self._mesh.vertex_count):
            if screen_z[i] < -0.05:
                continue
            painter.drawEllipse(QPointF(screen_x[i], screen_y[i]), 1.5, 1.5)

    # States that warrant the chromatic-aberration rim. The "active"
    # states — the orb is thinking, speaking, or signalling an error
    # — get the extra edginess; calm states stay clean.
    _CHROMATIC_ABERRATION_STATES = frozenset({
        OrbState.THINKING, OrbState.SPEAKING, OrbState.ERROR,
    })

    def _draw_chromatic_aberration(
        self, painter: QPainter, cx: float, cy: float,
        r: float, snap: StateSnapshot,
    ) -> None:
        """Fake RGB-shift rim added on top of the wireframe.

        Real chromatic aberration is a per-pixel channel offset on a
        rendered image; doing it for real in QPainter would require
        an offscreen QImage + QPainter.drawImage at three offsets,
        which is doable but unnecessary. The eye reads the same
        effect from two extra coloured rim outlines drawn at small
        horizontal offsets relative to the orb body.

        Triggered only for active states (THINKING / SPEAKING /
        ERROR). Calm states (IDLE / LISTENING) stay clean — the rim
        would compete with the soft halo bloom otherwise.

        Implementation:
        - Red rim shifted -2 px on x, alpha 80 (out of 255).
        - Blue rim shifted +2 px on x, alpha 80.
        - Composition: ``CompositionMode_Plus`` so the rim adds to
          the underlying body colour rather than overwriting it.
        """
        if snap.state not in self._CHROMATIC_ABERRATION_STATES:
            return

        # A 2 px offset either side: small enough that the orb does
        # not disintegrate visually.
        shift = 2.0

        prev_mode = painter.compositionMode()
        try:
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
            painter.setBrush(Qt.BrushStyle.NoBrush)

            red_pen = QPen(QColor(255, 30, 30, 80), 2.0)
            red_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(red_pen)
            painter.drawEllipse(QPointF(cx - shift, cy), r, r)

            blue_pen = QPen(QColor(30, 90, 255, 80), 2.0)
            blue_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(blue_pen)
            painter.drawEllipse(QPointF(cx + shift, cy), r, r)
        finally:
            painter.setCompositionMode(prev_mode)

    def _draw_inner_highlight(
        self, painter: QPainter, cx: float, cy: float,
        r: float, snap: StateSnapshot,
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
        snap: StateSnapshot,
    ) -> None:
        """Small bright dots orbiting the orb on a slow drift.

        Disabled entirely when the orb was constructed with
        ``particles_enabled=False`` (driven by ``cfg.ui.orb_particles_enabled``).
        """
        if self._particles is None:
            return
        orbits = self._particles.orbits
        t = snap.time_seconds
        # Particle radius lives between orb_r and glow_r.
        radius_span = glow_r - orb_r
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
            alpha = 140
            # Tinted core.
            core = QColor(base_color)
            core.setAlpha(alpha)
            grad = QRadialGradient(px, py, size * 2.2)
            grad.setColorAt(0.0, QColor(255, 255, 255, alpha))
            grad.setColorAt(0.5, core)
            grad.setColorAt(1.0, QColor(core.red(), core.green(), core.blue(), 0))
            painter.setBrush(QBrush(grad))
            painter.drawEllipse(QPointF(px, py), size, size)

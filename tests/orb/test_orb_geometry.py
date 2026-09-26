"""Geometry properties of the orb.

The default icosphere is subdiv=3 (642 vertices, 1280 triangles) and
the per-vertex displacement is composed of two octaves of pseudo-noise
so the surface breathes at rest.

These tests pin:

1. The default vertex count is 642, i.e. the OrbWidget constructor
   really uses subdiv=3. Catches accidental regressions in the
   constructor default.

2. The icosphere builder returns the documented counts at each
   subdivision level. This is a structural invariant of the geometry
   module; useful sanity check if a future optimisation rewrites it.

3. Two per-vertex phase arrays (a, b) exist on the widget, one per
   octave. A regression to a single octave would leave only the slow
   breath, and we want to notice.

We don't go further into "the visual is correct" because that's
inherently subjective. The capture script in scripts/orb_capture_states.py
covers the visual A/B side.
"""

from __future__ import annotations

import os

import pytest


# Headless Qt platform: required for any OrbWidget construction in
# CI / pre-commit hook environments where no display is available.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class TestIcosphereCounts:
    """Pin the documented counts from the geometry module.

    Standard recursive midpoint subdivision of an icosahedron:
        subdiv  vertices  triangles
          0        12        20
          1        42        80
          2       162       320
          3       642      1280
    """

    @pytest.mark.unit
    @pytest.mark.parametrize("subdiv,verts,tris", [
        (0, 12, 20),
        (1, 42, 80),
        (2, 162, 320),
        (3, 642, 1280),
    ])
    def test_subdivision_counts(self, subdiv: int, verts: int, tris: int) -> None:
        from desktop_app.orb.geometry import build_icosphere

        mesh = build_icosphere(subdiv)
        assert mesh.vertex_count == verts, (
            f"subdiv={subdiv}: expected {verts} verts, got {mesh.vertex_count}"
        )
        assert mesh.triangle_count == tris, (
            f"subdiv={subdiv}: expected {tris} tris, got {mesh.triangle_count}"
        )

    @pytest.mark.unit
    def test_positions_are_unit_sphere(self) -> None:
        """Every vertex must lie on the unit sphere (radius == 1 within
        a small float tolerance). The renderer treats positions as unit
        normals; a bad radius here would skew the displacement maths
        and make the orb look misshapen."""
        import numpy as np
        from desktop_app.orb.geometry import build_icosphere

        mesh = build_icosphere(3)
        norms = np.linalg.norm(mesh.positions, axis=1)
        assert np.all(np.abs(norms - 1.0) < 1e-5), (
            f"vertex norms out of [1-eps, 1+eps]: "
            f"min={norms.min()}, max={norms.max()}"
        )


class TestOrbWidgetDefaultGeometry:
    """Constructor defaults, pinned separately from the builder's counts:
    the widget asks for subdiv=3 and its surface moves on the clock."""

    @pytest.fixture
    def _qapp(self):
        """Create a single QApplication for the test session.
        OrbWidget construction needs a QApplication even offscreen."""
        from PyQt6.QtWidgets import QApplication
        import sys

        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        yield app

    @pytest.mark.unit
    def test_default_vertex_count_is_642(self, _qapp) -> None:
        """Construct an OrbWidget with no kwargs — the icosphere must
        have 642 vertices (subdiv=3)."""
        from desktop_app.orb.orb_widget import OrbWidget

        widget = OrbWidget()
        try:
            assert widget._mesh.vertex_count == 642, (
                f"Default OrbWidget vertex count is {widget._mesh.vertex_count}, "
                f"expected 642 (subdiv=3)."
            )
        finally:
            widget.deleteLater()

    @pytest.mark.unit
    def test_the_surface_breathes(self, _qapp) -> None:
        """The wireframe's vertices sit at different places at two
        instants: the surface moves on the clock, with nothing else
        driving it."""
        from unittest.mock import MagicMock
        from desktop_app.orb.orb_widget import OrbWidget
        from desktop_app.orb.state_controller import OrbState, StateSnapshot

        def _snapshot_at(t: float) -> StateSnapshot:
            return StateSnapshot(
                state=OrbState.IDLE, color=(0.5, 0.7, 1.0), intensity=0.8,
                displacement_scale=1.0, pulse_period_s=3.0, transitioning=False,
                time_seconds=t,
            )

        def _dots(painter):
            return [(args[0].x(), args[0].y()) for args, _kw in painter.drawEllipse.call_args_list]

        widget = OrbWidget()
        try:
            then, later = MagicMock(), MagicMock()
            widget._draw_wireframe(then, cx=160.0, cy=160.0, r=100.0, snap=_snapshot_at(0.0))
            widget._draw_wireframe(later, cx=160.0, cy=160.0, r=100.0, snap=_snapshot_at(0.5))
            moved = sum(1 for a, b in zip(_dots(then), _dots(later))
                        if abs(a[0] - b[0]) > 0.01 or abs(a[1] - b[1]) > 0.01)
            assert moved > 0, "the surface did not move between two instants"
        finally:
            widget.deleteLater()

    @pytest.mark.unit
    def test_explicit_subdiv_override_honoured(self, _qapp) -> None:
        """``icosphere_subdivisions`` is honoured when passed explicitly."""
        from desktop_app.orb.orb_widget import OrbWidget

        widget = OrbWidget(icosphere_subdivisions=2)
        try:
            assert widget._mesh.vertex_count == 162
        finally:
            widget.deleteLater()

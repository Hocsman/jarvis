"""Geometry properties for the Phase 2D orb refinement.

Phase 1 shipped a subdiv=2 icosphere (162 vertices, 320 triangles).
Phase 2D bumps the default to subdiv=3 (642 verts, 1280 tris) and
composes the per-vertex displacement from three octaves of
pseudo-noise so the surface reads as live even at idle.

These tests pin:

1. The default vertex count is 642 — i.e. the OrbWidget constructor
   really uses subdiv=3, not the Phase 1 subdiv=2 default. Catches
   accidental regressions in the constructor default.

2. The icosphere builder returns the documented counts at each
   subdivision level. This is a structural invariant of the geometry
   module; useful sanity check if a future optimisation rewrites it.

3. Three per-vertex phase arrays (a, b, c) exist on the widget — one
   per octave. A regression that drops back to two octaves would
   wash the surface motion out at silent states (no audio input
   means the only octave with non-trivial amplitude is the slow one),
   and we want to notice.

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
        a small float tolerance). The shader treats positions as unit
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
    """Constructor defaults: Phase 2D widget has subdiv=3 + three
    phase arrays. A regression that flipped the default back to
    subdiv=2 would still pass the icosphere counts test above but
    would fail here — which is the failure mode we care about."""

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
                f"expected 642 (subdiv=3). Did the default subdiv flip back to 2?"
            )
        finally:
            widget.deleteLater()

    @pytest.mark.unit
    def test_three_phase_arrays_exist(self, _qapp) -> None:
        """Three octaves means three phase arrays. A regression that
        dropped to two octaves would still let the orb breathe at
        idle (slow octave alone) but would wash out the audio-driven
        detail layer. Test name names the attributes explicitly so
        a future renamer notices."""
        from desktop_app.orb.orb_widget import OrbWidget

        widget = OrbWidget()
        try:
            for attr in ("_vertex_phase_a", "_vertex_phase_b", "_vertex_phase_c"):
                assert hasattr(widget, attr), (
                    f"Missing phase array {attr!r}; orb octave composition "
                    f"is incomplete. Re-add the third phase or drop the "
                    f"multi-octave promise from the docs."
                )
                # Each phase array must be sized to the mesh.
                arr = getattr(widget, attr)
                assert arr.shape[0] == widget._mesh.vertex_count, (
                    f"{attr} has {arr.shape[0]} entries but mesh has "
                    f"{widget._mesh.vertex_count} vertices."
                )
        finally:
            widget.deleteLater()

    @pytest.mark.unit
    def test_explicit_subdiv_override_honoured(self, _qapp) -> None:
        """Tests / future debug modes can still pass subdiv=2 or 4 via
        kwarg. The default-flip didn't make the parameter read-only."""
        from desktop_app.orb.orb_widget import OrbWidget

        widget = OrbWidget(icosphere_subdivisions=2)
        try:
            assert widget._mesh.vertex_count == 162
        finally:
            widget.deleteLater()

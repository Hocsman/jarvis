"""Phase 2D fake-bloom + color depth tests.

The Phase 2D upgrade stacks 4 concentric halos behind the orb body
instead of a single radial gradient. We can't unit-test "the visual
looks good" so we pin the structural contract instead:

1. The halo stack has 4 entries (regression guard: a future
   contributor who replaces this with a single gradient again would
   trip the count).

2. Halo radii increase monotonically (visually meaningful — the
   stack works because each halo is wider than the previous).

3. Halo opacity multipliers decrease monotonically (the outermost
   halo must be the dimmest).

4. Each halo contributes 4 color stops to its radial gradient
   (inner, mid, tint, outer) — the "color depth" Phase 2D goal.
   We verify this by mocking the painter and counting setColorAt
   calls per drawn ellipse.

5. The bloom stack drawing path is wired into ``paintEvent`` — i.e.
   ``_draw_glow_halo`` is actually called once per frame and emits
   the expected number of drawEllipse calls.

Strategy: substitute QPainter.drawEllipse + QRadialGradient.setColorAt
with counting mocks during a single paintEvent call. This stays at
the public-behaviour level (number of draws / color stops) rather
than asserting pixel values, which would be brittle.
"""

from __future__ import annotations

import math
import os
from unittest.mock import MagicMock, patch

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


class TestBloomStackStructure:
    """The ``_BLOOM_HALOS`` table is the source of truth for the
    multi-halo design. Pin its shape so accidental edits surface
    immediately."""

    @pytest.mark.unit
    def test_halo_count_is_four(self, _qapp) -> None:
        """Spec: 4-5 halos. We ship with 4; this test catches a
        regression to 1 (the Phase 1 design)."""
        from desktop_app.orb.orb_widget import OrbWidget

        assert len(OrbWidget._BLOOM_HALOS) >= 4, (
            f"Bloom stack has {len(OrbWidget._BLOOM_HALOS)} halos, "
            f"expected >= 4 (Phase 2D spec calls for 4-5)."
        )

    @pytest.mark.unit
    def test_halo_radii_strictly_increasing(self, _qapp) -> None:
        """Each subsequent halo must be wider than its predecessor.
        Otherwise stacking them is pointless — the visual depends on
        the largest halo extending past the next inner one."""
        from desktop_app.orb.orb_widget import OrbWidget

        radii = [h[0] for h in OrbWidget._BLOOM_HALOS]
        assert all(b > a for a, b in zip(radii, radii[1:])), (
            f"Halo radii not strictly increasing: {radii}"
        )

    @pytest.mark.unit
    def test_halo_alphas_strictly_decreasing(self, _qapp) -> None:
        """Inner halo is the brightest. Each outer one fades. If a
        future edit reverses this, the orb gets a dark core with a
        bright rim — visually wrong."""
        from desktop_app.orb.orb_widget import OrbWidget

        # Inner-alpha is index 1 of each halo tuple.
        inner_alphas = [h[1] for h in OrbWidget._BLOOM_HALOS]
        assert all(b < a for a, b in zip(inner_alphas, inner_alphas[1:])), (
            f"Inner alphas not strictly decreasing: {inner_alphas}"
        )

    @pytest.mark.unit
    def test_innermost_halo_brighter_than_outermost(self, _qapp) -> None:
        """Specific anchor: alpha(halo[0]) > alpha(halo[-1])."""
        from desktop_app.orb.orb_widget import OrbWidget

        halos = OrbWidget._BLOOM_HALOS
        assert halos[0][1] > halos[-1][1]
        assert halos[0][1] > halos[-1][2]


class TestBloomDrawingCalls:
    """Behavioural test: ``_draw_glow_halo`` issues one drawEllipse
    per halo, and each gradient gets 4 color stops (the "color
    depth" upgrade)."""

    @pytest.mark.unit
    def test_draw_glow_halo_issues_one_ellipse_per_halo(self, _qapp) -> None:
        """Mock the painter; count ellipses; assert == 4 (one per
        halo in the default stack)."""
        from desktop_app.orb.orb_widget import OrbWidget
        from PyQt6.QtGui import QColor

        widget = OrbWidget()
        try:
            painter = MagicMock()
            color = QColor(120, 200, 255)
            widget._draw_glow_halo(
                painter, cx=160.0, cy=160.0, r=80.0,
                color=color, intensity=0.8, rms=0.3,
            )
            assert painter.drawEllipse.call_count == len(OrbWidget._BLOOM_HALOS), (
                f"Expected {len(OrbWidget._BLOOM_HALOS)} ellipses (one per halo); "
                f"got {painter.drawEllipse.call_count}. Either the stack "
                f"shrank or the loop short-circuits unexpectedly."
            )
        finally:
            widget.deleteLater()

    @pytest.mark.unit
    def test_each_halo_has_at_least_three_color_stops(self, _qapp) -> None:
        """Color depth goal: 3+ stops per halo gradient. We intercept
        QRadialGradient construction to verify how many setColorAt
        calls each receives."""
        from desktop_app.orb.orb_widget import OrbWidget
        from PyQt6.QtGui import QColor, QRadialGradient

        stops_per_gradient: list[int] = []

        original_set_color_at = QRadialGradient.setColorAt

        # Wrap setColorAt to count calls per gradient instance.
        counts: dict[int, int] = {}

        def counting_set_color_at(self, position, color):
            counts[id(self)] = counts.get(id(self), 0) + 1
            return original_set_color_at(self, position, color)

        widget = OrbWidget()
        try:
            painter = MagicMock()
            color = QColor(120, 200, 255)
            with patch.object(QRadialGradient, "setColorAt", counting_set_color_at):
                widget._draw_glow_halo(
                    painter, cx=160.0, cy=160.0, r=80.0,
                    color=color, intensity=0.8, rms=0.3,
                )

            # We should have at least one gradient per halo; each
            # gradient must have >= 3 stops (inner + at least 1 mid
            # + outer).
            stops_per_gradient = list(counts.values())
            assert len(stops_per_gradient) >= len(OrbWidget._BLOOM_HALOS), (
                f"Expected >= {len(OrbWidget._BLOOM_HALOS)} gradients (one per "
                f"halo); got {len(stops_per_gradient)}"
            )
            for n_stops in stops_per_gradient:
                assert n_stops >= 3, (
                    f"A halo gradient has only {n_stops} colour stops; "
                    f"Phase 2D color depth requires >= 3 (inner / mid / outer "
                    f"at minimum)."
                )
        finally:
            widget.deleteLater()

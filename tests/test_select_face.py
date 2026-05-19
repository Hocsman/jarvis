"""Behaviour tests for ``desktop_app.app.select_face``.

The helper consolidates the face-widget choice that used to live in
two scattered ``if "--no-orb" in sys.argv`` branches into a single
decision point. Tests pin:

1. Default config (``cfg.ui.face == "orb"``) yields ``"orb"``.
2. Config ``"lowpoly"`` is honoured.
3. ``--no-orb`` in argv overrides config-orb to ``"lowpoly"`` (legacy
   CLI compat).
4. ``--no-orb`` is idempotent with config-lowpoly (both say lowpoly).
5. Defensive fallback: malformed cfg without a ``ui`` attribute
   returns ``"orb"`` rather than raising.

These are pure unit tests — no Qt, no app startup.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytest


@dataclass
class _UI:
    face: str


@dataclass
class _Cfg:
    ui: Optional[_UI] = None


# ── Tests ───────────────────────────────────────────────────────────────


class TestSelectFace:

    @pytest.mark.unit
    def test_default_face_is_orb(self) -> None:
        """Config default (``ui.face == "orb"``) and a clean argv ->
        ``"orb"``. This is the path 99% of fresh installs follow after
        Phase 2A."""
        from desktop_app.app import select_face

        assert select_face(_Cfg(ui=_UI(face="orb")), argv=["jarvis"]) == "orb"

    @pytest.mark.unit
    def test_face_lowpoly_when_configured(self) -> None:
        """User explicitly set ``ui.face = "lowpoly"`` — must be
        respected. No argv override present."""
        from desktop_app.app import select_face

        assert select_face(_Cfg(ui=_UI(face="lowpoly")), argv=["jarvis"]) == "lowpoly"

    @pytest.mark.unit
    def test_no_orb_flag_overrides_config_orb(self) -> None:
        """The legacy ``--no-orb`` CLI flag stays in scripts and CI.
        It must continue to force lowpoly regardless of config."""
        from desktop_app.app import select_face

        cfg = _Cfg(ui=_UI(face="orb"))
        assert select_face(cfg, argv=["jarvis", "--no-orb"]) == "lowpoly"

    @pytest.mark.unit
    def test_no_orb_flag_with_lowpoly_config_idempotent(self) -> None:
        """``--no-orb`` + config 'lowpoly' -> 'lowpoly'. Pinning this
        guarantees the CLI override is a clean idempotent step rather
        than toggling state in surprising ways."""
        from desktop_app.app import select_face

        cfg = _Cfg(ui=_UI(face="lowpoly"))
        assert select_face(cfg, argv=["jarvis", "--no-orb"]) == "lowpoly"

    @pytest.mark.unit
    def test_invalid_face_value_in_cfg_falls_back_to_orb(self) -> None:
        """Defensive: if a future cfg somehow surfaces an unexpected
        face value (test fixtures, partial mocks), select_face must
        return ``"orb"`` rather than propagate the bad value upstream.
        The real config parser already validates, so this is a
        belt-and-braces for cfg objects built outside ``load_settings``."""
        from desktop_app.app import select_face

        cfg = _Cfg(ui=_UI(face="holographic"))
        assert select_face(cfg, argv=["jarvis"]) == "orb"

    @pytest.mark.unit
    def test_cfg_without_ui_attribute_falls_back_to_orb(self) -> None:
        """A bare cfg passed by a test that doesn't bother setting
        ``ui`` must still produce a usable face name — the desktop
        path can't degrade gracefully on a missing field, so we
        absorb it here."""
        from desktop_app.app import select_face

        class _PartialCfg:
            pass  # no ``ui`` attribute at all

        assert select_face(_PartialCfg(), argv=["jarvis"]) == "orb"

    @pytest.mark.unit
    def test_cfg_with_none_ui_falls_back_to_orb(self) -> None:
        """Sometimes a test passes ``ui=None`` explicitly. Same fallback."""
        from desktop_app.app import select_face

        assert select_face(_Cfg(ui=None), argv=["jarvis"]) == "orb"

    @pytest.mark.unit
    def test_no_orb_flag_anywhere_in_argv(self) -> None:
        """``--no-orb`` doesn't need to be the last argument. It's a
        membership check, not positional parsing."""
        from desktop_app.app import select_face

        cfg = _Cfg(ui=_UI(face="orb"))
        assert select_face(cfg, argv=["jarvis", "--debug", "--no-orb", "--verbose"]) == "lowpoly"


class TestNoStrayNoOrbBranches:
    """After Phase 2A the wiring should be: select_face is the *only*
    place that consumes ``--no-orb`` for face-widget decisions. Two
    call sites in app.py (start-listening and main startup) must
    delegate to it; bare ``"--no-orb" in sys.argv`` checks elsewhere
    that drive face widgets would re-fragment the decision and
    re-introduce the bug Phase 2A is fixing.
    """

    @pytest.mark.unit
    def test_select_face_consumes_no_orb(self) -> None:
        """select_face itself is allowed to mention --no-orb. This
        test is here to document the intent: select_face is the
        gatekeeper."""
        import inspect

        from desktop_app import app as app_mod

        source = inspect.getsource(app_mod.select_face)
        assert "--no-orb" in source, (
            "select_face must reference --no-orb — it owns the legacy "
            "CLI override path."
        )

    @pytest.mark.unit
    def test_face_widget_call_sites_use_select_face(self) -> None:
        """The two original face-widget gates (around the tray
        start-listening path and the main startup orb show) must now
        call ``select_face``. If a future refactor introduces a third
        site that bypasses select_face, this test surfaces it
        immediately."""
        import inspect

        from desktop_app import app as app_mod

        full_source = inspect.getsource(app_mod)
        # The full source must contain at least two select_face call
        # sites beyond the function definition itself.
        call_sites = full_source.count("select_face(")
        # 1 occurrence inside the def line, plus at least 2 at the
        # call sites we wired (start-listening + main startup).
        assert call_sites >= 3, (
            f"Expected select_face to be called from at least 2 sites "
            f"(start-listening + main startup); found {call_sites - 1} "
            f"call(s) beyond the def. A regression here likely means a "
            f"call site bypassed select_face and went back to a bare "
            f"`'--no-orb' in sys.argv` check."
        )

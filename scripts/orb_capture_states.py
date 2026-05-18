"""Capture one screenshot per orb state for visual A/B review.

Phase 2D adds several visual upgrades (bloom stack, chromatic
aberration, denser geometry, particle audio coupling). The eye is
the only reasonable judge for these — automated assertions can pin
that we DREW the right number of ellipses but not that the result
looks good. This script generates one PNG per orb state so you can
compare side-by-side before / after a change.

Usage:
    PYTHONPATH=src .mamba_env/bin/python scripts/orb_capture_states.py

Output:
    docs/img/orb_state_idle.png
    docs/img/orb_state_listening.png
    docs/img/orb_state_thinking.png
    docs/img/orb_state_speaking.png
    docs/img/orb_state_error.png

The script forces ``QT_QPA_PLATFORM=offscreen`` before constructing
the QApplication so it runs headless in CI / over SSH. The orb is
allowed a few render frames to settle into the requested state
before each grab (the cubic-ease transitions take ~250 ms each).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Offscreen Qt — must be set BEFORE the QApplication is created.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# States we want a frame for. ERROR is special: it's normally
# transient (auto-fades back), so we capture it during the visible
# peak by triggering and grabbing inside the controller's fade
# window. The simpler approach: just set the state and grab; the
# ERROR overlay is at full intensity at t=0 of the fade.
_STATES = ["idle", "listening", "thinking", "speaking", "error"]


def _settle(widget, render_seconds: float) -> None:
    """Process Qt events for ``render_seconds`` so timers fire and
    the cubic-ease transition completes before the grab."""
    from PyQt6.QtCore import QCoreApplication

    end = time.monotonic() + render_seconds
    while time.monotonic() < end:
        QCoreApplication.processEvents()
        time.sleep(0.016)  # ~60 Hz


def _capture_state(orb_window, state_name: str, out_path: Path) -> None:
    from desktop_app.orb.state_controller import OrbState

    controller = orb_window._orb.state_controller()
    state = getattr(OrbState, state_name.upper())

    # ERROR needs the dedicated trigger; set_state() forwards to it.
    controller.set_state(state)
    # Let the cubic-ease + a couple of FFT frames settle.
    _settle(orb_window, render_seconds=0.6)

    # Grab the orb widget (not the whole window — the window has
    # the frameless translucent background which complicates PNG
    # diffing).
    pixmap = orb_window._orb.grab()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not pixmap.save(str(out_path), "PNG"):
        raise RuntimeError(f"failed to save {out_path}")
    print(f"  📸 {state_name:<10}  →  {out_path}", flush=True)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "docs" / "img"
    print(f"🟠 Orb capture — output dir: {out_dir}", flush=True)
    print(f"   states: {', '.join(_STATES)}", flush=True)
    print(flush=True)

    from PyQt6.QtWidgets import QApplication
    from desktop_app.orb.orb_window import OrbWindow

    app = QApplication.instance() or QApplication(sys.argv)
    orb_window = OrbWindow()
    # Position off-screen — we don't want a flash on the user's
    # actual desktop while running, even though offscreen QPA
    # already keeps the window invisible.
    orb_window.move(-10000, -10000)
    orb_window.show_orb()
    # Initial settle for the orb's first paint pass.
    _settle(orb_window, render_seconds=0.3)

    try:
        for state_name in _STATES:
            out_path = out_dir / f"orb_state_{state_name}.png"
            _capture_state(orb_window, state_name, out_path)
    finally:
        orb_window.close()
        # Give Qt a moment to release resources before exit.
        _settle(orb_window, render_seconds=0.1)

    print(flush=True)
    print(f"✅ Captured {len(_STATES)} states in {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

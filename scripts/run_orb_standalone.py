"""Standalone orb launcher for debug.

No tray, no FaceWindow, no daemon. Just a QApplication + OrbWindow
so we can isolate whether the orb itself renders or whether something
in the desktop_app context is hiding it.

Usage:
    PYTHONPATH=src .mamba_env/bin/python scripts/run_orb_standalone.py
"""

from __future__ import annotations

import sys


def main() -> int:
    from PyQt6.QtWidgets import QApplication
    from desktop_app.orb.orb_window import OrbWindow

    app = QApplication(sys.argv)
    win = OrbWindow()
    win.show_orb()
    print("🟠 Orb shown (standalone). Close the window or Ctrl+C to exit.", flush=True)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

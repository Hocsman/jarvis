"""How a dictation hotkey reads on screen.

Plain string work, kept apart from ``dictation_engine`` so the settings
window and the setup wizard can label their hotkey dropdown without loading
sounddevice, numpy or pynput.
"""

from __future__ import annotations

import platform


def format_hotkey_display(combo: str) -> str:
    """Format a hotkey string for human-readable display.

    On Windows, ``cmd`` is shown as ``Win``.  On macOS, ``cmd`` stays as
    ``Cmd`` and ``alt`` becomes ``Option``.  Key parts are title-cased and
    joined with `` + ``.
    """
    system = platform.system().lower()
    parts = [p.strip().lower() for p in combo.split("+") if p.strip()]

    display_parts: list[str] = []
    for part in parts:
        if part in ("cmd", "super", "win"):
            if system == "windows":
                display_parts.append("Win")
            else:
                display_parts.append("Cmd")
        elif part == "alt" and system == "darwin":
            display_parts.append("Option")
        else:
            display_parts.append(part.capitalize())

    return " + ".join(display_parts)


def dictation_hotkey_choices() -> list[tuple[str, str]]:
    """The dictation hotkey presets as ``(value, label)`` pairs for a
    dropdown, the platform default's label ending in ``(default)``."""
    # The list is read from config on each call rather than bound at import,
    # so both windows always show what config holds.
    from .. import config

    default = config._default_dictation_hotkey()
    choices = []
    for combo in config.DICTATION_HOTKEY_PRESETS:
        label = format_hotkey_display(combo)
        choices.append((combo, f"{label} (default)" if combo == default else label))
    return choices

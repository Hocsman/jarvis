"""Orb UI for Jarvis.

A frameless, translucent always-on-top window that renders an icosphere
whose colour, intensity and surface motion follow the assistant's
state. Lives in the desktop_app process. Its state comes from its host:
the desktop app hands the floating orb a provider reading the shared
``JarvisState``, and the chat window drives its own through
``set_state``. It has no audio input.

Public surface (lazy)
---------------------
``__getattr__`` defers submodule imports so the state controller can
be used without Qt: touching ``OrbWindow`` or ``OrbWidget`` imports the
Qt stack, touching ``OrbState`` or ``StateController`` does not.
"""

from __future__ import annotations

from typing import Any


__all__ = [
    "OrbWindow",
    "OrbWidget",
    "OrbState",
    "StateController",
]


# Mapping of public name -> (submodule, attribute) so each access
# loads only the file that owns it. Names not in this map raise
# ``AttributeError`` as a normal package would.
_PUBLIC: dict[str, tuple[str, str]] = {
    "OrbWindow":       (".orb_window",       "OrbWindow"),
    "OrbWidget":       (".orb_widget",       "OrbWidget"),
    "OrbState":        (".state_controller", "OrbState"),
    "StateController": (".state_controller", "StateController"),
}


def __getattr__(name: str) -> Any:
    entry = _PUBLIC.get(name)
    if entry is not None:
        import importlib
        mod = importlib.import_module(entry[0], __name__)
        return getattr(mod, entry[1])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

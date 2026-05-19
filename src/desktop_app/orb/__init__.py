"""Reactive orb UI for Jarvis.

A frameless, translucent always-on-top window that renders an icosphere
deformed by 3 audio bands (bass/mid/high) plus state-driven colour and
intensity. Lives in the desktop_app process; reads state from the
shared ``JarvisStateManager`` (face_widget.py) and audio from the
observer hook registered on the daemon's voice listener.

Public surface (lazy)
---------------------
``__getattr__`` defers submodule imports so partial builds (e.g.
running just the audio_bus tests before orb_widget exists) work
cleanly. Touching ``OrbWindow`` triggers the GL stack import; touching
``AudioBus`` does not.

Phase 1 scope
-------------
- Bundled / same-process mode only: audio is tapped directly in
  memory. In subprocess dev mode the audio falls back to a synthetic
  envelope and the window shows a discreet ``DEV`` badge.
- ERROR state is renderable end-to-end (colour, fade, auto-recovery)
  but not yet wired to the daemon's exception path. Phase 2 will add
  the daemon-side trigger.
"""

from __future__ import annotations

from typing import Any


__all__ = [
    "OrbWindow",
    "OrbWidget",
    "OrbState",
    "StateController",
    "AudioBus",
    "FFTAnalyser",
    "register_audio_observer",
    "unregister_audio_observer",
]


# Mapping of public name -> (submodule, attribute) so each access
# loads only the file that owns it. Names not in this map raise
# ``AttributeError`` as a normal package would.
_PUBLIC: dict[str, tuple[str, str]] = {
    "OrbWindow":      (".orb_window",      "OrbWindow"),
    "OrbWidget":      (".orb_widget",      "OrbWidget"),
    "OrbState":       (".state_controller","OrbState"),
    "StateController":(".state_controller","StateController"),
    "AudioBus":       (".audio_bus",       "AudioBus"),
    "FFTAnalyser":    (".audio_bus",       "FFTAnalyser"),
}


def __getattr__(name: str) -> Any:
    entry = _PUBLIC.get(name)
    if entry is not None:
        import importlib
        mod = importlib.import_module(entry[0], __name__)
        return getattr(mod, entry[1])
    if name in ("register_audio_observer", "unregister_audio_observer"):
        from jarvis.listening import listener as _lst
        return getattr(_lst, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""Re-entrancy guard for ``debug_log``'s settings reload.

The settings parser itself logs (a kept auxiliary pin announces itself),
and ``_is_debug_enabled`` reloads the settings. Without a guard, a log
line emitted mid-reload re-enters the reload — hundreds of nested config
parses until the recursion limit, all swallowed by the broad except, so
the failure would be a stall that points nowhere near its cause.
"""

from __future__ import annotations

from types import SimpleNamespace

import src.jarvis.debug as debug


def test_a_debug_call_from_inside_the_settings_reload_does_not_recurse(monkeypatch, capsys):
    reloads = []

    def _reloading_settings():
        reloads.append(1)
        debug.debug_log("parser line", "config")
        return SimpleNamespace(voice_debug=True)

    monkeypatch.setattr(debug, "_cached_voice_debug", None)
    monkeypatch.setattr(debug, "_last_check_time", 0.0)
    monkeypatch.setattr(debug, "load_settings", _reloading_settings)
    monkeypatch.setattr(debug, "_SINK_PATH", "")

    debug.debug_log("outer line", "config")

    assert len(reloads) == 1
    err = capsys.readouterr().err
    assert "outer line" in err

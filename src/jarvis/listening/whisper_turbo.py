"""Whether the installed faster-whisper can load ``large-v3-turbo``.

Shared by the listener, which falls back to ``large-v3`` when it cannot, and
the setup wizard, which then leaves the model out of its options. The wizard
asks on its UI thread, so the answer comes from the package's installed
metadata: importing faster-whisper would load ctranslate2 to read a version
string.
"""

from __future__ import annotations

import importlib.metadata

from ..debug import debug_log


def is_faster_whisper_turbo_supported() -> bool:
    """True when the installed faster-whisper is 1.1.0 or later. A missing
    install or an unreadable version reads as False."""
    try:
        from packaging.version import Version

        # The name stays a literal: PyInstaller ships the metadata of the
        # distributions its analysis sees passed to ``version()``, and the
        # frozen build has no other reason to carry faster-whisper's.
        installed = importlib.metadata.version("faster-whisper")
        return Version(installed) >= Version("1.1.0")
    except Exception as exc:
        debug_log(f"faster-whisper version unreadable, large-v3-turbo off: {exc!r}", "voice")
        return False

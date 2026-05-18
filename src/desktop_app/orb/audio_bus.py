"""Backwards-compat re-export of the audio band bus.

The implementation moved to ``jarvis.utils.audio_bands`` in Phase 2C
so the daemon process can publish band frames over shared memory
without violating the layering rule (daemon must not import
``desktop_app``). All existing imports of
``desktop_app.orb.audio_bus`` keep working through this shim.

New code should import from ``jarvis.utils.audio_bands`` directly.
"""

from __future__ import annotations

from jarvis.utils.audio_bands import (
    AudioBus,
    BAND_BASS_HZ,
    BAND_MID_HZ,
    BandReading,
    DEFAULT_EMA_ALPHA,
    FFTAnalyser,
    FFT_WINDOW,
    INTERNAL_SAMPLE_RATE,
    _RingBuffer,
)

__all__ = [
    "AudioBus",
    "BAND_BASS_HZ",
    "BAND_MID_HZ",
    "BandReading",
    "DEFAULT_EMA_ALPHA",
    "FFTAnalyser",
    "FFT_WINDOW",
    "INTERNAL_SAMPLE_RATE",
    "_RingBuffer",
]

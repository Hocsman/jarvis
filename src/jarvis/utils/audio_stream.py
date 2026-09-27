"""Audio input stream negotiation across voice listener and dictation engine.

Provides format negotiation (mono first, then bounded native-rate / multi-channel
alternatives) for audio hardware that rejects 1-channel mono or 16 kHz capture.
"""

from __future__ import annotations

import contextlib
from typing import Any, Callable, Optional

try:
    import sounddevice as sd
except (ImportError, OSError):
    sd = None

from ..debug import debug_log
from .audio_lock import portaudio_lock


def is_input_format_error(exc: Exception) -> bool:
    """Distinguish unsupported capture formats from access/device failures."""
    code = exc.args[1] if len(exc.args) > 1 else None
    message = str(exc).lower()
    return code in (-9998, -9997) or any(part in message for part in (
        'invalid number of channels', 'invalid channel count',
        'invalid sample rate', 'paerrorcode -9998', 'paerrorcode -9997',
    ))


def open_input_stream(
    sample_rate: int,
    frame_ms: int,
    device_kwargs: dict[str, Any],
    *,
    callback: Optional[Callable] = None,
    serialise: bool = True,
    sd_backend: Optional[Any] = None,
    lock: Optional[Any] = None,
) -> tuple[Any, int, int]:
    """Open mono first, then bounded native-rate/channel alternatives on the same input."""
    active_sd = sd_backend if sd_backend is not None else sd
    if active_sd is None:
        raise OSError("sounddevice not available")

    active_lock = lock if lock is not None else portaudio_lock

    candidates = [(sample_rate, 1)]
    last_error = None
    for rate, channels in candidates:
        try:
            with active_lock if serialise else contextlib.nullcontext():
                stream = active_sd.InputStream(
                    samplerate=rate,
                    channels=channels,
                    dtype='float32',
                    blocksize=max(1, int(rate * frame_ms / 1000)),
                    callback=callback,
                    **device_kwargs,
                )
            debug_log(f"Input format accepted: {rate} Hz, {channels} channel(s)", "voice")
            return stream, rate, channels
        except Exception as exc:
            if not is_input_format_error(exc):
                raise
            last_error = exc
            debug_log(f"Input format rejected: {rate} Hz, {channels} channel(s): {exc}", "voice")
            if len(candidates) == 1:
                try:
                    info = (
                        active_sd.query_devices(device_kwargs['device'])
                        if 'device' in device_kwargs
                        else active_sd.query_devices(kind='input')
                    )
                    native_rate = int(info.get('default_samplerate', sample_rate))
                    max_channels = int(info.get('max_input_channels', 1))
                except Exception:
                    raise exc
                rates = list(dict.fromkeys(r for r in (sample_rate, native_rate) if r > 0))
                counts = list(dict.fromkeys(c for c in (1, 2, max_channels) if 0 < c <= max_channels))
                candidates.extend((r, c) for c in counts for r in rates if (r, c) != candidates[0])
    if last_error is not None:
        raise last_error
    raise OSError("No valid audio input format found")

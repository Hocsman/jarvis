"""Audio input selection and stream negotiation across voice listener and dictation engine.

Resolves the configured ``voice_device`` to one input, re-reads PortAudio's
device list when no stream is open, opens output streams counted as open,
and provides format negotiation (mono first, then bounded native-rate /
multi-channel alternatives) for audio hardware that rejects 1-channel mono
or 16 kHz capture.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Sequence

try:
    import sounddevice as sd
except (ImportError, OSError):
    sd = None

from ..debug import debug_log
from .audio_lock import open_stream_count, portaudio_lock, track_stream, unguarded_open


class InputDeviceNotFound(LookupError):
    """``voice_device`` names a microphone that no input device matches."""


class IndexNamesNoInput(InputDeviceNotFound):
    """``voice_device`` is an index that names no input in the first device
    list of the process: no microphone connected later is the one it means."""


class DeviceListUnavailable(OSError):
    """PortAudio was terminated to read its device list afresh and would not
    initialise again. It stays down, and every stream open fails, until a
    later reading initialises it."""


# The device list PortAudio read first in this process, kept when it is first
# read afresh: an index setting names the device at that position in it.
# Read and written under portaudio_lock.
_first_list: Optional[list] = None


def refresh_input_devices(sd_backend: Optional[Any] = None) -> Optional[list]:
    """PortAudio's device list read afresh, or None while a stream is open.

    PortAudio reads its devices when it initialises, so a microphone connected
    since is found only after it is terminated and initialised again, which
    closes every stream in the process. That happens here under
    ``portaudio_lock`` and only while no stream is open
    (``audio_lock.open_stream_count``); otherwise the answer is None and the
    caller asks again later. A PortAudio that will not initialise again, tried
    twice, raises ``DeviceListUnavailable``.
    """
    global _first_list
    active_sd = sd_backend if sd_backend is not None else sd
    if active_sd is None:
        raise DeviceListUnavailable("sounddevice is not available")
    with portaudio_lock:
        if open_stream_count():
            return None
        try:
            if _first_list is None:
                _first_list = list(active_sd.query_devices())
            # sounddevice has no public way to read the list afresh:
            # `_terminate`, `_initialize` and `_initialized` are its internals,
            # as pinned in requirements.txt (sounddevice==0.4.7). It counts
            # nested initialisations in `_initialized`, and PortAudio drops
            # its device list only once all of them are terminated, so
            # `_terminate` runs that many times before one `_initialize`.
            # The fallback count of one is for stand-ins without the counter.
            initialised = getattr(active_sd, "_initialized", 1)
            for _ in range(initialised if isinstance(initialised, int) else 1):
                active_sd._terminate()
            try:
                active_sd._initialize()
            except Exception as first_failure:
                active_sd._initialize()
                debug_log(f"PortAudio initialised on its second try: {first_failure}", "voice")
            return list(active_sd.query_devices())
        except Exception as exc:
            raise DeviceListUnavailable(f"PortAudio could not read its devices afresh: {exc}") from exc


def resolve_input_device(
    voice_device: Optional[str],
    list_devices: Callable[[], Sequence[Mapping[str, Any]]],
) -> Optional[int]:
    """The PortAudio input index ``voice_device`` selects, or None for the system default.

    Blank, ``default`` and ``system`` (any case) mean the system default input.
    Anything ``int()`` reads is an index: the input at that position in the
    first device list of the process. Once the list has been read afresh,
    the index selects that device by its exact name, never whatever sits at
    its position now. Any other text is a name: the first input-capable
    device whose name contains it, ignoring case. ``list_devices`` is not
    called for the system default. An index or a name that selects no input
    raises ``InputDeviceNotFound`` rather than falling back to another
    microphone, ``IndexNamesNoInput`` for an index that named none in the
    first list; the message leaves the value out, since a device name can
    carry a person's.
    """
    wanted = (voice_device or '').strip().lower()
    if wanted in ('', 'default', 'system'):
        return None
    devices = list_devices()
    try:
        index = int(wanted)
    except ValueError:
        for idx, dev in enumerate(devices):
            if dev.get('max_input_channels', 0) > 0 and wanted in dev.get('name', '').lower():
                return idx
        raise InputDeviceNotFound("voice_device selects no input device")
    return _device_at_first_index(index, devices)


def _device_at_first_index(index: int, devices: Sequence[Mapping[str, Any]]) -> int:
    """Where the input at ``index`` in the first device list sits in ``devices``."""
    first = _first_list if _first_list is not None else devices
    if not (0 <= index < len(first) and first[index].get('max_input_channels', 0) > 0):
        raise IndexNamesNoInput("voice_device is an index that names no input device")
    if _first_list is None:
        return index
    name = first[index].get('name')
    for idx, dev in enumerate(devices):
        if dev.get('max_input_channels', 0) > 0 and dev.get('name') == name:
            return idx
    raise InputDeviceNotFound("the input voice_device names is not in the device list")


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
    """Open mono first, then bounded native-rate/channel alternatives on the same input.

    Each stream opened is tracked as open (``audio_lock.track_stream``) until
    it is closed. ``serialise=False`` opens without ``portaudio_lock`` for the
    Windows permission probe, inside ``unguarded_open``.
    """
    active_sd = sd_backend if sd_backend is not None else sd
    if active_sd is None:
        raise OSError("sounddevice not available")

    active_lock = lock if lock is not None else portaudio_lock

    candidates = [(sample_rate, 1)]
    last_error = None
    for rate, channels in candidates:
        try:
            with active_lock if serialise else unguarded_open():
                stream = active_sd.InputStream(
                    samplerate=rate,
                    channels=channels,
                    dtype='float32',
                    blocksize=max(1, int(rate * frame_ms / 1000)),
                    callback=callback,
                    **device_kwargs,
                )
                track_stream(stream)
            debug_log(f"Input format accepted: {rate} Hz, {channels} channel(s)", "voice")
            return stream, rate, channels
        except Exception as exc:
            if not is_input_format_error(exc):
                raise
            last_error = exc
            debug_log(f"Input format rejected: {rate} Hz, {channels} channel(s): {exc}", "voice")
            if len(candidates) == 1:
                try:
                    with active_lock:
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


def open_output_stream(sd_backend: Any, **kwargs: Any) -> Any:
    """``sd_backend.OutputStream(**kwargs)``, counted as open from its
    construction until it is closed (``audio_lock.track_stream``).

    Both happen in one hold of ``portaudio_lock``, so no reading of the device
    list can fall between them and terminate PortAudio under the new stream.
    The caller may already hold the lock, to start the stream in the same hold.
    """
    with portaudio_lock:
        stream = sd_backend.OutputStream(**kwargs)
        track_stream(stream)
    return stream

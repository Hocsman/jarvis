"""Process-wide serialisation of PortAudio stream lifecycle calls.

PortAudio explicitly documents opening and closing streams as not thread-safe
(https://github.com/PortAudio/portaudio/wiki/Tips_Threading), and its Windows host
APIs abort the whole process on internal assertion failures.
Jarvis coordinates PortAudio across independent threads: the voice listener run
loop, the Windows microphone permission check thread, the dictation engine worker
threads, the TTS playback thread, and the thinking tune player.

Every stream lifecycle call (constructing an InputStream/OutputStream, calling
start/stop/close/abort, and blocking sd.play which opens a stream internally)
runs inside portaudio_lock. The lock is not held across playback waits or other
blocking work, and no other lock should be acquired while holding it: acquire it
innermost, around the direct sounddevice call only.

Accepted exceptions:
- The dictation beep holds the lock across its blocking sd.play (sub-200ms, and
  splitting play and wait would leave the internal stream teardown unguarded).
- The Windows microphone permission probe opens and starts its stream without
  the lock: that open can hang indefinitely when Windows blocks microphone
  access, and hanging while holding this lock would freeze every audio user in
  the process. It runs before capture first opens, and again mid-session once a
  microphone that was waited for appears; mid-session the listener runs it
  only at a moment when no stream is counted open, so it does not overlap her
  voice, the tune or a dictation, though one of those may still start while it
  runs. It resolves the microphone inside ``unguarded_open`` (below), so no
  re-read moves the list between resolving and opening.

Re-reading the device list
--------------------------
PortAudio reads its device list when it initialises, and only then: a microphone
connected later is seen once PortAudio is terminated and initialised again
(``jarvis.utils.audio_stream.refresh_input_devices``). Terminating closes every
stream in the process, and a stream object used after that is a native
use-after-free, so the list is re-read under this lock and only while no stream
is open:

- every stream is handed to ``track_stream`` in the same hold of the lock as its
  construction (``audio_stream.open_input_stream`` and ``open_output_stream``
  do both), and counts as open until its ``closed`` reads True, however it was
  closed; the dictation beep's ``sd.play`` is not tracked, since it opens and
  closes inside one hold;
- an open made without the lock is wrapped in ``unguarded_open``, which takes
  the lock to begin, so it cannot start while a re-read runs and no re-read
  starts until it is over;
- a stream whose state cannot be read counts as open: a mistake keeps the old
  list, it never closes a stream under its owner;
- a device query that may run while another thread re-reads the list takes the
  lock, and a device index is only used against the list it was read from, so
  resolving a microphone and opening it happen in one hold of the lock.
"""

import threading
from contextlib import contextmanager

# Re-entrant so a guarded close inside a guarded open-fallback path is safe.
portaudio_lock = threading.RLock()

# Every stream constructed and not yet seen closed, and how many opens are
# running without the lock. Both are read and written under portaudio_lock.
_streams: list = []
_unguarded_opens = 0


def _is_closed(stream) -> bool:
    try:
        return stream.closed is True
    except Exception:
        return False


def track_stream(stream) -> None:
    """Count ``stream`` as open until it is closed.

    Called right after the stream's construction succeeds, in the same hold
    of ``portaudio_lock`` (or inside ``unguarded_open``), by the openers in
    ``jarvis.utils.audio_stream``; streams are opened through those.
    """
    with portaudio_lock:
        _streams[:] = [s for s in _streams if not _is_closed(s)]
        _streams.append(stream)


@contextmanager
def unguarded_open():
    """Mark an open that runs without ``portaudio_lock`` for its duration.

    Entering waits for a device-list re-read in progress; while inside, none
    starts. Track the stream before leaving, so it stays counted once open.
    """
    global _unguarded_opens
    with portaudio_lock:
        _unguarded_opens += 1
    try:
        yield
    finally:
        with portaudio_lock:
            _unguarded_opens -= 1


def open_stream_count() -> int:
    """Streams that may still be open: those tracked and not seen closed, and
    opens running without the lock."""
    with portaudio_lock:
        _streams[:] = [s for s in _streams if not _is_closed(s)]
        return len(_streams) + _unguarded_opens

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
- The Windows microphone permission probe opens its stream without the lock: that
  open can hang indefinitely when Windows blocks microphone access, and hanging
  while holding this lock would freeze every audio user in the process.
"""

import threading

# Re-entrant so a guarded close inside a guarded open-fallback path is safe.
portaudio_lock = threading.RLock()

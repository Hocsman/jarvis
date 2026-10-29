# Talking Face

## Why

She can speak through a face: a portrait the user chooses, animated so that
its lips follow her words closely enough to lip-read. A neural model (Ditto)
draws each frame from her own audio a fraction of a second before that audio
is heard, and the frame is stamped with the moment the sound leaves the
speakers, so a front end that shows frames on time shows lips and voice
together.

This file covers the renderer: the process that draws the face, what it is
fed, what it publishes, the model files it needs, and how it fails. Kyutai
feeds it (`output/tts.spec.md`, "Kyutai") and a front end shows it; each of
them states its side in its own spec.

## Privacy

- The portrait is the file named to the renderer when it starts, in its
  environment, which other users of the machine cannot read, rather than on its
  command line, which they can. The renderer reads it on this machine. It is not
  copied, not cached and not sent anywhere.
- Her audio reaches the renderer through a pipe between two processes of this
  machine. It is not written to disk and not sent anywhere.
- Frames live in shared memory and nowhere else. No frame is written to disk.
- The model files are fetched once, from the Hugging Face repository
  `digital-avatar/ditto-talkinghead`, when the renderer starts and finds them
  missing. The request names files and carries nothing of the user's: no
  Hugging Face token is sent, even when one is saved on the machine.

## Off unless asked for

`face_enabled` is false by default. The daemon starts a renderer when it
starts, and only when `face_enabled` is true, `face_portrait_path` names a
file and the speech engine is Kyutai, the one engine that feeds a face. It
hands the renderer to the engine and stops it when it shuts down. Otherwise
no renderer is started, nothing is fetched, and the GPU does no work for a
face.

## Hardware and installation

The renderer needs an NVIDIA GPU with CUDA, PyTorch built with CUDA, TensorRT
(`tensorrt-cu12`), OpenCV (`opencv-python-headless`) and `einops`. None of the
last three is in `requirements.txt`: like `moshi`, which brings `einops` with
it, they are installed by hand, OpenCV in a 4.x release because
`requirements.txt` keeps numpy below 2. Without any of them the renderer
reports why it cannot start and stops; nothing else changes. It runs from a
source install only: the packaged app carries no TensorRT, and `FaceRenderer`
says so instead of starting a child. Everything around the model (the
timeline, the shared memory, the handle) needs only numpy.

## The process

`FaceRenderer` (`client.py`) is the handle on the renderer. It starts the
renderer as a child process, `python -m jarvis.output.face.process`, with the
interpreter of the process that holds it, and passes it the framing on its
command line and the portrait in its environment (`JARVIS_FACE_PORTRAIT`). The model, TensorRT and the GPU memory they take all live in that
child, so a renderer that crashes, hangs or runs out of memory costs the face
and never the voice.

- **To the renderer**, through its stdin, go three kinds of message: audio
  (float32 samples, their sample rate, and the time the first of them will be
  heard), a cut (see Interrupting) and quit.
- **From the renderer**, on its stdout, comes one JSON object per line:
  `ready` with the renderer's delay once it can draw, `watched` whenever
  someone starts or stops watching the face, or `failed` with a reason.
  Everything else it prints goes to its stderr, which the handle logs.
- **`FaceRenderer` never raises to its caller and never blocks it.** Sending
  is a write to a pipe from a worker thread fed by an unbounded queue, so a
  renderer that stops reading costs memory, not time, until it is declared
  dead; what cannot be sent (a time or a rate of the wrong kind) is dropped
  and logged. A watchdog declares dead a child that has not taken a message
  for five seconds, and kills it. A failure to start, a `failed` line, a child
  that exits, a pipe that breaks and a child that stops reading each turn the
  face off for the rest of the session; the reason is logged once. `ready()`
  is true only between a `ready` line and the first of those, and only
  reports: it changes nothing. Nothing is sent before `ready` or after a
  failure.
- **The renderer stops when the daemon does.** The end of its stdin, a
  message cut short and a message it cannot read each end it like quit, so a
  renderer is never left running without the daemon that started it.

## The clock

Every time on this channel is `time.perf_counter()`. On the platforms the
renderer runs on (Windows and Linux) it reads one clock for every process of
the machine, so a time stamped in the daemon means the same instant in the
renderer and in a front end.

**Heard at.** Audio is handed over with the time its first sample will leave
the speakers, and the renderer lays it on its timeline at that time. Audio is
mono at any sample rate; the renderer resamples it to the 16 kHz Ditto reads.
Where two pieces of audio overlap on the timeline, the later one wins.

**Delay.** The renderer's delay is 0.75 s, and it is the contract between the
renderer and whoever feeds it: audio that will be heard at time H must be
handed over by H minus the delay, and the frame drawn from it is then
published before H. Fed as described under Chunks, Ditto published every
frame at most 0.66 s after its stamp minus the delay, so at least 0.09 s
before its stamp (measured next to Kyutai and Whisper on an RTX 5070 Ti); that
0.09 s is the margin. Audio handed over later is still laid on the timeline,
but the frames drawn from it may come after their moment, and nothing waits
for them.

**Chunks.** The renderer feeds Ditto the timeline 200 ms (five frames) at a
time. Each chunk carries 120 ms of audio before its five frames and 85 ms
after them, which is what Ditto's streaming audio encoder reads, and it is fed
the moment the clock reaches the end of that window minus the delay. A chunk
is never fed early: the audio after it might not have arrived yet. One
sampling of the motion model yields one chunk's frames. `timeline.py` holds
this geometry and Ditto's stream takes it from there.

**Stamps.** Every frame is stamped with the time its moment of audio is heard:
frame n of a chunk whose five frames start at time S is stamped S + n / 25.
A front end shows, at any instant, the newest frame whose stamp has passed,
and never one whose stamp has not.

## Watching and resting

The face costs the GPU nothing while no one can see it.

- A front end that shows the face writes a heartbeat (the current time) into
  the shared memory every time it reads.
- While the latest heartbeat is less than two seconds old, the renderer feeds
  Ditto continuously, silence included, so the face keeps blinking and
  moving between sentences.
- Otherwise it rests, whatever audio it is handed: no chunk, no frame, no GPU
  work. A viewer who arrives while she speaks sees her from the next chunk
  that can still be drawn on time.
- Frames still inside Ditto when it rests are never published.
- The renderer tells its handle each time that changes (`watched`), and
  Kyutai holds her voice back for the face only while someone watches:
  with nobody watching she speaks at once.

## Interrupting

`cut(at)` means she stops talking at `at`. The renderer drops every sample
heard at or after `at`, so what it draws from then on is silence: the mouth
closes. Frames drawn before the cut but stamped after it show words she will
not say. The cut withdraws those already published, and the renderer refuses
to publish those still being drawn, whichever cut since they were drawn
covers them: a second cut never brings back what an earlier one took. The
face holds its last frame until the first frame drawn after the cut is due,
at most one delay later.

## Shared memory

`frames.py` owns one shared memory segment per user, created by the renderer
and named after the user so that two users of one machine never share a face.
It holds a header and a ring of 24 frames of 512 by 512 RGB pixels.

- **The header** holds a tag and the format version, the frame size, the
  number of slots, the sequence number of the last frame published, the
  viewer heartbeat and the renderer heartbeat.
- **A slot** holds a sequence number, a stamp and the pixels. The renderer
  writes the pixels and the stamp first and the sequence number last, and a
  cut withdraws a slot by clearing its sequence number. A reader copies a slot
  and keeps the copy only if the sequence number was the same before and
  after, so a copy torn by a concurrent write is thrown away, never shown.
- **The renderer beats** into the header on every turn of its loop, resting
  included. A reader that finds the beat more than two seconds old shows
  nothing and lets go of the segment, so a renderer started after it is found
  again; a new renderer reuses a segment a front end still holds.
- **Only its user opens it.** On Linux the segments of every user live in one
  directory, so a segment that does not belong to the user, or that others may
  open, is used by neither side. Windows keeps such names per session.
- **Readers never hold the renderer up**, and the renderer never waits for a
  reader. Twenty-four slots hold about one second of frames, more than the
  delay keeps ahead of the clock.
- **The reader is in `jarvis`.** `read_face_frame()` returns the frame to show
  now, or None when there is no renderer or nothing is due. A front end
  imports it from `jarvis`, as it does the assistant's state
  (`state.spec.md`): nothing on the renderer's side knows a front end exists.

## The model

Ditto (`antgroup/ditto-talkinghead`, Apache-2.0) lives under `ditto/` with its
licence. Only the code the streaming renderer runs is kept, and every file
changed from the original says so at its top.

- **Its configuration is a pickle**, and a pickle can run code. It is read
  with an unpickler that admits numpy arrays and nothing else, and any other
  object makes the load fail. The model paths it names are resolved inside the
  model folder, never against the working directory.
- **The portrait is measured once**, when the renderer starts: face detection
  and landmarks (ONNX Runtime, on the CPU), then the appearance and motion
  features Ditto animates. The openness of the eyes and the direction of the
  gaze, which the motion model is conditioned on, are measured from Ditto's
  203-point landmarks. The models that measure the portrait are released once
  it is measured.
- **Framing.** Ditto animates a square around the face. The renderer publishes
  that square as it is, so everything on screen moves with the head, and never
  pastes it back into the still portrait, where hair and background outside the
  square would stay put. The framing picks the size of the square: `head` (the
  default), 2.8 times the size of the face, shows the whole head; `face`, 2.3
  times, frames it closer. Before measuring, the renderer widens the portrait by
  stretching and blurring its own edges, so that the square never runs off the
  picture.
- **Speed.** Ditto runs in its streaming mode, with ten sampling steps and five
  new frames per step (an overlap of 75). The motion model is replayed from a
  CUDA graph in half precision. The warping network and the decoder, which do
  most of the work, run as TensorRT engines in half precision. The audio
  encoder runs on the CPU, which spares the GPU for the voice and Whisper, on
  four threads that do not spin while they wait.
  Together they keep up with real time next to Kyutai and Whisper on an RTX
  5070 Ti.

## Model files and engines

- **Weights** live in `~/.local/share/jarvis/models/ditto/`: the files listed in
  `weights.py`, about 2.2 GB, from one pinned revision of the repository. A
  file is downloaded into a staging folder, checked there against its
  SHA-256, and only then moved into place, so a file at its final path has
  always matched; a file that does not match is deleted, never loaded.
- **TensorRT engines** are built from those weights on the first start on a
  given GPU: the two networks are exported to ONNX, then compiled, which took
  thirty seconds on an RTX 5070 Ti and keeps the GPU busy meanwhile. They are
  kept under `engines/<key>/`, where the key names the GPU, the TensorRT
  version and the weights by their published SHA-256. An engine whose key does
  not match is built again, never loaded: an engine built elsewhere can crash
  the process that loads it.
- **Warm-up.** Ditto's first seconds run slowly while kernels load and memory
  is laid out, so the renderer feeds three seconds of silence nobody sees, then
  waits for what is still inside Ditto to come out, before it reports `ready`.

## Failure

Initialisation is attempted once per session. A missing package, no GPU, a
portrait with no face in it, a download that fails or an engine that cannot be
built each end it with a `failed` line that says which, and the renderer exits.
A renderer that dies while drawing ends the same way. In every case the face
is off until the daemon starts again, and nothing she says depends on it.

A frame that comes after its stamp is not a failure: a front end shows the
newest frame due and skips the rest. When Ditto falls behind, so that a frame
it still owes should already be on screen, the renderer skips the chunk that
falls due rather than queue it behind, so the face catches up with the voice
instead of trailing it. It counts the frames it published after their stamp
and the chunks it skipped, and logs both when it rests.

## Configuration

| Key | Default | Meaning |
|-----|---------|---------|
| `face_enabled` | `false` | Start the renderer with the daemon (Kyutai only) |
| `face_portrait_path` | `""` | The portrait to animate: a still image of one face seen from the front |
| `face_framing` | `head` | `head` shows the whole head, `face` frames the face closer |

## Testing

The suite loads neither ONNX Runtime nor the GPU (`tests/native_guard.py`), so
it holds everything around the model to the contract, and the renderer itself
is checked by hand.

- `tests/test_face_timeline.py`: audio laid on the timeline at its time,
  resampled as scipy would, joined across small pieces, overlapping, cut and
  forgotten; the chunk windows, when each is due, the stamps, feeding while
  watched, resting otherwise even while she speaks, a viewer arriving
  mid-sentence, a feeder that fell behind.
- `tests/test_face_frames.py`: the shared memory between two processes, the
  newest frame due, nothing early, torn copies thrown away, cuts withdrawing
  and refusing frames (two cuts included), the heartbeats, a reader that
  outlives its renderer and finds the next one.
- `tests/test_face_process.py`: the renderer with Ditto faked; messages read,
  cut and the end of the daemon, frames published at their stamps, warm-up
  frames and frames left inside Ditto at a rest never shown, frames after a
  cut refused, late frames counted, a chunk skipped while Ditto owes a frame already due, nothing
  fed while no one watches, `watched` reported as it changes, `failed` with
  its reason.
- `tests/test_face_client.py`: the renderer process faked; `ready`,
  `watched` and `failed`, a child that dies, cannot start or ignores quit, the packaged app
  refused, the portrait in the environment and not the command line, messages
  in order, what cannot be sent dropped, a line that is not an event, a caller
  never blocked by a child that stops reading.
- `tests/test_face_pickle.py`: the configuration unpickler admits numpy arrays
  and refuses anything else without running it.
- `tests/test_face_files.py`: the portrait widened around the original pixels
  and refused when unreadable (the image cases need OpenCV and are skipped
  without it); weights fetched only when missing, without a token, and never
  put in place when their digest does not match; an engine key that changes
  with the GPU, TensorRT and the weights; engines built once, then found.
- `tests/test_face_daemon.py`: no renderer unless the face is enabled, a
  portrait is named and the voice is Kyutai; the renderer started with its
  portrait and framing and handed to the voice; the settings' defaults.
- `tests/test_kyutai_face.py`: Kyutai's side (`output/tts.spec.md`).
- `scripts/check_talking_face.py`, by hand on a machine with the GPU, the
  packages and the weights: the real renderer fed a WAV the way a speech engine
  feeds it, read the way a front end reads it, and how late each frame was.

# Talking Face

## Why

She can speak through a face: a portrait the user chooses, animated so that
its lips follow her words closely enough to lip-read. A neural model (Ditto)
draws each frame from her own audio a fraction of a second before that audio
is heard, and the frame is stamped with the moment the sound leaves the
speakers, so a front end that shows frames on time shows lips and voice
together.

This file covers the renderer: the process that draws the face, what it is
fed, what it publishes, the model files it needs, and how it fails. A speech
engine feeds it and a front end shows it; each of them states its side in its
own spec.

## Privacy

- The portrait is the file named to the renderer when it starts. The renderer
  reads it on this machine. It is not copied, not cached and not sent anywhere.
- Her audio reaches the renderer through a pipe between two processes of this
  machine. It is not written to disk and not sent anywhere.
- Frames live in shared memory and nowhere else. No frame is written to disk.
- The model files are fetched once, from the Hugging Face repository
  `digital-avatar/ditto-talkinghead`, when the renderer starts and finds them
  missing. The request names files and carries nothing of the user's.

## Hardware and installation

The renderer needs an NVIDIA GPU with CUDA, PyTorch built with CUDA, TensorRT
(`tensorrt-cu12`) and OpenCV (`opencv-python-headless`). TensorRT and OpenCV
are not in `requirements.txt`: like `moshi`, they are installed by hand,
OpenCV in a 4.x release because `requirements.txt` keeps numpy below 2. Without any of them the renderer reports why it cannot start
and stops; nothing else changes. It runs from a source install only: the
packaged app carries no TensorRT, and `FaceRenderer` says so instead of
starting a child.

## The process

`FaceRenderer` (`client.py`) is the handle on the renderer. It starts the
renderer as a child process, `python -m jarvis.output.face.process`, with the
interpreter of the process that holds it, and passes it the portrait and the
framing. The model, TensorRT and the GPU memory they take all live in that
child, so a renderer that crashes, hangs or runs out of memory costs the face
and never the voice.

- **To the renderer**, through its stdin, go three kinds of message: audio
  (float32 samples, their sample rate, and the time the first of them will be
  heard), a cut (see Interrupting) and quit.
- **From the renderer**, on its stdout, comes one JSON object per line:
  `ready` with the renderer's delay once it can draw, or `failed` with a
  reason. Everything else it prints goes to its stderr, which the handle
  logs.
- **`FaceRenderer` never raises to its caller and never blocks it.** Sending
  is a write to a pipe from a worker thread fed by an unbounded queue, so a
  renderer that stops reading costs memory, not time, until it is declared
  dead. A child that has not read for five seconds is dead, and is killed. A
  failure to start, a `failed` line, a child that exits, a pipe that breaks
  and a child that stops reading each turn the face off for the rest of the
  session; the reason is logged once. `ready()` is true only between a `ready`
  line and the first of those.

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
is never fed early: the audio after it might not have arrived yet.

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
- Otherwise it feeds only the stretch of timeline that holds audio, and rests
  in between: no chunk, no frame, no GPU work.
- Frames still inside Ditto when it rests come out when feeding resumes. They
  keep the stamps they were given, so they are already past, and a front end
  never shows them.

## Interrupting

`cut(at)` means she stops talking at `at`. The renderer drops every sample
heard at or after `at`, so what it draws from then on is silence: the mouth
closes. Frames drawn before the cut but stamped after it show words she will
not say, so the cut also raises the generation in the shared memory, and a
front end does not show a frame stamped after the cut that was drawn in an
earlier generation. The face holds its last frame until the first frame drawn
after the cut is due, at most one delay later.

## Shared memory

`frames.py` owns one shared memory segment per user, created by the renderer
and named after the user so that two users of one machine never share a face.
It holds a header and a ring of 24 frames of 512 by 512 RGB pixels.

- **The header** holds the format version, the frame size, the number of
  slots, the sequence number of the last frame published, the viewer
  heartbeat, the current generation and the time of the last cut.
- **A slot** holds a sequence number, a stamp, a generation and the pixels.
  The renderer writes the pixels and the stamp first and the sequence number
  last. A reader copies a slot and keeps the copy only if the sequence number
  was the same before and after, so a copy torn by a concurrent write is
  thrown away, never shown.
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
  object makes the load fail.
- **The portrait is measured once**, when the renderer starts: face detection
  and landmarks (ONNX Runtime, on the CPU), then the appearance and motion
  features Ditto animates. The openness of the eyes and the direction of the
  gaze, which the motion model is conditioned on, are measured from Ditto's
  203-point landmarks.
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
  encoder runs on the CPU, which spares the GPU for the voice and Whisper.
  Together they keep up with real time next to Kyutai and Whisper on an RTX
  5070 Ti.

## Model files and engines

- **Weights** live in `~/.local/share/jarvis/models/ditto/`: the files listed in
  `weights.py`, about 2.2 GB, from one pinned revision of the repository. A
  file is downloaded under a temporary name and renamed once complete, then
  checked against its SHA-256; a file that does not match is deleted, never
  loaded.
- **TensorRT engines** are built from those weights on the first start on a
  given GPU: the two networks are exported to ONNX, then compiled, which takes
  about two minutes. They are kept under `engines/<key>/`, where the key names
  the GPU, the TensorRT version and the weights. An engine whose key does not
  match is built again, never loaded: an engine built elsewhere can crash the
  process that loads it.
- **Warm-up.** Ditto's first seconds run slowly while kernels load and memory
  is laid out, so the renderer feeds three seconds of silence before it
  reports `ready`. Feeding the face only starts then.

## Failure

Initialisation is attempted once per session. A missing package, no GPU, a
portrait with no face in it, a download that fails or an engine that cannot be
built each end it with a `failed` line that says which, and the renderer exits.
A renderer that dies while drawing ends the same way. In every case the face
is off until the daemon starts again, and nothing she says depends on it.

A frame that comes after its stamp is not a failure: a front end shows the
newest frame due and skips the rest. The renderer counts the frames it
published after their stamp and logs the count when it rests.

## Testing

The suite loads neither ONNX Runtime nor the GPU (`tests/native_guard.py`), so
it holds everything around the model to the contract, and the renderer itself
is checked by hand.

- `tests/test_face_timeline.py`: audio laid on the timeline at its time,
  resampled, joined across small pieces, overlapping, cut and forgotten; the
  chunk windows, when each is due, the stamps, feeding while watched, waking
  for audio and resting after it.
- `tests/test_face_frames.py`: the shared memory between two processes, the
  newest frame due, nothing early, torn copies thrown away, the generation and
  the cut, the heartbeat, a renderer that stopped beating.
- `tests/test_face_client.py`: the renderer process faked; `ready` and
  `failed`, a child that dies or cannot start, messages in order, a caller
  never blocked by a child that stops reading.
- `tests/test_face_pickle.py`: the configuration unpickler admits numpy arrays
  and refuses anything else without running it.
- `tests/test_face_files.py`: the portrait widened around the original pixels
  and refused when unreadable; weights fetched only when missing and deleted
  when their digest does not match; an engine key that changes with the GPU,
  TensorRT and the weights.
- `scripts/check_talking_face.py`, by hand on a machine with the GPU, the
  packages and the weights: the real renderer fed a WAV the way a speech engine
  feeds it, read the way a front end reads it, and how late each frame was.

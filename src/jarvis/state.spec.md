# Assistant State Specification

What the assistant is doing right now, as one value the whole machine can read. Implemented in `src/jarvis/state.py`.

## Ownership

The state belongs to the core. The voice pipeline publishes it as it works, and whatever shows the assistant to the user (the orb, the dashboard, the face) reads it. It is defined in `jarvis`, and a front end imports it from `jarvis`; `jarvis` imports no front end (`desktop_app.spec.md`).

That direction is checked, not just stated: `tests/test_jarvis_does_not_import_desktop_app.py` parses every module under `src/jarvis` and fails on any import of `desktop_app`, including one inside a function or a `try` block. Nothing the pipeline needs to tell a front end may live on the front end's side of that line.

Publishing needs the standard library and debug logging only. No Qt and no desktop package is loaded to publish, so a headless run publishes exactly as a desktop run does, and nothing reads it.

## States

`JarvisState` is a closed enum. Each value is a plain string, which is what crosses processes.

| Value | Meaning | Published by |
|-------|---------|--------------|
| `ASLEEP` | The daemon is not running or not ready | The holder when it is created; the desktop app while the daemon is down or a setup wizard is open |
| `IDLE` | Awake, waiting for the wake word | The listener when its microphone first starts capturing, when her speech ends while the microphone is waited for (below), when a hot window ends and when a stop word silences her, and again when the engine is done with the reply that stop silenced, unless a question he started since owns the state; the reply engine when the stop tool dismisses the conversation; the listener whenever it stops the thinking tune (a reply is about to be spoken, or there is none) and when a question ends with nothing in it (below); the daemon when a dictation ends |
| `LISTENING` | Collecting a query, or a hot window is open | The listener on hearing the wake word or a hot-window follow-up; the listening state manager when a collection starts and when a hot window opens |
| `THINKING` | A query is with the reply engine | The listener's reply thread when it hands a query to the engine; the listener when a question ends with nothing in it while a reply is on its way (below) |
| `SPEAKING` | A speech engine is speaking | Each speech engine (Piper, Kokoro, Chatterbox, Kyutai) when speech starts; the listener when a question ends with nothing in it while she speaks (below) |
| `DICTATING` | Hold-to-dictate recording is active | The daemon's dictation-start callback |
| `DICTATION_PROCESSING` | Captured dictation is being transcribed and pasted | The daemon's dictation-processing callback |

The end of speech publishes nothing. What follows it (a hot window, so `LISTENING`, or `IDLE`) is the daemon's call, and an engine that stops must not guess.

While the microphone is waited for (`listening.spec.md`, Capture format and health), nothing hears a follow-up, so no hot window opens: when her speech ends the listener publishes `IDLE`, unless a dictation owns the state. Waiting for a microphone publishes nothing of its own. At start-up `ASLEEP` stands until the first capture, except that a reminder or a report she speaks meanwhile publishes `SPEAKING`, then `IDLE` when it ends, and a dictation ends on `IDLE`. A microphone lost mid-session leaves the state to the same publishers: a question being collected or a hot window open at the loss still ends on time, on what it publishes when it ends.

A question that ends with nothing in it ends on what she is doing. Her name alone opens a question with nothing in it yet, and `LISTENING` is published while she waits for it (`listening.spec.md`, Wake Word Mode). When his silence runs out with nothing collected, the listener publishes `SPEAKING` while she speaks, `THINKING` while a reply of hers is on its way, and `IDLE` otherwise. A reply stopped since it was handed over is not on its way, though the engine may still be finishing it. While he dictates the listener publishes nothing: `DICTATING` or `DICTATION_PROCESSING` stands until the daemon ends the dictation.

## Holder

`get_jarvis_state()` returns the process-wide `JarvisStateManager`. Its `state` property reads, `set_state(state)` publishes.

- **A file carries the state across processes.** The daemon runs as a subprocess in a development checkout and as a thread inside the bundled app; a file named `jarvis_state` in the system temp directory reads the same in both. It holds one enum value and nothing else.
- **Readers read the file every time**, so a reader in one process sees what a writer in another published. The in-memory value is the fallback when the file is missing, unreadable or holds a value no release defines, and the only copy on a machine with no usable temp directory.
- **A new holder always starts `ASLEEP`** and writes that to the file. The file carries state across processes during a session, not across launches, so a value left by a previous session never shows.
- **Publishing never raises.** A file that cannot be written costs the viewer its view of the state, never the user their answer, so publishers call `set_state` bare and do not wrap it. The failure is logged through `debug_log`.
- **Publishes are ordered within a process.** Timer threads, the reply thread and the speech threads publish concurrently. The memory value and the file are updated as one step, so publishes land in the order they are admitted and both end up holding the last one, and a publish held up on the disk cannot land over a later one. Readers are not held up by a publish in progress. The order is per publish call: whether a state is still true when its publisher gets its turn is the publisher's check to make. Two processes publishing to the one file are not ordered against each other.

## Readers

The desktop app reads the state and does not subscribe to it: it is published from daemon threads, and from another process in a development checkout, so polling the holder is the one mechanism that works in every layout.

- The floating orb's `StateController` polls a provider each frame (`desktop_app.spec.md`, Orb).
- The dashboard bridge polls at 5 Hz and maps the value to the HUD orb's accent and label.
- The face polls each frame.

The desktop app also publishes one value, `ASLEEP`, when the daemon stops and while the setup wizard is open, so the face and the orb do not look ready while nothing is listening.

## Tests

- `tests/test_jarvis_state.py`: the holder's contract (starts asleep, shared through the file, falls back, never raises, publishes stay ordered under concurrency) and the headless import guarantee, checked in a clean interpreter.
- `tests/test_the_pipeline_publishes_its_state.py` and `TestTheListenerPublishesItsState` in `tests/test_hot_window_input.py`: each stage publishes the state it enters, read back through the same reader a viewer uses.
- `TestHerNameAloneThenSilence` in `tests/test_hot_window_input.py`, and the tests on her name alone and on what follows a stop in `tests/test_elle_entend_pendant_qu_elle_reflechit.py`: a question that ends with nothing in it ends on what she is doing, a reply that was stopped keeps neither the tune nor `THINKING`, and its end leaves the tune and the state of a question asked since alone.
- `tests/conftest.py` gives every test its own state file, and `evals/conftest.py` every eval, so a run never moves the orb of an app the developer has open; `evals/test_evals_keep_the_core_out_of_the_repository.py` checks it for the evals.

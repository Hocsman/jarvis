# Dictation Engine Specification

## Overview

WisprFlow-like dictation: hold a hotkey to record speech, release to type the
transcription into the focused application. Completely independent from the
assistant pipeline (no wake words, intent judge, profiles, or TTS).

## Configuration

| Key                            | Type   | Default (per-platform)                         | Description                                     |
|--------------------------------|--------|------------------------------------------------|-------------------------------------------------|
| `dictation_enabled`           | bool   | `true`                                         | Master switch for the feature                   |
| `dictation_hotkey`            | string | Win: `"ctrl+cmd"`, macOS/Linux: `"ctrl+alt"`   | Hold-to-record hotkey combination               |
| `dictation_filler_removal`    | bool   | `false`                                        | LLM filler word removal, always on the machine  |
| `dictation_custom_dictionary` | list   | `[]`                                           | Custom replacements in `"wrong -> right"` format|

Defaults are aligned with WisprFlow. Modifier-only combos are supported
(e.g. `"ctrl+cmd"` activates when both keys are held, with no extra trigger
key required).

The hotkey is configurable as a dropdown in both the setup wizard and settings
window. The presets are `DICTATION_HOTKEY_PRESETS` in `jarvis/config.py`,
beside `_default_dictation_hotkey()`, and every platform's default is one of
them. Both dropdowns are built by `dictation_hotkey_choices()` in
`hotkeys.py`: each preset labelled with `format_hotkey_display()`, the platform
default's label ending in `(default)`. `hotkeys.py` is plain string work and
imports neither sounddevice, numpy nor pynput, so labelling a dropdown loads
none of the engine's audio or keyboard libraries. A preset added to the list
reaches both windows. A hotkey written by hand that no preset matches is added
to the dropdown as its raw value and selected, so opening either window and
saving keeps it.

## Core Flow

### Hold-to-Dictate (Standard Mode)

1. **Press hotkey** → start recording audio into buffer, play start beep,
   publish the state `DICTATING`, pause main voice listener.
2. **Hold hotkey** → audio frames accumulate in a dedicated
   `sounddevice.InputStream`.
3. **Release hotkey** → stop recording, play stop beep,
   publish `DICTATION_PROCESSING`, transcribe via shared Whisper model, apply
   post-processing pipeline, paste result into focused app via clipboard,
   publish `IDLE`, resume main voice listener.

Only a fresh press of the hotkey starts a recording. The OS repeats a held
key's press, and a repeat never starts one, so a press that was refused (no
microphone, Whisper not loaded) is tried again only once the hotkey is
released and pressed again.

The published state therefore moves through three distinct values across a
dictation cycle: `DICTATING` while recording, `DICTATION_PROCESSING` while the
captured audio is being transcribed / post-processed / pasted, and back to
`IDLE` once the cycle completes. A front end reading it (`state.spec.md`) can
show the user that their voice input has been accepted and is being processed.

### Hands-Free Mode (Double-Tap)

1. **Quick press-and-release** (hold < 0.4 s) followed by a **second tap**
   within 0.4 s → enters hands-free mode. Recording continues until
   explicitly stopped.
2. **Stop triggers** — re-press the hotkey *or* press Escape.
3. Same post-processing pipeline as standard mode.

## Post-Processing Pipeline

After transcription, text passes through these stages in order:

1. **Custom dictionary** — case-insensitive whole-word regex replacements
   from `dictation_custom_dictionary`. Each entry is `"wrong -> right"`.
**Dictation never leaves the machine.** The cleanup pass is pinned to the local backend rather than following `llm_provider`, unlike the four contexts that read his own sentences and honour a pin. Those carry one sentence he addressed to the assistant; this carries everything he dictates, all day, into applications that have nothing to do with her. Nothing local answering costs the cleanup and nothing else: the raw transcript is what comes back, as it already does for a timeout or an unparseable reply.

2. **LLM filler removal** (optional) — when `dictation_filler_removal` is
   enabled, sends the text to the local backend (same model as the
   assistant), whatever `llm_provider` is set to with a prompt to remove filler words (um, uh, like, you know,
   etc.) while preserving meaning. Uses a 5-second timeout; falls back to the
   unprocessed text on failure.

## Architecture

- **`pynput`** for global hotkey detection (cross-platform).
- **Clipboard-based paste** (`Ctrl+V` / `Cmd+V`) for text insertion — more
  reliable than character-by-character typing, handles Unicode.
- **Shared Whisper model** via lazy reference (`lambda: voice_thread.model`)
  and backend info — no double memory usage.
- **Separate `sounddevice.InputStream`** for dictation audio — avoids
  modifying the complex listener code.
- **Pause flag** on the main listener to prevent dictation speech being
  interpreted as commands.
- **No PortAudio work on the pynput hook thread.** The hotkey callbacks only
  flip state; stream open/start runs on a worker thread (`_begin_recording`).
  Windows silently unhooks slow hook callbacks, and opening streams on hook
  threads risks race conditions with other audio threads. If recording is
  stopped before the worker finishes opening the stream, the worker closes it
  rather than leaking a live capture.
- **All PortAudio stream lifecycle calls** (open/start/stop/close, and
  `sd.play` for beeps) run under the process-wide
  `jarvis.utils.audio_lock.portaudio_lock`: PortAudio documents stream
  open/close as not thread safe, and unserialised calls abort the whole app
  on Windows.

### Optional Dependencies

- `sounddevice`, `numpy`, and `pynput` are optional at import time: the
  module must always import cleanly even when they are broken or missing
  (e.g. `import sounddevice` raises **OSError**, not ImportError, when the
  PortAudio shared library is absent; `pynput` can raise display errors on
  headless Linux). The daemon's dictation start-up relies on this, and in
  the bundled app the daemon runs inside the desktop app's process: a
  missing audio backend must never crash the app.

### Audio Device Handling

- The engine accepts an optional `voice_device` parameter, passed through from
  the daemon's configured device, and resolves it on every press with the
  shared resolver the wake-word listener also uses (`resolve_input_device` in
  `jarvis.utils.audio_stream`), so both record from the same microphone:
  blank, `default` or `system` is the system default input, a number is a
  device index, and any other text is the first input-capable device whose
  name contains it, ignoring case.
- PortAudio reads the device list once, when the app starts, and every press
  resolves against that list. A microphone connected later is found once the
  app is quit and reopened.
- The device is resolved first, before the Whisper check and before anything
  announces a recording. A name that matches no input (the microphone was not
  connected when the app started, so the listener refused it too) records
  nothing: no other input is opened, no start beep plays, and the console says
  the selected microphone was not found. A failed device lookup is handled the
  same way, without the console line.
- A microphone unplugged while the app runs is still in that list, so its name
  still matches: the start beep plays, its stream fails to open, the console
  says the microphone could not be opened, and the recording ends with nothing
  captured.
- The stream opens at the selected input's native sample rate (requesting
  16 kHz fails on many devices, e.g. PortAudio error -50 on macOS), with
  `open_input_stream`'s format negotiation, and stores the rate in
  `_stream_sample_rate`.
- If the stream rate differs from the Whisper target rate, audio is resampled
  via linear interpolation before transcription.

## Edge Cases

| Case                      | Behaviour                                         |
|---------------------------|----------------------------------------------------|
| Whisper not yet loaded    | Play "not ready" beep, skip                        |
| Max recording duration    | No cap: the user controls when to stop by releasing the hotkey. A cap would paste prematurely mid-dictation and restart recording |
| Empty transcription       | No paste occurs                                    |
| Concurrent with assistant | Dictation works independently; pauses listener     |
| macOS permissions         | `pynput` requires Accessibility permissions        |
| macOS 26+ (Tahoe)         | `pynput` disabled — TSM main-thread assertion crash |
| Linux / Wayland           | `pynput` requires X11 (limited Wayland support)    |
| Audio rate mismatch       | Resample via linear interpolation to Whisper rate  |
| Named microphone missing at startup | Records nothing, opens no other input, no start beep; console line |
| Named microphone unplugged while running | Start beep, stream fails to open, console line, nothing recorded |
| Hotkey held after a refused press | Key repeat does not retry it; the next fresh press does |
| LLM filler removal fails  | Falls back to raw transcription (5 s timeout)     |
| Custom dictionary empty   | No-op, text passes through unchanged               |

## Thread Safety

- `threading.Lock` around shared Whisper model transcription calls.
- Dedicated audio stream; never touches the listener's stream.
- The `pynput` key handlers (`_on_key_press` / `_on_key_release`) must return
  quickly — Windows silently removes low-level keyboard hooks that take more
  than ~5 s to return, leaving pynput in an inconsistent state that can
  segfault on the next `Controller` call from the paste thread. `_stop_recording`
  therefore only flips state under the lock and dispatches audio-stream
  teardown, beep playback, transcription, and paste to a background thread.
  The `discard=True` path keeps the synchronous teardown so shutdown can
  reliably wait for everything to finish.

## Beeps

Two short beeps generated the same way as the existing `TunePlayer` sonar ping:
- **Start beep** — higher pitch (700 Hz), signals recording started.
- **Stop beep** — lower pitch (440 Hz), signals recording stopped.

## Setup Wizard

The setup wizard includes a dedicated Dictation page (between the Whisper and
MCP Servers steps) that allows users to:
- Enable/disable dictation
- Turn filler word removal on or off
- Choose the hotkey from a dropdown of presets
- View tips about hold-to-dictate and double-tap hands-free mode

Each control starts from the value in `config.json`, so re-running the wizard
keeps what was set in Settings.

## Dependencies

- `pynput>=1.7.6` — global hotkey detection and keyboard simulation.

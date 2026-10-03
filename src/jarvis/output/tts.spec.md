# Text-to-Speech Engines

## Why

Everything she says aloud leaves through one of three engines in
`tts.py`. The listener, the echo detector, the hot window and the
reminder scheduler all lean on what those engines promise, and most of
those promises are not visible in a signature: that a completion
callback means the words were heard, that an interrupted reply opens no
window, that the text spoken is never sent anywhere. This file writes
them down.

It covers the engines, the queue in front of them, the Piper voice
files, and the configuration that selects them. It does not cover how a
reply is cut into sentences (`streaming.spec.md`), how the listener
decides to speak or to stop (`listening.spec.md`), or what the thinking
tune sounds like (`tune_player.py`, described only where it meets the
voice).

## Privacy

Synthesis is local. The text she speaks never leaves the machine. The
only network traffic in this module is fetching a model the first time
it is needed (a Piper voice, the Kokoro weights), and that request
carries a file name, never speech.

## Engines and selection

`create_tts_engine(engine=...)` returns one of three engines. The name
is compared case-insensitively.

| `tts_engine` | Class | What it is |
|--------------|-------|------------|
| `piper` (default) | `PiperTTS` | Light local neural voice, one `.onnx` model per voice, fetched on first use |
| `kokoro` | `KokoroTTS` | Kokoro-82M, more natural, real time on CPU, multilingual through espeak-ng |
| `chatterbox` | `ChatterboxTTS` | Heavy PyTorch voice with emotion control and cloning, runs from source only |

Any other name yields Piper. The config layer lowercases `tts_engine` and
falls back to `piper` for a value that is not one of the three, so the
factory only ever sees a valid name from the daemon.

An engine built with `enabled=False` accepts every call and does
nothing: `speak` queues nothing, `start` starts nothing. Callers test
`tts.enabled` before relying on speech.

### The shared surface

The three engines are interchangeable to the listener. Each exposes:

| Member | Meaning |
|--------|---------|
| `enabled` | Whether the engine speaks at all |
| `start()` | Create the worker thread. Idempotent |
| `stop()` | Interrupt, end the worker, wait up to two seconds for it |
| `speak(text, completion_callback=None, duration_callback=None, language=None)` | Queue one item (see below). `language` is the language the user was heard in. Piper chooses its voice from it; Kokoro and Chatterbox accept it and ignore it |
| `interrupt()` | Stop talking (see Interrupting) |
| `is_speaking()` | Whether an item is being synthesised or played |
| `get_last_spoken_text()` | The processed text of the item most recently started |

`speak` starts the worker if nobody has. Calling `start` on Piper loads
the voice on the calling thread (downloading it if absent) so the wait
happens at launch rather than at the first word. Kokoro warms its
pipeline on a background thread named `kokoro-init`; a `speak` that
arrives before the load finishes waits on the same lock. On Windows
Kokoro first imports its package on the calling thread, because loading
PyTorch, SciPy and OpenBLAS DLLs concurrently deadlocks the Windows
loader. Chatterbox loads its model on `start`.

## The queue

One worker thread per engine takes items from a FIFO queue and speaks
them one at a time. An item is `(text, completion_callback,
duration_callback)`, and for Piper a fourth field, the language. The
language belongs to the item, not to the engine: sentences of two
replies in two languages can wait in the queue together, each spoken in
its own voice.

**Text is processed when it is queued.** `_preprocess_for_speech`
replaces links with a spoken description of the domain and strips
markdown (bold, italics, code, headings, quotes, bullets, and numbered
markers only when two or more adjacent lines form a real list). The
processed text is what the worker speaks and what
`get_last_spoken_text()` returns. Words inside the markup are kept.

**Empty text is dropped, unless it carries a callback.** Whitespace
with no completion callback queues nothing. Empty text *with* a
completion callback is the end-of-reply marker: it speaks nothing, and
when the worker reaches it, after everything queued before it, it fires
its callback. A streamed reply whose last sentence ended on punctuation
leaves no tail for the callback to ride on, so it travels as a marker of
its own (`streaming.spec.md`, "One reply is still one turn").

**Callbacks belong to their item.** They are queued with the text and
installed by the worker as it dequeues that item. They are never
engine-wide settings written by `speak`, because a reply spoken sentence
by sentence would then fire the last chunk's callback after the first
chunk finished and open the hot window mid-reply. An early chunk carries
no callback at all.

## Speaking state and completion

`is_speaking()` is true from the moment the worker starts an item until
that item is finished, interrupted or abandoned. It describes the item
in hand. It is false while the worker is idle, in the instant between
two items, and for items still waiting in the queue. The end of a reply
is therefore signalled by the completion callback, never inferred from
`is_speaking()` going false.

**The completion callback fires if and only if the item's audio played
to its end.** It does not fire when the engine could not initialise,
when synthesis produced no audio, when an exception was caught, or when
the item was interrupted. The reminder scheduler settles a reminder as
said from this callback (`reminders.spec.md`: delivery settles it, never
queueing), so a broken voice must leave the reminder owed rather than
recording that she said it. The end-of-reply marker is the one item that
fires without playing anything; it has no audio to play.

**The duration callback** receives the exact length of the item's audio
in seconds, computed from the synthesised samples, after synthesis and
before playback starts. It does not fire when nothing was synthesised.

**The orb.** When an item starts, the engine sets the SPEAKING state on
the shared state holder if that holder can be imported, and ignores the
failure if it cannot. Speech never depends on it. Nothing in this module
clears the state when speech ends; whoever owns the transitions does.

## Interrupting

`interrupt()` is what a spoken stop word calls, and it means stop
talking, completely. The contract is the one `streaming.spec.md` states
("Interruption still stops everything"):

- the item playing stops within a fraction of a second (the output
  stream is aborted, and the playback loops poll every 50 ms, 100 ms for
  Chatterbox);
- items queued before the call are discarded without being synthesised;
- an item still being synthesised is abandoned;
- no completion callback fires for any of them, and the end-of-reply
  marker does not fire its callback either: an interrupted reply is
  unheard, and a hot window opened by it would invite the user to answer
  something she did not finish saying;
- items queued after the call are spoken normally, so a reply that
  follows a stop is not swallowed.

`interrupt()` is safe from any thread, safe when nothing is speaking,
and safe to call repeatedly. `stop()` calls it. Chatterbox can only
interrupt playback: the generation of one item is a single model call
that cannot be cut short.

## What she owes the echo detector

The engines do not own the echo reference. The listener records what was
said with `track_tts_start`, accumulating the sentences of one reply
(`streaming.spec.md`, "The constraint that shapes everything"). The
engine owes the listener four things:

1. `is_speaking()`, which gates stop-word and echo handling while she
   talks;
2. `get_last_spoken_text()`, which is the current chunk only. It is not
   the echo reference, and must not be used as one;
3. the duration callback, which replaces the detector's estimate with the
   real length when the listener passed one. A streamed reply attaches it to
   its tail only, so earlier chunks are timed by estimate;
4. the completion callback, which is what opens the hot window.

Where no exact duration is known the detector estimates from `tts_rate`
(words per minute). Piper, Kokoro and Chatterbox ignore `tts_rate`
themselves: Piper's pace is `length_scale`, Kokoro's is `speed`, and
`tts_rate` and `tts_voice` are accepted by the factory for interface
compatibility only.

## The thinking tune

`TunePlayer` is not an engine. It loops a synthesised pad while the
reply is being prepared and is owned by the listener, which starts it
when a query is dispatched (never while she is already speaking) and
stops it when speech begins or when there is nothing to say. It opens its
output stream through the same `sounddevice` API and under the same
process-wide `portaudio_lock` as the engines, so stopping it releases the
device in milliseconds and the voice can open it straight away. The tune
and the voice must not overlap: the tune is stopped no later than the
first spoken chunk.

## Audio output

Each item is synthesised whole, then played. Piper and Kokoro play it
through a `sounddevice.OutputStream` (mono; Piper in `int16` at the
voice's own sample rate, Kokoro in `float32` at 24 kHz) in blocks of
1024 frames so an abort lands quickly. Chatterbox writes a temporary WAV
and plays it through the pygame mixer. Opening, starting, aborting and
closing a stream all happen under `portaudio_lock`
(`listening.spec.md`, "Serialised PortAudio Lifecycle").

## Piper voices

### Which language wins

The voice follows the language the reply is written in. The engine knows
two candidates for it, and they can disagree:

- `response_language`, when set. The persona prompt makes the model write
  in it whatever language it is spoken to, so it is the language of the
  reply. It outranks detection: a user who set `français` and asks a
  question in English is answered in French, and the voice has to read
  French.
- The language the user was heard in, which the listener passes as
  `language`. With `response_language` empty (auto) the persona pins no
  language, so unless the reply engine's English-only instruction applies
  (`reply.spec.md`, "System message composition") she answers in the
  language she was spoken to, and that is the one to follow.

When neither is known, as before the first transcription or for a line
spoken with no utterance behind it, no language is claimed and the
default voice speaks.

### Which voice

An item is spoken by the first of these that applies:

1. `tts_piper_voices`, when it names the language of the reply. Keys are
   languages in any spelling a user might write: the ISO 639-1 code
   Whisper reports (`fr`), the language's own name (`français`) or its
   English name (`French`), compared without case or accents, and a
   regional variant (`fr-FR`, `pt_BR`) counts as its base language. A
   language nobody listed in the table below may still be a key. A value
   is the path of a `.onnx` model, or a bare voice name, which is a file
   in the models directory and so is fetched by name on first use.
2. The default voice, loaded when the engine starts so the wait happens
   at launch rather than at the first word:
   1. A `tts_piper_model_path` the user set. Naming a voice is a choice,
      and neither the language nor a failed download overrides it. It is
      also the voice for every language the map does not name.
   2. Otherwise the voice for the configured `response_language`, from a
      table of eleven languages (French, English, Spanish, German,
      Italian, Dutch, Portuguese, Polish, Russian, Turkish, Chinese).
      Each language lists the spellings a user might write: its own name,
      its English name and its ISO 639-1 code, compared without case or
      accents, so `français`, `Francais` and `fr` meet.
   3. Otherwise, for an empty or unlisted language, the fallback voice
      (`PIPER_FALLBACK_VOICE`). A language nobody listed speaks with it
      rather than falling silent.

The built-in table serves the configured language only. A language that
was merely heard never selects a built-in voice: that would download a
voice nobody asked for, mid-conversation, on the strength of a detector's
guess. To have the voice follow what was heard, the user lists the voices
in `tts_piper_voices`.

Supporting another language in the table is one row. The table pairs a
language with a voice trained for it, and a test holds each row to that.

A mapped voice is loaded the first time a reply needs it, then kept, and
is played at its own sample rate. If it cannot be loaded (no file, no
network) the default voice speaks the item instead, and the path is not
tried again for the life of the engine. A reply in the wrong accent is
worse than one in the right voice and better than silence.

The choice is logged at debug level: the language, the voice file, and
whether it came from the map or is the fallback.

### Fetching

Piper voices are named `{lang}_{REGION}-{speaker}-{quality}` and live in
the `rhasspy/piper-voices` repository on Hugging Face, which fixes the
URL of both files: the `.onnx` model and its `.onnx.json` config. They
are stored in `~/.local/share/jarvis/models/piper/`. Resolving the
directory creates nothing; only a download creates it.

When either file of the resolved voice is missing, the engine downloads
the voice named by the basename of the path. A file already on disk is
not fetched again. Each file is written to a `.tmp` name and renamed, so
a partial download is never taken for a model. An HTTP 429 is retried
with a doubling wait (two, four, eight, sixteen seconds, four retries);
any other failure gives up, and a partial download never takes the final
file name. A name that
does not have the three dash-separated parts cannot be fetched.

### Falling back

A voice that cannot be fetched must not cost her speech. When the
download fails and the voice was chosen by language (not pinned), the
engine speaks with the fallback voice if both its files are already on
disk, and says so. With no such file, or with a pinned voice, nothing is
substituted: initialisation fails, the reason is kept in `_init_error`,
and each item prints it and speaks nothing. A missing `piper` package or
a model that fails to load fails the same way.

Initialisation of the default voice is attempted once per engine, and
each mapped voice once per path. Whatever the outcome is kept for the
life of the engine, so a failed fetch is not retried on the next
sentence.

## Kokoro and Chatterbox

**Kokoro** takes its voice from `tts_kokoro_voice` (default `ff_siwis`)
and its language from `tts_kokoro_lang_code` (default `f`, French), and
nothing else: `response_language` does not reach it. The model weights
download once from Hugging Face on first use. Non-English phonemisation
needs `libespeak-ng`; `PHONEMIZER_ESPEAK_LIBRARY` is pointed at a known
install location when the user has not set it. Initialisation is
attempted once, as for Piper.

**Chatterbox** takes no language and speaks English. It loads on CUDA
when asked and available, on CPU otherwise. If its dependencies or model
are missing it warns and every item is skipped.

## Language

Piper is the only engine whose voice follows a language. Which language,
and which wins when the configured one and the one heard disagree, is
decided under "Piper voices". Kokoro and Chatterbox accept the language
`speak` is given and ignore it: Kokoro's voice and language are set by
hand, and Chatterbox speaks English.

## Configuration

| Key | Applies to | Meaning |
|-----|------------|---------|
| `tts_enabled` | all | Whether she speaks |
| `tts_engine` | all | `piper`, `kokoro` or `chatterbox` |
| `tts_voice`, `tts_rate` | all | Interface compatibility; `tts_rate` also feeds the echo detector's timing estimate |
| `response_language` | Piper | The language of every reply when set, and so the voice's; chooses the default voice when no model path is pinned |
| `tts_piper_voices` | Piper | Language to voice (path or bare name), chosen per reply. A language it does not name speaks with the default voice |
| `tts_piper_model_path` | Piper | Pins the default voice; fetched by basename if absent |
| `tts_piper_speaker` | Piper | Speaker index for multi-speaker models |
| `tts_piper_length_scale` | Piper | Pace: below 1.0 is faster |
| `tts_piper_noise_scale`, `tts_piper_noise_w` | Piper | Expressiveness and rhythm |
| `tts_piper_sentence_silence` | Piper | Carried on the engine; synthesis does not apply it |
| `tts_kokoro_voice`, `tts_kokoro_lang_code`, `tts_kokoro_speed` | Kokoro | Voice, language code, pace |
| `tts_chatterbox_device`, `_audio_prompt`, `_exaggeration`, `_cfg_weight` | Chatterbox | Device, cloning sample, emotion, quality |

## Testing

- The factory returns the right engine for each name, case-insensitively,
  and Piper for an unknown one.
- A disabled engine accepts every call and queues nothing.
- A completion callback travels with its own item: earlier chunks carry
  none, the last carries its own, and an end-of-reply marker is queued
  for a callback with no text.
- The completion callback does not fire when the engine could not start
  or the item was interrupted.
- An interrupt during one item leaves the items queued behind it
  unspoken, and fires neither their callbacks nor the end-of-reply
  marker's.
- A missing voice that cannot be fetched falls back to a cached fallback
  voice, never to another voice when pinned, and reports the reason when
  there is nothing to fall back to.
- Each listed language gets a voice trained for it, in any spelling, and
  an unlisted one gets the fallback.
- A pinned model path beats the configured language, and is the voice for
  every language `tts_piper_voices` leaves out.
- A French reply picks the French voice, an unmapped language picks the
  default voice, and with `response_language` empty the voice follows the
  language heard. When `response_language` is set it outranks the
  language heard.
- A mapped voice that cannot be loaded leaves the reply spoken in the
  default voice, and is not fetched again on the next sentence.
- Every voice is played at its own sample rate.
- The language passed to `speak` travels with its item to the worker.
- Markdown and links are stripped before speech.

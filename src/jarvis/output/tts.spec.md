# Text-to-Speech Engines

## Why

Everything she says aloud leaves through one of four engines in
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
it is needed (a Piper voice, the Kokoro weights, the Kyutai model and
the file of its voice), and that request carries a file name, never
speech.

## Engines and selection

`create_tts_engine(engine=...)` returns one of four engines. The name
is compared case-insensitively.

| `tts_engine` | Class | What it is |
|--------------|-------|------------|
| `piper` (default) | `PiperTTS` | Light local neural voice, one `.onnx` model per voice, fetched on first use |
| `kokoro` | `KokoroTTS` | Kokoro-82M, more natural, real time on CPU, multilingual through espeak-ng |
| `chatterbox` | `ChatterboxTTS` | Heavy PyTorch voice with emotion control and cloning, runs from source only |
| `kyutai` | `KyutaiTTS` | Kyutai TTS 1.6B (English and French), the most natural of the four, faster than real time on a recent NVIDIA GPU, runs from source only |

Any other name yields Piper. The config layer lowercases `tts_engine` and
falls back to `piper` for a value that is not one of the four, so the
factory only ever sees a valid name from the daemon.

The daemon builds the engine from the settings in one function,
`build_tts_engine`, which carries every `tts_*` setting and
`response_language` to the factory. A setting that is read, shown in the
settings window and not passed on there would be silently ignored, so
that road is what the tests follow from the config file to the voice.

An engine built with `enabled=False` accepts every call and does
nothing: `speak` queues nothing, `start` starts nothing. Callers test
`tts.enabled` before relying on speech.

### The shared surface

The four engines are interchangeable to the listener. Each exposes:

| Member | Meaning |
|--------|---------|
| `enabled` | Whether the engine speaks at all |
| `start()` | Create the worker thread. Idempotent |
| `stop()` | Interrupt, end the worker, wait up to two seconds for it |
| `speak(text, completion_callback=None, duration_callback=None, language=None)` | Queue one item (see below). `language` is the language the user was heard in. Piper chooses its voice from it; Kokoro, Chatterbox and Kyutai accept it and ignore it |
| `interrupt()` | Stop talking (see Interrupting) |
| `is_speaking()` | Whether an item is being synthesised or played |
| `get_last_spoken_text()` | The processed text of the item most recently started |

`speak` starts the worker if nobody has. Calling `start` on Piper loads
the voice on the calling thread (downloading it if absent) so the wait
happens at launch rather than at the first word. Kokoro warms its
pipeline on a background thread named `kokoro-init`, ending with one
throwaway synthesis whose audio is discarded (of a digit, which has a
reading in every language, where a word belongs to one), so the
model's first-call cost (CUDA kernel loading can take tens of seconds)
is paid at startup rather than on the first reply; a warm-up failure
leaves the pipeline loaded and the error surfaces at speak time. A
`speak` that arrives
before the load finishes waits on the same lock. On Windows
Kokoro first imports its package on the calling thread, because loading
PyTorch, SciPy and OpenBLAS DLLs concurrently deadlocks the Windows
loader. Chatterbox loads its model on `start`. Kyutai warms its model on
a background thread named `kyutai-init`, ending with one throwaway
synthesis whose audio is discarded: the first synthesis captures the CUDA
graphs the model replays afterwards, which takes tens of seconds, so it is
paid at startup rather than on the first reply. On Windows `start` first
sets the compile switch (see "Kyutai") and imports the package on the
calling thread, for the loader reason given for Kokoro.

## The queue

One worker thread per engine takes items from a FIFO queue and speaks
them one at a time. An item is `(text, completion_callback,
duration_callback)`, and for Piper a fourth field, the language. The
language belongs to the item, not to the engine: sentences of two
replies in two languages can wait in the queue together, each spoken in
its own voice. Kyutai's fourth field is the count of interruptions at the
moment the item was queued (see "Interrupting").

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

**Speaking state.** When an item starts, the engine publishes SPEAKING
through the shared state (`state.spec.md`). Publishing never raises, so
the engine calls it bare and speech never depends on it. Nothing in this
module clears the state when speech ends; whoever owns the transitions
does.

**First sound.** Once an item's audio stream is running (Piper, Kokoro
and Kyutai: after the output stream starts; Chatterbox: after playback
begins), the engine calls `_sound_started()`, which marks the latency
journal (`jarvis/latency.spec.md`). That is not where SPEAKING is
published: SPEAKING goes out when an item starts being synthesised, before
anything is audible, so marking there would understate the wait by the
synthesis time of the first chunk. The call sits outside the audio locks,
never raises, and an engine added later has to make it.

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

Kyutai keeps the whole contract. `interrupt()` counts the interruption and
every item carries the count it was queued under, so an item queued before
the call is dropped, callbacks and end-of-reply marker included, when the
worker reaches it, and one queued after is spoken. The count is compared
again at every frame the model produces, so an item still being
synthesised is abandoned within one frame of audio rather than run to its
end, and an interruption that lands between the worker taking an item and
starting it cannot be lost.

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
stops it when the finished reply is handed over for speaking, or when
there is nothing to say. It opens its output stream through the same
`sounddevice` API and under the same process-wide `portaudio_lock` as the
engines, so stopping it releases the device in milliseconds and the voice
can open it straight away.

## Audio output

Each item is synthesised whole, then played. Piper and Kokoro play it
through a `sounddevice.OutputStream` (mono; Piper in `int16` at the
voice's own sample rate, Kokoro and Kyutai in `float32` at 24 kHz) in blocks of
1024 frames so an abort lands quickly. Chatterbox writes a temporary WAV
and plays it through the pygame mixer. Opening, starting, aborting and
closing a stream all happen under `portaudio_lock`
(`listening.spec.md`, "Serialised PortAudio Lifecycle").

## Piper voices

### Which language wins

The voice should follow the language the reply is written in. The engine
never reads the reply to find out; it holds two candidates for it, and
they can disagree:

- `response_language`, when set. The persona prompt asks the model to
  write in it whatever language it is spoken to, so the engine assumes
  the reply is in it. It outranks detection: a user who set `français`
  and asks a question in English is expected to be answered in French,
  and the voice has to read French.
- The language the user was heard in, which the listener passes as
  `language`. With `response_language` empty (auto) the persona pins no
  language, so the engine assumes she answers in the language she was
  spoken to and follows that.

Both are assumptions about what the model writes. When the model writes
in another language, the voice chosen for the assumed one reads that text
with its own accent (see "Language" for the case the engine itself
provokes).

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
   language nobody listed in the table below may still be a key, but it
   has to be keyed by its ISO 639-1 code (`ja`), because only the table
   knows a language's names: with `response_language` empty the language
   arrives as the code the recogniser reports, and a key such as
   `Japanese` never matches it. A value is the path of a `.onnx` model,
   or a bare voice name, which is a file in the models directory and so is
   fetched by name on first use.
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
      accents, so `français`, `Francais` and `fr` meet. A regional variant
      (`fr-FR`, `pt_BR`) counts as its base language here exactly as it
      does for a map key, so one `response_language` is never French to
      the map and unknown to the default voice.
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

The choice is logged at debug level when it changes: the language, the
voice file, and whether it came from the map or is the default voice. A
reply of many sentences names its voice once.

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
nothing else: `response_language` does not reach it. The voice may be a
single voice, a comma blend (`ff_siwis,af_bella`, which Kokoro itself
blends in equal parts), or a weighted blend (`ff_siwis:0.8,af_bella:0.2`)
— weights need not sum to one, they are normalised by their sum, a voice
without a weight counts as 1, and the blend is built as the weighted
mean of the voice packs at initialisation, before the warm-up synthesis
that already rides it. A spec whose weights cannot be parsed, or whose
packs cannot be loaded, is handed to Kokoro as the raw string, failing
the way an unknown voice name would. The model weights
download once from Hugging Face on first use. Non-English phonemisation
needs `libespeak-ng`; `PHONEMIZER_ESPEAK_LIBRARY` is pointed at a known
install location when the user has not set it. Initialisation is
attempted once, as for Piper.

**Chatterbox** takes no language and speaks English. It loads on CUDA
when asked and available, on CPU otherwise. The `chatterbox-tts` package
is not in `requirements.txt` — it pins `torch==2.6.0`, which cannot
coexist with the newer CUDA torch builds Kokoro uses on recent GPUs — so
running the engine means installing it by hand; if its dependencies or
model are missing it warns and every item is skipped.

## Kyutai

`KyutaiTTS` runs Kyutai TTS 1.6B (`kyutai/tts-1.6b-en_fr`) through the
`moshi` package. It speaks English and French and reads the language from
the text itself. Its voice comes from `tts_kyutai_voice` and nothing else:
`response_language` does not reach it.

**Voice.** The value is either a name in the `kyutai/tts-voices`
repository (a recording with a precomputed embedding, for example
`cml-tts/fr/577_394_000070-0001_enhanced.wav`), fetched on first use, or
the path of a `.safetensors` embedding file, used as it is. A path that
does not exist fails initialisation: a voice the user named is never
replaced by another. The default is a French female voice.

**Hardware.** The engine needs an NVIDIA GPU (CUDA). Without one,
initialisation fails and the reason is kept in `_init_error`; each item
prints it and speaks nothing. There is no CPU mode: a model of this size
synthesises far slower than real time there.

**Compilation.** `torch.compile` needs Triton, which has no Windows build.
When Triton cannot be imported the engine sets `NO_TORCH_COMPILE=1`,
unless the user already set it, before `moshi` is first imported, because
the package reads the variable at import. The CUDA graphs stay on: with
them the model runs faster than real time, without them several times
slower, so the engine never sets `NO_CUDA_GRAPH`.

**Synthesis.** One item is one model call: the text is prepared as a
single-speaker script and conditioned on the voice, with the sampling
temperature and guidance coefficient the model was released with (fixed,
not settings). Frames are decoded to audio as the model produces them, and
the first audible frame arrives about a second in. Like the other engines
the item is played once all of it exists, so the exact duration is known
before playback starts.

**Failure.** Initialisation is attempted once and its outcome is kept for
the life of the engine. A missing `moshi` package, a missing GPU, a model
that fails to load, a voice that cannot be resolved: each fails
initialisation with its reason, and nothing is said. A synthesis that
raises, or produces no audio, fires no completion callback.

**Installation.** `moshi` is not in `requirements.txt`: it declares a
`torch` range older than the CUDA builds used here, so the README installs
it by hand without its dependencies.

## Language

Piper is the only engine whose voice follows a language. Which language,
and which wins when the configured one and the one heard disagree, is
decided under "Piper voices". Kokoro, Chatterbox and Kyutai accept the
language `speak` is given and ignore it: Kokoro's voice and language are
set by hand, Chatterbox speaks English, and Kyutai speaks the English or
French it finds in the text, so a reply in another language is read with
the wrong sounds.

The voices only read what the model writes, and the voice is chosen from
the language the engine expects, not from the text. The reply engine
separately appends "Always respond in English regardless of the language
the user speaks in" to the system message whenever the engine is Piper or
Chatterbox, whatever `response_language` and `tts_piper_voices` say. It
contradicts a configured `response_language`, and which of the two the
model follows is up to the model. A model that obeys the English line
writes English, and the voice mapped to the language the user spoke (or to
`response_language`) then reads English text with that voice's accent.
The voice map therefore helps only when the model writes in the mapped
language. The line is not added for Kokoro or Kyutai.

## Configuration

| Key | Applies to | Meaning |
|-----|------------|---------|
| `tts_enabled` | all | Whether she speaks |
| `tts_engine` | all | `piper`, `kokoro`, `chatterbox` or `kyutai` |
| `tts_voice`, `tts_rate` | all | Interface compatibility; `tts_rate` also feeds the echo detector's timing estimate |
| `response_language` | Piper | The language of every reply when set, and so the voice's; chooses the default voice when no model path is pinned |
| `tts_piper_voices` | Piper | Language to voice (path or bare name), chosen per reply. A language of the built-in table may be keyed in any of its spellings; any other language is keyed by its ISO 639-1 code. A language it does not name speaks with the default voice |
| `tts_piper_model_path` | Piper | Pins the default voice; fetched by basename if absent |
| `tts_piper_speaker` | Piper | Speaker index for multi-speaker models |
| `tts_piper_length_scale` | Piper | Pace: below 1.0 is faster |
| `tts_piper_noise_scale`, `tts_piper_noise_w` | Piper | Expressiveness and rhythm |
| `tts_piper_sentence_silence` | Piper | Carried on the engine; synthesis does not apply it |
| `tts_kokoro_voice`, `tts_kokoro_lang_code`, `tts_kokoro_speed` | Kokoro | Voice, language code, pace |
| `tts_kyutai_voice` | Kyutai | A name in the `kyutai/tts-voices` repository, or the path of a `.safetensors` embedding file |
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
- Each engine marks the first sound once its audio stream is running, and
  not when SPEAKING is published.
- Kokoro's initialisation ends with one throwaway synthesis (audio
  discarded, nothing queued or played), and a warm-up failure leaves the
  pipeline loaded rather than failing init.
- A weighted voice blend is the normalised weighted mean of its packs
  (weights need not sum to one; a part without a weight counts as 1),
  the warm-up rides it, an equal blend stays a raw string for Kokoro,
  and an unparseable weight falls back to the raw string.
- A regional variant (`fr-FR`, `fr_FR`, `pt_BR`) chooses the same default
  voice as its base language, and a variant of an unlisted language still
  gets the fallback. The map and the default voice read one value as the
  same language.
- A language outside the table is matched by its ISO code: a key `ja`
  (or `ja-JP`) matches the language heard as `ja`, a key `Japanese` does
  not.
- A pinned model path beats the configured language, and is the voice for
  every language `tts_piper_voices` leaves out.
- A French reply picks the French voice, an unmapped language picks the
  default voice, and with `response_language` empty the voice follows the
  language heard. When `response_language` is set it outranks the
  language heard.
- A mapped voice that cannot be loaded leaves the reply spoken in the
  default voice, and is not fetched again on the next sentence.
- Every voice is played at its own sample rate.
- The debug log names the voice once for a reply of many sentences, and
  again when the voice changes.
- The language passed to `speak` travels with its item to the worker.
- Settings read from the config file reach the engine: a language listed in
  `tts_piper_voices` is spoken in its voice when the daemon builds the
  engine, and `response_language` travels the same road.
- Markdown and links are stripped before speech.
- `kyutai` resolves to `KyutaiTTS` in any case, from the `tts_kyutai_voice`
  setting, and the config layer keeps it as a valid `tts_engine`.
- A Kyutai engine without a GPU, or without the `moshi` package, speaks
  nothing, fires no completion callback, reports the reason, and does not
  retry on the next item.
- With Triton absent the compile switch is set before `moshi` is first
  imported, a value the user set is left alone, and the CUDA graphs are
  never switched off.
- Kyutai's initialisation ends with one throwaway synthesis (audio
  discarded, nothing queued or played).
- A voice name is resolved through the model's voice repository, a
  `.safetensors` path is used as it is, and a path that does not exist
  fails initialisation.
- Kyutai's audio is played at the codec's rate, the duration callback
  gets its exact length before playback, and the completion callback
  fires only after the audio played to its end.
- A Kyutai interrupt drops the items queued before it without their
  callbacks, speaks the ones queued after, and abandons a synthesis in
  flight at its next frame.
- Kyutai marks the first sound once its output stream is running.

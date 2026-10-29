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
| `stop()` | Interrupt, end the worker (and Kyutai's player), wait up to two seconds for each thread. The engine counts as started until they are gone, so an item arriving meanwhile waits in the queue for the next worker |
| `speak(text, completion_callback=None, duration_callback=None, language=None)` | Queue one item (see below). `language` is the language the user was heard in. Piper chooses its voice from it; Kokoro, Chatterbox and Kyutai accept it and ignore it |
| `interrupt()` | Stop talking (see Interrupting) |
| `is_speaking()` | Whether an item is being synthesised or played |
| `get_last_spoken_text()` | The processed text of the item most recently started |
| `voice_delay()` | How much later than usual her voice will be heard: the talking face's delay while a face is ready and watched (Kyutai, see "The face"), otherwise 0 |

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
synthesis whose audio is discarded: the first synthesis pays the model's
one-off costs (kernel and module loading, allocator warm-up), which take
tens of seconds, so they are paid at startup rather than on the first
reply. On Windows `start` first
sets the compile switch (see "Kyutai") and imports the package on the
calling thread, for the loader reason given for Kokoro.

## The queue

One worker thread per engine takes items from a FIFO queue and speaks
them one at a time. An item is `(text, completion_callback,
duration_callback)`, then the engine's own fields: for Piper the
language, and last in every engine the count of interruptions at the
moment the item was queued (see "Interrupting"). The language belongs to
the item, not to the engine: sentences of two replies in two languages
can wait in the queue together, each spoken in its own voice. Kyutai has a
second thread, the player, that plays what the worker has synthesised, in
the same order (see "Kyutai"); for it "the worker speaks the item" reads
"the worker synthesises it and the player plays it".

**Text is processed when it is queued.** `_preprocess_for_speech`
replaces links with a spoken description of the domain and strips
markdown (bold, italics, code, headings, quotes, bullets, and numbered
markers only when two or more adjacent lines form a real list). The
processed text is what the worker speaks and what
`get_last_spoken_text()` returns. Words inside the markup are kept.

**Empty text is dropped, unless it carries a callback.** Whitespace
with no completion callback queues nothing. Empty text *with* a
completion callback is the end-of-reply marker: it speaks nothing, and
when the engine reaches it, after everything queued before it has been
said, it fires its callback (for Kyutai, the player reaches it, so the
marker waits for the audio of the items before it to finish playing). A
streamed reply whose last sentence ended on punctuation leaves no tail
for the callback to ride on, so it travels as a marker of its own
(`streaming.spec.md`, "One reply is still one turn").

**Callbacks belong to their item.** They are queued with the text and
installed by the worker as it dequeues that item (Kyutai carries them
with the item to its player). They are never engine-wide settings written
by `speak`, because a reply spoken sentence by sentence would then fire
the last chunk's callback after the first chunk finished and open the hot
window mid-reply. An early chunk carries no callback at all.

## Speaking state and completion

`is_speaking()` is true from the moment the worker starts an item until
that item is finished, interrupted or abandoned. It describes the item
in hand. It is false while the worker is idle, in the instant between
two items, and for items still waiting in the queue. The end of a reply
is therefore signalled by the completion callback, never inferred from
`is_speaking()` going false. Kyutai takes the next item up while the
current one is still playing, so for it the item in hand is the earliest
one not yet played: `is_speaking()` stays true from the first item taken up
until the last one has been played, interrupted or abandoned, with no
instant between two items.

**The completion callback fires if and only if the item's audio played
to its end.** It does not fire when the engine could not initialise,
when synthesis produced no audio, when an exception was caught, or when
the item was interrupted. The reminder scheduler settles a reminder as
said from this callback (`reminders.spec.md`: delivery settles it, never
queueing), so a broken voice must leave the reminder owed rather than
recording that she said it. The end-of-reply marker is the one item that
fires without playing anything; it has no audio to play.

**The duration callback** receives the exact length of the item's audio
in seconds, computed from the synthesised samples, once synthesis has
ended. Piper, Kokoro and Chatterbox play after synthesising, so it arrives
before playback starts. Kyutai plays while it synthesises, so it arrives
during playback, or before it when the item was prepared while the previous
one played. It does not fire when nothing was synthesised.

**Speaking state.** When an item starts, the engine publishes SPEAKING
through the shared state (`state.spec.md`). Publishing never raises, so
the engine calls it bare and speech never depends on it. An item that a
stop word has already cancelled when the worker takes it up publishes
nothing: the listener answers a stop with IDLE, and a SPEAKING after it
would stay on the orb until the next wake word. Nothing in this module
clears the state when speech ends; whoever owns the transitions does.

**First sound.** Once an item's audio stream is running (Piper, Kokoro
and Kyutai: after the output stream starts; Chatterbox: after playback
begins; a Kyutai item drawn on the talking face: when it is heard, see
"Kyutai"), the engine calls `_sound_started()`, which marks the latency
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
- items queued before the call are discarded, so that none of them is
  heard: Piper, Kokoro and Chatterbox never synthesise them, and Kyutai,
  which prepares items while one plays, drops those it has already
  prepared (see below);
- an item still being synthesised is abandoned;
- no completion callback fires for any of them, and the end-of-reply
  marker does not fire its callback either: an interrupted reply is
  unheard, and a hot window opened by it would invite the user to answer
  something she did not finish saying;
- items queued after the call are spoken normally, so a reply that
  follows a stop is not swallowed.

`interrupt()` is safe from any thread, safe when nothing is speaking,
and safe to call repeatedly. `stop()` calls it. Chatterbox cannot cut a
generation short: the item is a single model call, so what it made for an
item interrupted meanwhile is thrown away when the call returns, never
played.

**How the contract is kept.** An engine counts its interruptions.
`interrupt()` adds one, and `speak` stamps every item with the count
current when it was queued. The worker drops an item whose stamp is no
longer current before doing anything else with it: no synthesis, no
callback, and for the end-of-reply marker no firing. The item in hand
compares the count at every step (before synthesis, between the pieces
the model returns, after synthesis, in the audio callback and in the
playback wait) and counts as unspoken once it has changed. Stamping at
the moment of queueing is what makes an interrupt that lands between the
worker taking an item up and starting it impossible to lose, and it is
what lets the reply that follows a stop be spoken, since its items carry
the new count.

Nothing upstream does this work. A stop word is heard on the listener's
own thread, which also dispatches the reply, so it is only heard once
generation has ended and the whole reply (every chunk, the tail and the
marker) is already waiting in the engine's queue.

An interrupted reply never reaches its completion callback, so nothing the
engine does takes the published state out of SPEAKING: the listener, which
heard the stop, publishes IDLE right after calling `interrupt()`
(`state.spec.md`).

Kyutai applies the count in both its threads. The worker drops a stale item
before preparing it and compares the count at every frame the model
produces, so an item still being synthesised is abandoned within one frame
of audio rather than run to its end. The player drops a prepared item whose
count is stale without opening a stream, fires nothing for it, and stops
the stream of the item it is playing; so the items prepared while the
interrupted one played are never heard. It drops an end-of-reply marker
whose count is stale in the same way, and that is where a stop word usually
finds the marker: the worker has handed it over while the last sentence
still plays. With a face attached, `interrupt()` also cuts the face at that
instant (`face/face.spec.md`, "Interrupting"), so her lips stop with her
voice.

## What she owes the echo detector

The engines do not own the echo reference. The listener records what was
said with `track_tts_start`, accumulating the sentences of one reply
(`streaming.spec.md`, "The constraint that shapes everything"). The
engine owes the listener five things:

1. `is_speaking()`, which gates stop-word and echo handling while she
   talks;
2. `get_last_spoken_text()`, which is the current chunk only. It is not
   the echo reference, and must not be used as one;
3. the duration callback, which replaces the detector's estimate with the
   real length when the listener passed one. A streamed reply attaches it to
   its tail only, so earlier chunks are timed by estimate. Kyutai produces
   the tail's length as soon as the tail is synthesised, which can be while
   earlier chunks are still heard: the rate the detector derives from it (the
   words of the whole reply over the tail's length) is then applied to those
   chunks too, which moves its window ahead of the speech, and an utterance
   of four words or fewer is judged by that window alone;
4. the completion callback, which is what opens the hot window;
5. `voice_delay()`, which the listener adds to the start of her speech it
   records, because a watched face holds her voice back by its delay.

Where no exact duration is known the detector estimates from `tts_rate`
(words per minute). Piper, Kokoro, Chatterbox and Kyutai ignore `tts_rate`
themselves: Piper's pace is `length_scale`, Kokoro's is `speed`, Kyutai has
no pace setting, and `tts_rate` and `tts_voice` are accepted by the factory
for interface compatibility only (Kyutai is handed neither).

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

Piper, Kokoro and Chatterbox synthesise each item whole, then play it;
Kyutai plays an item as it is synthesised (see "Kyutai"). Piper, Kokoro and
Kyutai play through a `sounddevice.OutputStream` (mono; Piper in `int16` at the
voice's own sample rate, Kokoro and Kyutai in `float32` at 24 kHz) in
blocks of 1024 frames so an abort lands quickly. Chatterbox writes a temporary WAV
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
the path of a `.safetensors` embedding file, used as it is. A name is
slash-separated segments of letters, digits, dots, hyphens and
underscores; anything else (a drive letter or an absolute path, a
backslash, `~`, `..`, a URL scheme) is refused before the network is
touched, because the repository would be asked for it and a local path
would travel in that request with the user's name and folders in it. A
refused value, and a `.safetensors` path that does not exist, fail
initialisation before the model is loaded: a voice the user named is never
replaced by another. The default is a French female voice.

**Hardware.** The engine needs an NVIDIA GPU (CUDA). Without one,
initialisation fails and the reason is kept in `_init_error`; each item
prints it and speaks nothing. The reason says which of three things is
wrong: PyTorch cannot be loaded (with its own error), this PyTorch was
built without CUDA support, or no NVIDIA GPU was found. There is no CPU
mode: a model of this size synthesises far slower than real time there.

**Compilation.** `torch.compile` needs Triton, which has no Windows build.
When Triton is not installed the engine sets `NO_TORCH_COMPILE=1`,
unless the user already set it, before `moshi` is first imported, because
the package reads the variable at import. The CUDA graphs stay on: with
them the model runs faster than real time, without them several times
slower, so the engine never sets `NO_CUDA_GRAPH`.

**Graph capture.** `moshi` captures its CUDA graphs in torch's default
"global" mode, in which a CUDA call made by any other thread during a
capture (an allocation, a synchronisation) invalidates it and aborts the
process. Whisper keeps the GPU busy in the listener's thread while she
speaks, and every synthesis captures afresh, so the engine has `moshi`
capture per thread instead (`capture_error_mode="thread_local"`), where
only the capturing thread is checked. The engine replaces the `cuda` that
`moshi.utils.compile` takes from torch, and does so before the model's
first synthesis (the warm-up). A capture mode `moshi` passes by keyword is
kept, and a `moshi.utils.compile` with no `cuda` to replace is left as it
is: the engine still works, without the protection.

**Synthesis.** One item is one model call: the text is prepared as a
single-speaker script and conditioned on the voice, with the sampling
temperature and guidance coefficient the model was released with (fixed,
not settings). Each synthesis builds the model's generation state afresh,
CUDA graphs included, so every sentence pays a set-up cost on top of the
audio itself. Frames are decoded to audio as the model produces them: the
first audible frame arrives a second or less after the item is taken up,
whatever its length, and the model then produces audio two to three times
faster than it is played. The exact duration is known once the last frame
is decoded.

**Playback.** The audio is played as it is produced, not once all of it
exists, and the next item is prepared while the current one plays. A second
thread, the player, started with the first item, takes the synthesised
items in the order the worker took them up. For each it waits until about a
quarter of a second of audio has been produced (or synthesis has ended, for
an item shorter than that), then opens an output stream whose callback
takes what the synthesis has produced so far, waits for the stream to end,
and fires the item's completion callback if the audio played to its end.
The worker never waits for the player: as soon as one item's synthesis
ends it takes up the next, so the silence between two sentences is the
opening of a stream, a few tenths of a second. If synthesis falls behind
playback the stream plays silence
until audio arrives, and the underruns are counted in the debug log. An
item whose synthesis fails after playback began stops where its audio
ends and fires no completion callback. `get_last_spoken_text()` names the
item whose playback began last.

**The face.** Kyutai is the engine a talking face follows
(`face/face.spec.md`). The daemon hands it the face's renderer when it builds
it (`create_tts_engine(face=...)`; the other engines take none). An item is
drawn on the face when, at the moment its first quarter second of audio is in
hand, someone is watching the face (a face that failed is not watched) and no
item played without the face is still to be heard: the face would otherwise
be told a moment her voice cannot keep. Any other item plays as above, at
once, and the face costs her nothing. A viewer who arrives in the middle of a
reply sees her lips from the first sentence that becomes ready once what she
said without them has been heard.

- An item drawn on the face is scheduled to be heard at the later of two
  moments: the renderer's delay from then, and the end of the previous item
  drawn on the face, with whatever its voice waited, plus a gap. The gap is
  at least 0.2 s, and at least what the player takes to close one stream,
  open the next and have it heard (0.15 s plus the sound card's latency); a
  stream that starts late anyway widens it by as much, up to 0.3 s more, for
  the rest of the session. A sentence that is ready early enough keeps that
  spacing from the one before; one that is ready later waits for the face's
  delay, as the first of a reply does.
- Its audio is handed to the renderer as it is synthesised, each piece
  stamped with the moment it will be heard.
- The player plays it at that moment. Its stream starts with silence when it
  opens early, timed from when the sound card says the first sample will
  leave the speakers, or from the stream's own latency when the card says
  nothing. The first sound is marked when she is heard, not when the stream
  opens. A stream that starts late moves the rest of the item, and the items
  scheduled after it, by as much.
- When synthesis falls behind playback the voice waits for its audio as it
  does without a face, and the pieces handed over after the wait are stamped
  later by as much. A stretch that reaches the face less than its delay
  before it is heard comes too late to be drawn: there the mouth stays closed
  until synthesis is ahead again. The debug log counts how far the voice ran
  behind its moment.
- A stop resets the schedule: the next reply waits for the face's delay
  only, not for what the stop cut.
- `voice_delay()` is the renderer's delay while it is watched, and 0
  otherwise, so a renderer that failed never holds her voice back.

**Failure.** Initialisation is attempted once and its outcome is kept for
the life of the engine. A `moshi` package (or one of its dependencies) that
cannot be imported, a missing GPU or CUDA build, a model that fails to
load, a voice that cannot be used: each fails initialisation with its
reason, and nothing is said. A synthesis that
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
- An interrupt discards the items queued before it, whether the stop
  lands during synthesis or during playback: none of them is heard, and
  they fire neither their callbacks nor the end-of-reply marker's. The
  item in hand is not heard (or, playing, is cut), and the items queued
  after the interrupt are spoken to the end with their callbacks, so are
  the ones after a second stop. Held for each engine. Piper, Kokoro and
  Chatterbox also never synthesise the discarded items; Kyutai may already
  have prepared some.
- An interrupt landing just as the worker takes an item up is not lost,
  and an interrupt with nothing to stop swallows nothing queued after it.
- An item that a stop word cancelled as the worker took it up publishes no
  SPEAKING, and the listener answers a stop word, heard by the text check
  or by the intent judge, with IDLE.
- Chatterbox does not play audio generated for an item interrupted during
  generation.
- A missing voice that cannot be fetched falls back to a cached fallback
  voice, never to another voice when pinned, and reports the reason when
  there is nothing to fall back to.
- Each listed language gets a voice trained for it, in any spelling, and
  an unlisted one gets the fallback.
- Each engine marks the first sound once its audio stream is running (a
  Kyutai item drawn on the talking face: when it is heard), and not when
  SPEAKING is published.
- Kyutai and a talking face (`tests/test_kyutai_face.py`): with no face, a
  face not ready or nobody watching, she speaks at once and nothing is handed
  over; otherwise her audio reaches the face stamped with when it is heard,
  an item shorter than the first stretch included, the voice waits for that
  moment, the first sound is marked then, the sentences of a reply keep their
  spacing, pieces handed over after the voice waited are stamped later by as
  much, the gap covers the sound card's latency and learns from a late start,
  nothing is drawn while an item played without the face is ahead, a
  face nobody watches any more is not drawn from the next item, the item after
  a synthesis that failed midway follows what was said, a stop cuts the face
  and resets the schedule, `voice_delay()` follows the face, and the factory
  hands the face to Kyutai only.
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
  retry on the next item. A PyTorch that cannot be loaded, one built
  without CUDA and a machine without a GPU give three different reasons.
- Kyutai's `start` begins loading in a background thread named
  `kyutai-init` before anything is said and a second `start` leaves the
  first worker in place; a sentence arriving during the load waits for it
  and the model loads once; a stop that lands during the load abandons the
  item before it is synthesised.
- With Triton absent the compile switch is set before `moshi` is first
  imported, a value the user set is left alone, and the CUDA graphs are
  never switched off.
- `moshi` captures its CUDA graphs per thread, the capture made while the
  model warms up and the one made for each sentence alike, and `moshi`'s own
  capture code asks torch for it; a capture mode `moshi` passes by keyword is
  kept, a second engine keeps the wrapper the first one installed, and a
  `moshi.utils.compile` with no `cuda` to replace still loads. That Kyutai
  can speak while Whisper transcribes on the same GPU is checked by hand
  with a reproduction, as it needs the GPU.
- Kyutai's initialisation ends with one throwaway synthesis (audio
  discarded, nothing queued or played).
- A voice name is resolved through the model's voice repository, a
  `.safetensors` path is used as it is, and a path that does not exist
  fails initialisation before the model is loaded. A value that is not a
  repository name (a local path, a URL, a path going up) never reaches the
  network.
- Kyutai's audio is played at the codec's rate, the duration callback
  gets its exact length once synthesis has ended, and the completion
  callback fires only after the audio played to its end.
- Kyutai starts playing before synthesis has ended, once a short stretch of
  audio is ready, and an item shorter than that plays when its synthesis
  ends.
- Kyutai prepares the next item while the current one plays: its synthesis
  begins before the current one has finished playing, the items are heard in
  the order they were queued, the completion callbacks fire in that order
  each after its own audio, and the end-of-reply marker fires after the last
  of them has played.
- A Kyutai stream that runs out of audio plays silence until more arrives
  instead of ending, an item whose synthesis fails midway fires no
  completion callback, and an item that produced no audio opens no stream.
- `is_speaking()` stays true across the hand-over from one Kyutai item to the
  next, and is false once the last has been played or dropped.
- A Kyutai interrupt drops the items queued before it without their
  callbacks, speaks the ones queued after, abandons a synthesis in flight at
  its next frame, stops the stream being played, never plays an item that
  was prepared but not yet playing, and drops an end-of-reply marker that
  is waiting behind the sentence being played.
- Stopping the Kyutai engine ends both its threads, including while a
  sentence plays with another prepared, and the engine speaks again
  afterwards, even when an item arrived while it was stopping.
- Kyutai marks the first sound once its output stream is running.

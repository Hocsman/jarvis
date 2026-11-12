# Listening Flow Specification v2

This document outlines the voice listening architecture. The system uses a **transcript-first** approach where speech is continuously transcribed, and an LLM intent judge extracts queries with full context.

## Architecture Overview

### Capture format and health

The input stream tries mono at the configured sample rate. Unsupported channel
counts or sample rates trigger bounded retries on the same selected input:
mono, stereo and the device's advertised maximum channel count, at the configured
and native rates, without duplicate attempts. Access and device-availability
errors are not retried as format failures. Both the Windows permission probe and
continuous capture use this negotiation and input selection.

The input comes from `voice_device` through `resolve_input_device`
(`jarvis.utils.audio_stream`), the resolver hold-to-dictate also uses: blank,
`default` or `system` is the system default input; any other text is the first
input-capable device whose name contains it, ignoring case, so output-only
devices are skipped; and a number is an index into the first device list the
process read. The index names the input at that position in that list, and
once the list has been read afresh it selects that input by its exact name,
never whatever device has taken its position since. An index that names no
input on the first list (out of range, or an output-only device) has no
microphone to wait for: listening ends at start-up with
`❌ Selected microphone not found. Choose an available input in Settings.`

Any other microphone that is not there is waited for, never replaced by
another input: a name that matches no input, the input an index names once it
has left the list, or a system default with no input at all. The console says
so once, without the device's name
(`🎤 Microphone not connected: waiting for it...`), and the listener reads the
device list afresh (see §0) every 2.5 s for the first minute of the wait, then
every 15 s; a new loss, or a hold-to-dictate press that misses the microphone,
brings back the 2.5 s readings. Once the same setting resolves, it opens it
and says `✅ Microphone connected: listening`. A microphone that resolves but
will not open or start is waited for the same way, from start-up as later,
unless the error reads as access denied before any wait, which ends listening;
while waiting it is tried again after each reading that holds it, its error
said once per wait. `🎤 Using audio device: ...` is said at start-up and once
per wait, not on every attempt.

On Windows the permission probe runs once the microphone is there, and, once
the session is under way, only on a tick when no stream is open in the
process. A probe that hangs past five seconds ends listening with the
access-blocked box, and so does an error that reads as access denied, but
only at start-up: a device just plugged in can fail with the same codes before
it is ready. Any other probe error is said once per wait, and the microphone
is waited for and probed again.

The wait looks for a stop or shutdown every tenth of a second and ends on it.
While she waits, the reply thread runs, so a reminder or a Claude Code report
is still spoken, but no hot window opens after it, since nothing would hear a
follow-up: her speech ends on `IDLE` (`state.spec.md`). The `Listening!`
banner waits for the first capture.
Multichannel samples are averaged to mono before framing and speech detection.

Frames always span the configured 10, 20 or
30 ms at the actual capture rate; unsupported frame durations use 20 ms. Partial
callback blocks are retained until a complete frame is available and discarded
on audio-state resets. WebRTC VAD receives a 16 kHz mono PCM copy, including when
the hardware captures at 44.1 or 48 kHz. Utterances retain native-rate samples
until resampling for Whisper, preserving their duration.

VAD errors emit a single warning and use the configured energy threshold instead
of silently discarding speech. Capture health is checked every five seconds with
a monotonic clock. Missing callbacks, silent samples, callback errors, PortAudio
status flags and dropped queue blocks are reported outside the audio callback.
Warnings are transition-based; dictation pauses suspend health checks. With
`voice_debug`, diagnostics include callback/frame counts, speech-frame counts,
peak level and capture rate, without saving microphone audio. Linux warnings
point users to PipeWire/PulseAudio recording-source routing.

A microphone lost while she listens is reopened, not abandoned. Once a second
the listener asks whether its stream is still alive: a stream that reports
itself inactive, or raises when asked, is dead, and so is one that has
delivered no callback for 10 s, twice the five-second health warning, so a
stall that the warning reports and that recovers on its own is left alone.
The check is suspended while dictation pauses the listener. A dead stream is
aborted and closed under the lock, the audio still queued from it and any
unfinished utterance are dropped, the console says once
`⚠️  Microphone lost: waiting for it to come back...`, and the listener waits
for the same `voice_device` exactly as at start-up, never another input. The
first reading of the device list comes at once, unless the dead stream lived
less than 5 s: a device that keeps dying as soon as it opens is then read on
the wait's schedule, not reopened once a second. When
the reopened stream delivers its first callback, the console says once
`✅ Microphone back: listening`; the `Listening!` banner is not repeated.

```
┌─────────────────────────────────────────────────────────────────┐
│                         Audio Stream                            │
└───────────────────────────┬─────────────────────────────────────┘
                            │
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
┌───────────────┐                  ┌───────────────┐
│     VAD       │                  │   TTS Output  │
│ (speech gate) │                  │   Tracking    │
└───────┬───────┘                  └───────────────┘
        │
        ▼
┌───────────────┐
│    Whisper    │
│ (transcribe)  │
└───────┬───────┘
        │
        ▼
┌───────────────────────────────────────┐
│     Rolling Transcript Buffer         │
│     (2 minutes, with timestamps)      │
│                                       │
│  Segments include:                    │
│  - text, start_time, end_time         │
│  - energy level                       │
│  - is_during_tts flag                 │
└───────────────────┬───────────────────┘
                    │
                    ▼ (on wake detection)
┌───────────────────────────────────────┐
│          Intent Judge LLM             │
│        (gemma4 or main)          │
│                                       │
│  Inputs:                              │
│  - Transcript buffer (recent)         │
│  - Wake word timestamp (if any)       │
│  - Last TTS text + finish time        │
│  - Current state                      │
│                                       │
│  Outputs:                             │
│  - directed: bool                     │
│  - query: "extracted clean query"     │
│  - stop: bool                         │
│  - confidence: high/medium/low        │
│  - reasoning: "brief explanation"     │
└───────────────────┬───────────────────┘
                    │
                    ▼
┌───────────────────────────────────────┐
│           Reply Engine                │
└───────────────────────────────────────┘
```

## Key Design Principles

### 0. Serialised PortAudio Lifecycle

All stream lifecycle calls (`InputStream`/`OutputStream` construction,
`start`/`stop`/`close`/`abort`) run under the process-wide
`jarvis.utils.audio_lock.portaudio_lock`, shared with the dictation engine,
TTS, and the thinking tune. PortAudio documents stream open/close as not
thread safe; unserialised calls across threads abort the whole app on
Windows. The run loop uses `_serialised_stream` instead
of the raw `with stream:` context manager. Two deliberate exceptions: the
Windows mic-permission probe opens its stream *without* the lock (that open
can hang indefinitely when Windows blocks mic access, and hanging while
holding the process-wide lock would freeze every audio user), and its
timeout path abandons a blocked stream instead of aborting/closing it from
another thread: the check thread may still be inside `start()`/`stop()`
on it, and a cross-thread close is a native use-after-free.

PortAudio reads the device list when it is initialised and keeps it, so a
microphone connected later is not in it. `refresh_input_devices`
(`jarvis.utils.audio_stream`) reads it afresh by terminating and
re-initialising PortAudio, which closes every stream in the process, so it
does so only when none is open. It returns the fresh list, or None when a
stream held it back; a PortAudio that will not initialise again, tried twice,
raises `DeviceListUnavailable`, and the listener logs that once per wait and
tries on the next reading. It holds `portaudio_lock` throughout, and the count
of open streams lives beside the lock (`jarvis.utils.audio_lock`): every
stream the process opens is constructed and tracked in one hold of the lock,
through `open_input_stream` (the listener's and dictation's) or
`open_output_stream` (each speech engine's and the thinking tune's), and
counted open until its `closed` is true; a stream whose state cannot be read
counts as open. The dictation beep's blocking `sd.play` is not tracked: it
opens and closes its stream inside one hold of the lock, so no reading can
fall in between. The permission probe, which opens without the lock, resolves
the microphone and opens it inside `unguarded_open()`, which marks the open
under the lock before it starts and keeps the mark until it returns, so a
reading never overlaps it either, even when the probe abandons a blocked
stream; mid-session it waits for a tick when no stream is open, so it does not
overlap her voice, the tune or a dictation. A count that is wrong can only
hold a reading back: the listener keeps waiting on the old list and logs when
a reading is held back, and once a reading has been held back for a minute
the console says so once, without naming a device
(`⏳ Microphone check waits for other audio to finish...`). Every
`query_devices` call runs under the lock (the settings window's included,
since the daemon can run in the desktop app's process), and a microphone is
resolved and opened in one hold of it: a position in the list is only valid
for the list it was read from. The lock stays innermost and is never held
across a wait.

### 1. Transcript-First

Instead of extracting post-wake-word audio, we:
- Continuously transcribe all speech (VAD-gated)
- Store transcripts with timestamps in a rolling buffer
- Let the intent judge extract the relevant query

**Benefits:**
- Pre-wake-word chatter naturally filtered: "blah blah Jarvis what time is it" → "what time is it"
- Full context available for intent understanding
- Echo detection via multi-layer approach (fuzzy text matching + LLM intent judge)

### 2. Text-Based Wake Detection

Wake word detection operates on the rolling transcript buffer. When Whisper produces text, it is checked for the configured wake word and aliases. This supports arbitrary wake words in any language.

**Names are whole words.** The wake word and each alias are names, found only where they stand as whole words: neither end of the name runs on into a letter, a digit or a combining mark of the word around it, in any alphabet. "Charismatique" does not hold the alias "charis", nor "mañana" the name "ana", whether its "ñ" is written as one character or as an "n" and a combining tilde. Punctuation around a name is no part of a word ("Jarvis, …", "…, Jarvis ?", "«Jarvis»"). The words of a name of several words ("jar is") are found whatever the spacing between them, and not when its last word starts a longer one ("jar isolé"). The exception is text that does not mark where its words end: Chinese characters, kana and Hangul (whose particles join the name before them), and the scripts of South-East Asia (Thai, Lao, Khmer, Myanmar, the Tai scripts). There a name is found among the words around it. A name configured empty is found nowhere.

**Close spellings.** Failing a whole-word match, each token of the text, split on whitespace and stripped of the punctuation and symbols at its ends in any script ("¿jarvas,", "«jarvas»", "「jarvas」"), is compared with every name using difflib's `SequenceMatcher`, and a ratio of at least `wake_fuzzy_ratio` (0.78 by default) counts as the name: "jarvas" for "jarvis". A token that was only punctuation ("?", "…", "-", which French sets apart with a space) is nothing and is weighed against no name, and a name configured empty is no name here either, so an empty wake word does not wake her on every question. A whole token is weighed against a whole name, so a name inside a much longer word scores low ("charismatique", 0.63 against "charis"), but a word one letter away from a short name passes ("charisme", 0.86 against "charis"), and so do ordinary words close to a short alias ("jours", 0.80 against "joris"; "avis", 0.89 against "javis").

**Taking the name out.** When the wake word is what decides that he is speaking to her, the words where a name stands whole are taken out of what he said, with the spacing around them, and every other word is kept as it was heard: "Jarvis, est-elle charismatique ?" is answered as "est-elle charismatique ?". Commas, full stops, question and exclamation marks, semicolons and colons left at the start of what remains are dropped. A token that matched only as a close spelling stays in.

### 3. Context-Aware Intent Judge

The intent judge receives full context and makes intelligent decisions:
- Knows what TTS said → can identify echo vs real speech
- Sees pre-wake-word context → can understand "...what do YOU think, Jarvis?"
- Extracts clean query → removes filler words, false starts

**Gating:** The judge is called only when there is an engagement signal — (a) a wake word was detected in the current utterance, (b) the utterance falls inside (or pending) a hot window, (c) TTS is currently speaking, or (d) his question is still being collected and nothing else took the utterance, where only its `stop` and `noise` are read (see "Text Whisper Makes Out of Silence"). Pure ambient speech skips the judge entirely. This keeps the synchronous audio loop from blocking up to `intent_judge_timeout_sec` on every background utterance, which would otherwise freeze the UI when Ollama is slow or contended.

**Alias normalisation:** Before the transcript is sent to the judge, every configured wake-word alias in each segment is replaced with the primary assistant name (case-insensitive, word-boundary-aware). Aliases are Whisper mishearings of the wake word (e.g. "Jervis", "Jaivis" for "Jarvis"); without this step the small judge model sees the alias, doesn't know it refers to the assistant, and can decide the user is addressing a different person. Normalisation happens at prompt-build time only — the raw transcript buffer is untouched.

**Wake-word removal in the extracted query:** The wake word is addressed TO the assistant, never part of the query content. The judge prompt explicitly instructs removing every occurrence of the wake word from the extracted `query` — at the start, end, or middle of the sentence, including when it sits next to a named entity (e.g. "movie called Possessor Jarvis" → film is "Possessor", not "Possessor Jarvis"). The only exception is when the user is literally talking *about* the assistant as a subject ("tell me about Jarvis"). This is enforced by prompt rule + example rather than post-hoc string stripping, because the LLM already understands the semantic distinction and can handle cases a regex would mishandle (e.g. proper names that contain the wake word, like "Jarvis Cocker").

**Model residency (`keep_alive`):** Each intent-judge request, and each start-up warm-up, asks Ollama to keep the model resident after the call. The default duration is 30 minutes, which avoids cold reloads between utterances, or the server's own residency when `OLLAMA_KEEP_ALIVE` in Yuba's environment sets a longer one: a request that names a residency replaces the server's for that model, so asking for less would cut short what the user set for Ollama (`at_least_server_keep_alive`, whole seconds sent with their unit). When `cfg.low_power_mode` is true, the duration is 1 minute whatever the server's, so the model can unload soon after an active exchange. The trade-off is latency: low-power sessions can pay a cold-load cost after idle periods, while default sessions keep the judge model (default `gemma4:e2b`, ~2 GB) in RAM/VRAM during active voice use.

### 4. The Microphone Is Always Read

The capture callback stamps each block with the time it was captured and queues it. The listening loop reads the queue frame by frame, and an utterance's start and end are read from those stamps, so audio the loop reaches late still says when it was spoken: the echo detector, the hot window and the latency journal compare these times with when she spoke, and the silence that ends a question being collected is counted in them.

**Replies run on a thread of their own.** Once a query is collected, the loop hands it over (`_hand_over`) and goes back to listening; the reply thread runs the queries one after the other, each through the reply engine and then out loud. While she prepares an answer and while she says it, what he says is still heard.

- **A question asked while a reply is on its way waits its turn.** It is answered after the one in hand, never alongside it, and only once her voice has stopped: a reply started while the last is still audible would replace the echo detector's record of it, and that one's window would open over the new one. At most `MAX_WAITING_TURNS` (2) wait; past that the oldest is dropped with a console line, so a wake word repeated by a film cannot stack up replies that would run minutes later.
- **A stop silences the reply.** Heard while a reply is being prepared, said or waits to be, a stop says nothing more of the reply in hand (no sentence streamed after it, no tail), drops the questions waiting behind it and a question still being collected, and interrupts the speech engine. Each stop is counted, and a turn handed over before the count moved is never said: one stopped while it waited for the query lock is not run at all. The engine finishes a turn already running on its own, unheard; its tools still run. A stopped turn is no longer a reply on its way, even while the engine finishes it: the thinking tune and THINKING are not kept for it, and when the engine is done with it, what he has started since (a question being collected or handed over) keeps its tune and the published state. Replies other features handed over are kept.
- **What counts as a stop.** While she speaks, any utterance holding a stop command, as before; while a reply is on its way, or one a stop silenced is still being finished by the engine, or his question is still being collected, an utterance of at most `STOP_WORDS_WHILE_THINKING` (3) words (a stop he repeats, "Jarvis, stop !", is still a stop, never a question), because a stop word inside a longer sentence ("Jarvis, how do I stop the timer?") is a question, and waits its turn. A stop the intent judge reads out of a sentence counts the same.
- **Her spoken question is answered by what he says to it.** A spoken question is settled by the next voice turn (`tools/policy.spec.md`), so a turn whose words he said before he finished hearing it is set aside, with a console line, and the question waits for his answer. A question he never heard, its reply having been stopped, is withdrawn as `expiré` before the next turn runs.
- **Speech ignored while a reply is on its way leaves her thinking.** The thinking tune is that reply's: an utterance that is not for her does not stop it, and the published state stays THINKING.
- **The reply thread is the only one that speaks.** Her answers and the replies other features hand over (`enqueue_reply`, "Speaking Something Produced Elsewhere") leave from it, so the echo detector's record of what she last said has one writer.
- **The turn keeps what was true when he asked.** The language he was heard in, his exact words and when he stopped speaking travel with the query: the loop goes on transcribing while the reply is prepared. His exact words are the last utterance collected into it, whole; one left out, such as a phrase Whisper made out of silence, does not answer for him.

**The loop still waits on Whisper and on the intent judge.** The queue holds `AUDIO_BACKLOG_SECONDS` (30 s), with room to spare over the longest the judge waits for an answer by default (`intent_judge_timeout_sec` plus 2 s), so a stall delays his words instead of dropping them. Nothing empties it on a hand-over, an accepted utterance or a stop: what it holds was said after the utterance just handled. A block that finds the queue full is dropped and counted (`⚠️ Audio capture: queue full; N blocks dropped`). The judge may now run while the reply engine works; on one local Ollama the two can wait on each other.

## Startup & Model Warmup

Before the listener announces "Listening!", it pre-loads every model the first engagement will need. All warmup output is grouped under a single `🔥 Warming up models...` header with indented child status lines, e.g.

```
  🔥 Warming up models...
     🎤 Whisper 'small' loaded on cpu
     💬 Chat model 'llama3.1' ready
     🧠 Intent judge 'gemma4:e2b' ready
     🔊 Voice 'kyutai' ready
🎙️  Listening! Try:
      "How's the weather, Jarvis?"          ← when location is known
      "How's the weather in [your city], Jarvis?"  ← when location is disabled or not configured
      "I just ate a Big Mac, Jarvis."
      "What are you thinking, Jarvis?"
      "What do you know about me, Jarvis?"
```

The weather example adapts to location availability: if `location_enabled` is true, a location source is configured (`location_auto_detect` or a manual `location_ip_address`), **and** the GeoLite2 database is present (`is_location_available()` returns true), the plain form is shown; otherwise the `[your city]` placeholder form is shown so the user understands they must substitute a real city name in their query.

On small models, a caveat line is appended above a more involved example to set expectations (`⚠️ Small model in use (…). Assume it can't infer. Spell out the steps for anything more involved:`). The Chrome MCP tip continues to appear as its own block when the browser tool is detected.

**What gets warmed:**
- **Whisper**: loading the model; additionally a silent-audio transcribe so the first real utterance doesn't pay the cold-decode cost. Both the MLX and faster-whisper backends do this. On faster-whisper that transcribe runs without the voice detector, which would cut the warm-up noise before the decoder ran; with `whisper_vad` on, the detector's model is then loaded on its own (`prepare_voice_detector`, about 0.1 s, kept for the life of the process), which otherwise the first utterance would pay. A detector that cannot load is announced there, at each start-up, and every utterance is transcribed without it ("Text Whisper Makes Out of Silence").
- **Chat model** (`cfg.llm_chat_model`): a minimal Ollama `/api/generate` request with the configured power-mode `keep_alive` so the weights stay resident.
- **Intent judge model** (`cfg.intent_judge_model`): same pattern. If it points at the same Ollama model as the chat model, a single warmup covers both roles (Ollama loads the weights once).
- **Tool router model**: only when `tool_selection_strategy == "llm"`. On a model of its own, it is warmed on its own thread on the router's prompt rather than the weights alone: `warm_up_tool_router` (the reply engine's, so it resolves the model, backend and catalogue as the routing call does) sends the router's system prompt and catalogue with an empty query, with the same `keep_alive` as the other warm-ups, and on Ollama the first exchange then reads only the facts, the dialogue and the sentence (`tools/selection.spec.md`, Router warm-up). When the resolved router model is the chat or intent-judge model, that model's warm-up covers the router and no second request goes to it: their calls ask Ollama for another context size than the router's, so a prompt computed for the router would be gone before its first call, and on a remote provider their warm-up already opens the connection. Its line is `🔧 Tool router '<model>' ready` either way.

- **The voice**: the speech engine the daemon started before the listener, which loads and warms on its own (`output/tts.spec.md`, "The shared surface"). The listener does not warm it; it waits for that warm-up to be over (`wait_until_ready`), so the first reply after a launch is not the one that waits for it: Kyutai's warm-up takes seconds to tens of seconds, and a reply that waited for it would stay silent that long once its text was ready. Its line is `🔊 Voice '<tts_engine>' ready`, or `⚠️  Voice '<tts_engine>' could not load` when the engine settled without a voice (the reason is the engine's to give when she would speak); a voice turned off has no line.

**Low-power mode:** When `cfg.low_power_mode` is true, the listener skips chat, intent-judge, and router warmup threads and prints `🌱 Low power mode: LLM warmup skipped`. Whisper still warms because speech recognition needs to be ready before the listener can accept input, and the voice is still waited for: it loads whatever the mode, and waiting for it costs nothing. The first LLM-backed engagement after startup or idle loads models on demand.

**Concurrency:** LLM warmups run in up to three daemon threads (chat, judge, router) started before Whisper loads, so they overlap with Whisper initialisation, and the voice has been loading since the daemon started it. After Whisper finishes, the listener joins the warmup threads, then waits for the voice, with a **single 60 s budget** counted from the `🔥 Warming up models...` header and shared across them all. If the budget is exhausted, the listener continues (with a `⏳ Some models still warming — continuing anyway` notice) and the first engagement pays the cold-load cost on demand; a voice still downloading on first use is the usual case.

**Best-effort semantics:** Every warmup path swallows its own errors and returns a bool. A failed warmup prints `⚠️ … warmup failed — will load on first use` but never blocks or crashes the listener — voice input is prioritised over startup latency.

## The Three Listening Modes

### 1. Wake Word Mode (Default)

System is waiting for wake word activation.

**Triggers:**
- Text-based detection finds the wake word or an alias in the transcript, as whole words or as a close spelling (see "Text-Based Wake Detection")

**On trigger:**
1. Start thinking beep immediately and publish the state LISTENING
2. Wait for utterance to complete (user finishes speaking)
3. Send transcript buffer + wake timestamp to intent judge
4. If `directed=true` and `query` exists, collect it, and hand it to the reply thread once he has stopped for `voice_collect_seconds`; a verdict that finds nothing but her name leaves the decision to the wake-word check (below)
5. If rejected, stop the beep and publish IDLE

**Her name with nothing after it opens a question with nothing in it yet.** When the wake-word check finds nothing after the wake word, she waits for his question as for any other. Nothing means no letter or digit in any script once the wake word and its aliases are taken out: "Jarvis ?", "¿Jarvis?", "Jarvis…" and "Jarvis ! Jarvis !" are her name alone. The console says `👂 Waiting for your question...` rather than announcing a query, shown at once even through a pipe; the thinking tune plays, never over her voice; LISTENING is published; and what he says before his silence has lasted `voice_collect_seconds` is his question. When the silence runs out with nothing collected (no letter or digit: punctuation Whisper makes out of the quiet is nothing either), nothing is handed over and the wait ends where he can see it: `💤 Nothing after the wake word, returning to wake word mode` is printed, and the published state goes back to what she is doing. While she speaks she is SPEAKING; while a reply of hers is on its way she is THINKING, and the tune, which is that reply's, plays on; otherwise the tune stops and she is IDLE. A reply stopped since it was handed over is not on its way (see "The Microphone Is Always Read"). While he dictates, nothing is published: the dictation's state stands. The hot window comes to the same wait when a wake word that can only be his leaves the decision to the wake-word check (see "Hot Window Mode").

**The intent judge does not answer her name as a question.** Outside the hot window, a directed verdict that finds nothing but her name, in what he said and in the `query` it reads, decides nothing, whatever its confidence: the wake-word check does, and she waits for his question as above. A query that holds more than her name is a question the judge read in his earlier words, and it is answered.

**A sentence heard while a question is still being collected completes it**, in this mode and in the hot window. Every path that accepts an utterance adds it to the question being collected, never puts it in its place, and she answers both: "Salut Yuba, comment tu vas ?" then "Tu m'entends Yuba ?" is one question. A collection that keeps being completed still ends after `voice_max_collect_seconds`. The silence that ends it is counted from the end of the last utterance collected, in the time its audio was captured: neither Whisper nor the judge adds to the wait, and audio the loop reads late, after a stall, is measured in its own time, so his next words still complete the question. Whatever joins it is marked processed, like every accepted segment, so the judge never sees it again as context it could restate. A stop heard while it is collected cancels it instead (see "The Microphone Is Always Read"). What he adds needs no wake word, so Whisper's inventions reach the question too: one the judge calls noise is left out (see "Text Whisper Makes Out of Silence").

### 2. Hot Window Mode

After TTS finishes, allow wake-word-free follow-up.

**Activation:** `echo_tolerance` seconds after TTS ends (allows echo to settle)

**Duration:** Configurable (default: 3 seconds)

**Behaviour:** Speech first passes through an early fuzzy echo check (rapidfuzz `partial_ratio`, threshold 70, with word-count guard to avoid catching mixed echo+speech). Pure echo is rejected with `🔇 Heard (echo)` **without calling the intent judge** — this keeps echo rejection instant and prevents it from blocking the audio loop. The early check stands aside while she speaks (see "During TTS") and while the thinking tune plays; with the tune playing, pure echo heard in the hot window reaches the judge, and the checks after its verdict set it aside with the same line. The hot window timer is **not** reset on echo rejection. Non-echo speech is sent to the intent judge, and if the judge rejects it, the rejection is overridden: non-echo speech in the hot window is accepted as a follow-up query. An echo verdict is overridden only on positive evidence (see "Overriding an echo verdict takes positive evidence"). Words that could be her own voice coming back, the ones the judge is shown "(during TTS)" (see "During TTS"), are taken by no override: whatever words the judge finds for leaving them out, a rejection that does not say echo or a directed verdict with no query, they are set aside as an echo verdict that stands sets them, with `🔇 Heard (echo)`. The listener decides it on the text as it stands once an echo prefix is stripped. In the hot window the judge's confidence changes none of this (it decides only whether words it calls said to someone else are set aside, see "Words He Says to Someone Else"): a verdict it is unsure of goes through the same overrides as one it is sure of, a directed verdict with no query takes what he said whatever its confidence, and every path a verdict takes there ends in a console line that says what became of the utterance, a stop aside: that one she obeys out loud. The judge's confidence is "low" whenever its answer leaves it out. The wake word is his call in the hot window as outside it, where it can only be his: an utterance carrying such a wake word that the verdict extracts no query from, her reply coming back whole aside, goes to the wake-word check, which answers what was heard with the wake word taken out, so a quiet answer of his that could be hers is still answered when he names her. A wake word can only be his when the wake-word check finds it in the utterance, finds none in her reply, and still finds it once the words of her reply are taken out. Her reply can name her, or say a word the check takes for her name ("avis" for "Jarvis"; with the default aliases, common words such as "jours" or "chairs" too), and coming back that word can take any spelling Whisper gives her name, so after such a reply no wake word heard can be told from hers, his quiet answer naming her included: it is set aside with the other words that could be hers. Whisper can also bend a word of hers into one ("servi" heard "servis", close to the alias "jervis"): that word is still hers as long as it stays as close to hers as the word test of "Overriding an echo verdict takes positive evidence" asks, and so is his wake word when it comes that close to a word of hers ("jarvis" to the "avis" of her "d'avis"). Two of her words merged into one ("Dave is" heard "Davis") or a sound spelled another way ("à vie" heard "avis") go beyond that test and pass for his. With no verdict (a timeout, or the judge's back-off after a connection error) the hot window tells his words from hers as it does behind a verdict: her reply coming back whole is set aside with `🔇 Heard (echo)`, and so are words that could be hers, unless they carry a wake word that can only be his, which leaves the decision to the wake-word check; anything else is his and is answered as he said it, a wake word in it included. Outside the hot window, a verdict below high confidence leaves the decision to the wake-word check. Speech, not text: a segment the judge calls noise is not speech, and nothing overrides that verdict (see "Text Whisper Makes Out of Silence"). And speech to her: words the judge is sure he said to someone else are not answered either (see "Words He Says to Someone Else").

**Mixed echo+speech handling:** When Whisper merges TTS echo and user speech into one chunk (e.g. mic picks up TTS then user speaks), the word-count guard detects the extra content and lets it through to the intent judge. The judge extracts the user's actual query from the mixed transcript. Post-judge echo checks also use the word-count guard and verify the judge's extracted query isn't itself echo before rejecting.

**Early salvage for echo-prefixed follow-ups:** Before the early fuzzy check rejects a chunk as pure echo, the listener calls `cleanup_leading_echo` to strip any TTS-tail prefix. If exact-word cleanup fails (for example because Whisper mis-transcribed the first echo word — *"explores"* → *"laws"* — breaking the word-level comparison), the listener falls back to `salvage_after_echo_tail`, which scans heard-text word boundaries right-to-left looking for the rightmost 5-word window that fuzzy-matches the TTS tail (`partial_ratio >= 85`) and keeps everything after it. This preserves short follow-ups (*"Who made it?"*) that the existing fuzzy-prefix salvage would otherwise truncate by one word because it prefers the shortest suffix. If the surviving remainder has at least `EchoDetector.min_salvage_words` words (default 3), carries words she did not say in a voice (the two proofs of "Overriding an echo verdict takes positive evidence"), and its voice went on after hers (see "During TTS"), it replaces the transcript segment text and is treated as the user's follow-up. The same minimum-word threshold is shared by the during-TTS and post-TTS merged-chunk salvage paths. The during-TTS one keeps a remainder on that count alone. So does the post-TTS one, unless the utterance's voice stopped with hers (see "During TTS"): from such an utterance it keeps nothing, and the utterance stays whole, and flagged. In the hot window the overrides still set a remainder aside when it could be hers.

**Timestamp-based detection:** `was_speech_during_hot_window(utterance_start_time, utterance_end_time)` compares the utterance's time range against the hot window's time span (from schedule to expiry). This eliminates race conditions between slow Whisper transcription and the expiry timer — if the user started speaking during the window, it counts as hot window input regardless of when the transcript arrives. Also handles **overlapping utterances** where VAD triggered during TTS (mic picking up echo) but the utterance extended into the hot window period.

**`could_be_hot_window` (intent judge context):** Derived from timestamp comparison — returns True if the hot window is active, activation is pending, the utterance started within the window span even after expiry, or the utterance overlaps with the span (started before, ended during).

**Expiry:** Timer-based, guaranteed to fire even if no audio. The `hot_window_seconds` clock starts the instant the window opens, before the state is published and the console is told, so the window is the same length however slow that announcement is (a state file on a busy disk, a console that blocks). An announcement the clock has already outrun is skipped.

### 3. During TTS

While TTS is playing, echo rejection and stop commands are handled with fast text-based checks (no LLM). This prevents self-loops where the mic picks up TTS output. After TTS finishes, the intent judge takes over.

**Stop detection:**
- Text-based: Check for "stop", "quiet", "shut up", etc., while she speaks and while a reply is on its way (prepared, or waiting its turn); while she only thinks, in an utterance of at most three words
- Intent judge can also detect stop commands, with the same effect
- However a stop is heard, the listener interrupts the engine, which drops the rest of the reply, and publishes IDLE: the interrupted reply is unheard, so its completion callback never fires and nothing else would take the published state out of SPEAKING (`state.spec.md`, `tts.spec.md` "Interrupting"). No hot window opens. Nothing more of the reply being prepared is said, and the questions waiting their turn are dropped (see "The Microphone Is Always Read").

**Echo handling:**
- The detector times her words from when the listener handed her the reply,
  plus the speech engine's `voice_delay()`: a talking face holds her voice
  back by its delay (`output/tts.spec.md`, "Kyutai").
- A transcript that could be her own voice coming back is flagged
  `is_during_tts=true`, shown to the judge as "(during TTS)", and the judge
  is told never to extract a query from it. A verdict that extracts one
  anyway is answered, as any directed verdict with a query is (see "Hot
  Window Mode"). Timing says when it could be hers: heard
  while she speaks, or started (the voice detector's pre-roll included)
  before her last word plus `echo_tolerance`.
- Timing does not decide alone. His answer to a question she ends her reply
  on often starts within that tolerance, so an utterance that carries words
  she did not say, loud enough to have been spoken (the two proofs of
  "Overriding an echo verdict takes positive evidence"), in a voice that
  went on after hers, is never flagged, whatever its timing. Her own words
  coming back keep the flag, as long as Whisper bends no more of them than
  that test allows (40% of the words heard without a close match in her
  reply), and so does a phrase Whisper makes of the quiet right after her:
  the first carries no words she did not say, the second no voice.
- Her voice coming back stops with hers. Through the speakers it clears the
  energy bar, as she speaks and as she stops, and Whisper can make words she
  never said of it (a figure heard as an hour, a unit as a name), so both
  proofs can hold on her own voice. After her last word, an utterance whose
  voice stopped before her last word plus `echo_tolerance` keeps the flag
  whatever its words: its voice stopped when the frame that closed it was
  captured, less the silence the voice detector waited for, counted in
  frames of the rate the microphone is read at. Her last word is when her
  reply finished playing, as its completion callback stamps it
  (`output/tts.spec.md`, "Speaking state and completion"), so quiet the
  engine plays after her last sound counts as hers: for Kyutai, the pause
  the model leaves after its last word and the time its stream takes to
  drain. The listener stamps it as it opens the hot window: with
  `hot_window_enabled` off no last word is stamped, and only her speaking is
  her time. His answer begun over her last word goes on after it and is not
  flagged. Kept on purpose: a short word of his said over her last word and
  stopping with it cannot be told from her voice, and is flagged with it; a
  wake word in it that can only be his still decides (see "Hot Window
  Mode"). A short answer said in the quiet after her last sound, whose voice
  stops before that quiet ends plus the tolerance, is flagged the same way.
  While she speaks her last word is not known, so there, as for an utterance
  whose end is not known, the flag rests on the two proofs. Her own voice
  clears the energy bar, so for her voice coming back they come down to its
  words.
- Intent judge uses this context to identify echo
- All of this is read on the utterance once the sentences of her reply that
  open and close it are out (see "Her sentences at the edges of what he
  said"), so his words between two sentences of hers are weighed alone.

## Rolling Transcript Buffer

### Design

```python
@dataclass
class TranscriptSegment:
    text: str              # Transcribed text
    start_time: float      # Unix timestamp when speech started
    end_time: float        # Unix timestamp when speech ended
    energy: float          # Audio energy level
    is_during_tts: bool    # Could be her own voice coming back (see "Echo handling")

class TranscriptBuffer:
    max_duration_sec: float = 120.0  # Ambient speech context for intent judging
```

### Memory Alignment

- **Transcript buffer** (`transcript_buffer_duration_sec`): Rolling raw ambient speech. Separate and potentially longer — in group conversations, 2+ minutes of context lets the intent judge synthesise a complete query with relevant information when someone decides to involve Jarvis later in the conversation.
- **Short-term memory** (`dialogue_memory_timeout`): Processed Jarvis interactions (user queries + assistant responses) of the current conversation, which a pause of `dialogue_memory_timeout` ends. Its latest part reaches the model word for word, within `dialogue_context_max_chars`, and its earlier part as the diary's summary (`reply.spec.md`, Recent Dialogue Context). The same duration drives the forced diary update interval.
- **Long-term memory (diary):** Forced update when unsaved messages reach `dialogue_memory_timeout` age, so a long conversation is summarised as it goes. Enrichment retrieves any relevant context from other conversations.

### Methods

- `add(text, start_time, end_time, energy, is_during_tts)`: Add segment
- `get_since(timestamp)`: Get all segments since a timestamp
- `get_around(timestamp, before_sec, after_sec)`: Get segments in time window
- `format_for_llm(segments)`: Format for intent judge input
- `prune()`: Remove segments older than max_duration

## Intent Judge

### Context Duration & Query Synthesis

The intent judge receives the full transcript buffer (default: 120 seconds / 2 minutes) and **synthesizes a complete query** using conversation context.

This enables Jarvis to **chime into ongoing conversations** between people. When someone asks "Jarvis, what do you think?", the judge uses context to understand what they were discussing and creates a complete, actionable query. Vague references like "that", "it", "this", "they" in the current segment are resolved using previous segments in the buffer (e.g. "I think dinosaurs are cool" + "What do you think about that Jarvis?" → "what do you think about dinosaurs being cool").

**Multi-topic disambiguation.** Real buffers often contain interleaved threads from ambient chatter — e.g. a sports conversation running alongside a purchase discussion. When the wake-word segment uses a vague reference or a topic-less question ("what's the price", "how much does it cost"), the judge must pick the thread whose subject fits the question's grammar (a purchasable thing for "price", a release for "when did it come out") and ignore unrelated threads. When resolving to a sub-item ("pro model", "the red one"), the query must include the parent noun/brand so it remains answerable without the transcript. The grammar-matching behaviour lives entirely in the judge's system prompt (no runtime code branch) and is exercised by the `buried_target_*` eval cases in `evals/test_intent_judge.py` — if the small model regresses on this behaviour, those evals catch it.

**Hot-window override.** In hot-window mode the user is always treated as directed; the topic-less / vague-reference heuristics above are subordinate. Short follow-ups like "tell me more", "and?", or "what else" stay directed rather than being rejected as undirected chatter, because the hot window only opens after a completed Jarvis exchange. The rule is about what the user says to her, so it covers neither a segment that is not speech at all nor words he plainly says to someone else (see "Words He Says to Someone Else").

**A remark to her is directed.** A greeting, thanks or a compliment said to her in the hot window is his speech to her, directed like a question or a statement, even when it asks for nothing: the prompt lists them with questions and statements as what is valid and directed there. Unmarked, such a remark is answered on any verdict but the two rejections the hot window never overrides: a not-directed noise one (see "Text Whisper Makes Out of Silence", where his side wins a contradiction) and a not-directed aside the judge is sure of (see "Words He Says to Someone Else"); marked, only a directed verdict with a query answers it. Right after her reply, with his question already answered and so no longer shown, a remark can read like the closing lines Whisper invents or, when it takes up words of her reply, like her reply coming back; a judge that calls it noise then leaves it unanswered, and how often it does is the judge model's (`docs/llm_contexts.md`). The prompt does not say that a remark needs no request: the judge calls her own words, misheard by Whisper and shown unmarked, noise because they hold no request, and every wording measured that said so led it to take her misheard words for his. The line names thanks where NOT SPEECH names a lone "thank you" among the text Whisper invents, so a lone "Thank you." after an English reply, Whisper's commonest invention there, can be taken for thanks to her and answered; a line that keeps it out measured no gain on the remarks. `evals/test_intent_judge.py` measures what the listener does with invented remarks shown as it shows them and with follow-ups that name another person, against what it must leave alone: his words to someone else, Whisper's inventions, her own sign-off misheard and a lone "Thank you." after an English reply (`-k a_remark_to_her`).

**Declarative statements addressed to the wake word.** Segments where the user shares information, feelings, or an action with the assistant — e.g. "Jarvis, I just ate a burger from McDonald's", "I'm feeling a bit tired today, Jarvis", "my flight got cancelled, Jarvis" — are directed and must be extracted verbatim (wake word removed) as the query. The wake word can appear at the start, middle, or end of the segment; position does not affect directedness. The judge must not reject these as "not a command or question": any segment where the wake word is used to address the assistant (as opposed to a narrative mention like "I told my friend about Jarvis") is directed, regardless of sentence mood.

**Imperative resolution.** The same mechanism covers imperatives that refer to a prior unanswered question. If a prior segment contains a question and the wake-word segment is an instruction like "answer that", "respond to that", "reply to that", "address that", "answer my question", or "go ahead and answer", the query is the prior question itself — not the literal imperative. Whisper tense variants of these imperatives ("answered that", "answers that", "answering that") are treated the same. If the current segment contains both an imperative and a new explicit question, the new question takes priority.

**Multi-person conversation example:**
```
[12:28:30] Person A: "I wonder what the weather will be like tomorrow"
[12:28:45] Person B: "Yeah, we should check before planning the picnic"
[12:29:00] Person A: "Jarvis, what do you think?"
```

The intent judge synthesizes: `"what do you think about the weather tomorrow for the picnic"`

### Text Whisper Makes Out of Silence

Whisper turns silence, breath and room noise into text: mostly the closing lines of videos and subtitle credits ("Thank you.", "ご視聴ありがとうございました", "Продолжение следует...", "Sous-titres réalisés par la communauté d'Amara.org"), often in a language nobody in the room speaks. The confidence filters below catch some of it, not all, and naming the one language he speaks (`whisper_language`) stops the ones in other languages at the source.

Most of it never comes out at all. With `whisper_vad` on, the default, faster-whisper's own voice detector (Silero, `vad_filter`) cuts the stretches of an utterance that hold no speech before the decoder hears them, so an utterance of breath, a chair or the room comes back empty and goes no further, its language included: with no `whisper_language`, Whisper still names a language for an utterance the detector emptied, guessed from the silence it pads it with, so only an utterance heard as something sets the language last heard. The filters after Whisper cannot do that work: `large-v3-turbo` gives such an invention a `no_speech_prob` of 0.0. Measured on one real room and voice (recordings kept on that machine and deleted after): on 18 windows of the room's noise the decoder alone wrote 17 phrases and with the detector none, and of 24 calls with her name it found the name in 22 alone and 23 with the detector; two of the 24 sentences came back less exact (16 word for word without it, 14 with it), a difference that sample cannot tell from chance. MLX Whisper has no voice detector, and on it the setting changes nothing. Dictation honours the same setting (`dictation.spec.md`). An utterance heard as nothing leaves a debug line with its length, the setting and whether the detector is on (none on MLX), so a detector that cut his voice shows in the log.

**A detector that fails never takes the utterance with it.** The detector is an ONNX model of its own, read from faster-whisper's `assets` folder through onnxruntime; when either is missing (a build that left the files out, an install without onnxruntime) every call that asks for it fails. The listener and dictation both call Whisper through `jarvis/listening/whisper_vad.py`:

- A call that fails with the detector is made again without it, and the utterance is heard from that second call.
- The detector is then loaded on its own. When that fails, the detector was what failed: `⚠️  Whisper's voice detector is unavailable (…): transcribing without it, so phrases may be invented out of noise` is printed, and no call asks for the detector again until the listener next starts. The failure is the process's, as the detector's model is: the listener and dictation share it, so it is printed once whichever of them saw it.
- Each start-up of the listener tries the detector afresh (`prepare_voice_detector`). The bundled app restarts her inside its own process (Settings, Restart Jarvis): that start-up uses a detector that loads by then, and prints the line again for one that still cannot.
- When the detector loads, the failure was something else (with no `whisper_language`, the call with the detector also detects the language on the GPU, where an error can pass) and the detector stays on.
- When the call without it fails too, the detector is not blamed: the utterance is lost as any transcription error loses it, and the detector stays on.
- A model that takes none of these arguments (`TypeError`) is called with the language alone.

The judge's verdict carries `noise`: true when the current segment is such an invention rather than words someone said, and then it is not directed. Words a person said are never noise, however short, thankful, odd or misheard; in doubt the judge says false, and a model that leaves the field out has said false.

- **In the hot window nothing overrides it.** A not-directed verdict with `noise` on an utterance that could be a hot-window follow-up is final: the listener marks the segment processed, so the judge never sees it again as context, prints `🔇 Heard (not speech)`, stops the thinking beep and returns. No hot-window override runs (non-echo speech, an echo verdict with words she did not say and a voice, the grace after the window), because they exist to keep his words, and a hallucination produces words nobody said that the energy test can still take for a voice while she is speaking.
- **While his question is collected it does not join it.** An utterance heard then that no other path took (no wake word, no hot window) would complete his question with no wake word, as a follow-up would, so it is judged as one, in hot-window mode. A verdict already given on it in hot-window mode is used as it is; one given in wake-word mode, while she speaks, read it without the question and without a follow-up's rules, so the judge is asked again. The question collected so far is put back in the transcript, just before the utterance judged: its segments are processed, and an addition read without it, misheard ("et pour à près de main aussi"), looks like something Whisper invented. Only `stop` and `noise` are read, never a query, so the question shown cannot be restated into one.
  - A stop cancels the question, as a stop word does.
  - Not directed and `noise`, the segment is marked processed, `🔇 Heard (not speech)` is printed and it is left out. The thinking tune of his question plays on, and the collection's silence is still counted from the end of his last words, so a phantom never holds his question back.
  - Any other verdict, or none, completes the question. With none, the judge's unavailability is printed.
  - Utterances of three words or fewer are judged here even while she speaks, though the judge skips them in her speech otherwise: Whisper's inventions are that short.
  - It costs a judge call for each utterance that gets this far while a question is collected, and a second one when the first was given in wake-word mode.
- **Elsewhere the verdict changes nothing.** The judge only runs there on a wake word or while she speaks, and Whisper does not invent her name, so a doubtful verdict still falls through to the text wake-word check as any other.
- **His side wins a contradiction.** A verdict that says both `noise` and directed is treated as directed.
- **The window stays open.** Like an echo, a noise verdict does not touch the hot window, so his real follow-up after it is still a follow-up.
- **Fails open.** With no verdict (judge unavailable) or a model that never sets `noise`, nothing is left out as noise: the hot window and the question being collected behave as without this rule. In the hot window, words that could be hers are still set aside (see "Hot Window Mode").

How well it works is the judge model's: a small local model can miss what a larger one catches, and a miss is answered as before. `evals/test_intent_judge.py` holds the inventions and the real follow-ups it is measured on (`TestIntentJudgeKnowsSilenceFromSpeech` on the suite's small model, `EVAL_INTENT_JUDGE_ARMS` on any), and `docs/llm_contexts.md` the measurement.

### Words He Says to Someone Else

He is not always alone: a guest in the room, someone on a call, the people watching a live stream. The hot window takes what he says without her name, and what he says then to one of them is not for her: answered, it has her reply to a thanks he gave someone else, over his conversation.

The judge's verdict carries `aside`: true when the current segment is plainly said to another person (he calls them by name, answers them, or talks to them about her), and then it is not directed. A follow-up that addresses nobody else is for her, and so is a request to her that mentions another person ("tell Sam I am running late"); in doubt the judge says false, and a model that leaves the field out has said false.

- **In the hot window a sure aside is final.** A not-directed verdict with `aside`, given with high confidence, on an utterance that could be a hot-window follow-up is not overridden: the listener prints `🔇 Heard (not to her)`, stops the thinking beep and returns, unless one of the cases below holds.
- **An aside the judge is not sure of is not one.** Below high confidence the verdict goes through the hot-window overrides as any other rejection of speech, and he is answered. It is the one place in the hot window where the judge's confidence decides: its doubtful asides are mostly his words to her (a warm remark said with nobody named, a question about someone), its sure ones his words to someone else, and a follow-up left unanswered costs him more than an aside answered.
- **Her name is his call.** An utterance that holds the wake word or an alias, spelled as configured and as whole words, is never set aside this way: the paths that weigh a wake word in the hot window decide, as for any other verdict. A word that is only close to her name does not count here, though the wake-word check takes it for her name elsewhere: a guest can be called one ("Boris", "Chris" with the default aliases), and the judge, which is shown her configured names as her name, was sure. A spelling of her name that is not configured is therefore not his call on a sure aside; adding it to `wake_aliases` makes it one.
- **His answer to her question is never an aside.** While a question she asked before acting waits for his spoken answer (`tools/policy.spec.md`, "Asking"), what he says goes on to the reader of answers whoever he names in it. Set aside, his "no" would leave the question open for a later turn to answer.
- **What he adds to his question joins it.** While his question is collected nothing is set aside this way, whatever the verdict says of an aside (see "Wake Word Mode").
- **Words that could be hers are her echo.** An utterance the listener marks "(during TTS)" keeps the paths that set it aside with `🔇 Heard (echo)`.
- **It stays in the transcript.** The segment is not marked processed, so when he then asks her about what was said the judge has it to resolve his question. Not everything called an aside is his: the judge also calls her own sign-off, misheard by Whisper as words she never said and shown unmarked, an aside (`docs/llm_contexts.md`), and that segment stays as context too.
- **The window is left as it is.** It is neither closed nor extended: a follow-up he begins before it expires is still a follow-up.
- **His side wins a contradiction.** A verdict that says both `aside` and directed is treated as directed.
- **Elsewhere the verdict changes nothing.** Outside the hot window nothing is answered without her name, whatever the verdict says of an aside.
- **Fails open.** With no verdict, or a model that never sets `aside`, his words in the hot window are answered as without this rule.

How well it works is the judge model's, and both of its mistakes cost him something: an aside it misses is answered, and a follow-up it takes for an aside is left unanswered. `evals/test_intent_judge.py` measures both on words to someone else and on the follow-ups that look most like them, the ones that name another person, ask her to pass something on, or open on the word an aside opens on (`-k a_remark_to_her`), and `docs/llm_contexts.md` holds the measurement.

### Input Format

```
Transcript (last 120 seconds):
[12:28:30] "I wonder what the weather will be like tomorrow"
[12:28:45] "Yeah, we should check before planning the picnic"
[12:29:00] "Jarvis what do you think"

Wake word detected at: 12:29:00.8 (text-based)
Last TTS: "The weather is sunny and 72 degrees"
TTS finished at: 12:28:02
Current state: wake_word_mode
```

### Output Format

```json
{
  "directed": true,
  "query": "what do you think about the weather tomorrow for the picnic",
  "stop": false,
  "noise": false,
  "aside": false,
  "confidence": "high",
  "reasoning": "synthesized context from conversation about weather and picnic"
}
```

**A directed verdict is acted on once it is decided.** The judge's answer is streamed, and its reasoning comes last. Nothing acts on the reasoning of a verdict that says she is spoken to, so that verdict is acted on as soon as `directed`, `query`, `stop` and `confidence` are written and the reasoning's key follows them: what precedes the key is closed into an object and parsed as the whole answer would be, its reasoning empty. When the whole answer parses, it gives the same verdict; what comes after the key is not read for the decision, so a tail the token cap cuts or that does not parse leaves the verdict standing. The debug log gets the reasoning once the answer is whole, on a line of its own, and says so when the whole answer reads otherwise on one of those fields; the debug line for a directed verdict whose wake word the check does not find prints its reasoning empty. A verdict that says she is not spoken to is read whole, since the echo check reads its reasoning, and so is one where any of those fields is missing before its reasoning. A backend that does not stream (Ollama) answers whole, and its verdict is read then. The judge waits for an answer `intent_judge_timeout_sec` plus 2 seconds at most, wall clock: a streamed call is bounded per read only, so a stream that keeps trickling is not cut by the HTTP client, and the judge stops waiting on it while its thread reads it to the end. Past that there is no verdict, and the judge does not back off.

### Multi-Layer Echo Detection

Echo detection uses a layered approach for reliability:

1. **Fuzzy text matching (safety net):** `rapidfuzz.fuzz.partial_ratio` compares transcript against last TTS text. Score ≥ 70 = echo. This runs before the intent judge and catches obvious echoes quickly, including in the hot window directed path.
2. **Intent judge (contextual):** Receives `last_tts_text` and timing context. Can identify echo even when fuzzy matching misses subtle cases, and can extract real user speech from mixed echo+speech chunks.

The fuzzy check acts as a fast, reliable safety net. The intent judge provides deeper understanding but may be unreliable with smaller models (e.g. gemma4).

3. **Overriding an echo verdict takes positive evidence.** When the judge says an utterance is her own echo, that verdict stands unless the utterance carries words she did not say: the share of heard words with no close match in her reply must exceed 40%. Her reply is the one the judge was shown: a reply that starts while the judge is thinking does not change what its verdict is weighed against. The override exists for barge-in, and barge-in is what that measures.

   Absence of a fuzzy match is not evidence of speech. `partial_ratio` looks for a contiguous window, so an echo Whisper decimated — words dropped from the middle, others corrupted — stops matching and used to read as new content, which had her run a full turn on her own voice. Per word nothing separates the two: `même` against `semaine` scores what a genuinely new word scores. Over the whole utterance it separates cleanly, and stably — measured on real transcripts, the two echoes sat at 0% and 18% and barge-in at 62% and 100%, unchanged whether the per-word bar is 70, 80 or 90.

   **And it takes a voice.** Text alone cannot separate a hallucination
   from a barge-in: Whisper turned background noise into "Thank you."
   over her French reply, the words-she-did-not-say test answered yes
   because there were, and she replied to it. A hallucination produces
   words *nobody* said, so that premise fails on it.

   What separates them is that one of them moved air. Overriding an echo
   verdict therefore needs both: words she did not say, and an utterance
   whose energy rises above the room's quiet level recorded when she
   started speaking, by the same `energy_spike_threshold` the detector
   already applies to this question in the post-TTS cooldown. Either test
   alone was fooled — text by a hallucination, energy by a door slamming.

   **The floor is the room, never her nor him** (`room_floor.py`). The
   microphone's last second as her reply starts is rarely the room: the
   thinking tune plays until her reply is handed over, and a reply spoken
   as it is written (`output/streaming.spec.md`) queues its first sentence
   while the tune still plays; the end of his question can be in it; a
   reply from elsewhere said the moment another ends (see "Speaking
   Something Produced Elsewhere") follows her voice. A floor read there
   sits at several times the room's level, and his soft answer right after
   her last word, louder than the room and softer than the tune, falls
   under the bar it sets and is shown to the judge as her echo. So:

   - The listening loop measures each frame's level as it reads it. A
     frame counts as the room when:
     - nothing of hers plays as the loop reads it, neither the thinking
       tune nor her voice;
     - it was captured at most `ROOM_FRAME_MAX_AGE_SECONDS` (0.5 s) before.
       The loop knows what plays when it reads a frame, not when the frame
       was captured, and it falls behind while Whisper or the judge holds
       it;
     - it holds a sound: samples of exactly nothing are a muted or missing
       input, not a quiet room.
   - The floor is the `ROOM_FLOOR_PERCENTILE` (10th) percentile of the last
     `ROOM_FLOOR_WINDOW_SECONDS` (15 s) of such frames, however long ago
     they were heard. The stretch is longer than the longest utterance the
     voice detector keeps whole at the default `max_utterance_ms` (12 s),
     so his words, a cough or a door fill less than nine-tenths of it and
     sit above the percentile; a steady room varies by a few percent from one
     frame to the next, so the percentile lies within a few percent of
     its mean. It follows the room within one stretch of quiet when a fan
     starts.
   - Before `ROOM_FLOOR_MIN_SECONDS` (1 s) of such frames has been heard
     there is no room to read: the floor is `FALLBACK_FLOOR` (0.0045), and
     a debug line says so.

   The floor is taken once per reply, when her first sentence is queued,
   and holds for the whole reply: a reply is weighed against the room as
   it began.

   Both together are fooled by her own voice when Whisper makes words she
   never said of it, since her voice through the speakers clears the energy
   bar, as she speaks and as she stops. After her last word, when the voice
   stops tells that one from him: her voice coming back stops with hers, so
   an echo verdict on an utterance whose voice stopped before her last word
   plus `echo_tolerance` stands, whatever its words (see "During TTS").
   While she speaks nothing does, and a hallucination there the judge names:
   a verdict that calls the utterance noise is never overridden (see "Text
   Whisper Makes Out of Silence").

   Fails open: with no TTS text to compare against, everything is the user's.
   The energy half fails open twice over — no baseline, or an energy of
   zero, which is the listener's "not measured" value rather than a claim
   of silence.

   The same two proofs, with when the voice stopped, decide which
   transcripts are flagged as possibly hers (`is_during_tts`, see "During
   TTS"), so the judge is not told in the first place that words with both,
   in a voice that went on after hers, are her echo, and no hot-window
   override takes the others heard in her time, whether the judge gave a
   verdict or none. A wake word among those words is weighed with them,
   unless it can only be his (see "Hot Window Mode"): a verdict that
   extracts nothing from an utterance carrying such a wake word is not
   weighed here at all, wherever in the hot window it was heard, and the
   wake-word check decides.

Example:
```
TTS: "The weather is sunny and 72 degrees"
TTS finished: 12:30:14

Transcript:
[12:30:15] "The weather is sunny and 72 degrees" ← Echo (fuzzy score 100, rejected)
[12:30:18] "Ni hao" ← Real speech (fuzzy score < 70, sent to judge)

Judge output: {"directed": true, "query": "Ni hao", "reasoning": "New speech directed at assistant"}
```

## Early Feedback (Beep & Published State)

To minimise perceived latency, audio and visual feedback starts **immediately after Whisper transcription**, before the intent judge runs. The visual half is the assistant's shared state (`src/jarvis/state.py`, see `state.spec.md`), which whatever shows the assistant to the user reads:

- **Wake word mode:** If the transcribed text carries the wake word, as "Text-Based Wake Detection" finds it, start the thinking beep and publish LISTENING.
- **Hot window:** If voice started during an active (or pending) hot window, start the thinking beep and publish LISTENING.
- **No trigger:** If neither condition is met, no feedback is given.

If the intent judge later rejects the query (and no hot window override applies), the beep is stopped and the state reverts to IDLE, unless a reply of hers is on its way: the tune is then that reply's, and it plays on. Her name followed by nothing ends as "Wake Word Mode" describes. This brief false-positive beep is acceptable — users prefer immediate acknowledgement over delayed but perfect accuracy.

**LISTENING is not published early during TTS** — the beep is suppressed while TTS is playing to avoid self-triggering.

## Timing Marks (Latency Journal)

The listener feeds the latency journal (`src/jarvis/latency.py`, see `latency.spec.md`). It binds the journal to its database and the user's settings when it is built; notes every non-empty transcript with the moment the user stopped speaking (the close of the utterance less the trailing silence the voice detector waited through); times the intent judge call and records what kind of answer it gave; opens the turn when it dispatches the query, hands the journal's stage callback to the reply engine, freezes the turn when the engine returns, and writes it once the reply has been queued for speech. The marks hold durations and enumerations, never text, and cannot change what the listener does: none raises, and with the journal off each costs a lock and a clock read.

## Configuration

```json
{
  "transcript_buffer_duration_sec": 120,

  "intent_judge_model": "gemma4:e2b",
  "intent_judge_timeout_sec": 15.0,

  "hot_window_seconds": 3.0,
  "echo_tolerance": 0.3
}
```

| Setting | Default | Description |
|---------|---------|-------------|
| `transcript_buffer_duration_sec` | 120 | Duration (seconds) for rolling ambient speech transcript. Provides conversation context so the intent judge can synthesise a complete query when someone involves Jarvis. Separate from dialogue memory. |
| `whisper_language` | `""` | Code of the one language Whisper transcribes in, for the listener and dictation alike, from `WHISPER_LANGUAGES` (Whisper's own list; "fr-FR" reads as "fr"). Empty (or `auto`): it detects the language of each utterance, which a multilingual user needs and which lets it invent phrases out of silence in any language and spell a short word in another script ("Стоп."). Named, it transcribes only that language, and that is the language reported for each utterance. A code Whisper does not know, or a language other than English with an English-only (`.en`) model, is announced once and read as empty: passed on, it would fail every transcription or report the wrong language. |
| `whisper_vad` | `true` | Whether faster-whisper's voice detector (Silero) cuts the stretches that hold no speech before Whisper decodes an utterance, for the listener and dictation alike ("Text Whisper Makes Out of Silence"). Off, Whisper decodes the whole utterance and writes phrases into noise. No effect on MLX Whisper, which has no detector. |
| `voice_collect_seconds` | 4.5 | Silence after which the question being collected is handed to the reply thread, counted from the end of the last utterance collected, in the time its audio was captured. |
| `voice_max_collect_seconds` | 180.0 | Longest a question is collected, however often it is completed. |
| `whisper_min_confidence` | 0.3 | Minimum `avg_logprob`-derived confidence score for a transcribed segment. Segments below this are discarded before the intent judge sees them. |
| `whisper_no_speech_threshold` | 0.5 | Hard cutoff on Whisper's `no_speech_prob` field. Any segment at or above this value is discarded **regardless of `avg_logprob`** — Whisper can be confident about a hallucinated phrase even when no real speech is present (e.g. the "MBC 뉴스" hallucination on background noise). This filter runs before the `avg_logprob` check so it catches high-confidence hallucinations that would otherwise survive. Applies to both the faster-whisper and MLX backends. |
| `whisper_vocabulary` | `[]` | Names Whisper is given as a hint, after `weather_city`. See "The names he uses". |

**Repetition guard.** Past the confidence filters, a transcript is
discarded when it loops: one word over half the tokens, the same word
three times running, a short character pattern repeating, or **a phrase
of two to ten words repeated three times back to back**. The last is not
covered by the others — measured on "There we go." thirteen times over
air-conditioning hum, no single word reaches a third of the tokens and
none ever falls twice in a row, because the other two sit between each
repetition.

Three repeats, never two: people say things twice ("oui oui, vas-y"), and
a guard that ate an emphatic repeat would cost real sentences. Each
hallucination that survives costs an intent-judge call on the cloud model
to be told it is not directed.

Note: the intent judge has no enable flag. A connection error puts it in a 30-second back-off during which `judge()` returns without calling the backend; the listener still takes its no-verdict path, so in the hot window his follow-up is kept, words that could be hers are set aside with `🔇 Heard (echo)` (see "Hot Window Mode"), and the unavailability is printed to standard output. Outside a hot window, text-based wake detection answers as usual. A request that outlives `intent_judge_timeout_sec` gives no verdict either, without the back-off, so the next engagement asks the judge again; so does a stream the server stops sending for that long after it has answered, which is a timeout, not a refusal. The audio loop waits `intent_judge_timeout_sec` plus 2 seconds at most, wall clock, for the judge's answer (see "A directed verdict is acted on once it is decided").

## State Transitions

```mermaid
stateDiagram-v2
    direction LR
    [*] --> WakeWord: System Starts

    WakeWord: Listening for Wake Word
    HotWindow: Listening for Follow-up
    DuringTTS: TTS Playing
    Thinking: Reply being prepared (the microphone is still read)

    WakeWord --> IntentJudge: Wake detected (text-based)
    IntentJudge --> WakeWord: Not directed / no query
    DuringTTS --> HotWindow: TTS ends + echo_tolerance
    HotWindow --> IntentJudge: Speech detected
    HotWindow --> WakeWord: Timer expires
    DuringTTS --> WakeWord: Stop command detected
    IntentJudge --> Thinking: Query handed to the reply thread
    Thinking --> DuringTTS: Her voice free, reply spoken
    Thinking --> WakeWord: Stop command detected
```

## Audio Pipeline

```
Microphone Audio
    ↓
Sounddevice Callback → _audio_q
    ↓
Main Loop: Get Frames → VAD Check
    ↓
Speech Detected → Accumulate Frames
    ↓
Silence Timeout → Whisper Transcription
    ↓
In her time, several segments → her sentences at the edges left out (✂️)
    ↓
Add to Transcript Buffer (with timestamps; flagged "(during TTS)" when it could be her echo)
    ↓
Wake Detection Check:
    └→ Text contains wake word? → Start thinking beep + publish LISTENING
    ↓
If wake detected OR in hot window:
    → Fuzzy echo check, unless she speaks or the thinking tune plays (partial_ratio ≥ 70 = echo → 🔇 Heard (echo), hot window timer untouched)
    → Send buffer + context to Intent Judge
    ↓
If no verdict, in hot window:
    → Her reply coming back whole: 🔇 Heard (echo)
    → Words that could be hers: 🔇 Heard (echo), unless they carry a wake word that can only be his: wake-word check, as below
    → Otherwise: collect what he said as a query
If judge rejects and says noise, in hot window:
    → Not speech: mark processed, ignore, hot window untouched
If judge rejects and says aside with high confidence, in hot window (words that could not be hers, no question being collected, no question of hers waiting for his answer, her name not in them as configured):
    → 🔇 Heard (not to her): left in the transcript, ignore, hot window untouched
If judge.directed, outside the hot window, and neither what was heard nor its query holds more than her name:
    → Wake-word check: her name alone, she waits for his question
If judge.directed and judge.query:
    → Verify wake word present (wake word mode) or non-echo (hot window)
    → Collect it (added to a question being collected), then hand it to the reply thread (it waits its turn)
If no query extracted, in hot window, and a wake word that can only be his (found in the utterance, none in her reply, still found once her words are taken out; not her reply coming back whole):
    → Wake-word check: collect what was heard, the wake word taken out
If judge.directed without a query, or rejects in any other way, in hot window, at any confidence:
    → Words that could be hers (heard in her time, and either their voice stopped with hers or they lack words she did not say in a voice): 🔇 Heard (echo)
    → Otherwise, not pure echo: collect what he said as a query
    → An echo verdict: only on words she did not say, in a voice, that could not be hers; otherwise 🔇 Heard (echo)
If nothing above took it and a question is being collected:
    → Judge it as a follow-up, his question before it (unless already judged as one)
    → Stop: cancel the question
    → Noise, not directed: mark processed, leave it out, silence clock untouched
    → Otherwise: mark processed, add it to the question
When his silence has lasted voice_collect_seconds:
    → A letter or digit collected: hand it to the reply thread
    → Nothing collected (her name alone, punctuation at most): 💤 Nothing after the wake word, nothing handed over;
      published state back to SPEAKING, THINKING (the reply's tune plays on) or IDLE (tune stopped);
      while he dictates, the dictation's state stands
```

## Keeping a Copy of an Utterance

Off unless `JARVIS_SAVE_AUDIO=<directory>` is set in the environment.
There is no config key, no default and no UI toggle: a microphone writing
his voice to disk is the most sensitive thing here, and the only switch
that cannot be left on by a past session is one that dies with the shell.
When it is live the listener says so at startup, naming the directory.

Each utterance is written as 16 kHz mono WAV with a JSON sidecar holding
Whisper's hypothesis, the model, the input device, the duration and the
RMS. `reference` and `keywords` are left empty for him to fill in:
deriving a reference from the hypothesis would score the recogniser
against itself.

Utterances that transcribed to nothing are kept. Silence and noise are
where an autoregressive decoder and a transducer part company, so a
corpus of successes only would measure the easy half.

The capture sits before the empty-transcript return and never raises:
the listener's loop reads the audio queue on a deadline, and a capture
that threw would cost the sentence it was meant to record.

`scripts/asr_bench.py` scores such a corpus across recognisers, reporting
word and character error rates alongside recall on the declared keywords.

## Speaking Something Produced Elsewhere

Some replies are not answers to an utterance: a tool approved by a click
on a card runs on its own worker, and the sentence describing what
happened has to be said out loud.

That worker never speaks. It calls `enqueue_reply(text)`, and the
listener's reply thread says it, through the same `_speak_reply` path it
uses for her own replies. Two speakers would both write the echo
detector's record of what was last said, and the next transcript would
then be compared against the wrong sentence — which is how a user's
answer gets deleted as an echo.

**The reply thread drains it on every pass**, between queries and
whether or not anyone speaks, so a narration never waits on the
microphone. A query being answered keeps it waiting until that reply has
been handed to the speech engine. A stop leaves it queued: it is owed by
another feature, and a reminder is a promise.

**The language is the one last heard.** A reply from elsewhere has no
utterance behind it, so it is spoken with the language the user was last
heard in, and the engine's configured `response_language` outranks that
(`tts.spec.md`, "Which language wins"). A reply of her own carries the
language heard for it, fixed when the query is handed to the reply
thread and shared by every sentence of it.

**One reply per pass, and none while she is already speaking.** The echo
detector holds the record of one reply at a time: the text she is
saying, when she began, its exact duration, and the floor of the voice
proof (see "And it takes a voice"). Starting a second reply while the
first is still audible would replace that record with the second's, and
the rest of the first would be compared against the wrong sentence.
Completion callbacks are not the constraint, since they travel with
their own queue item (`tts.spec.md`, "The queue"). Anything still queued
waits for the reply thread's next pass, a few milliseconds away when no
query is running.

## Fallback Behaviour

When components are unavailable, the system degrades gracefully:

| Component | Unavailable Behaviour |
|-----------|---------------------|
| Intent Judge | Simple text-based wake word + query extraction; in the hot window his words are still kept and words that could be hers set aside |
| 16 kHz sample rate | Stream at device native rate, resample to 16 kHz for Whisper |
| Transcript Buffer | Process each utterance independently |
| `large-v3-turbo` on faster-whisper older than 1.1.0 | Load `large-v3` instead and say so (`⚠️ large-v3-turbo is not supported…`). MLX always runs it |

Whether the installed faster-whisper runs `large-v3-turbo` is `is_faster_whisper_turbo_supported()` in `jarvis/listening/whisper_turbo.py`, the one check the listener and the setup wizard share. It reads the version from the package's installed metadata (`importlib.metadata.version("faster-whisper")`) and never imports the package, so the wizard can ask on its UI thread without loading ctranslate2. A missing install or an unreadable version reads as unsupported. The distribution name stays a literal in that call: PyInstaller's analysis ships the metadata of the names it sees passed to `version()`, which is how the frozen build can answer at all.

## Download Recovery

Whisper model loading handles transient download failures automatically:

### Visible Progress and Resume

Before `WhisperModel` (faster-whisper) or the first MLX `transcribe` call runs, the listener pre-downloads the model's HuggingFace snapshot with `download_snapshot_with_progress` (`jarvis/utils/hf_download.py`). faster-whisper silences huggingface_hub's progress bars and the daemon never has a TTY, so without this a multi-GB first-run download printed nothing. The helper watches the repo's `blobs` cache directory and prints a throttled `⬇️ Downloading … X/Y MB · rate` line (or `X MB so far` when the total size cannot be learned), plus a "still downloading" line if the transfer stalls. A warm cache prints nothing. Pre-download failures never block loading: `WhisperModel`'s own offline/cache paths take over. Resume on interruption is huggingface_hub's `.incomplete` + HTTP `Range` mechanism, so a killed download continues where it stopped. On Windows, a WinError 1314 (symlink privilege not held) escaping huggingface_hub's support probe at the pointer-creation step is caught and retried once with plain file copies, so the completed download is not thrown away.

### Corrupted Cache Recovery

If the HuggingFace model cache is corrupted (e.g. from an interrupted download), the system detects the CTranslate2 "unable to open file" error, deletes the parent `models--` cache directory, and retries the download once. If the retry also fails, a message guides the user to manually delete the cache.

### Rate Limit Retry (HTTP 429)

When HuggingFace returns HTTP 429 (Too Many Requests), both faster-whisper and MLX Whisper backends retry up to 4 times with exponential backoff (2s, 4s, 8s, 16s). Progress messages inform the user of each retry attempt. If all retries are exhausted, the user is advised to wait and restart.

## Future: Acoustic Echo Cancellation

Currently, echo is handled at the transcript level via fuzzy text matching and the intent judge. True acoustic echo cancellation (AEC) would:
- Require the audio output signal (reference)
- Process in real-time with adaptive filtering
- Add 10-50ms latency

**Current recommendation:** The transcript-level echo detection (fuzzy matching + intent judge) is sufficient and simpler. Consider AEC only if transcript-level detection proves inadequate in practice.

## The word that wakes her

`whisper_min_confidence` (0.3) is computed from `avg_logprob`, and that
is a poor judge of a proper noun the model has never seen — which is
exactly why "Yuba" comes back as "Youba", "Nuba", "Juba". The segment
carrying the wake word scores low, is dropped, and the sentence that
follows arrives with no wake word in it and is ignored. She then looks
broken, which is the one failure a wake word cannot afford.

A segment about to be dropped for confidence is kept when it matches the
wake word, using the same matcher the detector runs a moment later. The
wake word is configured, not a language pattern, and keeping the segment
decides nothing: the detector and the intent judge both still apply.

Everything else stays dropped. In one afternoon's logs, 38 segments went
to this filter and 36 were a television and a conversation in the room —
lifting the bar for all of them would send every one to the intent judge.

## The names he uses

Whisper pulls a proper noun it has rarely seen towards a word it knows:
the name of his town has come back in three spellings that are not its
own, and one of them sent a weather request to a village in another
country.

Every transcription is given his names as a hint: `weather_city`, then
the entries of `whisper_vocabulary`, each once, joined with commas
(`whisper_hint` in `jarvis/listening/vocabulary.py`). faster-whisper
receives it as `hotwords`, MLX as `initial_prompt`. Both place the text
where Whisper keeps what it heard before, which steers spelling and
leaves language detection alone, since detection reads the audio first;
with a `whisper_language` named, the hint is given the same way.
The names are configured, not a language list. With none configured,
Whisper is called exactly as without the feature, with no argument for a
hint.

On faster-whisper the hint goes through `jarvis/listening/whisper_vad.py`
with the call's other arguments: a call made again without a voice
detector that failed still carries it, and a model that takes only the
language (`TypeError`) is called without it ("Text Whisper Makes Out of
Silence").

**Her name is never in it.** A name the wake-word matcher recognises,
against `wake_word` and `wake_aliases`, is left out: the hint makes
Whisper write its words more readily, and writing hers more readily is a
false wake.

**The hint may change how a word is spelled, never whether anything was
said.** Over noise Whisper writes the hint back, and its scores cannot
tell that from speech: measured, noise that comes out "Thank you."
without the hint comes out as his town with it, clearing both confidence
filters, and in a hot window it would be taken as a question. A
transcript at least three quarters of whose words are the hint's own
(`is_the_hint_repeated`) is therefore read again without the hint, and
that second reading is the one used whole: its text, the language it
reports and its segments, which the listener reads afterwards ("Her
sentences at the edges of what he said"). The second pass runs only
then. With `whisper_vad` on, an utterance the voice detector empties
comes back empty whatever the hint; the second reading is what holds for
the noise the detector lets through, with the setting off, after the
detector has failed, and on MLX, which has none. The threshold sits
between the two shapes measured: a fourteen-name list read back over
noise was 88 to 94% hint words, often with a stray word of its own
tacked on; a sentence that merely names one of them is half or less. The
price is a reply that is nothing but his names, such as his town on its
own: it is transcribed as it would be without the hint.

**Keep the list short.** Each name also attracts words that sound like
it, mostly words Whisper was already getting wrong: with his town in the
hint, another town whose name sounds like it came out as his 54 times in
112, where it never came out right with or without the hint, while three
more towns with names close to his came out right as often as without
it. A long list adds nothing for any one name: with fourteen names, his
town came out right no more often than with it alone.

Measured on 2,059 synthetic clips (Kokoro, Piper and Windows voices, each
clean, under pink noise, under babble and through a far microphone; then
noise and silence) with large-v3-turbo, through this path with and
without `weather_city`, the voice detector off and no language named,
the two arms alternating clip by clip: his town came out right in 502
sentences of 672 against 271, the far microphone gaining most (118 of
168 against 43). Noise and silence gave none of their 219 transcripts as
his town; what the confidence filters let through of them is unchanged
(121 against 123). The 496 sentences without the name gave no hint word
and no false wake, either way.
`evals/test_noise_never_becomes_his_city.py` holds the real model to the
noise half, with the voice detector and without it.

## Salvaging his speech out of a merged echo chunk

Whisper sometimes hands back one segment holding the tail of her reply and the start of his. `salvage_after_echo_tail` finds the rightmost five-word window resembling her last twenty words and keeps what follows.

**What is discarded has to look like her, not just the seam.** Checking only the seam was a destructive defect: when he corrects her he necessarily reuses her wording, so the winning window is found inside his own sentence and the correction is thrown away. Measured, "no my next meeting is at three o'clock can you check again" came back as "can you check again".

The prefix test uses `ratio`, not `partial_ratio`: the discarded half must be substantially the same string as her tail rather than merely contain a fragment of it. Over two replies, real echo prefixes score 57-60 and his own words 34-52, where `partial_ratio` puts them at 94-95 against 81-88 and separates nothing. The threshold sits at 55 and ties lean towards keeping — her words left at the front of his sentence are noise the intent judge reads past, while a wrong cut destroys the correction and the caller overwrites the transcript buffer, so nothing downstream can recover it.

The trigger upstream is purely temporal: he began speaking within the echo tolerance of her finishing. "There was no echo" is therefore exactly the case this fires on, which is why the text-level test has to carry the decision.

A known, measured limit: the scan takes the rightmost matching window, so it eats the first word of his real speech — "who else is coming" comes back as "else is coming". Left alone deliberately, and pinned by a test so a future change has something to compare against.

## Her sentences at the edges of what he said

Whisper makes several segments of one utterance. When he speaks over her, or the moment she stops, the first or the last of them can be a sentence of her reply coming back through the microphone, and the rest his. Left together, her sentence goes to the model stuck to his question, or her sentences on both sides drown the few words of his that say he spoke, and the whole utterance is taken for her echo.

So an utterance heard in her time (see "During TTS") that Whisper made of more than one segment loses the sentences of her reply that open and close it. The transcript buffer, the "(during TTS)" mark, the echo checks and strippers, the judge and the wake-word check all read what is left. The copy kept of the utterance (see "Keeping a Copy of an Utterance"), the repetition filter and the `📝 Heard` line read it as Whisper heard it; the console then says `✂️ Left out her own words, kept: "…"`.

**A segment is a sentence of hers when it is one run of her reply**, the reply the echo detector holds, whole (`output/streaming.spec.md`): `partial_ratio` of 85 or more (`_HER_SENTENCE_MATCH`). Measured segment by segment on two days of real transcripts, hers scored 87 and above and his 65 and below; corrections of his written in her own wording for the measure scored 70 to 81.

- The run is read in one direction. A segment longer than her whole reply is weighed whole against it (`ratio`): `partial_ratio` lays the shorter text along the longer, so it would ask whether her reply is a run of the segment, and a segment holding her short reply and then the start of his question would be taken out, his question with it.
- A segment under four words (`_HER_SENTENCE_MIN_WORDS`) is never hers. Three words in a row are few enough for him to say after a long reply of hers, and what is taken out cannot be recovered; her words left at the edge of his are noise the judge reads past.

**Only the edges, and only when words of his are left.** A sentence of hers between two of his stays where it is. Nothing is taken out when what would be left is under `EchoDetector.min_salvage_words` words, counted as every salvage counts them, or carries no words she did not say (the share of "Overriding an echo verdict takes positive evidence"): a word or two after her sentences is what Whisper makes of the quiet after her voice ("Merci."), and split from them it would stand alone as words she never said, in the hot window, where such words are answered.

Outside her time nothing is taken out: a sentence of hers he repeats then is his.

Kept on purpose:

- A sentence of his, four words or more, that repeats words of hers nearly as she said them, at an edge of an utterance heard in her time, cannot be told from her sentence bent by Whisper, and is taken out with it.
- Her sentences followed by three words or more that nobody said to her (a television, a phrase Whisper makes of the room) leave those words standing alone, unmarked, as they stand when Whisper hears them as an utterance of their own. While she speaks nothing is made of them without her name; in the hot window the rules for words that could be hers apply to them as to any utterance.
- The cut works on Whisper's segments, never inside one: a segment holding a sentence of hers and words of his is left to the strippers of "Salvaging his speech out of a merged echo chunk".
- Words are counted between spaces, so in a script written without them a sentence counts as one word and nothing is taken out.

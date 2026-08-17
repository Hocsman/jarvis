# Speaking Before the Sentence Is Finished

## Why

She writes the whole reply, then starts speaking it. On a three-sentence
answer that is several seconds of silence where a person would already
be talking — and the complaint that prompted this was exactly that: an
exchange that should feel like one with another person.

The cost is not the model. A turn makes four cloud round-trips before she
opens her mouth (intent judge, memory keywords, tool router, chat model),
and only the last is the chat model. But the largest single block is the
one after all of them: waiting for the final token before the first word.
Speaking each sentence as it closes reclaims the whole generation time,
and it is the only lever that changes how the exchange *feels* rather
than how long it takes.

The plumbing exists. `on_token` runs from `run_reply_engine` through
`chat_with_messages` to the backend. Nothing consumes it on the voice
path.

## The constraint that shapes everything

**Echo detection compares what the microphone hears against the last
thing she said, and it must keep seeing the whole reply.**

`EchoDetector.track_tts_start(text)` *replaces* `_last_tts_text`. Speaking
sentence by sentence through the existing call would leave it holding
only the last sentence, and every guard downstream reads it:

- the fuzzy `partial_ratio` safety net,
- the intent judge, which receives it as context,
- `_carries_speech_she_did_not_say`, which decides whether to override an
  echo verdict and whether a stripped remainder is speech.

An echo of sentence one, arriving while sentence three is being spoken,
would be compared against sentence three, match nothing, and be taken for
the user. She would start answering her own earlier sentences — the exact
failure fixed on 2026-08-16, reintroduced by the feature.

So `_last_tts_text` accumulates across the sentences of one reply and
resets when a new reply begins. The reference text is the reply, never
the chunk.

## Segmentation

A sentence closes on terminal punctuation followed by whitespace or end
of text: `.`, `!`, `?`, `…`, and their full-width counterparts `。`,
`！`, `？` so the rule does not privilege a script. Punctuation is
orthographic, not lexical — no word list, no language detection.

A chunk is spoken when it closes **and** carries enough to be worth a
synthesis call; below that it waits for the next one. Decimals,
abbreviations and ellipses will occasionally split a sentence early. That
is acceptable: a slightly clipped prosody costs less than the silence it
replaces, and the text reaching the user is unchanged.

If the stream ends with an unterminated tail, that tail is spoken as-is.

## What must not change

**One reply is still one turn.** The completion callback that opens the
hot window fires once, when the last chunk finishes — not per sentence.
Firing per sentence would open the window mid-reply and invite her own
next sentence in as a follow-up.

**Interruption still stops everything.** `stop` mid-reply drops the
queued chunks as well as the one playing, and cancels what has not been
synthesised yet.

**The printed reply is unchanged.** What appears in the transcript and
what is stored in dialogue memory is the whole text, assembled, exactly
as today. Streaming is a delivery detail; it must not become a difference
in what was said.

**A tool-calling turn speaks nothing.** Content is empty on those turns;
only the final prose is spoken, as now.

## Failure

Fails back, not open. If the backend does not stream, or streaming raises
mid-reply, the reply is spoken whole by the existing path — the current
behaviour is the fallback, so the worst case is today's latency rather
than a lost answer.

## Testing

- A three-sentence reply is spoken as three chunks, in order.
- The echo reference after the third chunk contains all three sentences,
  and an echo of the first is still recognised.
- The hot window opens once, after the last chunk.
- A stop during chunk one leaves chunks two and three unspoken.
- The assembled text equals the non-streamed text, character for
  character.
- The control: a backend without streaming still produces a spoken reply.

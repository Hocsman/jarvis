"""
Evals for the Intent Judge LLM.

Deduplicated suite: 22 cases covering all behaviour axes from the original 59.
See PR description / commit message for the dedup rationale.

The not-speech cases also run on any judge model: name one or more in
`EVAL_INTENT_JUDGE_ARMS`. One test then reads every case on every arm,
alternating arms call by call (the order rotates each call, the cases are
shuffled each round), and prints, per arm, how many inventions and how many
of his words the listener would answer in the hot window (and how many of
those on a verdict below high confidence that extracted no query, which the
hot window weighs as a sure one), and how many would join a question it is
still collecting, where a stop drops the question and is a miss either way;
a row under 20 calls gets counts and no rate. A second test reads the
collecting cases with the question shown, as the listener shows it, and
hidden, the two alternating call by call on each arm (`-k shown` runs it
alone). A third reads his answer to a question she ended her reply on,
begun the moment she stops, unmarked, as the listener shows it when he is
heard in a voice that goes on after hers, and marked "(during TTS)", as it
shows an answer too quiet to prove he spoke, which it answers only on a
directed verdict with a query, alternating the same way (`-k his_answer`
runs it alone). A fourth reads her last sentence coming back as words she
never said, its voice stopping with hers, unmarked, as the judge is shown
it when the two proofs alone decide, and marked, as the listener shows an
utterance whose voice stops with hers (`-k rewritten` runs it alone, after
a check that needs no model: the listener shows each case to the judge
whole, marked only when its voice stops with hers).
Arms go through `MockConfig`, so a cloud model
goes to the provider the `EVAL_LLM_*` variables name and is billed there,
and a bare local tag runs on the local Ollama. Only these synthetic
transcripts leave the machine. `EVAL_INTENT_JUDGE_ROUNDS` sets the rounds
(4 by default).

    EVAL_LLM_PROVIDER=anthropic EVAL_LLM_API_KEY_ENV=YUBA_ANTHROPIC_KEY \\
    EVAL_INTENT_JUDGE_ARMS="gemma4:e2b,claude-haiku-5-5" \\
      python -m pytest evals/test_intent_judge.py -k "arms or reaches_the_judge" -s
"""

import dataclasses
import os
import random

import pytest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass
from typing import Optional, List, Union

from helpers import JUDGE_MODEL, JUDGE_BASE_URL, MockConfig, is_judge_llm_available
from jarvis.listening.listener import _carries_speech_she_did_not_say

NOT_SPEECH_ARMS = [a.strip() for a in os.environ.get("EVAL_INTENT_JUDGE_ARMS", "").split(",") if a.strip()]
NOT_SPEECH_ROUNDS = int(os.environ.get("EVAL_INTENT_JUDGE_ROUNDS", "4") or 4)


# =============================================================================
# Test Data
# =============================================================================

@dataclass
class IntentJudgeTestCase:
    """Test case for intent judge evaluation."""
    name: str
    transcript: str
    last_tts_text: str
    in_hot_window: bool
    wake_timestamp: Optional[float]
    expected_directed: bool
    expected_query_contains: Optional[Union[str, List[str]]]
    expected_query_not_contains: Optional[Union[str, List[str]]] = None
    expected_stop: bool = False


# Single-segment cases - one per distinct behaviour axis.
INTENT_JUDGE_TEST_CASES = [
    # Wake word + simple question (canonical directed+extract)
    IntentJudgeTestCase(
        name="wake_word_simple_question",
        transcript="Jarvis what time is it",
        last_tts_text="",
        in_hot_window=False,
        wake_timestamp=1000.5,
        expected_directed=True,
        expected_query_contains="time",
        expected_query_not_contains="jarvis",
    ),
    # Wake word at sentence end, adjacent to a named entity. Regression guard:
    # the judge previously left "Jarvis" in the query, causing the reply engine
    # to treat "Possessor Jarvis" as the film title instead of "Possessor".
    IntentJudgeTestCase(
        name="wake_word_trailing_after_named_entity",
        transcript="what do you know about the movie called Possessor Jarvis",
        last_tts_text="",
        in_hot_window=False,
        wake_timestamp=1001.5,
        expected_directed=True,
        expected_query_contains="possessor",
        expected_query_not_contains="jarvis",
    ),
    # Wake word mid-sentence (not at start, not at end). Ensures the judge
    # removes every occurrence, not just the leading one.
    IntentJudgeTestCase(
        name="wake_word_mid_sentence",
        transcript="hey Jarvis what's the weather in London",
        last_tts_text="",
        in_hot_window=False,
        wake_timestamp=1000.3,
        expected_directed=True,
        expected_query_contains="weather",
        expected_query_not_contains="jarvis",
    ),
    # Wake word + command/imperative addressed to the assistant (not a question)
    IntentJudgeTestCase(
        name="wake_word_command_timer",
        transcript="Jarvis set a timer for 5 minutes",
        last_tts_text="",
        in_hot_window=False,
        wake_timestamp=1000.5,
        expected_directed=True,
        expected_query_contains="timer",
        expected_query_not_contains="jarvis",
    ),
    # Wake word + statement/command to remember something
    IntentJudgeTestCase(
        name="wake_word_statement_remember",
        transcript="Jarvis remind me to call mum at 5pm",
        last_tts_text="",
        in_hot_window=False,
        wake_timestamp=1000.5,
        expected_directed=True,
        expected_query_contains="mum",
    ),
    # Wake word + casual share-of-information statement (no explicit command
    # or question). Regression guard: the judge previously rejected these as
    # "not directed" because the sentence was a statement about the user's
    # own action rather than a command or question, even though the wake
    # word was clearly addressed to the assistant.
    IntentJudgeTestCase(
        name="wake_word_share_statement_burger",
        transcript="Jarvis, I just ate a burger from McDonald's.",
        last_tts_text="",
        in_hot_window=False,
        wake_timestamp=1000.5,
        expected_directed=True,
        expected_query_contains="burger",
        expected_query_not_contains="jarvis",
    ),
    IntentJudgeTestCase(
        name="wake_word_share_statement_feeling",
        transcript="Jarvis I'm feeling a bit tired today",
        last_tts_text="",
        in_hot_window=False,
        wake_timestamp=1000.5,
        expected_directed=True,
        expected_query_contains="tired",
        expected_query_not_contains="jarvis",
    ),
    # Wake word at the END of a declarative statement. Position of the wake
    # word must not affect directedness — this pattern must also be directed.
    IntentJudgeTestCase(
        name="wake_word_share_statement_trailing",
        transcript="My flight just got cancelled, Jarvis",
        last_tts_text="",
        in_hot_window=False,
        wake_timestamp=1001.5,
        expected_directed=True,
        expected_query_contains="flight",
        expected_query_not_contains="jarvis",
    ),
    # Wake word at the END of a declarative statement that contains a
    # capitalised brand/product name immediately before "Jarvis". Regression:
    # gemma4:e2b misread "big Mac Jarvis" as the compound name "Mac Jarvis",
    # treating "Jarvis" as a surname rather than the wake word, and returned
    # directed=false despite its own reasoning stating it found the wake word.
    IntentJudgeTestCase(
        name="wake_word_trailing_after_capitalised_brand",
        transcript="I just ate a big Mac Jarvis",
        last_tts_text="",
        in_hot_window=False,
        wake_timestamp=1001.5,
        expected_directed=True,
        expected_query_contains="big Mac",
        expected_query_not_contains="jarvis",
    ),
    # Self-contained imperative with an intentionally open subject ("something",
    # "anything", "a joke") — these are valid queries and must not be treated
    # as vague references or standalone "re-issue prior question" imperatives.
    # Regression: gemma4:e2b was returning directed=false with reasoning "no
    # extractable query" on "Jarvis say something please" because it conflated
    # the open subject with a topic-less question.
    IntentJudgeTestCase(
        name="wake_word_open_imperative_say_something",
        transcript="Jarvis say something please",
        last_tts_text="",
        in_hot_window=False,
        wake_timestamp=1000.5,
        expected_directed=True,
        expected_query_contains="say something",
        expected_query_not_contains="jarvis",
    ),
    IntentJudgeTestCase(
        name="wake_word_open_imperative_tell_me_a_joke",
        transcript="Jarvis tell me a joke",
        last_tts_text="",
        in_hot_window=False,
        wake_timestamp=1000.5,
        expected_directed=True,
        expected_query_contains="joke",
        expected_query_not_contains="jarvis",
    ),
    IntentJudgeTestCase(
        name="wake_word_open_imperative_tell_me_anything",
        transcript="Jarvis tell me anything",
        last_tts_text="",
        in_hot_window=False,
        wake_timestamp=1000.5,
        expected_directed=True,
        expected_query_contains="anything",
        expected_query_not_contains="jarvis",
    ),
    IntentJudgeTestCase(
        name="wake_word_open_imperative_give_me_advice",
        transcript="Jarvis give me advice please",
        last_tts_text="",
        in_hot_window=False,
        wake_timestamp=1000.5,
        expected_directed=True,
        expected_query_contains="advice",
        expected_query_not_contains="jarvis",
    ),
    IntentJudgeTestCase(
        name="wake_word_open_imperative_surprise_me",
        transcript="Jarvis surprise me",
        last_tts_text="",
        in_hot_window=False,
        wake_timestamp=1000.5,
        expected_directed=True,
        expected_query_contains="surprise",
        expected_query_not_contains="jarvis",
    ),
    # Same-segment context synthesis (distinct from simple wake+Q)
    IntentJudgeTestCase(
        name="context_synthesis_weather_opinion",
        transcript="I think the weather is great today in London. What do you think, Jarvis?",
        last_tts_text="",
        in_hot_window=False,
        wake_timestamp=1000.8,
        expected_directed=True,
        expected_query_contains="weather",
    ),
    # Echo + user follow-up in hot window
    IntentJudgeTestCase(
        name="echo_plus_followup_extracted",
        transcript="London has 8 hours of daylight. That's quite cool. Tell me more.",
        last_tts_text="On this day, London receives around 7-8 hours of daylight.",
        in_hot_window=True,
        wake_timestamp=None,
        expected_directed=True,
        expected_query_contains="more",
    ),
    # Stop command during TTS
    IntentJudgeTestCase(
        name="stop_command_during_tts",
        transcript="stop",
        last_tts_text="Let me tell you about the history of...",
        in_hot_window=False,
        wake_timestamp=None,
        expected_directed=True,
        expected_query_contains=None,
        expected_stop=True,
    ),
    # No wake word, not hot window -> not directed
    IntentJudgeTestCase(
        name="no_wake_word_casual_speech",
        transcript="I think the weather is nice today",
        last_tts_text="",
        in_hot_window=False,
        wake_timestamp=None,
        expected_directed=False,
        expected_query_contains=None,
    ),
    # Wake word only mentioned in narrative -> not directed
    IntentJudgeTestCase(
        name="mentioned_in_narrative_past_tense",
        transcript="I told my friend about Jarvis yesterday",
        last_tts_text="",
        in_hot_window=False,
        wake_timestamp=1000.8,
        expected_directed=False,
        expected_query_contains=None,
    ),
    # Hot window simple follow-up
    IntentJudgeTestCase(
        name="hot_window_simple_followup",
        transcript="What about next week?",
        last_tts_text="The weather this weekend will be rainy.",
        in_hot_window=True,
        wake_timestamp=None,
        expected_directed=True,
        expected_query_contains="next week",
    ),
]


@dataclass
class MultiSegmentTestCase:
    """Test case with multiple transcript segments (realistic buffer state)."""
    name: str
    segments: list
    last_tts_text: str
    in_hot_window: bool
    wake_timestamp: Optional[float]
    expected_directed: bool
    expected_query_contains: Optional[Union[str, List[str]]]
    expected_query_not_contains: Optional[Union[str, List[str]]] = None
    expected_stop: bool = False
    aliases: Optional[List[str]] = None


MULTI_SEGMENT_TEST_CASES = [
    # Real-logs scenario: echo + rejected similar + wake retry
    MultiSegmentTestCase(
        name="echo_plus_rejected_similar_plus_wake_retry",
        segments=[
            ("and relatively windy, about 11 kilometers per hour", False),
            ("Okay, well, what about any new movies tomorrow?", False),
            ("Jarvis, what about new movies tomorrow?", False),
        ],
        last_tts_text="Tomorrow's weather in Kensington looks a bit gloomy, with overcast conditions expected. It'll be quite cool, around 6°C, and relatively windy, about 11 km/h.",
        in_hot_window=False,
        wake_timestamp=1004.5,
        expected_directed=True,
        expected_query_contains="movies",
        expected_query_not_contains="weather",
    ),
    # Hot window with echo in buffer + user follow-up
    MultiSegmentTestCase(
        name="buffer_echo_then_followup_hot_window",
        segments=[
            ("The weather is sunny and warm", False),
            ("What about the weekend?", False),
        ],
        last_tts_text="The weather today is sunny and warm, around 20 degrees.",
        in_hot_window=True,
        wake_timestamp=None,
        expected_directed=True,
        expected_query_contains="weekend",
        expected_query_not_contains="sunny",
    ),
    # Stop command with TTS echoes in buffer
    MultiSegmentTestCase(
        name="multiple_echoes_then_interrupt",
        segments=[
            ("Let me tell you about", True),
            ("the history of", True),
            ("Jarvis stop", False),
        ],
        last_tts_text="Let me tell you about the history of ancient Rome.",
        in_hot_window=False,
        wake_timestamp=1002.0,
        expected_directed=True,
        expected_query_contains=None,
        expected_stop=True,
    ),
    # No wake word in multi-segment buffer
    MultiSegmentTestCase(
        name="no_wake_word_in_buffer",
        segments=[
            ("How are you?", False),
        ],
        last_tts_text="",
        in_hot_window=False,
        wake_timestamp=None,
        expected_directed=False,
        expected_query_contains=None,
    ),
    # Context synthesis with prior ambient speech that must be filtered
    MultiSegmentTestCase(
        name="context_synthesis_with_prior_ambient",
        segments=[
            ("Did you see the game last night?", False),
            ("Yeah it was amazing", False),
            ("The food here is excellent. Jarvis, what's the best dish to order?", False),
        ],
        last_tts_text="",
        in_hot_window=False,
        wake_timestamp=1004.0,
        expected_directed=True,
        expected_query_contains="dish",
        expected_query_not_contains="game",
    ),
    # Multi-person conversation: context synthesis across speakers without explicit pronoun
    MultiSegmentTestCase(
        name="multi_person_weather_discussion",
        segments=[
            ("I wonder what the weather will be like tomorrow", False),
            ("Yeah we should check before planning the picnic", False),
            ("Jarvis what do you think", False),
        ],
        last_tts_text="",
        in_hot_window=False,
        wake_timestamp=1004.0,
        expected_directed=True,
        expected_query_contains="weather",
    ),
    # Multi-person + vague reference ("that" = iPhone from earlier segment)
    MultiSegmentTestCase(
        name="multi_person_vague_reference",
        segments=[
            ("The new iPhone looks pretty cool", False),
            ("I heard the camera is amazing", False),
            ("Jarvis how much does that cost", False),
        ],
        last_tts_text="",
        in_hot_window=False,
        wake_timestamp=1004.0,
        expected_directed=True,
        expected_query_contains="iphone",
    ),
    # User statement follow-up in hot window (not an echo of TTS question)
    MultiSegmentTestCase(
        name="user_followup_statement_after_question_nihilism",
        segments=[
            ("Some people find that appealing", True),
            ("While others see it as a bleak outlook", True),
            ("What are your thoughts on nihilism", True),
            ("I think it's way more ridiculous than absurdism. Absurdism is the way to go.", False),
        ],
        last_tts_text="Nihilism is an interesting philosophical position. Some people find it appealing, while others see it as a bleak outlook. What are your thoughts on nihilism?",
        in_hot_window=True,
        wake_timestamp=None,
        expected_directed=True,
        expected_query_contains="absurdism",
        expected_query_not_contains="what are your thoughts",
    ),
    # Cross-segment vague reference ("that" -> dinosaurs)
    MultiSegmentTestCase(
        name="cross_segment_dinosaur_opinion",
        segments=[
            ("I think dinosaurs are cool", False),
            ("What do you think about that Jarvis", False),
        ],
        last_tts_text="",
        in_hot_window=False,
        wake_timestamp=1002.5,
        expected_directed=True,
        expected_query_contains="dinosaur",
    ),
    # Imperative resolution: "answer that" -> re-issue prior question
    MultiSegmentTestCase(
        name="cross_segment_answer_that_weather",
        segments=[
            ("Sorry, how's the weather today?", False),
            ("Jarvis, answer that", False),
        ],
        last_tts_text="",
        in_hot_window=False,
        wake_timestamp=1002.5,
        expected_directed=True,
        expected_query_contains="weather",
        expected_query_not_contains="answer that",
    ),
    # Imperative resolution with unrelated noise between Q and imperative
    MultiSegmentTestCase(
        name="cross_segment_answer_that_with_noise",
        segments=[
            ("How tall is Mount Everest", False),
            ("Charlie sands to that", False),
            ("Jarvis answer that", False),
        ],
        last_tts_text="",
        in_hot_window=False,
        wake_timestamp=1004.5,
        expected_directed=True,
        expected_query_contains="everest",
        expected_query_not_contains="answer that",
    ),
    # Whisper tense variant of imperative ("answered that")
    MultiSegmentTestCase(
        name="cross_segment_answered_that_whisper_variant",
        segments=[
            ("Sorry, how's the weather today?", False),
            ("Jarvis answered that", False),
        ],
        last_tts_text="",
        in_hot_window=False,
        wake_timestamp=1002.5,
        expected_directed=True,
        expected_query_contains="weather",
        expected_query_not_contains="answered that",
    ),
    # Multi-word imperative variant
    MultiSegmentTestCase(
        name="cross_segment_go_ahead_and_answer",
        segments=[
            ("What's the capital of Portugal", False),
            ("Jarvis go ahead and answer", False),
        ],
        last_tts_text="",
        in_hot_window=False,
        wake_timestamp=1002.5,
        expected_directed=True,
        expected_query_contains="portugal",
        expected_query_not_contains="go ahead and answer",
    ),
    # Imperative superseded by new explicit question in same segment
    MultiSegmentTestCase(
        name="cross_segment_imperative_superseded_by_new_question",
        segments=[
            ("How's the weather today?", False),
            ("Jarvis, answer that — actually, what time is it?", False),
        ],
        last_tts_text="",
        in_hot_window=False,
        wake_timestamp=1002.5,
        expected_directed=True,
        expected_query_contains="time",
        expected_query_not_contains="weather",
    ),
    # Cross-segment follow-up in hot window (topic extension)
    MultiSegmentTestCase(
        name="cross_segment_hot_window_followup",
        segments=[
            ("The capital of France is Paris", True),
            ("What about Germany", False),
        ],
        last_tts_text="The capital of France is Paris, known as the City of Light.",
        in_hot_window=True,
        wake_timestamp=None,
        expected_directed=True,
        expected_query_contains="germany",
    ),
    # Alias (Whisper mishearing) should be treated as the wake word. Without
    # alias normalisation the small model sees "Jervis" and decides the user
    # is addressing a different person.
    MultiSegmentTestCase(
        name="alias_treated_as_wake_word",
        segments=[
            ("Jervis, what time is it in London?", False),
        ],
        last_tts_text="",
        in_hot_window=False,
        wake_timestamp=1000.8,
        expected_directed=True,
        expected_query_contains="time",
        aliases=["jervis", "jaivis", "jervis", "javis"],
    ),
    # Alias mid-utterance after narrative context — the model must still
    # recognise the addressee as the assistant and resolve the vague reference.
    MultiSegmentTestCase(
        name="alias_after_narrative_context",
        segments=[
            ("The new iPhone looks pretty cool", False),
            ("I heard the camera is amazing", False),
            ("Jaivis how much does that cost", False),
        ],
        last_tts_text="",
        in_hot_window=False,
        wake_timestamp=1004.0,
        expected_directed=True,
        expected_query_contains="iphone",
        aliases=["jervis", "jaivis", "jervis", "javis"],
    ),
    # Buried target sentence amid interleaved unrelated chatter (multi-topic
    # disambiguation). Two separate topics coexist in the buffer — iPhone
    # pricing thread and an unrelated Yankees game discussion. The wake-word
    # segment contains a vague reference ("it") that must resolve to the
    # correct thread (iPhone), not the most recent unrelated topic.
    MultiSegmentTestCase(
        name="buried_target_amid_unrelated_chatter",
        segments=[
            ("The new iPhone looks pretty cool", False),
            ("Did you see the Yankees game last night", False),
            ("I heard the camera is amazing on that phone", False),
            ("Yeah that was a great play in the ninth inning", False),
            ("Jarvis how much does it cost", False),
        ],
        last_tts_text="",
        in_hot_window=False,
        wake_timestamp=1008.5,
        expected_directed=True,
        expected_query_contains="iphone",
        expected_query_not_contains="yankees",
    ),
    # Same buried-target disambiguation, but the wake-word question has no
    # explicit pronoun ("what's the price" instead of "how much does it cost").
    # The judge must still resolve the topic from prior segments — a query of
    # "what's the price" is not answerable alone.
    MultiSegmentTestCase(
        name="buried_target_topicless_question",
        segments=[
            ("so anyway the meeting ran really long yesterday", False),
            ("did you catch the ball game", False),
            ("the new iPhone is out", False),
            ("yeah they lost again though", False),
            ("I want the pro model", False),
            ("Jarvis what's the price", False),
        ],
        last_tts_text="",
        in_hot_window=False,
        wake_timestamp=1010.5,
        expected_directed=True,
        # Parent-noun rule: resolving to a sub-item ("pro model") must also
        # include the parent noun/brand ("iPhone") — "pro model" alone is
        # not self-contained.
        expected_query_contains=["iphone", "pro"],
        expected_query_not_contains="ball game",
    ),
    # Vague reference "they" — the AirPods are the only plural antecedent
    # that can be cost-queried, so "how much do they cost" must resolve to
    # the AirPods thread and include the brand/noun in the query.
    MultiSegmentTestCase(
        name="buried_target_plural_vague_ref_they",
        segments=[
            ("the AirPods sound great", False),
            ("yeah the bass is really punchy", False),
            ("Jarvis how much do they cost", False),
        ],
        last_tts_text="",
        in_hot_window=False,
        wake_timestamp=1006.5,
        expected_directed=True,
        expected_query_contains="airpods",
    ),
    # Hot-window override: a topic-less follow-up ("tell me more") in hot
    # window must stay directed=true even though a topic-rich earlier buffer
    # would otherwise trigger the topic-resolution heuristic. The HOT WINDOW
    # rule must win over the "topic-less question" vague-reference rule.
    MultiSegmentTestCase(
        name="hot_window_override_topicless_followup",
        segments=[
            ("the new iPhone is out", False),
            ("I want the pro model", False),
            ("tell me more", False),
        ],
        last_tts_text="The iPhone 16 Pro has a titanium frame and a new camera system.",
        in_hot_window=True,
        wake_timestamp=None,
        expected_directed=True,
        expected_query_contains=None,
    ),
    # Wake word mid-utterance after narrative buffer, addressing the assistant.
    # Real-world case: user was discussing Mata Hari in the background, then
    # turned to the assistant with "Jarvis, do you know what she's talking about,
    # about Mata Hari?". The small model mis-classified as "not directed" with
    # reasoning that contradicted the verdict. The wake word is mid-utterance
    # here but the trailing clause addresses the assistant directly ("do YOU
    # know"), so this must be DIRECTED.
    MultiSegmentTestCase(
        name="wake_word_after_narrative_addresses_assistant",
        segments=[
            ("The dude was a lie upon the lie", False),
            ("Mata Hari was never a traitor, she was an honest woman", False),
            ("Jarvis, do you know what she's talking about, about Mata Hari?", False),
        ],
        last_tts_text="",
        in_hot_window=False,
        wake_timestamp=1004.5,
        expected_directed=True,
        expected_query_contains="mata hari",
    ),
]


@dataclass
class NotSpeechCase:
    """Right after her reply, or while his question is still being collected,
    the current segment is either text Whisper made out of silence or words
    he said. Shaped like ``MultiSegmentTestCase`` so the same runner takes
    it: hot-window mode, no wake word, which is how the listener asks in
    both situations."""
    name: str
    segments: list
    last_tts_text: str
    expected_noise: bool
    in_hot_window: bool = True
    wake_timestamp: Optional[float] = None
    aliases: Optional[List[str]] = None
    collecting: bool = False


_FR_ASKED = "Tu peux m'indiquer la météo pour demain ?"
_FR_REPLY = "Pour demain, le ciel sera dégagé, entre 6 et 15 degrés. Prévoyez une petite laine, Monsieur."
_EN_ASKED = "what's the weather like tomorrow"
_EN_REPLY = "Tomorrow will be clear, between 6 and 15 degrees. Bring a jumper."


def _after(asked, reply, current):
    return [(asked, False), (reply, True), (current, False)]


# Field trace 2026-10-29: each of the French-side phantoms below was heard in
# the hot window after a French reply, and answered. Whisper's inventions on
# silence are the closing lines of videos and subtitle credits, in whatever
# language its training data paired with silence, which is why the cases
# cross languages both ways.
NOT_SPEECH_TEST_CASES = [
    NotSpeechCase("russian_outro_after_french", _after(_FR_ASKED, _FR_REPLY, "Продолжение следует..."),
                  _FR_REPLY, expected_noise=True),
    NotSpeechCase("japanese_outro_after_french",
                  _after(_FR_ASKED, _FR_REPLY, "ご視聴ありがとうございました"),
                  _FR_REPLY, expected_noise=True),
    NotSpeechCase("lone_english_thanks_after_french", _after(_FR_ASKED, _FR_REPLY, "Thank you."),
                  _FR_REPLY, expected_noise=True),
    NotSpeechCase("french_subtitle_credit_after_english",
                  _after(_EN_ASKED, _EN_REPLY, "Sous-titres réalisés par la communauté d'Amara.org"),
                  _EN_REPLY, expected_noise=True),
    NotSpeechCase("german_subtitle_credit_after_english",
                  _after(_EN_ASKED, _EN_REPLY, "Untertitel im Auftrag des ZDF, 2017"),
                  _EN_REPLY, expected_noise=True),
    NotSpeechCase("video_outro_after_english",
                  _after(_EN_ASKED, _EN_REPLY, "Thanks for watching, see you in the next video!"),
                  _EN_REPLY, expected_noise=True),
    # Two the prompt neither quotes nor translates.
    NotSpeechCase("korean_news_sign_off_after_french",
                  _after(_FR_ASKED, _FR_REPLY, "MBC 뉴스 이덕영입니다."),
                  _FR_REPLY, expected_noise=True),
    NotSpeechCase("spanish_subscribe_call_after_english",
                  _after(_EN_ASKED, _EN_REPLY, "¡Suscríbete al canal y dale like!"),
                  _EN_REPLY, expected_noise=True),
    # His words: short, thankful, off-topic, odd or misheard, they are still
    # speech. A lone "Thank you." in an English conversation is left out on
    # purpose: without the audio it is Whisper's commonest invention as much
    # as his thanks, and a thanks left unanswered costs him nothing.
    NotSpeechCase("french_follow_up", _after(_FR_ASKED, _FR_REPLY, "Et pour après-demain ?"),
                  _FR_REPLY, expected_noise=False),
    NotSpeechCase("french_thanks", _after(_FR_ASKED, _FR_REPLY, "Merci beaucoup, c'est parfait."),
                  _FR_REPLY, expected_noise=False),
    NotSpeechCase("french_odd_remark", _after(_FR_ASKED, _FR_REPLY, "Ah, je crois que tu as bugué là, non ?"),
                  _FR_REPLY, expected_noise=False),
    NotSpeechCase("english_follow_up", _after(_EN_ASKED, _EN_REPLY, "and what about the weekend?"),
                  _EN_REPLY, expected_noise=False),
    NotSpeechCase("english_thanks", _after(_EN_ASKED, _EN_REPLY, "thank you, that's really helpful"),
                  _EN_REPLY, expected_noise=False),
    NotSpeechCase("english_pushback", _after(_EN_ASKED, _EN_REPLY, "don't you already know that?"),
                  _EN_REPLY, expected_noise=False),
    NotSpeechCase("french_lone_thanks", _after(_FR_ASKED, _FR_REPLY, "Merci."),
                  _FR_REPLY, expected_noise=False),
    NotSpeechCase("french_misheard_follow_up", _after(_FR_ASKED, _FR_REPLY, "et pour à près de main ?"),
                  _FR_REPLY, expected_noise=False),
    NotSpeechCase("english_misheard_follow_up", _after(_EN_ASKED, _EN_REPLY, "and what about the wee can?"),
                  _EN_REPLY, expected_noise=False),
    NotSpeechCase("english_off_topic_remark", _after(_EN_ASKED, _EN_REPLY, "did you see the game last night"),
                  _EN_REPLY, expected_noise=False),
]


# While his question is being collected the judge sees what Whisper wrote
# before his wake word, then the question collected so far, which the
# listener puts back before the utterance it judges. Field trace 2026-11-02,
# Whisper pinned to French: "C'est parti." and "Sous-titrage FR 2021" out of
# silence before "Hey Yuba, comment tu vas ?", then "Sous-titrage ST' 501"
# inside the collection window, which joined the question and was answered.
_FR_BEFORE = [("C'est parti.", False), ("– Sous-titrage FR 2021", False)]
_EN_BEFORE = [("Thank you.", False), ("Thanks for watching!", False)]


def _while_asking(before, asked, current):
    return [*before, (asked, False), (current, False)]


# Her reply ends on a question and he answers at once, his first word within
# the echo tolerance of her last. The listener shows his answer to the judge
# unmarked when it carries words she did not say, in a voice that goes on
# after her last word plus the tolerance; an answer too quiet to prove he
# spoke, or short enough for its voice to stop with hers, is marked
# "(during TTS)", which the prompt says never to extract.
_FR_ASKS_BACK = "Bien reçu, Monsieur. Je reste à votre écoute : que souhaitez-vous savoir ce soir ?"
_EN_ASKS_BACK = "Done. Is there anything else you would like to know tonight?"
_FR_EARLIER = "Note que la réunion est déplacée à jeudi."
_EN_EARLIER = "add milk to the shopping list"

ANSWER_TO_HER_QUESTION_CASES = [
    NotSpeechCase("french_weather_answer",
                  _after(_FR_EARLIER, _FR_ASKS_BACK, "Quel temps fera-t-il demain matin à Lyon ?"),
                  _FR_ASKS_BACK, expected_noise=False),
    NotSpeechCase("french_reminder_answer",
                  _after(_FR_EARLIER, _FR_ASKS_BACK, "Rappelle-moi d'arroser les plantes samedi."),
                  _FR_ASKS_BACK, expected_noise=False),
    NotSpeechCase("french_cooking_answer",
                  _after(_FR_EARLIER, _FR_ASKS_BACK, "Combien de temps faut-il pour cuire un œuf dur ?"),
                  _FR_ASKS_BACK, expected_noise=False),
    NotSpeechCase("english_question_answer",
                  _after(_EN_EARLIER, _EN_ASKS_BACK, "yes, how far is the moon from the earth?"),
                  _EN_ASKS_BACK, expected_noise=False),
    NotSpeechCase("english_reminder_answer",
                  _after(_EN_EARLIER, _EN_ASKS_BACK, "remind me to call the plumber on Friday"),
                  _EN_ASKS_BACK, expected_noise=False),
]


# Her reply ends on a reading, and her last sentence comes back as words she
# never said: Whisper hears the figure as an hour and the unit as a name.
# Through the speakers her voice clears the energy bar, so both proofs hold
# on it; its voice stops with hers, so the listener shows it marked
# "(during TTS)". Each case reaches the judge as written: the early echo
# check does not take it for her reply, and no stripper of her words cuts
# it to a remainder (`test_her_voice_rewritten_reaches_the_judge_whole`
# drives each through the listener).
# Her voice moved air: `expected_noise` is false, which grants it the voice
# the listener measures.
HER_VOICE_REWRITTEN_CASES = [
    NotSpeechCase("french_weight_rewritten",
                  _after("Combien pèse le colis ?",
                         "Le colis est bien arrivé, Monsieur. Il pèse 3,2 kg.",
                         "Il paie ce trois heures deux, Cagé."),
                  "Le colis est bien arrivé, Monsieur. Il pèse 3,2 kg.", expected_noise=False),
    NotSpeechCase("french_temperature_rewritten",
                  _after("Il fait combien dehors ?",
                         "Le ciel est couvert, Monsieur. Il fait 11,8 °C.",
                         "Il fait onze heures huit, Céline."),
                  "Le ciel est couvert, Monsieur. Il fait 11,8 °C.", expected_noise=False),
    NotSpeechCase("french_distance_rewritten",
                  _after("C'est loin, la gare ?",
                         "La gare est à vingt minutes à pied. Le trajet fait 1,6 km.",
                         "Le trajet faisait une heure six, Camille."),
                  "La gare est à vingt minutes à pied. Le trajet fait 1,6 km.",
                  expected_noise=False),
    NotSpeechCase("english_temperature_rewritten",
                  _after("how warm is it outside",
                         "It's a mild evening. It's currently 52.3 °F.",
                         "It's cold, Ren, 5:23, Fiona."),
                  "It's a mild evening. It's currently 52.3 °F.", expected_noise=False),
    NotSpeechCase("english_weight_rewritten",
                  _after("how heavy is the parcel",
                         "The parcel has arrived. It weighs 2.4 kg.",
                         "It waits two four, Kegan."),
                  "The parcel has arrived. It weighs 2.4 kg.", expected_noise=False),
]


def _a_listener_whose_judge_gives_no_verdict():
    """A VoiceListener with its audio stubbed, configured as the unit tests
    configure it, whose judge gives no verdict and keeps every call."""
    cfg = MagicMock()
    for key, value in dict(
            whisper_model="small", whisper_device="auto", whisper_compute_type="int8",
            whisper_backend="faster-whisper", whisper_language=None, sample_rate=16000,
            vad_enabled=False, vad_aggressiveness=2, echo_tolerance=0.3,
            echo_energy_threshold=2.0, hot_window_seconds=3.0, hot_window_enabled=True,
            voice_collect_seconds=2.0, voice_max_collect_seconds=60.0, voice_device=None,
            voice_debug=False, voice_min_energy=0.0045, tune_enabled=False,
            wake_word="jarvis", wake_aliases=[], wake_fuzzy_ratio=0.78,
            stop_commands=["stop", "quiet"], tts_rate=200,
            transcript_buffer_duration_sec=120.0).items():
        setattr(cfg, key, value)
    tts = MagicMock(enabled=True)
    tts.is_speaking.return_value = False
    with patch("jarvis.listening.listener.webrtcvad", None), \
            patch("jarvis.listening.listener.sd", None), \
            patch("jarvis.listening.listener.np", None), \
            patch("jarvis.listening.listener.create_intent_judge", return_value=None):
        from jarvis.listening.listener import VoiceListener
        listener = VoiceListener(MagicMock(), cfg, tts, MagicMock())
    listener._intent_judge = MagicMock(available=True, last_failure_reason="")
    listener._intent_judge.judge.return_value = None
    return listener


def _whisper_hears(listener, text: str, *, started: float, voice_ended: float,
                   level: float) -> None:
    """One utterance through the audio loop's own path, from the voice
    detector closing it now to the decision: it started at *started* on the
    capture clock, its voice went on until *voice_ended*, its audio sits at
    *level*, and Whisper (faked) makes *text* of it."""
    import time
    from types import SimpleNamespace
    import numpy as np

    cfg = listener.cfg
    cfg.whisper_min_audio_duration, cfg.whisper_min_confidence = 0.15, 0.3
    cfg.whisper_no_speech_threshold = 0.5
    listener._whisper_backend, listener._whisper_device = "faster-whisper", "cuda"
    listener.model = MagicMock()
    listener.model.transcribe.return_value = (
        [SimpleNamespace(text=f" {text}", avg_logprob=-0.1, no_speech_prob=0.01)],
        SimpleNamespace(language=None))
    listener.echo_detector.track_utterance_timing(started, 0.0)
    closed = max(time.time(), voice_ended)
    # The detector counts the silence after the voice in frames.
    listener._silence_frames = round((closed - voice_ended) * 1000 / listener._frame_ms)
    listener._frame_captured_at = closed
    listener._utterance_frames = [np.full(listener._frame_samples, level, dtype=np.float32)
                                  for _ in range(50)]
    listener.is_speech_active = True
    listener._finalize_utterance()


@pytest.mark.parametrize("voice_stops_with_hers", [False, True],
                         ids=["voice_on_after_hers", "voice_with_hers"])
@pytest.mark.parametrize("case", HER_VOICE_REWRITTEN_CASES, ids=lambda case: case.name)
def test_her_voice_rewritten_reaches_the_judge_whole(case, voice_stops_with_hers):
    """Needs no model. A case measures what the judge is shown only if the
    listener shows it whole. Heard from 1.2 s before the last word of a
    reply that played for six seconds, at the level her voice reaches the
    microphone, it is neither taken for her reply coming back nor cut to
    what follows her words: the judge is shown it unmarked when its voice
    goes on after hers, where the two proofs decide, and marked
    "(during TTS)" when its voice stops with hers."""
    import time

    heard, reply = case.segments[-1][0], case.last_tts_text
    floor = 0.005
    listener = _a_listener_whose_judge_gives_no_verdict()
    detector = listener.echo_detector
    try:
        detector.track_tts_start(reply, baseline_energy=floor,
                                 exact_duration=6.0, start_delay=-6.0)
        last_word = time.time()
        detector.track_tts_finish()
        listener.state_manager.schedule_hot_window_activation()
        while (not listener.state_manager.is_hot_window_active()
               and time.time() < last_word + 1.0):
            time.sleep(0.01)
        tolerance = detector.echo_tolerance
        voice_ended = last_word + (tolerance / 3 if voice_stops_with_hers else 2 * tolerance)
        time.sleep(max(0.0, voice_ended - time.time()))
        with patch("builtins.print"):
            _whisper_hears(listener, heard, started=last_word - 1.2, voice_ended=voice_ended,
                           level=floor * detector.energy_spike_threshold * 3.0)

        calls = listener._intent_judge.judge.call_args_list
        assert calls, "the listener set the case aside before the judge saw it"
        shown = calls[-1].kwargs
        assert shown["current_text"] == heard.lower(), "the judge was shown part of the case"
        judged = [segment for segment in shown["segments"]
                  if segment.text.lower().strip() == heard.lower()]
        assert judged and judged[-1].is_during_tts == voice_stops_with_hers
    finally:
        listener._stop_thinking_tune()
        listener.state_manager.stop()


def _marked_as_hers(case: NotSpeechCase) -> NotSpeechCase:
    """The case with the current segment marked "(during TTS)"."""
    text, _ = case.segments[-1]
    return dataclasses.replace(case, segments=[*case.segments[:-1], (text, True)])


COLLECTING_TEST_CASES = [
    NotSpeechCase("french_subtitle_number_while_asking",
                  _while_asking(_FR_BEFORE, "comment tu vas ?",
                                "Sous-titrage ST' 501"),
                  "", expected_noise=True, collecting=True),
    NotSpeechCase("french_amara_credit_while_asking",
                  _while_asking(_FR_BEFORE, "tu peux me rappeler d'appeler le garage demain ?",
                                "Sous-titres réalisés par la communauté d'Amara.org"),
                  "", expected_noise=True, collecting=True),
    NotSpeechCase("french_video_outro_while_asking",
                  _while_asking(_FR_BEFORE, "c'est quoi la capitale de l'Australie ?",
                                "Merci d'avoir regardé cette vidéo !"),
                  "", expected_noise=True, collecting=True),
    NotSpeechCase("french_subtitle_credit_after_a_quiet_room",
                  _while_asking([], "comment tu vas ? ça va aujourd'hui ?",
                                "Sous-titrage ST' 501"),
                  "", expected_noise=True, collecting=True),
    NotSpeechCase("english_subtitle_credit_while_asking",
                  _while_asking(_EN_BEFORE, "what's on my calendar tomorrow",
                                "Subtitles by the Amara.org community"),
                  "", expected_noise=True, collecting=True),
    NotSpeechCase("english_video_outro_while_asking",
                  _while_asking(_EN_BEFORE, "play something relaxing",
                                "See you in the next video!"),
                  "", expected_noise=True, collecting=True),
    NotSpeechCase("english_outro_after_her_last_reply",
                  _while_asking(_EN_BEFORE, "and how long does it take to get there",
                                "Thanks for watching, and don't forget to subscribe."),
                  _EN_REPLY, expected_noise=True, collecting=True),
    # The rest of his question, said with no wake word, sometimes with no
    # question in it at all.
    NotSpeechCase("french_second_question_while_asking",
                  _while_asking(_FR_BEFORE, "comment tu vas ?",
                                "Ça va aujourd'hui ?"),
                  "", expected_noise=False, collecting=True),
    NotSpeechCase("french_added_detail_while_asking",
                  _while_asking(_FR_BEFORE, "tu peux me trouver un restaurant italien ?",
                                "Enfin, je veux dire pour ce week-end."),
                  "", expected_noise=False, collecting=True),
    NotSpeechCase("french_check_while_asking",
                  _while_asking(_FR_BEFORE, "mets-moi un minuteur de dix minutes",
                                "Tu m'entends ?"),
                  "", expected_noise=False, collecting=True),
    NotSpeechCase("french_misheard_addition_while_asking",
                  _while_asking([], "quel temps fera-t-il demain ?",
                                "et pour à près de main aussi"),
                  "", expected_noise=False, collecting=True),
    NotSpeechCase("english_added_detail_while_asking",
                  _while_asking(_EN_BEFORE, "find me a good bakery",
                                "I mean the one in Lisbon."),
                  "", expected_noise=False, collecting=True),
    NotSpeechCase("english_second_question_while_asking",
                  _while_asking(_EN_BEFORE, "what time does the museum close",
                                "and is it open on Sundays?"),
                  "", expected_noise=False, collecting=True),
    NotSpeechCase("english_addition_after_her_last_reply",
                  _while_asking([], "book a table at that restaurant",
                                "for two people, please"),
                  _EN_REPLY, expected_noise=False, collecting=True),
]


# Cases known to fail with the small model on the current prompt.
# Track regressions / future prompt improvements here.
KNOWN_FAILING_CASES: set = {
    # gemma4:e2b calls subtitle credits noise but not the closing lines of a
    # video or a broadcast (docs/llm_contexts.md, "2. Intent Judge").
    "russian_outro_after_french",
    "japanese_outro_after_french",
    "lone_english_thanks_after_french",
    "video_outro_after_english",
    "korean_news_sign_off_after_french",
    "spanish_subscribe_call_after_english",
    "french_video_outro_while_asking",
    "english_video_outro_while_asking",
    "english_outro_after_her_last_reply",
    # Half the time or more: the field case itself, a subtitle credit with a
    # number.
    "french_subtitle_number_while_asking",
}


# =============================================================================
# Helper Functions
# =============================================================================

def _as_substring_list(value):
    """Normalise an expected_query_contains / _not_contains value to a list."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def create_transcript_segment(
    text: str,
    start_time: float = 1000.0,
    is_during_tts: bool = False,
    processed: bool = False,
):
    """Create a TranscriptSegment for testing."""
    from jarvis.listening.transcript_buffer import TranscriptSegment
    return TranscriptSegment(
        text=text,
        start_time=start_time,
        end_time=start_time + 2.0,
        energy=0.01,
        is_during_tts=is_during_tts,
        processed=processed,
    )


def run_intent_judge(case: IntentJudgeTestCase):
    """Run the intent judge on a test case."""
    from jarvis.listening.intent_judge import IntentJudge, IntentJudgeConfig

    judge = IntentJudge(IntentJudgeConfig(
        assistant_name="Jarvis",
        model="gemma4:e2b",
        timeout_sec=10.0,
    ))

    if not judge.available:
        return None

    segments = [create_transcript_segment(case.transcript)]

    return judge.judge(
        segments=segments,
        wake_timestamp=case.wake_timestamp,
        last_tts_text=case.last_tts_text,
        last_tts_finish_time=999.0 if case.last_tts_text else 0.0,
        in_hot_window=case.in_hot_window,
        current_text=case.transcript,
    )


def run_intent_judge_multi_segment(case: "MultiSegmentTestCase", judge=None,
                                   current_text: Optional[str] = None):
    """Run the intent judge on a multi-segment test case (on ``gemma4:e2b``
    unless a judge is given). The segment judged is the last one not marked
    "(during TTS)", unless *current_text* names it."""
    from jarvis.listening.intent_judge import IntentJudge, IntentJudgeConfig

    judge = judge or IntentJudge(IntentJudgeConfig(
        assistant_name="Jarvis",
        aliases=list(case.aliases or []),
        model="gemma4:e2b",
        timeout_sec=10.0,
    ))

    if not judge.available:
        return None

    segments = []
    base_time = 1000.0
    for i, (text, is_during_tts) in enumerate(case.segments):
        segments.append(create_transcript_segment(
            text=text,
            start_time=base_time + (i * 2.0),
            is_during_tts=is_during_tts,
        ))

    if current_text is None:
        current_text = ""
        for text, is_during_tts in reversed(case.segments):
            if not is_during_tts:
                current_text = text
                break

    return judge.judge(
        segments=segments,
        wake_timestamp=case.wake_timestamp,
        last_tts_text=case.last_tts_text,
        last_tts_finish_time=999.0 if case.last_tts_text else 0.0,
        in_hot_window=case.in_hot_window,
        current_text=current_text,
    )


def is_intent_judge_available() -> bool:
    """Check if the intent judge model is available."""
    import requests
    try:
        resp = requests.get("http://127.0.0.1:11434/api/tags", timeout=2)
        if resp.status_code != 200:
            return False
        data = resp.json()
        models = [m.get("name", "") for m in data.get("models", [])]
        return any("gemma4" in m for m in models)
    except Exception:
        return False


def _skip_if_not_intent_judge_phase():
    """Intent judge tests are fixed to gemma4:e2b and would run twice under the
    multi-model eval matrix. Skip during the large-model phase to keep runtime
    down; they still run once during the small-model (gemma4) phase."""
    if "gemma4" not in JUDGE_MODEL:
        pytest.skip(f"Intent judge tests only run in the gemma4 phase (current: {JUDGE_MODEL})")


# =============================================================================
# Tests
# =============================================================================

class TestIntentJudgeAccuracy:
    """Evals for intent judge accuracy."""

    @pytest.mark.parametrize("case", INTENT_JUDGE_TEST_CASES, ids=lambda c: c.name)
    def test_intent_judge_case(self, case: IntentJudgeTestCase):
        _skip_if_not_intent_judge_phase()
        if not is_intent_judge_available():
            pytest.skip("Intent judge model (gemma4) not available")

        if case.name in KNOWN_FAILING_CASES:
            pytest.xfail(f"Known issue: {case.name} needs prompt improvement")

        result = run_intent_judge(case)

        if result is None:
            pytest.fail("Intent judge returned None")

        print(f"\n{'='*60}")
        print(f"Test Case: {case.name}")
        print(f"Transcript: {case.transcript}")
        print(f"TTS: {case.last_tts_text[:50]}..." if case.last_tts_text else "TTS: None")
        print(f"Mode: {'hot_window' if case.in_hot_window else 'wake_word'}")
        print(f"{'='*60}")
        print(f"Result: directed={result.directed}, query='{result.query}', stop={result.stop}")
        print(f"Confidence: {result.confidence}")
        print(f"Reasoning: {result.reasoning}")
        print(f"{'='*60}")

        assert result.directed == case.expected_directed, (
            f"Expected directed={case.expected_directed}, got {result.directed}. "
            f"Reasoning: {result.reasoning}"
        )
        assert result.stop == case.expected_stop, (
            f"Expected stop={case.expected_stop}, got {result.stop}. "
            f"Reasoning: {result.reasoning}"
        )
        for needle in _as_substring_list(case.expected_query_contains):
            assert needle.lower() in (result.query or "").lower(), (
                f"Expected query to contain '{needle}', "
                f"got '{result.query}'. Reasoning: {result.reasoning}"
            )
        if result.query:
            for needle in _as_substring_list(case.expected_query_not_contains):
                assert needle.lower() not in result.query.lower(), (
                    f"Expected query to NOT contain '{needle}', "
                    f"got '{result.query}'. Reasoning: {result.reasoning}"
                )


class TestIntentJudgePromptQuality:
    """Tests for intent judge prompt construction quality."""

    def test_hot_window_mode_indicated_in_prompt(self):
        from jarvis.listening.intent_judge import IntentJudge

        judge = IntentJudge()
        segments = [create_transcript_segment("hello")]

        prompt = judge._build_user_prompt(
            segments=segments,
            wake_timestamp=None,
            last_tts_text="Test TTS",
            last_tts_finish_time=999.0,
            in_hot_window=True,
        )

        assert "HOT WINDOW" in prompt

    def test_tts_text_included_for_echo_detection(self):
        from jarvis.listening.intent_judge import IntentJudge

        judge = IntentJudge()
        segments = [create_transcript_segment("The weather is nice")]
        tts_text = "The weather today is nice and sunny"

        prompt = judge._build_user_prompt(
            segments=segments,
            wake_timestamp=None,
            last_tts_text=tts_text,
            last_tts_finish_time=999.0,
            in_hot_window=True,
        )

        assert "nice and sunny" in prompt

    def test_system_prompt_has_echo_guidance(self):
        from jarvis.listening.intent_judge import IntentJudge

        judge = IntentJudge()
        prompt = judge._build_system_prompt()

        assert "echo" in prompt.lower()
        assert "(during TTS)" in prompt


class TestIntentJudgeFallback:
    """Tests for intent judge fallback behaviour."""

    def test_returns_none_when_ollama_unavailable(self):
        from jarvis.listening.intent_judge import IntentJudge, IntentJudgeConfig

        from types import SimpleNamespace

        # The endpoint lives on the settings object the judge dispatches
        # through, not on its own config: every LLM call goes via
        # `get_auxiliary_backend(cfg, model)`.
        judge = IntentJudge(IntentJudgeConfig(
            cfg=SimpleNamespace(
                llm_provider="ollama",
                ollama_base_url="http://127.0.0.1:99999",
            ),
            timeout_sec=1.0,
        ))

        segments = [create_transcript_segment("test")]
        result = judge.judge(segments)

        assert result is None


class TestIntentJudgeMultiSegment:
    """Evals for intent judge with realistic multi-segment transcript buffers."""

    @pytest.mark.parametrize("case", MULTI_SEGMENT_TEST_CASES, ids=lambda c: c.name)
    def test_multi_segment_case(self, case: MultiSegmentTestCase):
        _skip_if_not_intent_judge_phase()
        if not is_intent_judge_available():
            pytest.skip("Intent judge model (gemma4) not available")

        if case.name in KNOWN_FAILING_CASES:
            pytest.xfail(f"Known issue: {case.name} needs prompt improvement")

        result = run_intent_judge_multi_segment(case)

        if result is None:
            pytest.fail("Intent judge returned None")

        print(f"\n{'='*60}")
        print(f"Test Case: {case.name}")
        print(f"Segments:")
        for text, is_tts in case.segments:
            marker = " (during TTS)" if is_tts else ""
            print(f"  - \"{text}\"{marker}")
        print(f"TTS: {case.last_tts_text[:50]}..." if case.last_tts_text else "TTS: None")
        print(f"Mode: {'hot_window' if case.in_hot_window else 'wake_word'}")
        print(f"{'='*60}")
        print(f"Result: directed={result.directed}, query='{result.query}', stop={result.stop}")
        print(f"Confidence: {result.confidence}")
        print(f"Reasoning: {result.reasoning}")
        print(f"{'='*60}")

        assert result.directed == case.expected_directed, (
            f"Expected directed={case.expected_directed}, got {result.directed}. "
            f"Reasoning: {result.reasoning}"
        )
        assert result.stop == case.expected_stop, (
            f"Expected stop={case.expected_stop}, got {result.stop}. "
            f"Reasoning: {result.reasoning}"
        )
        for needle in _as_substring_list(case.expected_query_contains):
            assert needle.lower() in (result.query or "").lower(), (
                f"Expected query to contain '{needle}', "
                f"got '{result.query}'. Reasoning: {result.reasoning}"
            )
        if result.query:
            for needle in _as_substring_list(case.expected_query_not_contains):
                assert needle.lower() not in result.query.lower(), (
                    f"Expected query to NOT contain '{needle}', "
                    f"got '{result.query}'. Reasoning: {result.reasoning}"
                )


def _answered_in_the_hot_window(case, verdict) -> bool:
    """What the listener does in the hot window with a verdict on a segment
    it did not mark as possibly hers: it dispatches a directed one, and its
    overrides take any rejection that does not say noise, whatever the
    confidence. A rejection whose reasoning says echo is overridden only on
    words she did not say, spoken in a voice: the words are weighed here as
    the listener weighs them, and his words carry a voice where text Whisper
    makes of silence carries none. The cases carry no wake word, so the
    wake-word check never takes the decision."""
    if verdict.directed:
        return True
    if verdict.noise:
        return False
    if "echo" in (verdict.reasoning or "").lower():
        return (not case.expected_noise
                and _carries_speech_she_did_not_say(case.segments[-1][0], case.last_tts_text))
    return True


def _answered_when_marked(case, verdict) -> bool:
    """What the listener does in the hot window with a verdict on a segment
    it marked "(during TTS)", words that could be hers: no override takes
    them, and the cases carry no wake word, so only a directed verdict with
    a query is dispatched."""
    return verdict.directed and bool(verdict.query)


def _answered_on_an_unsure_verdict(case, verdict) -> bool:
    """Answered in the hot window on a verdict below high confidence that
    extracted no query: what weighing the judge's unsure verdicts as its
    sure ones lets through. A directed verdict with a query is dispatched
    whatever its confidence."""
    return (_answered_in_the_hot_window(case, verdict) and verdict.confidence != "high"
            and not (verdict.directed and verdict.query))


def _joins_his_question(verdict) -> bool:
    """What the listener does with a verdict on a segment heard while his
    question is collected: a stop cancels the question, a not-directed noise
    verdict keeps the segment out, and anything else joins it."""
    return not verdict.stop and not (verdict.noise and not verdict.directed)


def _taken(case, verdict) -> bool:
    """Whether the listener would act on the segment as his words."""
    if case.collecting:
        return _joins_his_question(verdict)
    return _answered_in_the_hot_window(case, verdict)


def _wrong(case, verdict) -> bool:
    """Whether the listener would get the segment wrong. While his question
    is collected a stop is wrong either way: it drops his question."""
    return _taken(case, verdict) == case.expected_noise or (case.collecting and verdict.stop)


def _without_his_question(case: NotSpeechCase) -> NotSpeechCase:
    """A collecting case as the judge would read it if the listener did not
    put the question back: its segments are processed, so it is not shown."""
    return dataclasses.replace(case, segments=case.segments[:-2] + case.segments[-1:])


def _judge_on(model: str):
    from jarvis.listening.intent_judge import IntentJudge, IntentJudgeConfig

    return IntentJudge(IntentJudgeConfig(
        assistant_name="Jarvis", model=model, cfg=MockConfig(), timeout_sec=30.0,
    ))


def _warm_judges() -> dict:
    """A judge per arm, each asked once unmeasured, so loading a model is
    not scored as a miss."""
    judges = {arm: _judge_on(arm) for arm in NOT_SPEECH_ARMS}
    cold = [arm for arm, judge in judges.items()
            if run_intent_judge_multi_segment(NOT_SPEECH_TEST_CASES[0], judge=judge) is None]
    assert not cold, f"no answer to a warm-up from {cold}: check the model names and the provider"
    return judges


def _interleaved(arms: list, cases: list):
    """Every case to every arm, the arms alternating call by call: their
    order rotates each call and the cases are shuffled each round."""
    rng = random.Random(0)
    for rnd in range(NOT_SPEECH_ROUNDS):
        order = list(cases)
        rng.shuffle(order)
        for i, case in enumerate(order):
            turn = (i + rnd) % len(arms)
            for arm in arms[turn:] + arms[:turn]:
                yield case, arm


def _rate(count: int, scored: int) -> str:
    """A rate only where there are calls enough to mean one."""
    return f" ({count / scored:.0%})" if scored >= 20 else ""


class TestIntentJudgeKnowsSilenceFromSpeech:
    """The hot-window overrides never reverse a not-directed ``noise``
    verdict, and such a segment does not join a question being collected, so
    it must be set on Whisper's inventions and never on his words. Whether he
    is directed is the other evals' question."""

    @pytest.mark.parametrize("case", NOT_SPEECH_TEST_CASES + COLLECTING_TEST_CASES,
                             ids=lambda c: c.name)
    def test_not_speech_case(self, case: NotSpeechCase):
        _skip_if_not_intent_judge_phase()
        if not is_intent_judge_available():
            pytest.skip("Intent judge model (gemma4) not available")

        result = run_intent_judge_multi_segment(case)

        if result is None:
            pytest.fail("Intent judge returned None")

        print(f"\nTest Case: {case.name} | current: \"{case.segments[-1][0]}\"")
        print(f"Result: noise={result.noise}, directed={result.directed}, stop={result.stop}, "
              f"query='{result.query}'")
        print(f"Reasoning: {result.reasoning}")

        if case.expected_noise:
            ok = result.noise and not result.directed
            msg = (f"Expected noise (not directed), got noise={result.noise}, "
                   f"directed={result.directed}. Reasoning: {result.reasoning}")
        else:
            ok = not result.noise
            msg = f"Expected his speech, got noise=True. Reasoning: {result.reasoning}"
        if ok and case.collecting and result.stop:
            ok, msg = False, f"A stop would drop his question. Reasoning: {result.reasoning}"
        if not ok and case.name in KNOWN_FAILING_CASES:
            pytest.xfail(f"Known gemma4:e2b limitation: {msg}")
        assert ok, msg


@pytest.mark.skipif(not NOT_SPEECH_ARMS, reason="name one or more judge models in EVAL_INTENT_JUDGE_ARMS")
class TestIntentJudgeNotSpeechArms:

    def test_arms_hear_the_same_segments_call_by_call(self, capsys):
        judges = _warm_judges()
        kinds = {True: "inventions", False: "his words"}
        situations = {False: ("after her reply", "answered"), True: ("while he asks", "joined")}
        # taken, noise, stop, calls, no verdict, answered on an unsure verdict
        tally = {(arm, situation, kind): [0, 0, 0, 0, 0, 0] for arm in NOT_SPEECH_ARMS
                 for situation, _ in situations.values() for kind in kinds.values()}
        misses = {arm: {} for arm in NOT_SPEECH_ARMS}   # case -> wrong outcomes
        for case, arm in _interleaved(NOT_SPEECH_ARMS, NOT_SPEECH_TEST_CASES + COLLECTING_TEST_CASES):
            judges[arm]._last_error_time = 0.0   # one timeout must not mute the arm
            verdict = run_intent_judge_multi_segment(case, judge=judges[arm])
            row = tally[(arm, situations[case.collecting][0], kinds[case.expected_noise])]
            row[3] += 1
            if verdict is None:
                row[4] += 1
                continue
            row[0] += int(_taken(case, verdict))
            row[1] += int(verdict.noise)
            row[2] += int(case.collecting and verdict.stop)
            row[5] += int(not case.collecting and _answered_on_an_unsure_verdict(case, verdict))
            if _wrong(case, verdict):
                misses[arm][case.name] = misses[arm].get(case.name, 0) + 1

        lines = [""]
        verbs = dict(situations.values())
        for (arm, situation, kind), (taken, noise, stop, calls, silent, unsure) in tally.items():
            scored = calls - silent
            stops = f", stop {stop}" if situation == situations[True][0] else ""
            unsure_part = f" ({unsure} on an unsure verdict)" if situation == situations[False][0] else ""
            lines.append(f"🧪 {arm:24} {situation:15} {kind:10}  {verbs[situation]} "
                         f"{taken}/{scored}{_rate(taken, scored)}{unsure_part}, noise {noise}/{scored}"
                         f"{stops}, no verdict {silent}")
        for arm, wrong in misses.items():
            for name, count in sorted(wrong.items()):
                lines.append(f"   ↳ {arm}: {name} wrong {count}/{NOT_SPEECH_ROUNDS}")
        with capsys.disabled():
            print("\n".join(lines))

        mute = [arm for arm in NOT_SPEECH_ARMS
                if all(row[3] == row[4] for (who, _, _), row in tally.items() if who == arm)]
        assert not mute, f"no answer at all from {mute}: check the model names and the provider"

    def test_his_question_shown_against_hidden_call_by_call(self, capsys):
        """The listener puts the question being collected back before the
        utterance it judges. Each arm reads every collecting case with the
        question shown and with it hidden, as it is once its segments are
        processed, the two alternating call by call."""
        judges = _warm_judges()
        kinds = {True: "inventions", False: "his words"}
        variants = {"shown": lambda case: case, "hidden": _without_his_question}
        arms = [(arm, variant) for arm in NOT_SPEECH_ARMS for variant in variants]
        tally = {(arm, kind): [0, 0, 0, 0] for arm in arms   # joined, stop, calls, no verdict
                 for kind in kinds.values()}
        misses = {}   # (arm, case) -> wrong outcomes
        for case, arm in _interleaved(arms, COLLECTING_TEST_CASES):
            model, variant = arm
            judges[model]._last_error_time = 0.0
            verdict = run_intent_judge_multi_segment(variants[variant](case), judge=judges[model])
            row = tally[(arm, kinds[case.expected_noise])]
            row[2] += 1
            if verdict is None:
                row[3] += 1
                continue
            row[0] += int(_joins_his_question(verdict))
            row[1] += int(verdict.stop)
            if _wrong(case, verdict):
                misses[(arm, case.name)] = misses.get((arm, case.name), 0) + 1

        lines = [""]
        for ((model, variant), kind), (joined, stop, calls, silent) in tally.items():
            scored = calls - silent
            lines.append(f"🧪 {model:24} question {variant:6} {kind:10}  joined "
                         f"{joined}/{scored}{_rate(joined, scored)}, stop {stop}, "
                         f"no verdict {silent}")
        for ((model, variant), name), count in sorted(misses.items()):
            lines.append(f"   ↳ {model} {variant}: {name} wrong {count}/{NOT_SPEECH_ROUNDS}")
        with capsys.disabled():
            print("\n".join(lines))

        mute = [arm for arm in arms
                if all(row[2] == row[3] for (who, _), row in tally.items() if who == arm)]
        assert not mute, f"no answer at all from {mute}: check the model names and the provider"

    def test_his_answer_unmarked_against_marked_call_by_call(self, capsys):
        """His answer to her question, begun the moment she stops. Each arm
        reads every case unmarked, as the listener shows it when he is heard
        in a voice that goes on after hers, and marked "(during TTS)", as it
        shows an answer too quiet to prove he spoke, the two alternating
        call by call. Directed is the judge taking it as his; answered is
        what the listener then does in the hot window, where a marked one
        is answered only on a directed verdict with a query."""
        judges = _warm_judges()
        variants = {"unmarked": lambda case: case, "marked": _marked_as_hers}
        answered_when = {"unmarked": _answered_in_the_hot_window,
                         "marked": _answered_when_marked}
        arms = [(arm, variant) for arm in NOT_SPEECH_ARMS for variant in variants]
        tally = {arm: [0, 0, 0, 0, 0] for arm in arms}   # directed, answered, noise, calls, no verdict
        misses = {}   # (arm, case) -> not directed
        for case, arm in _interleaved(arms, ANSWER_TO_HER_QUESTION_CASES):
            model, variant = arm
            judges[model]._last_error_time = 0.0
            shown = variants[variant](case)
            verdict = run_intent_judge_multi_segment(
                shown, judge=judges[model], current_text=shown.segments[-1][0])
            row = tally[arm]
            row[3] += 1
            if verdict is None:
                row[4] += 1
                continue
            row[0] += int(verdict.directed)
            row[1] += int(answered_when[variant](shown, verdict))
            row[2] += int(verdict.noise)
            if not verdict.directed:
                misses[(arm, case.name)] = misses.get((arm, case.name), 0) + 1

        lines = [""]
        for (model, variant), (directed, answered, noise, calls, silent) in tally.items():
            scored = calls - silent
            lines.append(f"🧪 {model:24} his answer {variant:8}  directed "
                         f"{directed}/{scored}{_rate(directed, scored)}, answered "
                         f"{answered}/{scored}, noise {noise}/{scored}, no verdict {silent}")
        for ((model, variant), name), count in sorted(misses.items()):
            lines.append(f"   ↳ {model} {variant}: {name} not directed {count}/{NOT_SPEECH_ROUNDS}")
        with capsys.disabled():
            print("\n".join(lines))

        mute = [arm for arm in arms if tally[arm][3] == tally[arm][4]]
        assert not mute, f"no answer at all from {mute}: check the model names and the provider"

    def test_her_voice_rewritten_unmarked_against_marked_call_by_call(self, capsys):
        """Her last sentence coming back as words she never said, its voice
        stopping with hers. Each arm reads every case unmarked, as the judge
        is shown it when the two proofs alone decide, where any verdict but
        noise has it answered, and marked "(during TTS)", as the listener
        shows an utterance whose voice stops with hers, where only a
        directed verdict with a query is answered, the two alternating call
        by call. Every answer is a miss: these are her words."""
        judges = _warm_judges()
        variants = {"unmarked": lambda case: case, "marked": _marked_as_hers}
        answered_when = {"unmarked": _answered_in_the_hot_window,
                         "marked": _answered_when_marked}
        arms = [(arm, variant) for arm in NOT_SPEECH_ARMS for variant in variants]
        tally = {arm: [0, 0, 0, 0, 0] for arm in arms}   # answered, directed, noise, calls, no verdict
        misses = {}   # (arm, case) -> answered
        for case, arm in _interleaved(arms, HER_VOICE_REWRITTEN_CASES):
            model, variant = arm
            judges[model]._last_error_time = 0.0
            shown = variants[variant](case)
            verdict = run_intent_judge_multi_segment(
                shown, judge=judges[model], current_text=shown.segments[-1][0])
            row = tally[arm]
            row[3] += 1
            if verdict is None:
                row[4] += 1
                continue
            answered = answered_when[variant](shown, verdict)
            row[0] += int(answered)
            row[1] += int(verdict.directed)
            row[2] += int(verdict.noise)
            if answered:
                misses[(arm, case.name)] = misses.get((arm, case.name), 0) + 1

        lines = [""]
        for (model, variant), (answered, directed, noise, calls, silent) in tally.items():
            scored = calls - silent
            lines.append(f"🧪 {model:24} her voice rewritten {variant:8}  answered "
                         f"{answered}/{scored}{_rate(answered, scored)}, directed "
                         f"{directed}/{scored}, noise {noise}/{scored}, no verdict {silent}")
        for ((model, variant), name), count in sorted(misses.items()):
            lines.append(f"   ↳ {model} {variant}: {name} answered {count}/{NOT_SPEECH_ROUNDS}")
        with capsys.disabled():
            print("\n".join(lines))

        mute = [arm for arm in arms if tally[arm][3] == tally[arm][4]]
        assert not mute, f"no answer at all from {mute}: check the model names and the provider"


class TestProcessedSegmentFiltering:
    """Tests for processed segment filtering in intent judge."""

    def test_processed_segment_not_reextracted(self):
        _skip_if_not_intent_judge_phase()
        if not is_intent_judge_available():
            pytest.skip("Intent judge model (gemma4) not available")

        from jarvis.listening.intent_judge import IntentJudge, IntentJudgeConfig

        judge = IntentJudge(IntentJudgeConfig(
            assistant_name="Jarvis",
            model="gemma4:e2b",
            timeout_sec=10.0,
        ))

        segments = [
            create_transcript_segment(
                text="Jarvis what's the weather in London",
                start_time=1000.0,
                processed=True,
            ),
            create_transcript_segment(
                text="Jarvis tell me a random topic",
                start_time=1010.0,
                processed=False,
            ),
        ]

        result = judge.judge(
            segments=segments,
            wake_timestamp=1010.0,
            last_tts_text="",
            last_tts_finish_time=0.0,
            in_hot_window=False,
            current_text="Jarvis tell me a random topic",
        )

        assert result is not None
        assert result.directed is True
        assert "random" in result.query.lower() or "topic" in result.query.lower(), (
            f"Expected query about 'random topic', got '{result.query}'."
        )
        assert "weather" not in result.query.lower(), (
            f"Query contains 'weather' from processed segment: '{result.query}'"
        )

        print(f"\n✅ Correctly extracted new query: '{result.query}'")

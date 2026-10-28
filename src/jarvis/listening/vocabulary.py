"""The names he uses, given to Whisper as a hint.

See listening.spec.md, "The names he uses".
"""

from __future__ import annotations

import re
from typing import Optional

from ..debug import debug_log
from .wake_detection import is_wake_word_detected

# A transcript at least this much made of the hint's own words is the
# hint repeated back. Measured: a fourteen-name list read back over noise
# sat at 88-94% hint words, a sentence that merely names one of them at
# 50% or less.
_HINT_SHARE = 0.75

_WORDS = re.compile(r"\w+", re.UNICODE)


def _her_names(cfg) -> tuple[str, list[str]]:
    wake = getattr(cfg, "wake_word", "")
    aliases = getattr(cfg, "wake_aliases", [])
    wake = wake.strip().lower() if isinstance(wake, str) else ""
    if not isinstance(aliases, (list, tuple)):
        aliases = []
    aliases = [a.strip().lower() for a in aliases if isinstance(a, str) and a.strip()]
    return wake, aliases


def whisper_hint(cfg) -> Optional[str]:
    """His names joined for Whisper, or None when there are none.

    `weather_city` first, then `whisper_vocabulary`, each once. A name
    the wake-word matcher recognises is left out: the hint makes Whisper
    write its words more readily, and writing hers more readily is a
    false wake.
    """
    vocabulary = getattr(cfg, "whisper_vocabulary", [])
    if not isinstance(vocabulary, (list, tuple)):
        vocabulary = []
    wake, aliases = _her_names(cfg)

    names: list[str] = []
    seen: set[str] = set()
    for name in [getattr(cfg, "weather_city", ""), *vocabulary]:
        if not isinstance(name, str) or not name.strip():
            continue
        name = name.strip()
        if name.casefold() in seen:
            continue
        if (wake or aliases) and is_wake_word_detected(name.lower(), wake or aliases[0], aliases):
            debug_log(f"'{name}' left out of the Whisper hint: it sounds like her name", "voice")
            continue
        seen.add(name.casefold())
        names.append(name)
    return ", ".join(names) or None


def is_the_hint_repeated(text: str, hint: Optional[str]) -> bool:
    """Whether a transcript is mostly the hint's own words.

    Over noise Whisper writes the hint back, and nothing in its scores
    tells that apart from speech. A sentence that merely names one of
    the names is mostly other words.
    """
    if not hint:
        return False
    words = _WORDS.findall((text or "").casefold())
    if not words:
        return False
    hint_words = set(_WORDS.findall(hint.casefold()))
    return sum(w in hint_words for w in words) / len(words) >= _HINT_SHARE

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

# How long the hint may be, in UTF-8 bytes. Whisper keeps 223 tokens of a
# hint and drops the rest without a word. A token is at least one byte,
# so a hint this long is read whole.
HINT_BYTES = 200

# Why a name is not in the hint.
SOUNDS_LIKE_HER = "sounds like her"
NO_ROOM = "no room"

_SEPARATOR = ", "
_WORDS = re.compile(r"\w+", re.UNICODE)


def _her_names(cfg) -> tuple[str, list[str]]:
    wake = getattr(cfg, "wake_word", "")
    aliases = getattr(cfg, "wake_aliases", [])
    wake = wake.strip().lower() if isinstance(wake, str) else ""
    if not isinstance(aliases, (list, tuple)):
        aliases = []
    aliases = [a.strip().lower() for a in aliases if isinstance(a, str) and a.strip()]
    return wake, aliases


def his_names(cfg) -> tuple[list[str], list[tuple[str, str]]]:
    """The names the hint is made of, and those left out with why.

    `weather_city` first, then `whisper_vocabulary`, each once. A name
    the wake-word matcher recognises, as close as `wake_fuzzy_ratio`
    allows, is left out: the hint makes Whisper write its words more
    readily, and writing hers more readily is a false wake. So is a name
    the hint has no room left for (`HINT_BYTES`): the names before it
    stay, and a shorter one after it can still fit.
    """
    vocabulary = getattr(cfg, "whisper_vocabulary", [])
    if not isinstance(vocabulary, (list, tuple)):
        vocabulary = []
    wake, aliases = _her_names(cfg)
    ratio = getattr(cfg, "wake_fuzzy_ratio", None)
    closeness = ({"fuzzy_ratio": float(ratio)}
                 if isinstance(ratio, (int, float)) and not isinstance(ratio, bool) else {})

    names: list[str] = []
    left_out: list[tuple[str, str]] = []
    seen: set[str] = set()
    room = HINT_BYTES
    for name in [getattr(cfg, "weather_city", ""), *vocabulary]:
        if not isinstance(name, str) or not name.strip():
            continue
        name = name.strip()
        if name.casefold() in seen:
            continue
        seen.add(name.casefold())
        if (wake or aliases) and is_wake_word_detected(
                name.lower(), wake or aliases[0], aliases, **closeness):
            left_out.append((name, SOUNDS_LIKE_HER))
            continue
        size = len(name.encode("utf-8")) + (len(_SEPARATOR) if names else 0)
        if size > room:
            left_out.append((name, NO_ROOM))
            continue
        room -= size
        names.append(name)
    return names, left_out


def whisper_hint(cfg) -> Optional[str]:
    """His names joined for Whisper, or None when there are none."""
    names, _left_out = his_names(cfg)
    return _SEPARATOR.join(names) or None


def announce_his_names(cfg) -> None:
    """Tell him how many of his names Whisper is given, and which are
    not: a name left out is one he would otherwise wait on in vain.
    Says nothing when he has configured none."""
    names, left_out = his_names(cfg)
    if not names and not left_out:
        return
    debug_log(f"whisper hint: {len(names)} names given, {len(left_out)} left out", "voice")
    print(f"     🔤 Names given to Whisper as a hint: {len(names)}", flush=True)
    for name, why in left_out:
        if why == SOUNDS_LIKE_HER:
            print(f"       ↪️ \"{name}\" left out: it sounds like the wake word", flush=True)
    crowded = [name for name, why in left_out if why == NO_ROOM]
    if crowded:
        shown = ", ".join(f"\"{name}\"" for name in crowded[:3])
        more = f" and {len(crowded) - 3} more" if len(crowded) > 3 else ""
        print(f"       ↪️ left out, the list is full: {shown}{more}", flush=True)


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

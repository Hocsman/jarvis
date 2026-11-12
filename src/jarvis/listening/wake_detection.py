"""Wake word and stop command detection logic."""

from typing import Iterator, List, Optional, Tuple
import difflib
import re
import unicodedata

from ..debug import debug_log


# Scripts written without spaces between their words, besides the wide
# forms of East Asian text: Thai, Lao, Myanmar, Khmer and the Tai scripts,
# whose word ends Unicode's line breaking leaves to a dictionary.
_SOUTH_EAST_ASIAN_BLOCKS = (
    (0x0E00, 0x0EFF),   # Thai, Lao
    (0x1000, 0x109F),   # Myanmar
    (0x1780, 0x17FF),   # Khmer
    (0x1950, 0x1AAF),   # Tai Le, New Tai Lue, Khmer Symbols, Buginese, Tai Tham
    (0xA9E0, 0xA9FF),   # Myanmar Extended-B
    (0xAA60, 0xAADF),   # Myanmar Extended-A, Tai Viet
)


def _written_without_spaces(ch: str) -> bool:
    """Whether *ch* belongs to text that does not mark where its words end:
    Chinese characters, kana, Hangul (whose particles join the name before
    them) and the other wide forms of East Asian text, and the scripts of
    South-East Asia."""
    if unicodedata.east_asian_width(ch) in ("W", "F", "H"):
        return True
    code = ord(ch)
    return any(low <= code <= high for low, high in _SOUTH_EAST_ASIAN_BLOCKS)


def _in_a_word(ch: str) -> bool:
    """Whether *ch* is part of a word in a script that writes its words
    apart: a letter, a digit, or a combining mark, which belongs to the
    letter before it."""
    if not (ch.isalnum() or ch == "_" or unicodedata.category(ch).startswith("M")):
        return False
    return not _written_without_spaces(ch)


def _names(wake_word: str, aliases: List[str]) -> List[str]:
    """The names she answers to, longest first. An empty one names nothing."""
    names = {name.strip() for name in [wake_word, *aliases] if name and name.strip()}
    return sorted(names, key=len, reverse=True)


def _is_a_mark(ch: str) -> bool:
    """Whether *ch* is punctuation or a symbol, in any script."""
    return unicodedata.category(ch)[0] in ("P", "S")


def _without_end_marks(token: str) -> str:
    """*token* with the punctuation and symbols at either end taken off."""
    start, end = 0, len(token)
    while start < end and _is_a_mark(token[start]):
        start += 1
    while end > start and _is_a_mark(token[end - 1]):
        end -= 1
    return token[start:end]


def _occurrences(text: str, name: str) -> Iterator[Tuple[int, int]]:
    """Every (start, end) where *name* occurs in *text*, overlapping ones
    included, the words of a name of several words apart by any spacing."""
    pattern = r"\s+".join(re.escape(word) for word in name.split())
    for match in re.finditer(f"(?=({pattern}))", text):
        yield match.start(1), match.end(1)


def _stands_alone(text: str, start: int, end: int) -> bool:
    """Whether text[start:end] is whole words: neither end of it runs on
    into a letter of the word around it."""
    joined_before = start > 0 and _in_a_word(text[start - 1]) and _in_a_word(text[start])
    joined_after = end < len(text) and _in_a_word(text[end - 1]) and _in_a_word(text[end])
    return not (joined_before or joined_after)


def _her_names_in(text: str, wake_word: str, aliases: List[str]) -> Tuple[List[Tuple[int, int]], Optional[str]]:
    """Where the wake word or an alias stands as whole words of *text*, as
    (start, end) spans, and the first name found inside a longer word, if
    any."""
    spans: List[Tuple[int, int]] = []
    inside_a_word = None
    for name in _names(wake_word, aliases):
        for start, end in _occurrences(text, name):
            if _stands_alone(text, start, end):
                spans.append((start, end))
            elif inside_a_word is None:
                inside_a_word = name
    return spans, inside_a_word


def is_wake_word_detected(text_lower: str, wake_word: str, aliases: List[str], fuzzy_ratio: float = 0.78) -> bool:
    """
    Check if text contains wake word using exact and fuzzy matching.

    The wake word and each alias are names, found only as whole words of
    *text_lower*: never inside a longer word, whatever the alphabet. In a
    script written without spaces between words, nothing says where a word
    ends, so a name is found among the words around it.

    Args:
        text_lower: Lowercase text to check
        wake_word: Primary wake word
        aliases: List of wake word aliases
        fuzzy_ratio: Threshold for fuzzy matching (0.0-1.0)

    Returns:
        True if wake word detected
    """
    if not text_lower or not text_lower.strip():
        return False

    # Exact match: the wake word or an alias, as whole words
    spans, inside_a_word = _her_names_in(text_lower, wake_word, aliases)
    if spans:
        return True
    if inside_a_word is not None:
        debug_log(f"wake word: '{inside_a_word}' only inside a longer word, not her name: "
                  f"\"{text_lower}\"", "wake")

    # Close spellings: each token stripped of the punctuation and symbols at
    # its ends, weighed against every name. A token that was only marks is
    # nothing.
    names = _names(wake_word, aliases)
    try:
        heard_tokens = [token for token in (_without_end_marks(t) for t in text_lower.split()) if token]
        for token in heard_tokens:
            for name in names:
                ratio = difflib.SequenceMatcher(a=name, b=token).ratio()
                if ratio >= fuzzy_ratio:
                    debug_log(f"wake word fuzzy match: '{name}' ~ '{token}' (ratio: {ratio:.3f})", "wake")
                    return True
    except Exception:
        pass

    return False


def names_her(text_lower: str, wake_word: str, aliases: List[str]) -> bool:
    """Whether *text_lower* holds the wake word or an alias as whole words,
    spelled as configured: her name said, not a word close to it. The names
    the intent judge is shown as hers are these (`_normalize_aliases`)."""
    return bool(text_lower and _her_names_in(text_lower, wake_word, aliases)[0])


def extract_query_after_wake(text_lower: str, wake_word: str, aliases: List[str]) -> str:
    """
    Extract the query portion after removing wake word.

    Removes the words where the wake word or an alias stands as whole
    words, as `is_wake_word_detected` finds them, with the spacing around
    each; every other word is kept as it was heard, one holding a name
    inside it included.

    Args:
        text_lower: Lowercase text containing wake word
        wake_word: Primary wake word
        aliases: List of wake word aliases

    Returns:
        Query text with wake word removed
    """
    if not text_lower:
        return ""

    spans, _ = _her_names_in(text_lower, wake_word, aliases)
    pieces: List[str] = []
    kept_from = 0
    for start, end in sorted(spans):
        if end <= kept_from:
            continue    # inside a name already taken out ("jar" in "jar is")
        start = max(start, kept_from)
        while start > kept_from and text_lower[start - 1].isspace():
            start -= 1
        while end < len(text_lower) and text_lower[end].isspace():
            end += 1
        pieces.append(text_lower[kept_from:start])
        kept_from = end
    pieces.append(text_lower[kept_from:])
    fragment = " ".join(piece for piece in pieces if piece)

    # Clean up punctuation that might be left after wake word removal
    fragment = re.sub(r"^[\s,.!?;:]+", "", fragment).strip()

    return fragment


def is_stop_command(text_lower: str, stop_commands: List[str], fuzzy_ratio: float = 0.8) -> bool:
    """
    Check if text contains a stop command.
    
    Args:
        text_lower: Lowercase text to check
        stop_commands: List of stop command phrases
        fuzzy_ratio: Threshold for fuzzy matching short inputs
    
    Returns:
        True if stop command detected
    """
    if not text_lower or not text_lower.strip():
        return False
    
    # Check for exact matches
    detected_commands = []
    for cmd in stop_commands:
        if cmd in text_lower:
            detected_commands.append(cmd)
    
    # Check fuzzy matches for short inputs (2 words or less)
    if len(text_lower.split()) <= 2:
        try:
            for word in text_lower.split():
                for cmd in stop_commands:
                    ratio = difflib.SequenceMatcher(a=cmd, b=word).ratio()
                    if ratio >= fuzzy_ratio:
                        detected_commands.append(f"{cmd}~{word}")
        except Exception:
            pass
    
    if detected_commands:
        debug_log(f"stop command detected: {detected_commands[0]} in '{text_lower}'", "voice")
        return True
    
    return False

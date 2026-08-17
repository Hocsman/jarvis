"""Cutting a stream of tokens into sentences worth speaking.

She used to write the whole reply before saying a word of it. This is the
piece that decides where a chunk ends, so the first sentence can leave
while the model is still writing the second.

See ``streaming.spec.md`` for what this must not break — chiefly that the
echo reference stays the whole reply, never the chunk.
"""

from __future__ import annotations

import re
from typing import List

# Terminal punctuation, orthographic rather than lexical: no word list and
# no language detection.
#
# Two shapes, because the scripts differ in what they guarantee. A
# full-width stop closes a sentence on its own — CJK writing puts no space
# after it, and the character is not used for anything else. An ASCII full
# stop is overloaded (decimals, abbreviations, initials), so it only
# closes when whitespace follows.
_CLOSES = re.compile(
    r"[。！？]+[\"'»”)\]]*"          # pleine chasse : se suffit
    r"|[.!?…]+[\"'»”)\]]*(?=\s)"    # ASCII : exige une espace derrière
)

# Below this, a chunk closed by ASCII punctuation waits for the next one.
# It absorbs the abbreviation problem: "M." is two characters, so it
# merges forward instead of becoming a chunk that says "M". Decimals
# behave the same — "24." is three.
#
# It does not apply to a full-width stop, and must not: the minimum exists
# to disambiguate an overloaded character, `。` is never overloaded, and a
# character count is itself script-dependent — ten characters is a whole
# sentence in Japanese and half a clause in French.
_MIN_CHUNK = 12
_FULL_WIDTH = re.compile(r"[。！？]")


class SentenceStreamer:
    """Feed it token deltas, it hands back sentences as they close."""

    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, delta: str) -> List[str]:
        """Add a delta and return whatever sentences are now complete."""
        if not delta:
            return []
        self._buffer += delta

        sorties: List[str] = []
        while True:
            m = None
            for candidat in _CLOSES.finditer(self._buffer):
                # The first break that leaves a chunk worth speaking.
                if (_FULL_WIDTH.search(candidat.group())
                        or len(self._buffer[:candidat.end()].strip()) >= _MIN_CHUNK):
                    m = candidat
                    break
            if m is None:
                break
            phrase = self._buffer[:m.end()].strip()
            self._buffer = self._buffer[m.end():]
            if phrase:
                sorties.append(phrase)
        return sorties

    def flush(self) -> str:
        """Whatever is left when the stream ends, terminated or not."""
        reste = self._buffer.strip()
        self._buffer = ""
        return reste

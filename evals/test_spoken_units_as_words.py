"""
End-to-end eval: a spoken reply says its units, signs and dates in words.

Measured on 05/11/2026 with the French Kyutai voice, each sentence
synthesised then transcribed back three times: a degree sign after a space
is misread 14 times in 15 ("11,2 °C" heard as "11h02 Uricol"), the "/h" of
"km/h" is lost 9 times in 9, a minus sign 12 times in 12 (so "-3 degrés" is
heard as "3 degrés"), a slash date 3 in 3, and an unbroken run of seven
digits 3 in 3. The same voice reads a decimal followed by a word, a time, a
percentage, a currency sign and a date in words right, and it reads a
number spelled out worse than its digits: it is the symbols that go wrong,
not the numbers.

So the rule (`SPOKEN_FORM`, src/jarvis/reply/prompts/system.py) asks the
model to write the words a person would say, in the reply's own language,
and it reaches a turn from the voice only (`origin="voix"`): a reply typed
in the chat window is read, not heard, and keeps its symbols.

Each case is a question whose tool result is full of those symbols:
weather in the weather tool's own format, a festival's dates, a city's
population, a train journey. A reply is scored on what the voice would
misread in it:

- a degree sign;
- "/h";
- a minus sign or hyphen right before a digit, unless it follows a letter
  or a digit (so "-3" counts, "J-3" and a range "3-5" do not);
- a date in digits, with slashes or dashes (05/11/2026, 2026-11-05);
- an unbroken run of five digits or more.

And on whether it kept its figures: the reply answers, and the figures the
question asks for are still in it, in digits (with the sign of a
temperature below zero, said or written). A number spelled out counts as
lost: the voice reads it worse than its digits, and a rule that turned
"-3,4 °C" into "moins trois virgule quatre degrés" would trade one misreading
for another.

`TestSpokenRepliesSayTheirUnits` asks every case once as a voice turn.
`TestBeforeAndAfter` compares the prompt without the rule against the prompt
with it: both arms are voice turns, the first with the rule emptied, and
they alternate call by call inside one loop (CLAUDE.md: a remote provider
drifts within a session). Set `EVAL_SPOKEN_UNITS_ROUNDS` to run it; each
round asks every case in both languages on both arms, so four rounds give
24 calls per arm and language. It holds the rule to more replies read right
(nothing misread and every figure in digits) in each language.

They take ``real_model_config``: the user's provider and models, sandboxed
memory, location off, no MCP server, and every tool offered rather than
routed. Tools are stubbed to return the case's invented result, so what
leaves the machine is the invented question, that result and the prompt
built around an empty memory.

Run:
    ./scripts/run_evals.sh --single spoken_units_as_words
    EVAL_SPOKEN_UNITS_ROUNDS=4 python -m pytest evals/test_spoken_units_as_words.py -k before -s
"""

from __future__ import annotations

import dataclasses
import os
import random
import re
import sys
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Tuple
from unittest.mock import patch

import pytest

from helpers import is_fallback_reply

ROUNDS = int(os.environ.get("EVAL_SPOKEN_UNITS_ROUNDS", "0") or 0)

# Below this many calls an arm's rate says nothing (CLAUDE.md, sample sizes).
_MIN_CALLS_FOR_A_RATE = 20

# What the voice misreads, as measured (see the module docstring).
MISREAD: Dict[str, "re.Pattern[str]"] = {
    "degree sign": re.compile("°"),
    "/h": re.compile(r"/\s*h\b", re.IGNORECASE),
    "minus before a digit": re.compile(r"(?<!\w)[-\u2212\u2013]\d"),
    "date in digits": re.compile(r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b|\b\d{4}-\d{2}-\d{2}\b"),
    "5+ digits": re.compile(r"\d{5,}"),
}

# Read right about half the time ("1 234 567"): counted and printed by the
# comparison, not scored, since it is not a symbol.
GROUPED_LONG_NUMBER = re.compile(r"\b\d{1,3}(?:[ \u00a0\u202f,.]\d{3}){2,}\b")

_LANGUAGES = {"fr": "français", "en": ""}


@dataclass(frozen=True)
class UnitsCase:
    id: str
    language: str
    asked: str
    # What the tool hands back, symbols and all.
    result: str
    # Each must match the folded reply (lower case, accents removed): the
    # figures the question asks for, in any form.
    facts: Tuple[str, ...]


def _weather(place: str, region: str, lines: str) -> str:
    """A result in the weather tool's own format."""
    return (
        f"Current weather in {place}, {region}, France:\n"
        f'Say the place as "{place}, {region}" in the reply, so the user can '
        "tell at once if it is the wrong place.\n\n" + lines
    )


_MILD = _weather("Valmorin", "Haute-Lisière", (
    "Conditions: Partly cloudy\n"
    "Temperature: 11.2°C (52.2°F)\n"
    "Feels like: 8.6°C (47.5°F)\n"
    "Humidity: 81%\n"
    "Wind: 8.6 km/h (gusts up to 23.4 km/h)"
))

_FREEZING = _weather("Brévanne", "Val-d'Ormel", (
    "Conditions: Clear sky\n"
    "Temperature: -3.4°C (25.9°F)\n"
    "Feels like: -7.8°C (18.0°F)\n"
    "Humidity: 72%\n"
    "Wind: 12.1 km/h\n\n"
    "Today's forecast (upcoming hours):\n"
    "  21:00 — -4.2°C, Clear sky\n"
    "  00:00 — -5.1°C, Clear sky\n"
    "  03:00 — -5.8°C, Fog"
))

_WINDY = _weather("Port-Avelin", "Côte-Brune", (
    "Conditions: Overcast\n"
    "Temperature: 9.8°C (49.6°F)\n"
    "Humidity: 88%\n"
    "Wind: 34.7 km/h (gusts up to 61.2 km/h)"
))

# The figures are looked for in digits: the voice reads a number spelled out
# worse than its digits (6 in 9 against 21 in 21), so a reply that spells
# them out has lost them as surely as one that drops them. The magnitude
# word of a rounded number is the exception, "un million" being how it is
# said.
_MILD_FACTS = (r"\b11\b",)
_BELOW_ZERO_FR = (
    r"(moins|[-\u2212\u2013])\s*[3-8]\b"
    r"|\b[3-8](?:[,.]\d)?\s*(?:degres?\s*)?(?:celsius\s*)?"
    r"(?:sous zero|en dessous de zero|en-dessous de zero|sous le zero)",
)
_BELOW_ZERO_EN = (
    r"(minus|[-\u2212\u2013])\s*[3-8]\b"
    r"|\b[3-8](?:[.,]\d)?\s*(?:degrees?\s*)?(?:celsius\s*)?below (?:zero|freezing)",
)
_WIND = (r"\b(34|35|61|60)\b",)
_POPULATION = (
    r"\b(1|un|one)\b[^.!?]{0,24}millions?"
    r"|1[\s,.\u202f\u00a0]?234[\s,.\u202f\u00a0]?567|\d{7}",
)
_JOURNEY_TIME = (r"\b[23]\s*(h|heures?|hours?)\b|2:54|\b54\b",)

CASES: List[UnitsCase] = [
    UnitsCase(
        id="fr mild weather", language="fr",
        asked="Il fait combien dehors à Valmorin ?",
        result=_MILD, facts=_MILD_FACTS,
    ),
    UnitsCase(
        id="fr freezing night", language="fr",
        asked="Est-ce qu'il va geler cette nuit à Brévanne ?",
        result=_FREEZING, facts=_BELOW_ZERO_FR,
    ),
    UnitsCase(
        id="fr wind", language="fr",
        asked="Il y a beaucoup de vent à Port-Avelin aujourd'hui ?",
        result=_WINDY, facts=_WIND,
    ),
    UnitsCase(
        id="fr festival dates", language="fr",
        asked="C'est quand, le festival des lanternes de Sorrevaux ?",
        result=(
            "**Content from top result:**\n"
            "Festival des lanternes de Sorrevaux, édition 2026. Dates : du "
            "14/11/2026 au 16/11/2026. Lieu : quai des Tanneurs, Sorrevaux. "
            "Affluence attendue : 48500 visiteurs. Entrée : 12,50 €."
        ),
        facts=(r"\b14\b", r"novembre|\b14/11\b|-11-"),
    ),
    UnitsCase(
        id="fr population", language="fr",
        asked="Combien d'habitants à Port-Avelin ?",
        result=(
            "**Content from top result:**\n"
            "Port-Avelin est la plus grande ville de la région Côte-Brune. "
            "Population (recensement 2025) : 1234567 habitants. Fondée en 1287."
        ),
        facts=_POPULATION,
    ),
    UnitsCase(
        id="fr train journey", language="fr",
        asked="Le train de Kerlivan-Plage à Valmorin, ça prend combien de temps, "
              "et le prochain part quand ?",
        result=(
            "**Content from top result:**\n"
            "Train de Kerlivan-Plage à Valmorin. Distance : 412,6 km. Vitesse "
            "moyenne : 142,3 km/h. Durée : 2 h 54. Prochain départ : 21/11/2026 "
            "à 07:42. Température prévue à l'arrivée : -1,5 °C."
        ),
        facts=_JOURNEY_TIME + (r"\b21\b",),
    ),
    UnitsCase(
        id="en mild weather", language="en",
        asked="What's the temperature in Valmorin right now?",
        result=_MILD, facts=_MILD_FACTS,
    ),
    UnitsCase(
        id="en freezing night", language="en",
        asked="Is it going to freeze tonight in Brévanne?",
        result=_FREEZING, facts=_BELOW_ZERO_EN,
    ),
    UnitsCase(
        id="en wind", language="en",
        asked="How windy is it in Port-Avelin today?",
        result=_WINDY, facts=_WIND,
    ),
    UnitsCase(
        id="en festival dates", language="en",
        asked="When is the Sorrevaux lantern festival?",
        result=(
            "**Content from top result:**\n"
            "Sorrevaux Lantern Festival 2026. Dates: 14/11/2026 to 16/11/2026. "
            "Venue: Tanners' Quay, Sorrevaux. Expected attendance: 48500 "
            "visitors. Entry: €12.50."
        ),
        facts=(r"\b14(?:th)?\b", r"november|\b14/11\b|\b11/14\b|-11-"),
    ),
    UnitsCase(
        id="en population", language="en",
        asked="How many people live in Port-Avelin?",
        result=(
            "**Content from top result:**\n"
            "Port-Avelin is the largest city of the Côte-Brune region. "
            "Population (2025 census): 1234567. Founded in 1287."
        ),
        facts=_POPULATION,
    ),
    UnitsCase(
        id="en train journey", language="en",
        asked="How long is the train from Kerlivan-Plage to Valmorin, and when "
              "is the next one?",
        result=(
            "**Content from top result:**\n"
            "Train from Kerlivan-Plage to Valmorin. Distance: 256.4 mi (412.6 km). "
            "Average speed: 88.4 mph (142.3 km/h). Journey time: 2 h 54. Next "
            "departure: 21/11/2026 at 07:42. Forecast temperature on arrival: "
            "-1.5°C."
        ),
        facts=_JOURNEY_TIME + (r"\b21(?:st)?\b",),
    ),
]


def _fold(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text.lower())
        if unicodedata.category(c) != "Mn"
    )


def misread_in(reply: str) -> List[str]:
    """What the voice would misread in ``reply``: the categories present."""
    return [name for name, pattern in MISREAD.items() if pattern.search(reply)]


def keeps_its_figures(reply: str, case: UnitsCase) -> bool:
    """The reply answers, and every figure the question asks for is in it."""
    if not reply.strip() or is_fallback_reply(reply):
        return False
    folded = _fold(reply)
    return all(re.search(fact, folded) for fact in case.facts)


def _config(real_model_config, case: UnitsCase):
    """The user's provider and models, the case's language, no MCP server:
    the user's servers have no part in a turn whose tools are stubbed.

    Every tool is offered. The router is not what this measures, and when it
    leaves `webSearch` out of a population question the model never sees the
    figures whose form is being scored."""
    return dataclasses.replace(
        real_model_config, mcps={}, response_language=_LANGUAGES[case.language],
        tool_selection_strategy="all",
    )


def _voice_turn(cfg, case: UnitsCase, *, rule: bool = True) -> str:
    """One voice turn on ``case``; ``rule=False`` empties the spoken-form rule,
    which is the prompt as it stood before it."""
    from contextlib import ExitStack

    from jarvis.memory.conversation import DialogueMemory
    from jarvis.memory.db import Database
    from jarvis.reply.engine import run_reply_engine
    from jarvis.tools.types import ToolExecutionResult

    db = Database(":memory:", sqlite_vss_path=None)
    try:
        with ExitStack() as stack:
            stack.enter_context(patch(
                "jarvis.reply.engine.run_tool_with_retries",
                return_value=ToolExecutionResult(success=True, reply_text=case.result),
            ))
            stack.enter_context(patch(
                "jarvis.reply.engine.extract_search_params_for_memory",
                return_value={"keywords": []},
            ))
            if not rule:
                stack.enter_context(patch(
                    "jarvis.reply.prompts.model_variants.SPOKEN_FORM", "",
                ))
            reply = run_reply_engine(
                db=db, cfg=cfg, tts=None, text=case.asked,
                dialogue_memory=DialogueMemory(inactivity_timeout=300),
                language=case.language, origin="voix",
            )
    finally:
        db.close()
    return reply or ""


def _to_terminal(capsys, lines: List[str]) -> None:
    from jarvis.utils.console import force_utf8_stream

    with capsys.disabled():
        force_utf8_stream(sys.stdout)
        print("\n".join(lines), flush=True)


# Read right in fewer than five replies in five with the rule, on Claude
# Haiku 5.5 (06/11/2026, 5 calls a case): the model sometimes copies the
# figure as the tool result wrote it.
_KNOWN_WEAK = {
    "fr freezing night":
        "3 in 5: '-3,4 degrés' copied with its minus sign, or the number "
        "spelled out: tracked, not blocking",
    "fr population":
        "2 in 5: '1234567' copied unrounded: tracked, not blocking",
    "en population":
        "3 in 5: '1234567' copied unrounded: tracked, not blocking",
}


def _each_case():
    for case in CASES:
        weak = _KNOWN_WEAK.get(case.id)
        yield pytest.param(
            case, id=case.id,
            marks=pytest.mark.xfail(reason=weak, strict=False) if weak else (),
        )


@pytest.mark.eval
class TestSpokenRepliesSayTheirUnits:
    """A voice turn's reply holds nothing the voice misreads, and keeps its
    figures."""

    @pytest.mark.parametrize("case", list(_each_case()))
    def test_the_reply_says_its_units_in_words(self, real_model_config, case):
        reply = _voice_turn(_config(real_model_config, case), case)

        print(f"\n  🗣️ {case.id}: {reply[:240]!r}")
        assert keeps_its_figures(reply, case), (
            f"The reply lost the figures it was asked for ({case.facts!r}). "
            f"Reply: {reply[:400]!r}"
        )
        assert not misread_in(reply), (
            f"The voice would misread {misread_in(reply)} in: {reply[:400]!r}"
        )


@pytest.mark.eval
@pytest.mark.skipif(ROUNDS < 1, reason="set EVAL_SPOKEN_UNITS_ROUNDS to compare")
class TestBeforeAndAfter:
    """The prompt without the rule against the prompt with it, call by call."""

    def test_the_rule_against_no_rule_call_by_call(self, real_model_config, capsys):
        from jarvis.reply.prompts.model_variants import SPOKEN_FORM

        assert SPOKEN_FORM, "no spoken-form rule to compare against"

        arms = {"without the rule": False, "with the rule": True}
        names = list(arms)
        languages = list(_LANGUAGES)
        # (arm, language) -> counters
        calls: Dict[Tuple[str, str], int] = {}
        clean: Dict[Tuple[str, str], int] = {}
        kept: Dict[Tuple[str, str], int] = {}
        # Both at once: nothing misread and every figure in digits.
        heard: Dict[Tuple[str, str], int] = {}
        by_kind: Dict[Tuple[str, str, str], int] = {}
        per_case: Dict[Tuple[str, str], List[int]] = {}   # clean, kept, calls
        misses: List[str] = []

        def rate(got: int, n: int) -> str:
            if n >= _MIN_CALLS_FOR_A_RATE:
                return f"{got}/{n} ({100 * got / n:.0f} %)"
            return f"{got}/{n}   (under {_MIN_CALLS_FOR_A_RATE} calls: counts only)"

        rng = random.Random(0)
        try:
            for rnd in range(ROUNDS):
                order = list(CASES)
                rng.shuffle(order)
                for i, case in enumerate(order):
                    cfg = _config(real_model_config, case)
                    turn = (i + rnd) % len(names)
                    for name in names[turn:] + names[:turn]:
                        reply = _voice_turn(cfg, case, rule=arms[name])
                        key = (name, case.language)
                        found = misread_in(reply)
                        figures = keeps_its_figures(reply, case)
                        calls[key] = calls.get(key, 0) + 1
                        clean[key] = clean.get(key, 0) + int(not found)
                        kept[key] = kept.get(key, 0) + int(figures)
                        heard[key] = heard.get(key, 0) + int(figures and not found)
                        for kind in found:
                            k = (name, case.language, kind)
                            by_kind[k] = by_kind.get(k, 0) + 1
                        if GROUPED_LONG_NUMBER.search(reply):
                            k = (name, case.language, "grouped long number")
                            by_kind[k] = by_kind.get(k, 0) + 1
                        row = per_case.setdefault((name, case.id), [0, 0, 0])
                        row[0] += int(not found)
                        row[1] += int(figures)
                        row[2] += 1
                        if found or not figures:
                            what = ", ".join(found) if found else "figures lost"
                            if found and not figures:
                                what += ", figures lost"
                            misses.append(f"{name} / {case.id} ({what}): {reply[:160]!r}")
                _to_terminal(capsys, [f"⏱️ round {rnd + 1}/{ROUNDS} done"])
        finally:
            lines = [
                "",
                f"📊 Spoken units: {len(CASES)} cases x {ROUNDS} rounds, "
                "arms interleaved call by call",
            ]
            for name in names:
                lines.append(f"   🧪 {name}")
                for language in languages:
                    key = (name, language)
                    n = calls.get(key, 0)
                    lines.append(
                        f"      🌐 {language}  read right {rate(heard.get(key, 0), n)}"
                        f"  (nothing misread {rate(clean.get(key, 0), n)}"
                        f", figures kept {rate(kept.get(key, 0), n)})"
                    )
                    for kind in list(MISREAD) + ["grouped long number"]:
                        count = by_kind.get((name, language, kind), 0)
                        if count:
                            note = "  (not scored)" if kind not in MISREAD else ""
                            lines.append(f"         ⚠️ {kind:<22} {count}/{n}{note}")
                for case in CASES:
                    c, k, n = per_case.get((name, case.id), [0, 0, 0])
                    lines.append(f"         💬 {case.id:<20} clean {c}/{n}, kept {k}/{n}")
            for miss in misses:
                lines.append(f"      ❌ {miss}")
            _to_terminal(capsys, lines)

        for language in languages:
            before = heard.get(("without the rule", language), 0)
            after = heard.get(("with the rule", language), 0)
            assert after > before, (
                f"In {language}, replies with the rule are no more often read right "
                f"(nothing misread, every figure in digits) than replies without it: "
                f"{after} against {before}."
            )

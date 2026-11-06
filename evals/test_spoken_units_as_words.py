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

Each units case is a question whose tool result is full of those symbols:
weather in the weather tool's own format, a festival's dates, a city's
population, a train journey. A reply is scored on what the voice would
misread in it:

- a degree sign;
- "/h";
- a minus sign or hyphen right before a digit, or one space before it,
  unless it follows a letter or a digit (so "-3" and "- 3" count, "J-3" and
  a range "3-5" do not);
- a date in digits, with slashes or dashes (05/11/2026, 05-11-2026,
  2026-11-05);
- an unbroken run of five digits or more.

And on whether it kept its figures: the reply answers, and the figures the
question asks for are still in it, in digits (with the sign of a
temperature below zero, said or written). A number spelled out counts as
lost: the voice reads it worse than its digits, and a rule that turned
"-3,4 °C" into "moins trois virgule quatre degrés" would trade one misreading
for another.

An identifier is not a quantity, and rounding it gives a wrong answer
rather than a misread one. The identifier cases ask for a postcode, a phone
number, a booking reference, a door code and a founding year, from a tool
result that writes them unbroken. Such a reply is right when every
identifier comes back with every digit in order, grouped by spaces at
most, in groups of five digits or fewer: never rounded, never spelled out,
and not left in a run the voice misreads.

`TestSpokenRepliesSayTheirUnits` and `TestIdentifiersSurvive` ask every
case once as a voice turn. `TestBeforeAndAfter` compares two arms of the
rule, both voice turns, alternating call by call inside one loop (CLAUDE.md:
a remote provider drifts within a session): the prompt without the rule
against the prompt with it, or, with `EVAL_SPOKEN_FORM_AGAINST` naming a
file that holds another wording, that wording against the prompt's. Set
`EVAL_SPOKEN_UNITS_ROUNDS` to run it; each round asks every case in both
languages on both arms, so four rounds give 24 units calls and 20
identifier calls per arm and language. Against no rule it holds the rule
to more replies read right (nothing misread and every figure in digits) and
to no fewer identifiers right, in each language; against another wording
it reports both arms and asserts nothing.

They take ``real_model_config``: the user's provider and models, sandboxed
memory, location off, no MCP server, every tool offered rather than
routed, and the Kyutai voice, the one the misreadings were measured on
(with Piper or Chatterbox the prompt tells the model to answer in English,
and the French cases would measure that conflict instead). Tools are
stubbed to return the case's invented result, so what leaves the machine
is the invented question, that result and the prompt built around an empty
memory.

Run:
    ./scripts/run_evals.sh --single spoken_units_as_words
    EVAL_SPOKEN_UNITS_ROUNDS=4 python -m pytest evals/test_spoken_units_as_words.py -k before -s
    EVAL_SPOKEN_UNITS_ROUNDS=4 EVAL_SPOKEN_FORM_AGAINST=other_rule.txt \\
        python -m pytest evals/test_spoken_units_as_words.py -k before -s
"""

from __future__ import annotations

import dataclasses
import os
import random
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from unittest.mock import patch

import pytest

from helpers import is_fallback_reply, rate, to_terminal

ROUNDS = int(os.environ.get("EVAL_SPOKEN_UNITS_ROUNDS", "0") or 0)
# A file holding another wording of the rule: the comparison then measures
# the prompt's rule against it instead of against no rule.
AGAINST = os.environ.get("EVAL_SPOKEN_FORM_AGAINST", "")

# What the voice misreads, as measured (see the module docstring).
MISREAD: Dict[str, "re.Pattern[str]"] = {
    "degree sign": re.compile("°"),
    "/h": re.compile(r"/\s*h\b", re.IGNORECASE),
    "minus before a digit": re.compile(r"(?<!\w)[-\u2212\u2013]\s?\d"),
    "date in digits": re.compile(
        r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b|\b\d{4}-\d{2}-\d{2}\b"
        r"|\b\d{1,2}-\d{1,2}-\d{2,4}\b"
    ),
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
    r"(?<!au )\bmoins\s*[3-8]\b|(?<!\w)[-\u2212\u2013]\s*[3-8]\b"
    r"|\b[3-8](?:[,.]\d)?\s*(?:degres?\s*)?(?:celsius\s*)?"
    r"(?:sous zero|en dessous de zero|en-dessous de zero|sous le zero)",
)
_BELOW_ZERO_EN = (
    r"\bminus\s*[3-8]\b|(?<!\w)[-\u2212\u2013]\s*[3-8]\b"
    r"|\b[3-8](?:[.,]\d)?\s*(?:degrees?\s*)?(?:celsius\s*)?below (?:zero|freezing)",
)
_WIND = (r"\b(34|35|61|60)\b",)
_POPULATION = (
    r"\b(?:1(?:[.,]\d+)?|un|one)\s+millions?\b"
    r"|1[\s,.\u202f\u00a0]?234[\s,.\u202f\u00a0]?567|\d{7}",
)
_JOURNEY_TIME = (
    r"\b[23]\s*(?:h|heures?|hours?)(?![a-z])|2:54|(?<![\d.,])54(?!\d)",
)

_POPULATION_FR = (
    "**Content from top result:**\n"
    "Port-Avelin est la plus grande ville de la région Côte-Brune. "
    "Population (recensement 2025) : 1234567 habitants. Fondée en 1287."
)
_POPULATION_EN = (
    "**Content from top result:**\n"
    "Port-Avelin is the largest city of the Côte-Brune region. "
    "Population (2025 census): 1234567. Founded in 1287."
)

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
        result=_POPULATION_FR, facts=_POPULATION,
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
        result=_POPULATION_EN, facts=_POPULATION,
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


@dataclass(frozen=True)
class IdentifierCase:
    id: str
    language: str
    asked: str
    result: str
    # Each must come back with every character, in order: never rounded,
    # never spelled out.
    identifiers: Tuple[str, ...]


# The phone numbers are in the ranges set aside for fiction (France
# 01 99 00, United Kingdom 020 7946 0).
ID_CASES: List[IdentifierCase] = [
    IdentifierCase(
        id="fr postcode", language="fr",
        asked="C'est quoi le code postal de Valmorin ?",
        result=(
            "**Content from top result:**\n"
            "Valmorin, commune de la région Haute-Lisière. Code postal : 47390. "
            "Altitude : 312 m. Population : 18650 habitants (2025)."
        ),
        identifiers=("47390",),
    ),
    IdentifierCase(
        id="fr phone number", language="fr",
        asked="Tu peux me donner le numéro de téléphone de la mairie de Brévanne ?",
        result=(
            "**Content from top result:**\n"
            "Mairie de Brévanne, 3 place des Tilleuls. Téléphone : 0199003572. "
            "Ouverte du lundi au vendredi, de 9 h à 17 h."
        ),
        identifiers=("0199003572",),
    ),
    IdentifierCase(
        id="fr booking reference", language="fr",
        asked="Dans mes notes, c'est quoi la référence de la réservation du gîte "
              "à Kerlivan-Plage ?",
        result=(
            "notes/vacances.md:\n"
            "Gîte des Mouettes, Kerlivan-Plage, du 21/11/2026 au 23/11/2026. "
            "Référence de réservation : 50318472. Arrivée à partir de 16 h."
        ),
        identifiers=("50318472",),
    ),
    IdentifierCase(
        id="fr door code", language="fr",
        asked="Dans mes notes, c'est quoi le digicode de l'immeuble d'Élise "
              "à Sorrevaux ?",
        result=(
            "notes/adresses.md:\n"
            "Élise Fontanel, 14 rue des Augustins, Sorrevaux. Digicode : 73914, "
            "puis interphone au nom de Fontanel, 3e étage."
        ),
        identifiers=("73914",),
    ),
    IdentifierCase(
        id="fr founding year", language="fr",
        asked="Port-Avelin, elle a été fondée en quelle année, et elle a combien "
              "d'habitants ?",
        result=_POPULATION_FR,
        identifiers=("1287",),
    ),
    IdentifierCase(
        id="en postcode", language="en",
        asked="What's the postcode of Valmorin?",
        result=(
            "**Content from top result:**\n"
            "Valmorin, a town in the Haute-Lisière region. Postcode: 47390. "
            "Elevation: 312 m. Population: 18650 (2025)."
        ),
        identifiers=("47390",),
    ),
    IdentifierCase(
        id="en phone number", language="en",
        asked="Can you give me the phone number of Brévanne town hall?",
        result=(
            "**Content from top result:**\n"
            "Brévanne Town Hall, 3 Linden Square. Telephone: 02079460718. "
            "Open Monday to Friday, 9am to 5pm."
        ),
        identifiers=("02079460718",),
    ),
    IdentifierCase(
        id="en booking reference", language="en",
        asked="In my notes, what's the booking reference for the cottage in "
              "Kerlivan-Plage?",
        result=(
            "notes/holidays.md:\n"
            "Gull Cottage, Kerlivan-Plage, 21/11/2026 to 23/11/2026. Booking "
            "reference: 50318472. Check-in from 4pm."
        ),
        identifiers=("50318472",),
    ),
    IdentifierCase(
        id="en door code", language="en",
        asked="In my notes, what's the door code for Elise's building in "
              "Sorrevaux?",
        result=(
            "notes/addresses.md:\n"
            "Elise Fontanel, 14 Augustine Street, Sorrevaux. Door code: 73914, "
            "then buzz Fontanel, third floor."
        ),
        identifiers=("73914",),
    ),
    IdentifierCase(
        id="en founding year", language="en",
        asked="When was Port-Avelin founded, and how many people live there?",
        result=_POPULATION_EN,
        identifiers=("1287",),
    ),
]

# What may stand between the characters of an identifier the reply kept:
# grouped by spaces it is also read right; a dot or a hyphen keeps it whole
# but is a symbol the voice may say.
_KEPT_GAP = "[   .\\-]"
_READ_GAP = "[   ]"
# An unbroken run of digits the voice reads right: a postcode's five. Seven
# in a row were misread 3 times in 3.
_LONGEST_GROUP = 5


def _written(identifier: str, gap: str) -> "re.Pattern[str]":
    body = f"{gap}?".join(re.escape(c) for c in identifier)
    return re.compile(rf"(?<![0-9A-Za-z]){body}(?![0-9A-Za-z])", re.IGNORECASE)


def identifier_kept(reply: str, identifier: str) -> bool:
    """Every character of ``identifier`` is in ``reply``, in order, at most
    grouped by spaces, dots or hyphens: not rounded, not spelled out."""
    return bool(_written(identifier, _KEPT_GAP).search(reply))


def identifier_readable(reply: str, identifier: str) -> bool:
    """Kept, grouped by spaces only, with no group longer than the voice
    reads right."""
    too_long = re.compile(rf"\d{{{_LONGEST_GROUP + 1},}}")
    return any(
        not too_long.search(m.group(0))
        for m in _written(identifier, _READ_GAP).finditer(reply)
    )


def identifiers_right(reply: str, case: IdentifierCase) -> bool:
    """The reply answers, and every identifier is kept and readable."""
    if not reply.strip() or is_fallback_reply(reply):
        return False
    return all(identifier_readable(reply, i) for i in case.identifiers)


def identifiers_kept(reply: str, case: IdentifierCase) -> bool:
    if not reply.strip() or is_fallback_reply(reply):
        return False
    return all(identifier_kept(reply, i) for i in case.identifiers)


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


def _config(real_model_config, case: "UnitsCase | IdentifierCase"):
    """The user's provider and models, the case's language, no MCP server:
    the user's servers have no part in a turn whose tools are stubbed.

    Every tool is offered. The router is not what this measures, and when it
    leaves `webSearch` out of a population question the model never sees the
    figures whose form is being scored."""
    return dataclasses.replace(
        real_model_config, mcps={}, response_language=_LANGUAGES[case.language],
        tool_selection_strategy="all", tts_engine="kyutai",
    )


# Where `get_system_prompts` reads the rule: replaced for an arm that
# carries another wording, emptied for the arm without it.
_RULE_AT = "jarvis.reply.prompts.model_variants.SPOKEN_FORM"


def _voice_turn(
    cfg, case: "UnitsCase | IdentifierCase", *, rule: Optional[str] = None,
) -> str:
    """One voice turn on ``case``. ``rule`` is the spoken-form rule's text
    for this turn: None keeps the prompt's own, "" leaves the rule out."""
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
            if rule is not None:
                stack.enter_context(patch(_RULE_AT, rule))
            reply = run_reply_engine(
                db=db, cfg=cfg, tts=None, text=case.asked,
                dialogue_memory=DialogueMemory(inactivity_timeout=300),
                language=case.language, origin="voix",
            )
    finally:
        db.close()
    return reply or ""


def _case(case_id: str) -> UnitsCase:
    return next(case for case in CASES if case.id == case_id)


class TestTheScoring:
    """The scoring itself, on written replies: no model is called."""

    @pytest.mark.parametrize("case_id, reply", [
        ("en population", "About 1.2 million people live in Port-Avelin."),
        ("fr population", "Port-Avelin compte environ 1,2 million d'habitants."),
        ("fr train journey", "Le trajet dure 2h54, prochain départ le 21 novembre."),
        ("en train journey", "It takes just under 3 hours; the next one leaves on 21 November."),
        ("fr freezing night", "Oui, il fera moins 3,4 degrés cette nuit."),
    ])
    def test_a_figure_in_the_form_the_rule_asks_for_is_kept(self, case_id, reply):
        assert keeps_its_figures(reply, _case(case_id))

    @pytest.mark.parametrize("case_id, reply", [
        ("en population", "About one point two million people live there."),
        ("fr population", "Environ un virgule deux million d'habitants."),
        ("fr freezing night", "Il gèlera au moins 5 heures cette nuit."),
    ])
    def test_a_figure_spelled_out_or_missing_is_lost(self, case_id, reply):
        assert not keeps_its_figures(reply, _case(case_id))

    @pytest.mark.parametrize("reply, kind", [
        ("Du 14-11-2026 au 16-11-2026.", "date in digits"),
        ("Le 14/11/2026.", "date in digits"),
        ("Il fera - 3 degrés.", "minus before a digit"),
        ("Il fait 11,2 °C.", "degree sign"),
        ("Le vent souffle à 34,7 km/h.", "/h"),
        ("Population : 1234567.", "5+ digits"),
    ])
    def test_what_the_voice_misreads_is_found(self, reply, kind):
        assert kind in misread_in(reply)

    @pytest.mark.parametrize("reply", [
        "Il fera moins 3,4 degrés à Brévanne, Val-d'Ormel, le 14 novembre 2026.",
        "Comptez 3-5 jours, environ 30 mille visiteurs.",
    ])
    def test_a_reply_in_words_has_nothing_misread(self, reply):
        assert misread_in(reply) == []

    @pytest.mark.parametrize("case_id, reply", [
        ("fr phone number", "Le numéro de la mairie est le 01 99 00 35 72."),
        ("en phone number", "You can reach them on 020 7946 0718."),
        ("fr booking reference", "La référence est 5031 8472."),
        ("fr postcode", "Le code postal de Valmorin est 47390."),
        ("en door code", "The door code is 73914."),
        ("en founding year", "It was founded in 1287 and has about 1.2 million people."),
    ])
    def test_an_identifier_whole_and_grouped_is_right(self, case_id, reply):
        case = next(c for c in ID_CASES if c.id == case_id)
        assert identifiers_right(reply, case)

    @pytest.mark.parametrize("case_id, reply, whole", [
        # Copied as the tool wrote it: whole, but a run the voice misreads.
        ("fr phone number", "Le numéro est le 0199003572.", True),
        ("fr phone number", "Le numéro est le 01.99.00.35.72.", True),
        # Rounded, spelled out or cut: not the identifier any more.
        ("fr booking reference", "La référence est environ 50 millions.", False),
        ("fr postcode", "C'est quarante-sept mille trois cent quatre-vingt-dix.", False),
        ("en booking reference", "The reference is 5031847.", False),
        ("en founding year", "It was founded around 1300.", False),
    ])
    def test_an_identifier_altered_or_unreadable_is_wrong(self, case_id, reply, whole):
        case = next(c for c in ID_CASES if c.id == case_id)
        assert not identifiers_right(reply, case)
        assert identifiers_kept(reply, case) is whole


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
    # Identifiers, 06/11/2026, 4 calls a case: never rounded or cut, but a
    # long one comes back as the tool wrote it, unbroken.
    "fr booking reference":
        "0 in 4: '50318472' copied unbroken: tracked, not blocking",
    "en booking reference":
        "0 in 4: '50318472' copied unbroken: tracked, not blocking",
    "en phone number":
        "2 in 4: '02079460718' copied unbroken: tracked, not blocking",
}


def _each_case(cases):
    for case in cases:
        weak = _KNOWN_WEAK.get(case.id)
        yield pytest.param(
            case, id=case.id,
            marks=pytest.mark.xfail(reason=weak, strict=False) if weak else (),
        )


@pytest.mark.eval
class TestSpokenRepliesSayTheirUnits:
    """A voice turn's reply holds nothing the voice misreads, and keeps its
    figures."""

    @pytest.mark.parametrize("case", list(_each_case(CASES)))
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
class TestIdentifiersSurvive:
    """A voice turn's reply gives a postcode, a phone number, a reference, a
    code or a year with every digit, in groups the voice reads."""

    @pytest.mark.parametrize("case", list(_each_case(ID_CASES)))
    def test_the_identifier_comes_back_whole(self, real_model_config, case):
        reply = _voice_turn(_config(real_model_config, case), case)

        print(f"\n  🗣️ {case.id}: {reply[:240]!r}")
        assert identifiers_kept(reply, case), (
            f"The reply altered {case.identifiers!r}: {reply[:400]!r}"
        )
        assert identifiers_right(reply, case), (
            f"The reply kept {case.identifiers!r} in a form the voice misreads: "
            f"{reply[:400]!r}"
        )


@pytest.mark.eval
@pytest.mark.skipif(ROUNDS < 1, reason="set EVAL_SPOKEN_UNITS_ROUNDS to compare")
class TestBeforeAndAfter:
    """Two arms of the rule, call by call: no rule against the prompt's, or
    the wording in `EVAL_SPOKEN_FORM_AGAINST` against the prompt's."""

    def test_the_rule_against_another_call_by_call(self, real_model_config, capsys):
        from jarvis.reply.prompts.model_variants import (
            SPOKEN_FORM, ModelSize, get_system_prompts,
        )

        assert SPOKEN_FORM, "no spoken-form rule to compare against"
        # Both arms would be the same prompt, and the comparison noise, if
        # the rule were read from anywhere the other arm does not replace.
        for size in ModelSize:
            with patch(_RULE_AT, ""):
                emptied = get_system_prompts(size, spoken=True).to_list()
            assert SPOKEN_FORM in get_system_prompts(size, spoken=True).to_list()
            assert SPOKEN_FORM not in emptied, (
                f"emptying {_RULE_AT} leaves the rule in the {size.name} prompt"
            )

        if AGAINST:
            with open(AGAINST, encoding="utf-8") as f:
                other = f.read().strip()
            assert other and other != SPOKEN_FORM, (
                f"{AGAINST} holds no other wording of the rule"
            )
            arms: Dict[str, Optional[str]] = {
                f"the rule in {os.path.basename(AGAINST)}": other,
                "the prompt's rule": None,
            }
        else:
            arms = {"without the rule": "", "with the rule": None}
        names = list(arms)
        languages = list(_LANGUAGES)
        # (arm, language) -> counters, units cases
        calls: Dict[Tuple[str, str], int] = {}
        clean: Dict[Tuple[str, str], int] = {}
        kept: Dict[Tuple[str, str], int] = {}
        # Both at once: nothing misread and every figure in digits.
        heard: Dict[Tuple[str, str], int] = {}
        by_kind: Dict[Tuple[str, str, str], int] = {}
        # (arm, language) -> counters, identifier cases
        id_calls: Dict[Tuple[str, str], int] = {}
        id_kept: Dict[Tuple[str, str], int] = {}
        id_right: Dict[Tuple[str, str], int] = {}
        per_case: Dict[Tuple[str, str], List[int]] = {}   # two scores, calls
        misses: List[str] = []

        def add(counter: Dict, key, n: int = 1) -> None:
            counter[key] = counter.get(key, 0) + n

        all_cases: List["UnitsCase | IdentifierCase"] = list(CASES) + list(ID_CASES)
        rng = random.Random(0)
        try:
            for rnd in range(ROUNDS):
                order = list(all_cases)
                rng.shuffle(order)
                for i, case in enumerate(order):
                    cfg = _config(real_model_config, case)
                    turn = (i + rnd) % len(names)
                    for name in names[turn:] + names[:turn]:
                        reply = _voice_turn(cfg, case, rule=arms[name])
                        key = (name, case.language)
                        row = per_case.setdefault((name, case.id), [0, 0, 0])
                        row[2] += 1
                        if isinstance(case, IdentifierCase):
                            whole = identifiers_kept(reply, case)
                            right = identifiers_right(reply, case)
                            add(id_calls, key)
                            add(id_kept, key, int(whole))
                            add(id_right, key, int(right))
                            row[0] += int(right)
                            row[1] += int(whole)
                            if not right:
                                what = "unreadable" if whole else "identifier lost"
                                misses.append(f"{name} / {case.id} ({what}): {reply[:160]!r}")
                            continue
                        found = misread_in(reply)
                        figures = keeps_its_figures(reply, case)
                        add(calls, key)
                        add(clean, key, int(not found))
                        add(kept, key, int(figures))
                        add(heard, key, int(figures and not found))
                        for kind in found:
                            add(by_kind, (name, case.language, kind))
                        if GROUPED_LONG_NUMBER.search(reply):
                            add(by_kind, (name, case.language, "grouped long number"))
                        row[0] += int(not found)
                        row[1] += int(figures)
                        if found or not figures:
                            what = ", ".join(found) if found else "figures lost"
                            if found and not figures:
                                what += ", figures lost"
                            misses.append(f"{name} / {case.id} ({what}): {reply[:160]!r}")
                to_terminal(capsys, [f"⏱️ round {rnd + 1}/{ROUNDS} done"])
        finally:
            lines = [
                "",
                f"📊 Spoken form: {len(CASES)} units cases and {len(ID_CASES)} "
                f"identifier cases x {ROUNDS} rounds, arms interleaved call by call",
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
                    m = id_calls.get(key, 0)
                    lines.append(
                        f"      🔢 {language}  identifiers right {rate(id_right.get(key, 0), m)}"
                        f"  (kept whole {rate(id_kept.get(key, 0), m)})"
                    )
                for case in all_cases:
                    a, b, n = per_case.get((name, case.id), [0, 0, 0])
                    if isinstance(case, IdentifierCase):
                        lines.append(f"         💬 {case.id:<20} right {a}/{n}, whole {b}/{n}")
                    else:
                        lines.append(f"         💬 {case.id:<20} clean {a}/{n}, kept {b}/{n}")
            for miss in misses:
                lines.append(f"      ❌ {miss}")
            to_terminal(capsys, lines)

        if AGAINST:
            # Which wording to keep weighs both measures at once: the run
            # reports them and leaves the judgement to whoever reads it.
            return
        for language in languages:
            before = heard.get(("without the rule", language), 0)
            after = heard.get(("with the rule", language), 0)
            assert after > before, (
                f"In {language}, replies with the rule are no more often read right "
                f"(nothing misread, every figure in digits) than replies without it: "
                f"{after} against {before}."
            )
            before = id_right.get(("without the rule", language), 0)
            after = id_right.get(("with the rule", language), 0)
            assert after >= before, (
                f"In {language}, the rule leaves fewer identifiers whole and "
                f"readable than no rule: {after} against {before}."
            )

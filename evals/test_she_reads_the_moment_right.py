"""
Eval: the moment she reads is the moment the sentence asks for.

LLM contexts #16 and #17, on 46 sentences, 34 of them in French. The
reference moment is fixed, so the answer each sentence asks for is
computed here and no judge is needed. A reminder is right when it lands
on the instant asked for, to the minute, or on the day when no hour was
named. A routine is right when its kind, weekday and hour are the ones
said. A sentence that asks for no repeating schedule is right when it is
refused.

Each failure is a promise kept at the wrong moment, or a refusal of one
she should have read. She says the time back, so a wrong day can be
caught in the same turn, but only by a user who listens for it.

What this measures is the model the configuration under test reads
with. `MockConfig` pins `reminder_model` to the suite's small model, so
the shape that matters is the chat in the cloud with that pin on this
machine. There the chat model reads, because the turn has already sent
it the sentence; the small local model reads only with the chat at home.
On the shape below, with arms interleaved over three runs of these
sentences, qwen2.5:3b read about half of them wrong and
gemini-2.5-flash read 137 of 138 right.

Run:
    EVAL_LLM_PROVIDER=openai_compatible \
    EVAL_LLM_BASE_URL=https://openrouter.ai/api/v1 \
    EVAL_LLM_API_KEY_ENV=OPENROUTER_API_KEY \
    EVAL_JUDGE_MODEL=google/gemini-2.5-flash \
    EVAL_SMALL_MODEL=qwen2.5:3b \
      ./scripts/run_evals.sh she_reads_the_moment_right
"""

from datetime import date, datetime, timedelta

import pytest

from conftest import requires_judge_llm
from helpers import MockConfig

# Tuesday 27 October 2026, 17:42, wall clock.
REF = datetime(2026, 10, 27, 17, 42)


def _a(day, month, hour, minute=0):
    return [datetime(2026, month, day, hour, minute)]


def _le(*days):
    """Any hour on one of these (day, month) dates: a day, and no hour."""
    return [date(2026, m, d) for d, m in days]


def _dans(**delta):
    return [REF + timedelta(**delta)]


RAPPELS = [
    ("rappelle-moi d'appeler maman dans vingt minutes", _dans(minutes=20)),
    ("dans deux heures, rappelle-moi de sortir le linge", _dans(hours=2)),
    ("rappelle-moi demain à 9 heures d'appeler le garage", _a(28, 10, 9)),
    ("rappelle-moi jeudi à 9h de payer le loyer", _a(29, 10, 9)),
    ("rappelle-moi lundi prochain d'envoyer la facture", _le((2, 11))),
    ("rappelle-moi le 3 novembre de renouveler mon passeport", _le((3, 11))),
    ("rappelle-moi ce soir à 20h de prendre mes médicaments", _a(27, 10, 20)),
    ("rappelle-moi à 18h30 de fermer les volets", _a(27, 10, 18, 30)),
    ("rappelle-moi samedi à 10 heures d'aller au marché", _a(31, 10, 10)),
    ("rappelle-moi demain matin de rappeler Paul",
     [datetime(2026, 10, 28, h) for h in range(6, 12)]),
    ("rappelle-moi après-demain à 14h d'aller chez le dentiste", _a(29, 10, 14)),
    ("rappelle-moi dans trois jours de vérifier le colis", _le((30, 10))),
    ("rappelle-moi vendredi à midi de réserver le restaurant", _a(30, 10, 12)),
    ("dans une demi-heure, rappelle-moi de sortir le gâteau du four",
     _dans(minutes=30)),
    ("rappelle-moi dimanche à 8h d'arroser les plantes", _a(1, 11, 8)),
    # "Prochain" is read both ways by people too: tomorrow, or next week's.
    ("rappelle-moi mercredi prochain à 15h d'appeler la banque",
     _a(28, 10, 15) + _a(4, 11, 15)),
    ("rappelle-moi à 7h demain de prendre le train", _a(28, 10, 7)),
    ("rappelle-moi le 15 décembre d'acheter les cadeaux", _le((15, 12))),
    ("rappelle-moi dans une heure et quart de relancer la machine",
     _dans(hours=1, minutes=15)),
    ("rappelle-moi jeudi prochain à 9h de payer le loyer",
     _a(29, 10, 9) + _a(5, 11, 9)),
    # Next week starts on Monday 2/11, or on Sunday 1/11 for a week that
    # starts on Sunday. Saturday 31/10 is this week either way.
    ("rappelle-moi la semaine prochaine de prendre rendez-vous chez le médecin",
     _le(*[(d, 11) for d in range(1, 9)])),
    ("rappelle-moi à 21h de sortir les poubelles", _a(27, 10, 21)),
    ("remind me in twenty minutes to call mum", _dans(minutes=20)),
    ("remind me tomorrow at 9 to call the garage", _a(28, 10, 9)),
    ("remind me on Thursday at 9am to pay the rent", _a(29, 10, 9)),
    ("remind me next Monday to send the invoice", _le((2, 11))),
    ("remind me on November 3rd to renew my passport", _le((3, 11))),
    ("remind me tonight at 8pm to take my pills", _a(27, 10, 20)),
    ("remind me on Saturday at 10 to go to the market", _a(31, 10, 10)),
    ("remind me the day after tomorrow at 2pm to go to the dentist",
     _a(29, 10, 14)),
]

_MATIN = list(range(6, 12))
_TOUTE_HEURE = list(range(24))

# (kind, weekday, hours accepted, minute or None for any), or None for a
# sentence that asks for no repeating schedule.
ROUTINES = [
    ("tous les lundis matin, fais-moi un point sur la météo",
     ("weekly", 0, _MATIN, None)),
    ("chaque jour à 7h, donne-moi les actualités", ("daily", None, [7], 0)),
    ("tous les vendredis à 18h, rappelle-moi de faire le point de la semaine",
     ("weekly", 4, [18], 0)),
    ("tous les soirs à 22h, dis-moi ce que j'ai prévu demain",
     ("daily", None, [22], 0)),
    ("chaque dimanche à 10h, fais-moi la liste des courses",
     ("weekly", 6, [10], 0)),
    ("tous les mercredis à 14h30, rappelle-moi le cours de piano",
     ("weekly", 2, [14], 30)),
    ("chaque matin à 8 heures, donne-moi la météo", ("daily", None, [8], 0)),
    ("tous les jeudis, vérifie mes objectifs",
     ("weekly", 3, _TOUTE_HEURE, None)),
    ("le samedi à 9h, toutes les semaines, rappelle-moi de faire le ménage",
     ("weekly", 5, [9], 0)),
    ("tous les mardis à midi, propose-moi une recette", ("weekly", 1, [12], 0)),
    ("demain à 9h, donne-moi la météo", None),
    ("toutes les heures, regarde mes mails", None),
    ("every Monday morning, give me a weather update",
     ("weekly", 0, _MATIN, None)),
    ("every day at 7am, read me the news", ("daily", None, [7], 0)),
    ("every Friday at 6pm, remind me to review the week",
     ("weekly", 4, [18], 0)),
    ("every Sunday at 10, make me a shopping list", ("weekly", 6, [10], 0)),
]


def _lands_on(due, accepted):
    for item in accepted:
        if isinstance(item, datetime):
            if abs((due - item).total_seconds()) <= 60:
                return True
        elif due.date() == item:
            return True
    return False


@pytest.mark.eval
@requires_judge_llm
@pytest.mark.parametrize("phrase,attendu", RAPPELS, ids=[p for p, _ in RAPPELS])
def test_a_reminder_lands_on_the_moment_asked_for(phrase, attendu):
    from jarvis.reminders.extract import ExtractionFailed, extract_reminder_time

    try:
        lu = extract_reminder_time(MockConfig(), phrase, now_local=REF)
    except ExtractionFailed as e:
        pytest.fail(f"{phrase!r} was refused ({e}). It names a moment, and "
                    f"a refusal costs the user saying it again.")

    print(f"\n  {phrase!r} -> {lu.due_local:%a %Y-%m-%d %H:%M}")

    assert _lands_on(lu.due_local, attendu), (
        f"{phrase!r} was read as {lu.due_local:%A %Y-%m-%d %H:%M} on "
        f"Tuesday 2026-10-27 17:42. A reminder at the wrong moment is a "
        f"promise broken on time."
    )


@pytest.mark.eval
@requires_judge_llm
@pytest.mark.parametrize("phrase,attendu", ROUTINES, ids=[p for p, _ in ROUTINES])
def test_a_routine_repeats_on_the_rhythm_asked_for(phrase, attendu):
    from jarvis.reminders.extract import ExtractionFailed
    from jarvis.routines.extract import extract_routine_rule

    if attendu is None:
        with pytest.raises(ExtractionFailed):
            lu = extract_routine_rule(MockConfig(), phrase)
            print(f"\n  {phrase!r} -> {lu.regle.to_json()}")
        return

    try:
        lu = extract_routine_rule(MockConfig(), phrase)
    except ExtractionFailed as e:
        pytest.fail(f"{phrase!r} was refused ({e}). It names a rhythm.")

    regle = lu.regle
    print(f"\n  {phrase!r} -> {regle.to_json()}")
    kind, weekday, heures, minute = attendu

    assert regle.kind == kind, (phrase, regle.to_json())
    if kind == "weekly":
        assert regle.weekday == weekday, (
            f"{phrase!r} runs on weekday {regle.weekday}, not {weekday}: a "
            f"routine on the wrong day runs there every week."
        )
    assert regle.hour in heures, (phrase, regle.to_json())
    if minute is not None:
        assert regle.minute == minute, (phrase, regle.to_json())

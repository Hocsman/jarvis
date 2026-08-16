"""What the bench measures, and what it refuses to hide.

His failure is not the average. Over a session, common French phrasing
came back clean while every proper noun broke: "Mythos 5" landed as
"met ce 5", "MISO 5", "Mesos 5" and "Mezus 5" on four passes, and
"Bagneux" as "Banyo". A whole-corpus word error rate averages that away —
one wrong word in fifteen reads as 6%, which looks like a good engine
even though the only word that carried meaning is the one it lost.

So the bench reports two numbers: the error rate, and the recall on the
words he marked as the ones that must survive. A near-miss is a miss:
"Banyo" does not count as "Bagneux", and any fuzzy matching here would
score the exact failure being investigated as a success.

It also refuses to be quiet about what it did not do. An engine that is
not installed is a line saying so, never an absence from the table, and
a clip with no reference is counted aloud rather than dropped.
"""

from __future__ import annotations

import json

import pytest

from scripts.asr_bench import (
    charger_corpus,
    normaliser,
    rappel_mots_cles,
    taux_erreur_caracteres,
    taux_erreur_mots,
)


# ── Normalisation ──────────────────────────────────────────────────────


def test_case_and_punctuation_are_not_errors():
    assert normaliser("La météo, à Bagneux !") == normaliser("la météo à bagneux")


def test_accents_survive_normalisation():
    """They carry the distinction the bench exists to measure: dropping
    them would make "mate" and "maté" the same word."""
    assert normaliser("météo") != normaliser("meteo")


# ── Taux d'erreur ──────────────────────────────────────────────────────


def test_a_perfect_transcription_scores_zero():
    assert taux_erreur_mots("la météo à Bagneux", "La météo à Bagneux !") == 0.0


def test_his_actual_failure_scores_badly():
    """Reference and hypothesis from the log on 2026-08-16."""
    taux = taux_erreur_mots("la météo à Bagneux", "Théo sur Banyo")

    assert taux > 0.7, taux


def test_an_empty_hypothesis_loses_every_word():
    assert taux_erreur_mots("la météo à Bagneux", "") == 1.0


def test_an_empty_reference_is_not_a_division_by_zero():
    assert taux_erreur_mots("", "n'importe quoi") is None


def test_characters_are_counted_too():
    """Word rate calls "Banyo" and "Bagneux" equally wrong as any other
    substitution; the character rate shows one was nearly right."""
    proche = taux_erreur_caracteres("Bagneux", "Banyo")
    loin = taux_erreur_caracteres("Bagneux", "zzzzzzz")

    assert proche < loin


# ── Rappel des mots qui comptent ───────────────────────────────────────


def test_a_keyword_that_survived_is_counted():
    trouves, total = rappel_mots_cles(["Bagneux", "météo"],
                                      "quelle est la météo à Bagneux")

    assert (trouves, total) == (2, 2)


def test_a_near_miss_is_a_miss():
    """The whole point. Fuzzy matching here would score the failure under
    investigation as a success."""
    trouves, total = rappel_mots_cles(["Bagneux"], "Théo sur Banyo")

    assert (trouves, total) == (0, 1)


def test_a_keyword_matches_whatever_the_case_and_the_punctuation():
    trouves, _ = rappel_mots_cles(["Bagneux"], "c'est à BAGNEUX, oui.")

    assert trouves == 1


def test_a_keyword_is_a_word_and_not_a_fragment():
    """"Yuba" inside "Yubatel" is not the word being looked for."""
    trouves, _ = rappel_mots_cles(["Yuba"], "j'ai vu Yubatel hier")

    assert trouves == 0


def test_a_multi_word_keyword_works():
    trouves, _ = rappel_mots_cles(["Mythos 5"], "le modèle Mythos 5 mène")

    assert trouves == 1


# ── Le corpus ──────────────────────────────────────────────────────────


def _clip(dossier, nom, **champs):
    base = {"audio": f"{nom}.wav", "hypothesis": "", "reference": "",
            "keywords": [], "sample_rate": 16000}
    base.update(champs)
    (dossier / f"{nom}.json").write_text(json.dumps(base), encoding="utf-8")
    (dossier / f"{nom}.wav").write_bytes(b"RIFF")


def test_a_clip_he_has_transcribed_is_scorable(tmp_path):
    _clip(tmp_path, "a", reference="la météo à Bagneux", keywords=["Bagneux"])

    pret, sans_reference = charger_corpus(tmp_path)

    assert len(pret) == 1
    assert sans_reference == 0
    assert pret[0].reference == "la météo à Bagneux"


def test_clips_without_a_reference_are_counted_not_dropped(tmp_path):
    """Silently skipping them would report a confident number over three
    clips out of forty and never say which forty."""
    _clip(tmp_path, "a", reference="la météo à Bagneux")
    _clip(tmp_path, "b")
    _clip(tmp_path, "c")

    pret, sans_reference = charger_corpus(tmp_path)

    assert len(pret) == 1
    assert sans_reference == 2


def test_a_clip_whose_audio_is_gone_is_not_scored(tmp_path):
    _clip(tmp_path, "a", reference="x")
    (tmp_path / "a.wav").unlink()

    pret, _ = charger_corpus(tmp_path)

    assert pret == []


def test_an_unreadable_sidecar_does_not_stop_the_run(tmp_path):
    _clip(tmp_path, "a", reference="la météo à Bagneux")
    (tmp_path / "b.json").write_text("{ pas du json", encoding="utf-8")

    pret, _ = charger_corpus(tmp_path)

    assert len(pret) == 1


# ── Les moteurs ────────────────────────────────────────────────────────


def test_an_engine_that_is_not_installed_says_so(tmp_path):
    """An engine missing from the table reads as an engine that lost.
    It has to be a line of its own saying why, with the way to fix it."""
    from scripts.asr_bench import moteurs_disponibles

    etats = moteurs_disponibles()

    assert etats, "le banc doit connaître au moins un moteur"
    for etat in etats:
        assert etat.nom
        if not etat.disponible:
            assert etat.installation, f"{etat.nom} n'explique pas comment l'installer"

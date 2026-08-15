"""A subsystem that speaks unprompted needs a switch he can reach.

`reminders_enabled` starts a thread that says things out loud at a time
he is not necessarily at the machine for. Its twin `routines_enabled` has
a field in the settings window, with a comment explaining exactly why.
This one had none: turning reminders off meant editing `config.json` by
hand and restarting.

`reminder_model` matters for a second reason. `reminders.spec.md` makes
it the way to keep the sentence he dictated — "rappelle-moi d'appeler
l'oncologue jeudi" — off the network, since that prompt carries his own
life and the reminder chain deliberately skips the cloud-safe rewrite.
A privacy control nobody can find is a privacy control nobody uses.
"""

from __future__ import annotations

import pytest


def _metadonnees():
    from src.desktop_app.settings_window import _build_field_metadata

    return [{"key": m.key, "category": m.category} for m in _build_field_metadata()]


def test_the_switch_for_what_speaks_on_its_own_is_reachable():
    cles = {m["key"] for m in _metadonnees()}

    assert "reminders_enabled" in cles


def test_the_model_that_keeps_his_sentence_home_is_reachable():
    cles = {m["key"] for m in _metadonnees()}

    assert "reminder_model" in cles


def test_every_reminder_field_lands_in_a_category_the_sidebar_shows():
    from src.desktop_app.settings_window import CATEGORIES

    connues = {cle for cle, _ in CATEGORIES}
    orphelines = [m["key"] for m in _metadonnees()
                  if m.get("category") not in connues]

    assert orphelines == [], f"champs sans catégorie affichée : {orphelines}"


def test_the_spec_lists_the_categories_the_code_has():
    """The window is generated from metadata, so the spec's ordered list
    is the contract. It said fifteen where the code had seventeen."""
    from pathlib import Path

    from src.desktop_app.settings_window import CATEGORIES

    # Compared on the words, not the emoji: the spec's older rows are
    # written without one, and the point is that the list is complete
    # rather than that it is decorated the same way.
    spec = Path("src/desktop_app/settings_window.spec.md").read_text(encoding="utf-8")
    manquantes = [
        libelle for _, libelle in CATEGORIES
        if libelle.split(" ", 1)[-1].strip() not in spec
    ]

    assert manquantes == [], f"catégories absentes de la spec : {manquantes}"

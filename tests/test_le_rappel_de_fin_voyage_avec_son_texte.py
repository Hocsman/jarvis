"""The completion callback belongs to its text, not to the engine.

Each engine keeps `self._completion_callback` and `speak()` overwrites
it, so the callback is a property of the engine rather than of the thing
being said. With one utterance per reply that is invisible. With a reply
spoken sentence by sentence it is not: the callback set for the last
chunk would fire after the first one to finish, and that callback is what
opens the hot window.

She would open the window in the middle of her own reply and take her
next sentence for a follow-up — which is the failure that three separate
guards were written for on 2026-08-16.

So the queue carries `(text, completion_callback, duration_callback)`,
and the worker installs them as it dequeues. Only the worker thread ever
writes them, and only for the item it is about to speak.
"""

from __future__ import annotations

import queue
import threading

import pytest


def _moteur(classe_nom):
    """An engine with just enough wiring to drive its queue, no audio."""
    import src.jarvis.output.tts as tts_mod

    classe = getattr(tts_mod, classe_nom)
    m = classe.__new__(classe)
    m._q = queue.Queue()
    m._stop = threading.Event()
    m._thread = object()          # already "started", so speak() enqueues
    m.enabled = True
    m._completion_callback = None
    m._duration_callback = None
    return m


def _draine(moteur):
    """Run the worker over what is queued, recording (text, callback)."""
    vus = []
    moteur._speak_once = lambda t: vus.append((t, moteur._completion_callback))

    def _boucle():
        while True:
            try:
                item = moteur._q.get_nowait()
            except queue.Empty:
                return
            texte, cb, dur = item
            if not texte:
                continue
            moteur._completion_callback = cb
            moteur._duration_callback = dur
            moteur._speak_once(texte)

    _boucle()
    return vus


MOTEURS = ["KokoroTTS", "PiperTTS", "ChatterboxTTS"]


@pytest.mark.parametrize("nom", MOTEURS)
def test_the_callback_arrives_with_the_chunk_it_was_given_to(nom):
    moteur = _moteur(nom)
    fin = lambda: None

    moteur.speak("Il fait beau à Bagneux.")
    moteur.speak("Vingt-quatre degrés.")
    moteur.speak("Prends une veste.", completion_callback=fin)

    vus = _draine(moteur)

    assert [cb for _, cb in vus] == [None, None, fin]


@pytest.mark.parametrize("nom", MOTEURS)
def test_the_earlier_chunks_carry_no_callback_at_all(nom):
    """The heart of it: an early chunk must not inherit the callback of a
    later one and open the hot window mid-reply."""
    moteur = _moteur(nom)
    fin = lambda: None

    moteur.speak("Première phrase ici.")
    moteur.speak("Dernière phrase.", completion_callback=fin)

    vus = _draine(moteur)

    assert vus[0][1] is None


@pytest.mark.parametrize("nom", MOTEURS)
def test_a_single_utterance_still_carries_its_callback(nom):
    """The control, and the path everything else in the app uses today."""
    moteur = _moteur(nom)
    fin = lambda: None

    moteur.speak("Une seule phrase.", completion_callback=fin)

    vus = _draine(moteur)

    assert len(vus) == 1
    assert vus[0][1] is fin


@pytest.mark.parametrize("nom", MOTEURS)
def test_the_text_is_unchanged_by_the_journey(nom):
    moteur = _moteur(nom)

    moteur.speak("Prends une veste quand même.")

    vus = _draine(moteur)

    assert "veste" in vus[0][0]

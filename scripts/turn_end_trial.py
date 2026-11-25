#!/usr/bin/env python3
"""Try the end-of-turn model on recordings, and time it.

Usage:
    python scripts/turn_end_trial.py [--voices DIR] [--closing S,S] [--wavs DIR] [--keep DIR]

What it does, on this machine only:

* times the model: how long it takes to open, and to give a verdict on
  three and eight seconds of sound;
* speaks a set of invented French sentences with each Piper voice found in
  ``--voices`` (default: the ones Jarvis downloaded), and asks the model
  about four kinds of turn, each preceded by the voice detector's pre-roll
  and followed by part of the silence that closes an utterance:
    - ``finished``: the whole sentence, which it should call finished;
    - ``paused``: the same sentence said with a pause after a word that
      cannot end it ("de", "à", "le"), cut at that pause, which it should
      call unfinished;
    - ``standalone``: paused where the words before the pause could be a
      request of their own ("Mets un minuteur, de douze minutes."), scored
      apart since a listener could hear either;
    - ``short``: short answers ("Oui.", "Vas-y."), which it should call
      finished;
  in quiet and over a little room noise, with each closing silence given in
  ``--closing`` (default: the one the listener hands the model, and 0.4 s);
* with ``--wavs``, reads 16 kHz mono WAV files whose name holds
  ``unfinished`` or ``finished`` and scores them the same way: a way to try
  it on his own recordings.

A synthetic voice says the words before a comma the way it ends a sentence,
which a person searching for a word does not: the ``paused`` rates measure
the model against that voice more than against a hesitation.

It prints rates and timings, never the sentences' audio, and writes nothing
unless ``--keep`` names a folder for the synthesised WAV files. The model
must already be in place (Jarvis downloads it on its first start); the
script downloads nothing.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jarvis.listening import turn_end  # noqa: E402

PRE_ROLL = 0.24
CLOSING_SILENCES = (turn_end.CLOSING_SILENCE_HEARD, 0.4)
THRESHOLDS = (0.5, 0.6, 0.7)
NOISE_DBFS = -50.0

# Each sentence once said whole and once with a pause where "|" sits, the
# words before it unable to end a sentence: he is looking for the next one.
SENTENCES = (
    "Quelle heure est-il à | Tokyo ?",
    "Est-ce que tu peux me dire quel temps il fera demain à | Valbrume ?",
    "Rappelle-moi d'appeler le | garagiste à dix-sept heures.",
    "Mets un minuteur de | douze minutes pour les pâtes.",
    "Je voudrais savoir combien de | calories il y a dans une pomme.",
    "Ajoute des œufs et | du lait à ma liste de courses.",
    "Qu'est-ce que je dois faire pour | mettre à jour mon ordinateur ?",
    "Lance la playlist de | jazz dans le salon.",
    "Combien de temps faut-il pour | cuire un œuf dur ?",
    "Tu peux me résumer les | nouvelles de ce matin ?",
    "Dis-moi ce que j'ai prévu pour | cet après-midi.",
    "Quelle est la capitale de | l'Australie ?",
    "Comment on dit bonne nuit en | japonais ?",
    "Je cherche une recette avec des | courgettes.",
    "Envoie un message à | Clara pour lui dire que j'arrive.",
    "À quelle heure ferme la | pharmacie du quartier ?",
    "Est-ce qu'il va pleuvoir à | Valbrume ce week-end ?",
    "Peux-tu baisser le | volume de la musique ?",
    "Note que la réunion est déplacée à | jeudi.",
    "Quel est le meilleur moment pour | planter des tomates ?",
    "Calcule vingt-trois fois | quarante-sept.",
    "Raconte-moi une histoire sur | les dragons.",
)
# Paused where the words before the pause could be a request of their own:
# only the rest of the sentence tells they were not. Scored apart, since a
# listener could hear either.
STANDALONE_PAUSES = (
    "Quelle heure est-il | à Tokyo ?",
    "Rappelle-moi | d'appeler le garagiste.",
    "Mets un minuteur | de douze minutes.",
    "Lance | la playlist de jazz.",
    "Dis-moi | ce que j'ai prévu cet après-midi.",
    "Note | que la réunion est déplacée à jeudi.",
    "Calcule | vingt-trois fois quarante-sept.",
    "Raconte-moi | une histoire courte.",
)
SHORT_ANSWERS = ("Oui.", "Non.", "Vas-y.", "D'accord.", "Oui, lance-le.", "Non merci.",
                 "C'est bon.", "Annule.")


def _resample(sound: np.ndarray, rate: int) -> np.ndarray:
    if rate == turn_end.SAMPLE_RATE:
        return sound
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(rate, turn_end.SAMPLE_RATE)
    return resample_poly(sound, turn_end.SAMPLE_RATE // g, rate // g).astype(np.float32)


def _speak(voice, text: str) -> np.ndarray:
    chunks = [chunk.audio_int16_array for chunk in voice.synthesize(text)]
    sound = np.concatenate(chunks).astype(np.float32) / 32768.0
    return _resample(sound, voice.config.sample_rate)


def _trim(sound: np.ndarray, floor: float = 0.01) -> np.ndarray:
    """The sound without the silence Piper puts around it."""
    loud = np.flatnonzero(np.abs(sound) > floor)
    return sound[loud[0]:loud[-1] + 1] if loud.size else sound


def _cut_at_the_pause(sound: np.ndarray) -> np.ndarray | None:
    """The sound up to the longest quiet stretch in its middle (the pause
    the comma made), or None when there is none of 80 ms or more."""
    frame = int(0.01 * turn_end.SAMPLE_RATE)
    n = sound.size // frame
    rms = np.sqrt(np.mean(sound[:n * frame].reshape(n, frame) ** 2, axis=1))
    quiet = rms < max(0.005, 0.05 * rms.max())
    best, best_len, start = None, 0, None
    for i in range(int(n * 0.15), int(n * 0.85)):
        if quiet[i] and start is None:
            start = i
        if (not quiet[i] or i == int(n * 0.85) - 1) and start is not None:
            if i - start > best_len:
                best, best_len = start, i - start
            start = None
    if best is None or best_len < 8:
        return None
    return sound[:best * frame]


def _turn(speech: np.ndarray, closing: float, noise: bool, rng) -> np.ndarray:
    """What the listener hands the model: the pre-roll, his voice, and the
    silence that closed the utterance."""
    sound = np.concatenate([np.zeros(int(PRE_ROLL * turn_end.SAMPLE_RATE), np.float32), speech,
                            np.zeros(int(closing * turn_end.SAMPLE_RATE), np.float32)])
    if noise:
        level = 10 ** (NOISE_DBFS / 20)
        sound = sound + (level * rng.standard_normal(sound.size)).astype(np.float32)
    return sound


def _detector() -> turn_end.TurnEndDetector:
    def no_download(**_kwargs):
        raise RuntimeError("the trial downloads nothing: start Jarvis once to fetch the model")

    detector = turn_end.TurnEndDetector(download=no_download)
    if not detector.load():
        print(f"❌ End-of-turn model unavailable: {detector.why}")
        sys.exit(1)
    return detector


def _time() -> turn_end.TurnEndDetector:
    """Opens the model, times that and its verdicts, and returns it."""
    print("⏱️  Timing")
    started = time.perf_counter()
    fresh = _detector()
    print(f"   🔌 opened in {(time.perf_counter() - started) * 1000:.0f} ms "
          f"({turn_end.THREADS} threads)")
    rng = np.random.default_rng(0)
    for seconds in (3, 8):
        sound = (0.05 * rng.standard_normal(seconds * turn_end.SAMPLE_RATE)).astype(np.float32)
        fresh.probability(sound)
        took = []
        for _ in range(200):
            t = time.perf_counter()
            fresh.probability(sound)
            took.append((time.perf_counter() - t) * 1000)
        took.sort()
        print(f"   🧭 verdict on {seconds} s: median {statistics.median(took):.1f} ms, "
              f"p90 {took[int(0.9 * len(took))]:.1f} ms, max {took[-1]:.1f} ms")
    return fresh


def _score(rows: list, label: str) -> None:
    """rows: (kind, probability). Prints, per kind, how often the verdict
    is the expected one at each threshold."""
    expected = {"finished": True, "short": True, "paused": False, "standalone": False}
    print(f"   {label}")
    for kind in ("finished", "paused", "standalone", "short"):
        ps = [p for k, p in rows if k == kind]
        if not ps:
            continue
        rates = []
        for threshold in THRESHOLDS:
            right = sum((p > threshold) == expected[kind] for p in ps)
            rates.append(f"{threshold:.1f}: {right}/{len(ps)} ({100 * right / len(ps):.0f}%)")
        median = statistics.median(ps)
        print(f"      {'✅' if expected[kind] else '⏸️ '} {kind:<9} median p={median:.2f}   "
              + "   ".join(rates))


def _piper(detector, voices_dir: Path, keep: Path | None, closings=CLOSING_SILENCES) -> None:
    from piper.voice import PiperVoice

    voices = sorted(p for p in voices_dir.glob("fr_*.onnx"))
    if not voices:
        print(f"⚠️  No French Piper voice in {voices_dir}")
        return
    print(f"🗣️  Piper corpus: {len(SENTENCES)} sentences, {len(STANDALONE_PAUSES)} paused where "
          f"the start could stand alone, {len(SHORT_ANSWERS)} short answers, {len(voices)} voices")
    rng = np.random.default_rng(1)
    by_condition: dict = {}
    by_voice: dict = {}
    skipped = 0
    for path in voices:
        voice = PiperVoice.load(str(path), str(path) + ".json")
        name = path.stem
        speech = []
        for sentence in SENTENCES:
            whole = _trim(_speak(voice, sentence.replace(" | ", " ")))
            paused = _trim(_speak(voice, sentence.replace(" | ", ", ")))
            cut = _cut_at_the_pause(paused)
            speech.append(("finished", whole))
            if cut is None:
                skipped += 1
            else:
                speech.append(("paused", cut))
        for sentence in STANDALONE_PAUSES:
            cut = _cut_at_the_pause(_trim(_speak(voice, sentence.replace(" | ", ", "))))
            if cut is None:
                skipped += 1
            else:
                speech.append(("standalone", cut))
        for answer in SHORT_ANSWERS:
            speech.append(("short", _trim(_speak(voice, answer))))
        for closing in closings:
            for noise in (False, True):
                key = (closing, noise)
                for i, (kind, s) in enumerate(speech):
                    sound = _turn(s, closing, noise, rng)
                    p = detector.probability(sound)
                    by_condition.setdefault(key, []).append((kind, p))
                    if key == (closings[0], True):
                        by_voice.setdefault(name, []).append((kind, p))
                    if keep is not None:
                        keep.mkdir(parents=True, exist_ok=True)
                        _write(keep / f"{name}_{i:02d}_{kind}_{closing}_{int(noise)}.wav", sound)
    if skipped:
        print(f"   ⚠️  {skipped} paused sentences had no pause long enough to cut at")
    print("📊 Verdicts expected / given, by threshold")
    for (closing, noise), rows in sorted(by_condition.items()):
        _score(rows, f"🔇 closing silence {closing} s, {'room noise' if noise else 'quiet'}")
    print(f"🎙️  By voice ({closings[0]} s closing silence, room noise)")
    for name, rows in sorted(by_voice.items()):
        _score(rows, name)


def _write(path: Path, sound: np.ndarray) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(turn_end.SAMPLE_RATE)
        w.writeframes((np.clip(sound, -1, 1) * 32767).astype(np.int16).tobytes())


def _read(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        if w.getnchannels() > 1:
            data = data.reshape(-1, w.getnchannels()).mean(axis=1)
    return _resample(data.astype(np.float32) / 32768.0, rate)


def _wavs(detector, folder: Path) -> None:
    rows = []
    for path in sorted(folder.glob("*.wav")):
        kind = ("paused" if "unfinished" in path.stem else
                "finished" if "finished" in path.stem else None)
        if kind is None:
            continue
        rows.append((kind, detector.probability(_read(path))))
    print(f"📁 {len(rows)} recordings from {folder}")
    _score(rows, "his recordings (unfinished counted as paused)")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--voices", default=str(Path("~/.local/share/jarvis/models/piper").expanduser()))
    parser.add_argument("--wavs", help="folder of WAV files named with finished/unfinished")
    parser.add_argument("--keep", help="folder to write the synthesised turns to")
    parser.add_argument("--closing", default=",".join(map(str, CLOSING_SILENCES)),
                        help="closing silences to try, in seconds, comma-separated")
    args = parser.parse_args(argv)

    detector = _time()
    closings = tuple(float(s) for s in args.closing.split(","))
    _piper(detector, Path(args.voices), Path(args.keep) if args.keep else None, closings)
    if args.wavs:
        _wavs(detector, Path(args.wavs))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Which recogniser survives his own recordings.

The question this answers is narrow and was not answerable from public
leaderboards: on *his* microphone, at *his* distance, in French, do the
proper nouns come back. A session on 2026-08-16 had common phrasing
transcribing cleanly while "Mythos 5" landed four different wrong ways
and "Bagneux" came back "Banyo" — which is the signature of an
autoregressive decoder's language prior pulling an unknown token toward a
familiar word, not of a bad microphone.

So the bench reports the error rate *and* the recall on the words he
marked as load-bearing, because a whole-corpus average hides exactly the
words that carried the meaning.

Everything runs locally. The corpus is his voice and never leaves the
machine.

Build a corpus:

    JARVIS_SAVE_AUDIO=~/yuba-corpus PYTHONPATH=src python -m desktop_app

Talk to her normally, then fill in the ``reference`` field of each JSON
with what you actually said, and list the words that must survive in
``keywords``. Then:

    python scripts/asr_bench.py ~/yuba-corpus
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

# ── Normalisation and scoring ──────────────────────────────────────────

_PONCTUATION = re.compile(r"[^\w\s]", re.UNICODE)
_ESPACES = re.compile(r"\s+", re.UNICODE)


def normaliser(texte: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace.

    Accents are kept. They carry the distinction being measured: folding
    them would make "mate" and "maté" the same word and quietly forgive
    the error class this bench exists to find. Unicode is normalised to
    NFC so a composed and a decomposed "é" compare equal.
    """
    if not isinstance(texte, str):
        return ""
    texte = unicodedata.normalize("NFC", texte)
    texte = _PONCTUATION.sub(" ", texte.lower())
    return _ESPACES.sub(" ", texte).strip()


def _distance(a: List[str], b: List[str]) -> int:
    """Levenshtein over two sequences, in O(len(b)) memory."""
    if not a:
        return len(b)
    precedent = list(range(len(b) + 1))
    for i, jeton_a in enumerate(a, start=1):
        courant = [i]
        for j, jeton_b in enumerate(b, start=1):
            courant.append(min(
                precedent[j] + 1,           # suppression
                courant[j - 1] + 1,         # insertion
                precedent[j - 1] + (jeton_a != jeton_b),  # substitution
            ))
        precedent = courant
    return precedent[-1]


def taux_erreur_mots(reference: str, hypothese: str) -> Optional[float]:
    """WER, or ``None`` when there is no reference to score against.

    ``None`` rather than 0.0: a clip he has not transcribed is a clip
    that was not measured, and scoring it as perfect would flatter every
    engine equally.
    """
    ref = normaliser(reference).split()
    if not ref:
        return None
    return _distance(ref, normaliser(hypothese).split()) / len(ref)


def taux_erreur_caracteres(reference: str, hypothese: str) -> Optional[float]:
    """CER. Kept alongside WER because WER calls every wrong word equally
    wrong, and "Banyo" for "Bagneux" is a different kind of failure from
    a word invented whole."""
    ref = list(normaliser(reference).replace(" ", ""))
    if not ref:
        return None
    hyp = list(normaliser(hypothese).replace(" ", ""))
    return _distance(ref, hyp) / len(ref)


def rappel_mots_cles(mots_cles: List[str], hypothese: str) -> Tuple[int, int]:
    """How many of the load-bearing words came back, and how many there were.

    Matched on whole words after normalisation, never fuzzily: "Banyo" is
    not "Bagneux", and a near-miss that scored as a hit would report the
    failure under investigation as a success. Multi-word keys match as a
    contiguous run.
    """
    if not mots_cles:
        return 0, 0
    jetons = normaliser(hypothese).split()
    trouves = 0
    for cle in mots_cles:
        attendu = normaliser(cle).split()
        if not attendu:
            continue
        fenetres = (jetons[i:i + len(attendu)]
                    for i in range(len(jetons) - len(attendu) + 1))
        if any(f == attendu for f in fenetres):
            trouves += 1
    return trouves, len([c for c in mots_cles if normaliser(c)])


# ── The corpus ─────────────────────────────────────────────────────────


@dataclass
class Extrait:
    audio: Path
    reference: str
    keywords: List[str] = field(default_factory=list)
    hypothesis: str = ""
    device: str = ""
    duration_sec: float = 0.0


def charger_corpus(dossier) -> Tuple[List[Extrait], int]:
    """Return the scorable clips and the count of those he has not yet
    transcribed. The second number is returned rather than logged so the
    caller has to say it out loud: a confident average over three clips
    out of forty is the shape of a measurement that measured nothing."""
    dossier = Path(dossier)
    prets: List[Extrait] = []
    sans_reference = 0
    for fiche in sorted(dossier.glob("*.json")):
        try:
            data = json.loads(fiche.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        audio = dossier / str(data.get("audio") or f"{fiche.stem}.wav")
        if not audio.exists():
            continue
        reference = str(data.get("reference") or "").strip()
        if not reference:
            sans_reference += 1
            continue
        mots = data.get("keywords") or []
        prets.append(Extrait(
            audio=audio,
            reference=reference,
            keywords=[str(m) for m in mots if str(m).strip()],
            hypothesis=str(data.get("hypothesis") or ""),
            device=str(data.get("device") or ""),
            duration_sec=float(data.get("duration_sec") or 0.0),
        ))
    return prets, sans_reference


# ── The engines ────────────────────────────────────────────────────────


@dataclass
class EtatMoteur:
    nom: str
    disponible: bool
    installation: str = ""
    detail: str = ""


def _mlx_dispo() -> bool:
    import importlib.util
    return importlib.util.find_spec("mlx_whisper") is not None


def _parakeet_dispo() -> bool:
    import importlib.util
    return importlib.util.find_spec("parakeet_mlx") is not None


MOTEURS = {
    "whisper-medium": {
        "repo": "mlx-community/whisper-medium-mlx",
        "type": "mlx-whisper",
    },
    "whisper-large-v3-turbo": {
        "repo": "mlx-community/whisper-large-v3-turbo",
        "type": "mlx-whisper",
    },
    "parakeet-tdt-0.6b-v3": {
        "repo": "mlx-community/parakeet-tdt-0.6b-v3",
        "type": "parakeet-mlx",
    },
}


def moteurs_disponibles() -> List[EtatMoteur]:
    """Every engine the bench knows, with why the missing ones are missing.

    An engine absent from the table reads as an engine that lost, so a
    missing one gets a line of its own and the command that would fix it.
    """
    etats = []
    for nom, spec in MOTEURS.items():
        if spec["type"] == "mlx-whisper":
            ok = _mlx_dispo()
            etats.append(EtatMoteur(
                nom, ok, "" if ok else "pip install mlx-whisper",
                detail=spec["repo"]))
        else:
            ok = _parakeet_dispo()
            etats.append(EtatMoteur(
                nom, ok, "" if ok else "pip install parakeet-mlx",
                detail=spec["repo"]))
    return etats


def transcrire(nom: str, audio: Path) -> str:
    """Run one engine over one clip. Raises on failure: a caught exception
    that returned "" would score as a total miss and look like a result."""
    spec = MOTEURS[nom]
    if spec["type"] == "mlx-whisper":
        import mlx_whisper
        out = mlx_whisper.transcribe(str(audio),
                                     path_or_hf_repo=spec["repo"],
                                     language=None)
        return str(out.get("text") or "").strip()

    from parakeet_mlx import from_pretrained
    modele = _CACHE_PARAKEET.setdefault(
        spec["repo"], from_pretrained(spec["repo"]))
    out = modele.transcribe(str(audio))
    return str(getattr(out, "text", "") or "").strip()


_CACHE_PARAKEET: dict = {}


# ── The run ────────────────────────────────────────────────────────────


def _mesurer(nom: str, extraits: List[Extrait]) -> dict:
    wer, cer, trouves, cles, rates = [], [], 0, 0, []
    for extrait in extraits:
        try:
            hypothese = transcrire(nom, extrait.audio)
        except Exception as e:
            rates.append(f"{extrait.audio.name}: {type(e).__name__}: {e}")
            continue
        w = taux_erreur_mots(extrait.reference, hypothese)
        c = taux_erreur_caracteres(extrait.reference, hypothese)
        if w is not None:
            wer.append(w)
        if c is not None:
            cer.append(c)
        t, n = rappel_mots_cles(extrait.keywords, hypothese)
        trouves += t
        cles += n
    return {
        "mesures": len(wer),
        "wer": sum(wer) / len(wer) if wer else None,
        "cer": sum(cer) / len(cer) if cer else None,
        "mots_cles": (trouves, cles),
        "rates": rates,
    }


def main(argv=None) -> int:
    parseur = argparse.ArgumentParser(description=__doc__)
    parseur.add_argument("corpus", help="dossier écrit par JARVIS_SAVE_AUDIO")
    parseur.add_argument("--moteurs", default="",
                         help="liste séparée par des virgules (défaut : tous)")
    args = parseur.parse_args(argv)

    extraits, sans_reference = charger_corpus(args.corpus)

    print(f"\n📁 Corpus : {args.corpus}")
    print(f"   🎧 {len(extraits)} extrait(s) transcrit(s) à la main, mesurables")
    if sans_reference:
        print(f"   ✍️  {sans_reference} en attente de leur champ `reference` "
              "— non mesurés")
    if not extraits:
        print("\n❌ Rien à mesurer. Remplis `reference` dans les fiches JSON.")
        return 1

    total_cles = sum(len(e.keywords) for e in extraits)
    if not total_cles:
        print("   ⚠️  Aucun mot-clé déclaré : le rappel sur les noms propres "
              "ne sera pas mesuré, et c'est la question posée.")

    demandes = [m.strip() for m in args.moteurs.split(",") if m.strip()]
    etats = [e for e in moteurs_disponibles()
             if not demandes or e.nom in demandes]

    print("\n🔧 Moteurs")
    for etat in etats:
        if etat.disponible:
            print(f"   ✅ {etat.nom}  ({etat.detail})")
        else:
            print(f"   ⛔ {etat.nom} — absent. Pour l'installer : "
                  f"{etat.installation}")

    utilisables = [e for e in etats if e.disponible]
    if not utilisables:
        print("\n❌ Aucun moteur disponible.")
        return 1

    resultats = {}
    for etat in utilisables:
        print(f"\n⏳ {etat.nom}…", flush=True)
        resultats[etat.nom] = _mesurer(etat.nom, extraits)

    print("\n📊 Résultats")
    print(f"   {'moteur':<26} {'WER':>7} {'CER':>7} {'noms propres':>14} "
          f"{'mesurés':>8}")
    for nom, r in resultats.items():
        wer = f"{r['wer']*100:.1f}%" if r["wer"] is not None else "—"
        cer = f"{r['cer']*100:.1f}%" if r["cer"] is not None else "—"
        t, n = r["mots_cles"]
        cles = f"{t}/{n}" if n else "—"
        print(f"   {nom:<26} {wer:>7} {cer:>7} {cles:>14} "
              f"{r['mesures']:>8}")

    for nom, r in resultats.items():
        if r["rates"]:
            print(f"\n⚠️  {nom} a échoué sur {len(r['rates'])} extrait(s) :")
            for ligne in r["rates"][:5]:
                print(f"      · {ligne}")

    if len(extraits) < 20:
        print(f"\n⚠️  {len(extraits)} extraits : trop peu pour départager deux "
              "moteurs proches. Vise une vingtaine avant de conclure.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

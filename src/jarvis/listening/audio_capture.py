"""Keeping a copy of an utterance, when he has asked for one.

The ASR bench measures which recogniser survives his proper nouns at his
distance on his microphone, and that question cannot be answered on a
read script or on somebody else's corpus. So the listener can write each
utterance to disk alongside what Whisper made of it.

This is a microphone writing his voice to a file, which is the most
sensitive thing in this codebase. The switch is therefore an environment
variable and nothing else: no config key, no default, no window with a
checkbox that a past session could have left on. It is announced on
startup whenever it is live, because a recording nobody is told about is
the failure mode that matters here.

Utterances that transcribed to nothing are kept. Silence and garbage are
exactly where an autoregressive decoder and a transducer part company,
and a corpus of successes only would measure the easy half.
"""

from __future__ import annotations

import json
import os
import uuid
import wave
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from ..debug import debug_log

ENV_VAR = "JARVIS_SAVE_AUDIO"


@dataclass
class UtteranceCapture:
    """Writes utterances to ``directory`` when one was asked for.

    ``directory`` is ``None`` when the environment variable is unset, and
    every method is then a no-op that touches no filesystem at all.
    """

    directory: Optional[Path] = None

    @classmethod
    def from_env(cls) -> "UtteranceCapture":
        raw = (os.environ.get(ENV_VAR) or "").strip()
        if not raw:
            return cls(None)
        return cls(Path(raw).expanduser())

    @property
    def enabled(self) -> bool:
        return self.directory is not None

    def announce(self) -> None:
        """Say it out loud, once, at startup."""
        if not self.enabled:
            return
        print(f"  🎙️  Utterance capture is ON — writing to {self.directory}",
              flush=True)
        print("     Every sentence heard is saved to disk until you unset "
              f"{ENV_VAR}.", flush=True)

    def save(self, audio, sample_rate: int, hypothesis: str, *,
             model: str = "", device: str = "") -> Optional[Path]:
        """Write one utterance and its sidecar. Returns the WAV's path.

        Never raises: this sits on the listener's loop, which drains a
        64-frame audio queue on a deadline, and a capture that threw would
        cost him the sentence it was meant to record.
        """
        if not self.enabled:
            return None
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            stem = (datetime.now().strftime("%Y%m%d-%H%M%S-%f")
                    + "-" + uuid.uuid4().hex[:6])
            chemin = self.directory / f"{stem}.wav"

            samples = np.asarray(audio, dtype=np.float32).flatten()
            pcm = np.clip(samples, -1.0, 1.0)
            pcm = (pcm * 32767.0).astype(np.int16)
            with wave.open(str(chemin), "wb") as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(int(sample_rate))
                f.writeframes(pcm.tobytes())

            rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
            (self.directory / f"{stem}.json").write_text(json.dumps({
                "audio": chemin.name,
                "hypothesis": hypothesis or "",
                # His to fill in. The bench scores nothing without it, and
                # guessing it from the hypothesis would score the
                # recogniser against itself.
                "reference": "",
                # Words that must survive. The failure being measured is
                # proper nouns, and overall WER hides them in the average.
                "keywords": [],
                "model": model,
                "device": device,
                "sample_rate": int(sample_rate),
                "duration_sec": round(samples.size / float(sample_rate or 1), 3),
                "rms": round(rms, 6),
                "recorded_at": datetime.now().isoformat(timespec="seconds"),
            }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            return chemin
        except Exception as e:
            debug_log(f"utterance capture failed: {e}", "voice")
            return None

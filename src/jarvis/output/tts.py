from __future__ import annotations
import platform
import subprocess
import threading
import queue
import shutil
import signal
import tempfile
import os
import re
import sys
import time
import unicodedata
import warnings
from pathlib import Path
from typing import Any, Callable, Mapping, Optional
from urllib.parse import urlparse

from ..debug import debug_log
from ..latency import get_latency_journal
from ..state import JarvisState, get_jarvis_state
from ..utils.audio_lock import portaudio_lock


def _sound_started() -> None:
    """The audio stream is running: the user can hear the reply from here on.

    The first call of a turn is the moment the latency journal measures to
    (``jarvis/latency.spec.md``). ``SPEAKING`` is published earlier, when a
    chunk begins to be synthesised, so marking there would understate the
    wait by the synthesis time of the first chunk. Called outside the audio
    locks so a journal write can never hold playback up.
    """
    get_latency_journal().mark_first_sound()


# ============================================================================
# Piper TTS Model Configuration
# ============================================================================
PIPER_VOICE_BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0"

# Spoken to when the configured language names no voice. Not a preferred
# language, just the one that has to be picked when nothing else applies.
PIPER_FALLBACK_VOICE = "en_GB-alan-medium"

def _normalise_language(value: str) -> str:
    """Lower-case and strip accents, so spelling variants meet as one key."""
    decompose = unicodedata.normalize("NFKD", value.strip().lower())
    return "".join(c for c in decompose if not unicodedata.combining(c))


# Which voice speaks which language. Piper names its voices by locale and
# speaker, so nothing in the string "français" yields ``fr_FR-siwis-medium``:
# the pairing has to be stated. Each language lists the spellings a user
# might write in ``response_language``: its own name, its English name and
# its ISO 639-1 code. Supporting one more language is one line here, and a
# language nobody has listed still speaks, with the fallback voice, rather
# than falling silent.
_PIPER_VOICES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("fr", "français", "french"), "fr_FR-siwis-medium"),
    (("en", "english"), PIPER_FALLBACK_VOICE),
    (("es", "español", "spanish"), "es_ES-davefx-medium"),
    (("de", "deutsch", "german"), "de_DE-thorsten-medium"),
    (("it", "italiano", "italian"), "it_IT-riccardo-x_low"),
    (("nl", "nederlands", "dutch"), "nl_NL-mls-medium"),
    (("pt", "português", "portuguese"), "pt_BR-faber-medium"),
    (("pl", "polski", "polish"), "pl_PL-darkman-medium"),
    (("ru", "русский", "russian"), "ru_RU-dmitri-medium"),
    (("tr", "türkçe", "turkish"), "tr_TR-dfki-medium"),
    (("zh", "中文", "chinese", "mandarin"), "zh_CN-huayan-medium"),
)


def _get_piper_models_dir() -> Path:
    """Where Piper voice models are kept.

    Resolving it creates nothing: only the download writes there, and it
    creates the directory itself.
    """
    return Path.home() / ".local" / "share" / "jarvis" / "models" / "piper"


# One language, one key. The first spelling of each row above is its ISO 639-1
# code, which is what Whisper reports, so every spelling a user might write
# reduces to the code the detector produces. Keyed on the normalised form, so a
# name may be written in the table the way its speakers write it: "türkçe" and
# "turkce" meet on the same key, and so do "Français" and "francais".
_LANGUAGE_CODES = {
    _normalise_language(nom): spellings[0] for spellings, _ in _PIPER_VOICES for nom in spellings
}

_VOICE_BY_CODE = {spellings[0]: voix for spellings, voix in _PIPER_VOICES}


def _language_code(value: str) -> str:
    """The key a language is compared under.

    A listed language reduces to its ISO code. A regional variant
    (``fr-FR``, ``pt_BR``) reduces to its base language. A language nobody
    listed is its own normalised spelling, so a map may still name it.
    """
    key = _normalise_language(value)
    if key in _LANGUAGE_CODES:
        return _LANGUAGE_CODES[key]
    base = re.split(r"[-_]", key, maxsplit=1)[0]
    return _LANGUAGE_CODES.get(base, base)


def _piper_voice_for_language(language: Optional[str]) -> str:
    """The voice that speaks ``language``, or the fallback when unlisted.

    The language is reduced through ``_language_code``, the same reduction a
    ``tts_piper_voices`` key goes through, so ``fr-FR`` is French to both.
    """
    if not language or not language.strip():
        return PIPER_FALLBACK_VOICE
    return _VOICE_BY_CODE.get(_language_code(language), PIPER_FALLBACK_VOICE)


def _voice_model_path(value: str) -> str:
    """The model file a voice setting names.

    A path is taken as written. A bare name, which is how Piper calls its
    voices, is a file in the models directory, so the name alone is enough
    for the engine to fetch it on first use.
    """
    value = value.strip()
    if value.lower().endswith(".onnx") or "/" in value or "\\" in value or value.startswith("~"):
        return value
    return str(_get_piper_models_dir() / f"{value}.onnx")


def _reply_language_code(
    response_language: Optional[str], detected_language: Optional[str]
) -> str:
    """The language a reply is expected to be written in, as a key.

    The configured language wins because the persona prompt asks the model
    to write in it whatever was spoken to her. When it is empty the reply is
    expected in the language she heard. Empty when neither is known.
    """
    language = response_language if (response_language or "").strip() else detected_language
    return _language_code(language) if (language or "").strip() else ""


def select_tts_voice(
    voices: Optional[Mapping[str, str]],
    fallback: str,
    response_language: Optional[str] = None,
    detected_language: Optional[str] = None,
) -> str:
    """The voice model that speaks a reply.

    ``response_language`` is the language the assistant is asked to write
    in, so when it is set it is taken as the language of the reply whatever
    was spoken to her. When it is empty she is expected to answer in the
    language she heard, and ``detected_language`` is the one to follow.

    ``voices`` maps a language to a voice. A language it does not name, or
    no known language at all, speaks with ``fallback``.
    """
    code = _reply_language_code(response_language, detected_language)
    if not voices or not code:
        return fallback
    for key, voice in voices.items():
        if _language_code(key) == code:
            return _voice_model_path(voice)
    return fallback


def _get_default_piper_model_path(language: Optional[str] = None) -> str:
    """Path to the voice that speaks ``language``.

    The name carries the voice: the engine downloads whatever the basename
    asks for, so choosing the language here is what makes a French install
    fetch a French voice.
    """
    return str(_get_piper_models_dir() / f"{_piper_voice_for_language(language)}.onnx")


def _download_piper_voice(voice_name: str, progress_callback: Optional[Callable[[str], None]] = None) -> Optional[str]:
    """
    Download a Piper voice model from HuggingFace.

    Args:
        voice_name: Voice name like "en_US-lessac-medium"
        progress_callback: Optional callback for progress messages

    Returns:
        Path to the downloaded model, or None if download failed
    """
    import requests

    def log(msg: str):
        if progress_callback:
            progress_callback(msg)
        debug_log(msg, "tts")

    # Parse voice name to construct URL
    # Format: {lang}_{region}-{name}-{quality}
    # Example: en_US-lessac-medium -> en/en_US/lessac/medium/en_US-lessac-medium.onnx
    parts = voice_name.split("-")
    if len(parts) < 3:
        log(f"Invalid voice name format: {voice_name}")
        return None

    lang_region = parts[0]  # e.g., "en_US"
    name = parts[1]         # e.g., "lessac"
    quality = parts[2]      # e.g., "medium"

    lang = lang_region.split("_")[0]  # e.g., "en"

    # Construct URLs
    base_path = f"{lang}/{lang_region}/{name}/{quality}/{voice_name}"
    onnx_url = f"{PIPER_VOICE_BASE_URL}/{base_path}.onnx"
    json_url = f"{PIPER_VOICE_BASE_URL}/{base_path}.onnx.json"

    # Target paths
    models_dir = _get_piper_models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = models_dir / f"{voice_name}.onnx"
    json_path = models_dir / f"{voice_name}.onnx.json"

    # Download with progress
    try:
        for url, target_path, desc in [
            (onnx_url, onnx_path, "model"),
            (json_url, json_path, "config"),
        ]:
            if target_path.exists():
                log(f"  {desc} already exists: {target_path.name}")
                continue

            log(f"  Downloading {desc}...")

            # Stream download with retry on rate limiting (HTTP 429)
            max_retries = 4
            response = None
            for attempt in range(max_retries + 1):
                response = requests.get(url, stream=True, timeout=60)
                try:
                    response.raise_for_status()
                    break  # Success
                except requests.exceptions.HTTPError as http_err:
                    response.close()
                    status = getattr(http_err.response, "status_code", None)
                    if status == 429 and attempt < max_retries:
                        wait = 2 ** (attempt + 1)
                        log(f"  ⏳ Rate limited by HuggingFace, retrying in {wait}s ({attempt + 1}/{max_retries})...")
                        time.sleep(wait)
                        continue
                    raise  # Non-429 or retries exhausted

            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0

            # Write to temp file first, then rename (atomic)
            temp_path = target_path.with_suffix(".tmp")
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0 and progress_callback:
                        pct = (downloaded / total_size) * 100
                        if downloaded % (1024 * 1024) < 8192:  # Log every ~1MB
                            log(f"  Downloading {desc}... {pct:.0f}%")

            # Rename temp to final
            temp_path.rename(target_path)
            log(f"  Downloaded {desc}: {target_path.name}")

        return str(onnx_path)

    except requests.RequestException as e:
        log(f"  Download failed: {e}")
        # Clean up partial downloads
        for p in [onnx_path, json_path]:
            tmp = p.with_suffix(".tmp")
            if tmp.exists():
                tmp.unlink()
        return None
    except Exception as e:
        log(f"  Download error: {e}")
        return None


# Default speaking rates for TTS estimation
DEFAULT_WPM = 200  # Default rate used in config (words per minute)
AUDIO_BUFFER_DELAY_SEC = 0.5  # Extra delay for audio buffer latency


def _estimate_tts_duration(text: str, wpm: int) -> float:
    """
    Estimate how long TTS audio will take to play.

    Args:
        text: The text being spoken
        wpm: Words per minute rate

    Returns:
        Estimated duration in seconds
    """
    # Count words (simple split on whitespace)
    words = len(text.split())

    # Calculate duration based on WPM
    if wpm <= 0:
        wpm = DEFAULT_WPM

    duration_sec = (words / wpm) * 60.0

    # Add buffer for audio latency
    return duration_sec + AUDIO_BUFFER_DELAY_SEC


def _extract_domain_description(url: str) -> tuple[str, bool]:
    """
    Extract a readable domain description from a URL.

    Returns:
        Tuple of (domain_description, is_homepage)
        - domain_description: e.g., "google.com"
        - is_homepage: True if URL points to homepage (no meaningful path)
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path.split('/')[0]

        # Remove common prefixes
        if domain.startswith('www.'):
            domain = domain[4:]

        # Check if it's a homepage (no path or just /)
        path = parsed.path.rstrip('/')
        is_homepage = not path or path == ''

        return domain, is_homepage
    except Exception:
        return url, True


_NUMBERED_MARKER_RE = re.compile(r"^\s*(\d+)[.)]\s+")


def _strip_markdown_for_speech(text: str) -> str:
    """Strip markdown formatting so TTS doesn't read syntax characters aloud.

    Small models often produce markdown (``**bold**``, bullet lists, headings)
    even when told to be conversational. Piper and similar engines read the
    syntax characters literally ("asterisk asterisk bold asterisk asterisk").
    This function removes the markup while preserving the words inside it.

    Handled:
    - Fenced code blocks ``` ```lang\\ncode\\n``` ``` → inner text only
    - Inline code ``` `x` ``` → ``x``
    - Bold ``**x**`` / ``__x__`` → ``x``
    - Italic ``*x*`` / ``_x_`` → ``x``
    - Strikethrough ``~~x~~`` → ``x``
    - Word-internal underscores (e.g. ``my_function``) are preserved so
      identifiers aren't mangled into concatenated words.
    - HTML tags ``<b>x</b>`` → ``x``
    - Leading heading markers ``# ``, ``## `` … at line start → removed
    - Setext heading underlines (``===`` / ``---`` beneath a title line) → removed
    - Leading blockquote markers ``> `` at line start → removed
    - Leading bullet markers ``- ``, ``* ``, ``+ `` at line start → removed
    - Leading numbered-list markers ``1. ``, ``2) ``: stripped only when the
      line is part of a real list — detected as ≥2 adjacent lines whose
      numbers are each ≤ 99. Prevents eating prose like "2024. The year...".
    """
    if not text:
        return text

    # Fenced code blocks: keep inner content, drop fences and language tag.
    text = re.sub(r"```[a-zA-Z0-9_-]*\n?([\s\S]*?)```", r"\1", text)

    # Inline code: keep inner content.
    text = re.sub(r"`([^`]+)`", r"\1", text)

    # Bold / strikethrough (before italic so the double-char form matches first).
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"~~([^~]+)~~", r"\1", text)

    # Italic with asterisk: single * not flanked by another *.
    text = re.sub(r"(?<!\*)\*([^*\s][^*]*?)\*(?!\*)", r"\1", text)
    # Italic with underscore: require word boundaries so we don't eat
    # underscores inside identifiers like "some_variable_name".
    text = re.sub(r"(?<!\w)_([^_\n]+?)_(?!\w)", r"\1", text)

    # HTML tags: drop tags, keep inner text. Safe here because TTS input is
    # assistant prose, not code discussing literal inequalities like "x<3".
    text = re.sub(r"<[^>]+>", "", text)

    # True list detection: a numbered line is a list item only if it's part
    # of a contiguous group of ≥2 such lines whose numbers are each ≤ 99.
    # This preserves prose like "2024. The year..." and "2023.\n2024." pairs
    # that are clearly years, not list markers.
    lines = text.split("\n")
    numbers = [
        int(m.group(1)) if (m := _NUMBERED_MARKER_RE.match(line)) else None
        for line in lines
    ]
    strip_numbered = [False] * len(lines)
    run_start: Optional[int] = None
    for i in range(len(lines) + 1):
        in_run = i < len(lines) and numbers[i] is not None and numbers[i] <= 99
        if in_run and run_start is None:
            run_start = i
        elif not in_run and run_start is not None:
            if i - run_start >= 2:
                for k in range(run_start, i):
                    strip_numbered[k] = True
            run_start = None

    cleaned: list[str] = []
    for i, line in enumerate(lines):
        # Setext heading underline: a line of only = or - (≥3 chars) directly
        # beneath a non-empty title line. Drop the underline; keep the title.
        if (
            i > 0
            and lines[i - 1].strip()
            and re.fullmatch(r"\s*(=+|-+)\s*", line)
            and len(line.strip()) >= 3
        ):
            continue
        stripped = re.sub(r"^\s*#{1,6}\s+", "", line)        # headings
        stripped = re.sub(r"^\s*>\s?", "", stripped)         # blockquotes
        stripped = re.sub(r"^\s*[-*+]\s+", "", stripped)     # bullets
        if strip_numbered[i]:
            stripped = _NUMBERED_MARKER_RE.sub("", stripped)
        cleaned.append(stripped)
    return "\n".join(cleaned)


def _preprocess_for_speech(text: str) -> str:
    """
    Preprocess text for TTS by converting links to readable descriptions and
    stripping markdown formatting.

    Handles:
    - Markdown links: [text](url) → "Link to domain.com with the text 'text'" or
      "Link to a page under domain.com with the text 'text'"
    - Raw URLs: https://domain.com → "domain.com homepage" or
      https://domain.com/path → "a page under domain.com"
    - Markdown formatting (bold, italic, code, headings, lists) → stripped so
      TTS engines don't read syntax characters (``**``, ``#``, ``-``) aloud.
    """
    # Pattern for markdown links: [text](url)
    markdown_link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'

    def replace_markdown_link(match: re.Match) -> str:
        link_text = match.group(1)
        url = match.group(2)
        domain, is_homepage = _extract_domain_description(url)

        if is_homepage:
            return f"Link to {domain} homepage with the text '{link_text}'"
        else:
            return f"Link to a page under {domain} with the text '{link_text}'"

    # Replace markdown links first
    result = re.sub(markdown_link_pattern, replace_markdown_link, text)

    # Pattern for raw URLs (not already processed as markdown)
    # Matches http://, https://, and www. prefixed URLs
    raw_url_pattern = r'(?<!\()(https?://[^\s<>\[\]()]+|www\.[^\s<>\[\]()]+)(?!\))'

    def replace_raw_url(match: re.Match) -> str:
        url = match.group(1)
        # Ensure URL has protocol for parsing
        if url.startswith('www.'):
            url = 'https://' + url
        domain, is_homepage = _extract_domain_description(url)

        if is_homepage:
            return f"{domain} homepage"
        else:
            return f"a page under {domain}"

    # Replace raw URLs
    result = re.sub(raw_url_pattern, replace_raw_url, result)

    # Strip any remaining markdown so TTS doesn't read syntax aloud.
    result = _strip_markdown_for_speech(result)

    return result


class ChatterboxTTS:
    """Experimental TTS implementation using Resemble AI's Chatterbox model."""

    def __init__(self, enabled: bool = True, voice: Optional[str] = None, rate: Optional[int] = None,
                 device: str = "cuda", audio_prompt_path: Optional[str] = None,
                 exaggeration: float = 0.5, cfg_weight: float = 0.5) -> None:
        self.enabled = enabled
        self.voice = voice  # Not used in Chatterbox, kept for interface compatibility
        self.rate = rate    # Not directly supported in Chatterbox, kept for interface compatibility
        self.device = device
        self.audio_prompt_path = audio_prompt_path
        self.exaggeration = exaggeration
        self.cfg_weight = cfg_weight

        # Threading and queue setup (same as TextToSpeech)
        self._q: queue.Queue[str] = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._is_speaking = threading.Event()
        self._last_spoken_text: str = ""
        self._completion_callback: Optional[Callable[[], None]] = None
        self._duration_callback: Optional[Callable[[float], None]] = None
        # Interruptions so far. Each queued item carries the count it was
        # queued under; an item whose count is no longer current was queued
        # before an interrupt and is not spoken (`tts.spec.md`, "Interrupting").
        self._epoch = 0

        # Chatterbox model (eagerly loaded during initialization)
        self._model = None
        self._model_error = None
        # Lazy initialization flags
        self._initialized = False
        self._init_lock = threading.Lock()

    def _initialize_with_logging(self) -> None:
        """Initialize Chatterbox with proper logging."""
        import sys

        print("🔧 [TTS] Initializing Chatterbox neural voice synthesis...", file=sys.stderr)

        try:
            print("📦 [TTS] Loading Chatterbox dependencies...", file=sys.stderr)

            # Import dependencies
            import torch
            import torchaudio as ta
            from chatterbox.tts import ChatterboxTTS as ChatterboxModel

            # Check device availability
            if self.device == "cuda" and not torch.cuda.is_available():
                print("⚠️  [TTS] CUDA requested but not available, falling back to CPU", file=sys.stderr)
                actual_device = "cpu"
            else:
                actual_device = self.device

            print(f"🚀 [TTS] Loading Chatterbox model on {actual_device.upper()}...", file=sys.stderr)

            # Load model with proper device specification
            self._model = ChatterboxModel.from_pretrained(device=actual_device)

            print("✅ [TTS] Chatterbox neural voice synthesis ready!", file=sys.stderr)

        except ImportError as e:
            self._model_error = f"Chatterbox dependencies not available: {e}"
            print(f"❌ [TTS] Missing dependencies: {self._model_error}", file=sys.stderr)
            warnings.warn(f"ChatterboxTTS initialization failed: {self._model_error}")
        except Exception as e:
            self._model_error = f"Failed to load Chatterbox model: {e}"
            print(f"❌ [TTS] Model loading failed: {self._model_error}", file=sys.stderr)
            warnings.warn(f"ChatterboxTTS initialization failed: {self._model_error}")

    def _ensure_initialized(self) -> None:
        """Initialize heavy dependencies only once, when actually needed."""
        if self._initialized or not self.enabled:
            return
        with self._init_lock:
            if self._initialized:
                return
            self._initialize_with_logging()
            self._initialized = True

    def _ensure_model(self) -> bool:
        """Check if Chatterbox model is loaded. Returns True if successful."""
        # Ensure lazy initialization happens before checking model
        self._ensure_initialized()
        if self._model is not None:
            return True
        if self._model_error is not None:
            return False
        return False

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        # Initialize on first actual start
        self._ensure_initialized()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        # Ensure any active speech is interrupted immediately
        try:
            self.interrupt()
        except Exception:
            pass
        self._stop.set()
        try:
            self._q.put_nowait(("", None, None, self._epoch))
        except Exception:
            pass
        self._thread.join(timeout=2.0)
        self._thread = None
        self._stop.clear()

    def speak(self, text: str, completion_callback: Optional[Callable[[], None]] = None,
              duration_callback: Optional[Callable[[float], None]] = None,
              language: Optional[str] = None) -> None:
        # One voice, whatever the language: accepted so the listener can hand
        # every engine the same call, and ignored.
        if not self.enabled:
            return
        if not text.strip() and completion_callback is None:
            return
        # Empty text with a callback is the end-of-reply marker. A streamed
        # reply whose last sentence ends on punctuation leaves no tail, so
        # the callback has nothing to ride on; it gets an item of its own,
        # spoken after everything queued before it.
        # Lazy start the worker thread and lazy init on first speak
        if self._thread is None:
            self.start()
        # The callbacks travel with their text. Set on the engine, they
        # would belong to whatever was said last rather than to this
        # utterance, and a reply spoken sentence by sentence would fire
        # the last chunk's callback after the first chunk finished —
        # opening the hot window mid-reply. Only the worker writes them,
        # as it dequeues the item it is about to speak. The count of
        # interruptions travels with them: an item queued before an
        # interrupt is dropped when the worker reaches it, one queued after
        # is spoken.
        _callbacks = (completion_callback, duration_callback)
        # Preprocess text for speech (convert links to readable descriptions)
        processed_text = _preprocess_for_speech(text)
        try:
            self._q.put_nowait((processed_text, *_callbacks, self._epoch))
        except Exception:
            pass

    def interrupt(self) -> None:
        """Stop current speech immediately and drop what is queued."""
        self._epoch += 1

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                texte, _fin, _duree, epoch = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            if epoch != self._epoch:
                debug_log("Chatterbox TTS dropped an item queued before an interrupt", "tts")
                continue
            if not texte:
                if _fin is not None:
                    try:
                        _fin()
                    except Exception as e:
                        debug_log(f"TTS end-of-reply callback error: {e}", "tts")
                continue
            self._completion_callback = _fin
            self._duration_callback = _duree
            text = texte
            try:
                self._speak_once(text, epoch)
            except Exception:
                continue

    def _speak_once(self, text: str, epoch: Optional[int] = None) -> None:
        if epoch is None:
            epoch = self._epoch

        def cancelled() -> bool:
            return self._epoch != epoch

        self._is_speaking.set()
        self._last_spoken_text = text
        interrupted = False
        # Positive rather than negative: `interrupted` only ever became
        # True inside the playback loop, so every other way out of this
        # method — the engine that could not initialise, the synthesis
        # that produced no audio, the exception caught below, an
        # interruption arriving before playback started — left it False
        # and fired the completion callback anyway. That callback settles
        # a reminder as spoken, and `reminders.spec.md` promises the row
        # stays owed until the speech has finished.
        spoken = False
        
        # SPEAKING goes out only for an item that is still live: a stop word
        # that landed since the worker took it up has been answered with IDLE
        # (`state.spec.md`), and a SPEAKING after it would stay on the orb
        # until the next wake word.
        if not cancelled():
            self._notify_speaking_state(True)

        try:
            # Check if model is available
            if not self._ensure_model():
                # Fall back to system TTS if Chatterbox fails
                warnings.warn("Chatterbox TTS not available, skipping speech synthesis")
                return

            # Generate audio using Chatterbox
            import tempfile
            import pygame
            import os

            if cancelled():
                debug_log("Chatterbox TTS interrupted before synthesis", "tts")
                return

            # Generate speech
            wav = self._model.generate(
                text,
                audio_prompt_path=self.audio_prompt_path,
                exaggeration=self.exaggeration,
                cfg_weight=self.cfg_weight
            )

            # The model call cannot be cut short; what it made for an item
            # interrupted meanwhile is thrown away rather than played.
            if cancelled():
                debug_log("Chatterbox TTS interrupted during synthesis", "tts")
                return

            # Calculate exact duration from audio samples
            exact_duration = wav.shape[-1] / self._model.sr
            debug_log(f"Chatterbox TTS synthesis complete: {exact_duration:.2f}s", "tts")

            # Notify listener of exact duration for precise echo detection
            if self._duration_callback is not None:
                try:
                    self._duration_callback(exact_duration)
                except Exception as e:
                    debug_log(f"Chatterbox TTS duration callback error: {e}", "tts")

            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_path = tmp_file.name

            try:
                # Save audio
                import torchaudio as ta
                ta.save(tmp_path, wav, self._model.sr)

                # Play audio using pygame (cross-platform)
                pygame.mixer.init(frequency=self._model.sr, size=-16, channels=1, buffer=1024)
                pygame.mixer.music.load(tmp_path)
                pygame.mixer.music.play()
                _sound_started()

                # Wait for playback to complete or interruption
                while pygame.mixer.music.get_busy():
                    if cancelled():
                        pygame.mixer.music.stop()
                        interrupted = True
                        break
                    pygame.time.wait(100)  # Check every 100ms

                spoken = not (interrupted or cancelled())

            finally:
                # Cleanup
                pygame.mixer.quit()
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

        except Exception as e:
            warnings.warn(f"Chatterbox TTS error: {e}")
        finally:
            self._is_speaking.clear()
            
            # Speech ended (nothing is published: the daemon manages what follows)
            self._notify_speaking_state(False)
            
            # Call completion callback if set and not interrupted
            if self._completion_callback is not None and spoken:
                try:
                    self._completion_callback()
                except Exception:
                    pass
                self._completion_callback = None
    
    def _notify_speaking_state(self, is_speaking: bool) -> None:
        """Publish SPEAKING when speech starts.

        When speaking ends the state is left alone: the daemon manages the
        transition out of SPEAKING.
        """
        if is_speaking:
            debug_log("setting state to SPEAKING (chatterbox)", "tts")
            get_jarvis_state().set_state(JarvisState.SPEAKING)

    # Loopback guard helpers (same interface as TextToSpeech)
    def is_speaking(self) -> bool:
        return self._is_speaking.is_set()

    def get_last_spoken_text(self) -> str:
        return self._last_spoken_text


class PiperTTS:
    """TTS implementation using Piper (local neural TTS with exact duration).

    Piper generates actual audio samples, enabling precise duration calculation
    instead of WPM-based estimation. Uses sounddevice for streaming playback
    with responsive interruption support.
    """

    def __init__(
        self,
        enabled: bool = True,
        voice: Optional[str] = None,
        rate: Optional[int] = None,
        model_path: Optional[str] = None,
        speaker: Optional[int] = None,
        length_scale: float = 1.0,
        noise_scale: float = 0.667,
        noise_w: float = 0.8,
        sentence_silence: float = 0.2,
        response_language: Optional[str] = None,
        voices: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.enabled = enabled
        self.voice = voice  # Not used in Piper, kept for interface compatibility
        self.rate = rate    # Not directly supported, use length_scale instead
        self.model_path = model_path
        self.response_language = response_language
        # Language to voice. A language it does not name speaks with the
        # voice `_resolved_model_path` picks.
        self.voices: dict[str, str] = dict(voices or {})
        self.speaker = speaker
        self.length_scale = length_scale
        self.noise_scale = noise_scale
        self.noise_w = noise_w
        self.sentence_silence = sentence_silence

        # Threading and queue setup (same pattern as other TTS engines)
        self._q: queue.Queue[str] = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._is_speaking = threading.Event()
        self._last_spoken_text: str = ""
        self._completion_callback: Optional[Callable[[], None]] = None
        self._duration_callback: Optional[Callable[[float], None]] = None
        # Interruptions so far. Each queued item carries the count it was
        # queued under; an item whose count is no longer current was queued
        # before an interrupt and is not spoken (`tts.spec.md`, "Interrupting").
        self._epoch = 0

        # Piper voice (lazy loaded)
        self._voice = None
        self._sample_rate: int = 22050  # Piper default, updated on model load
        self._initialized = False
        self._init_lock = threading.Lock()
        self._init_error: Optional[str] = None

        # Voices the map asked for, loaded on first use and kept, keyed by
        # model path. A path that could not be loaded is remembered too, so a
        # voice that cannot be fetched is not retried on every sentence.
        self._mapped_voices: dict[str, tuple[Any, int]] = {}
        self._unloadable_voices: set[str] = set()
        self._mapped_lock = threading.Lock()
        # The last choice named in the debug log, so a reply of many
        # sentences names its voice once and a change of voice names it again.
        self._last_voice_logged: Optional[tuple[str, str]] = None

        # Audio stream for interruption
        self._audio_stream = None
        self._audio_lock = threading.Lock()

    def _cached_fallback_voice(self) -> Optional[tuple[str, str]]:
        """The fallback voice and its config, if both already sit on disk.

        Offered only when the language chose the voice. A pinned
        ``model_path`` is a choice, and quietly speaking with another voice
        would answer a question the user did not ask.
        """
        if self.model_path:
            return None
        secours = os.path.expanduser(_get_default_piper_model_path(None))
        config = secours + ".json"
        if os.path.exists(secours) and os.path.exists(config):
            return secours, config
        return None

    def _resolved_model_path(self) -> str:
        """The voice this engine speaks with.

        A path the user pinned wins outright: naming a voice is a choice, and
        the configured language never overrides it. Otherwise the language
        decides, which is the whole point of having asked for one.
        """
        return self.model_path or _get_default_piper_model_path(self.response_language)

    def _load_voice_file(
        self, model_path: str, may_fall_back: bool
    ) -> tuple[Optional[Any], int, Optional[str]]:
        """Load the voice at ``model_path``, fetching it first when absent.

        Returns ``(voice, sample_rate, error)``. ``voice`` is None when the
        voice could not be loaded, and ``error`` says why.

        When the fetch fails and ``may_fall_back`` allows it, a fallback
        voice already on disk speaks in its place.
        """
        try:
            # Expand user path (e.g., ~/models/voice.onnx)
            model_path = os.path.expanduser(model_path)
            config_path = model_path + ".json"

            # Auto-download if model doesn't exist
            if not os.path.exists(model_path) or not os.path.exists(config_path):
                # Extract voice name from path for download
                voice_name = os.path.basename(model_path).replace(".onnx", "")

                print(f"🔊 Downloading Piper voice: {voice_name}", file=sys.stderr, flush=True)
                print("   This is a one-time download (~60MB)...", file=sys.stderr, flush=True)

                def progress(msg):
                    print(msg, file=sys.stderr, flush=True)

                downloaded_path = _download_piper_voice(voice_name, progress_callback=progress)

                if not downloaded_path:
                    secours = self._cached_fallback_voice() if may_fall_back else None
                    if secours is None:
                        error = f"Failed to download voice: {voice_name}"
                        debug_log(f"Piper TTS init failed: {error}", "tts")
                        return None, 0, error
                    model_path, config_path = secours
                    repli = os.path.basename(model_path).replace(".onnx", "")
                    debug_log(
                        f"{voice_name} unavailable, speaking with {repli}", "tts"
                    )
                    print(
                        f"⚠️  Voice {voice_name} unavailable, speaking with {repli}",
                        file=sys.stderr, flush=True,
                    )
                else:
                    model_path = downloaded_path
                    config_path = model_path + ".json"
                    print("✓ Voice downloaded successfully!", file=sys.stderr, flush=True)

            # Final check that files exist
            if not os.path.exists(model_path):
                error = f"Model file not found: {model_path}"
                debug_log(f"Piper TTS init failed: {error}", "tts")
                return None, 0, error

            if not os.path.exists(config_path):
                error = f"Model config not found: {config_path}"
                debug_log(f"Piper TTS init failed: {error}", "tts")
                return None, 0, error

            debug_log(f"Piper TTS loading model: {model_path}", "tts")

            # Import piper and load model
            from piper.voice import PiperVoice

            voice = PiperVoice.load(model_path, config_path)
            sample_rate = voice.config.sample_rate
            debug_log(f"Piper TTS voice loaded: sample_rate={sample_rate}", "tts")
            return voice, sample_rate, None

        except ImportError as e:
            error = f"piper-tts not installed: {e}"
        except Exception as e:
            error = f"Failed to load Piper model: {e}"
        debug_log(f"Piper TTS init failed: {error}", "tts")
        return None, 0, error

    def _ensure_initialized(self) -> bool:
        """Initialise the default Piper voice. Returns True if successful.

        The default voice is the one every language the map does not name
        speaks with: the pinned model, else the voice for the configured
        language. If it is not on disk it is downloaded automatically.
        """
        if self._initialized:
            return self._voice is not None
        if not self.enabled:
            return False

        with self._init_lock:
            if self._initialized:
                return self._voice is not None

            model_path = self._resolved_model_path()
            if not self.model_path:
                debug_log(
                    f"No model configured, voice for {self.response_language!r}: {model_path}",
                    "tts",
                )

            voice, sample_rate, error = self._load_voice_file(
                model_path, may_fall_back=not self.model_path
            )
            if voice is not None:
                self._voice = voice
                self._sample_rate = sample_rate
            else:
                self._init_error = error

            self._initialized = True
            return self._voice is not None

    def _load_mapped_voice(self, model_path: str) -> Optional[tuple[Any, int]]:
        """A voice the map names, loaded on first use and kept.

        None when it cannot be loaded, and then it is not tried again: a
        fetch that fails retries a rate-limited server for half a minute,
        which would hold every later sentence in that language hostage.
        """
        with self._mapped_lock:
            if model_path in self._mapped_voices:
                return self._mapped_voices[model_path]
            if model_path in self._unloadable_voices:
                return None
            voice, sample_rate, error = self._load_voice_file(model_path, may_fall_back=False)
            if voice is None:
                self._unloadable_voices.add(model_path)
                debug_log(f"Piper TTS voice unavailable, using the default: {error}", "tts")
                return None
            self._mapped_voices[model_path] = (voice, sample_rate)
            return voice, sample_rate

    def _name_the_choice(self, language: str, voice_path: str, source: str) -> None:
        """Say in the debug log which voice speaks, when that is news.

        A streamed reply reaches the engine a sentence at a time, so naming
        the voice for each one would bury the log. It is named when the
        language or the voice differs from the last item's.
        """
        choice = (language, voice_path)
        if choice == self._last_voice_logged:
            return
        self._last_voice_logged = choice
        debug_log(
            f"voice for {language or 'an unknown language'}: "
            f"{os.path.basename(voice_path)} ({source})",
            "tts",
        )

    def _voice_for(self, detected_language: Optional[str]) -> Optional[tuple[Any, int]]:
        """The loaded voice that speaks a reply, with its sample rate.

        None when there is no voice at all, which is when the default voice
        itself could not be loaded.
        """
        default_path = self._resolved_model_path()
        chosen = select_tts_voice(
            self.voices, default_path,
            response_language=self.response_language,
            detected_language=detected_language,
        )
        is_default = os.path.expanduser(chosen) == os.path.expanduser(default_path)
        self._name_the_choice(
            _reply_language_code(self.response_language, detected_language),
            chosen,
            "default" if is_default else "mapped",
        )
        if not is_default:
            mapped = self._load_mapped_voice(chosen)
            if mapped is not None:
                return mapped
        if not self._ensure_initialized():
            return None
        return self._voice, self._sample_rate

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        # Initialize model eagerly at startup (downloads if needed)
        # This provides better UX - download happens during startup, not first speech
        self._ensure_initialized()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        try:
            self.interrupt()
        except Exception:
            pass
        self._stop.set()
        try:
            self._q.put_nowait(("", None, None, None, self._epoch))
        except Exception:
            pass
        self._thread.join(timeout=2.0)
        self._thread = None
        self._stop.clear()

    def speak(self, text: str, completion_callback: Optional[Callable[[], None]] = None,
              duration_callback: Optional[Callable[[float], None]] = None,
              language: Optional[str] = None) -> None:
        """Queue ``text`` to be spoken.

        ``language`` is the language the user was heard in. It travels with
        the item, and the voice is chosen from it when the item is spoken:
        sentences of two replies in two languages can wait in the queue
        together. The configured ``response_language`` outranks it.
        """
        if not self.enabled:
            return
        if not text.strip() and completion_callback is None:
            return
        # Empty text with a callback is the end-of-reply marker. A streamed
        # reply whose last sentence ends on punctuation leaves no tail, so
        # the callback has nothing to ride on; it gets an item of its own,
        # spoken after everything queued before it.
        # Lazy start the worker thread
        if self._thread is None:
            self.start()
        # The callbacks travel with their text. Set on the engine, they
        # would belong to whatever was said last rather than to this
        # utterance, and a reply spoken sentence by sentence would fire
        # the last chunk's callback after the first chunk finished —
        # opening the hot window mid-reply. Only the worker writes them,
        # as it dequeues the item it is about to speak. The count of
        # interruptions travels with them: an item queued before an
        # interrupt is dropped when the worker reaches it, one queued after
        # is spoken.
        _callbacks = (completion_callback, duration_callback)
        # Preprocess text for speech
        processed_text = _preprocess_for_speech(text)
        try:
            self._q.put_nowait((processed_text, *_callbacks, language, self._epoch))
        except Exception:
            pass

    def interrupt(self) -> None:
        """Stop current speech immediately and drop what is queued."""
        self._epoch += 1
        with self._audio_lock:
            if self._audio_stream is not None:
                try:
                    with portaudio_lock:
                        self._audio_stream.abort()
                except Exception:
                    pass

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                texte, _fin, _duree, _langue, epoch = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            if epoch != self._epoch:
                debug_log("Piper TTS dropped an item queued before an interrupt", "tts")
                continue
            if not texte:
                if _fin is not None:
                    try:
                        _fin()
                    except Exception as e:
                        debug_log(f"TTS end-of-reply callback error: {e}", "tts")
                continue
            self._completion_callback = _fin
            self._duration_callback = _duree
            text = texte
            try:
                self._speak_once(text, _langue, epoch)
            except Exception as e:
                debug_log(f"Piper TTS error in _speak_once: {e}", "tts")
                continue

    def _speak_once(self, text: str, language: Optional[str] = None,
                    epoch: Optional[int] = None) -> None:
        if epoch is None:
            epoch = self._epoch

        def cancelled() -> bool:
            return self._epoch != epoch

        self._is_speaking.set()
        self._last_spoken_text = text
        interrupted = False
        # Positive rather than negative: `interrupted` only ever became
        # True inside the playback loop, so every other way out of this
        # method — the engine that could not initialise, the synthesis
        # that produced no audio, the exception caught below, an
        # interruption arriving before playback started — left it False
        # and fired the completion callback anyway. That callback settles
        # a reminder as spoken, and `reminders.spec.md` promises the row
        # stays owed until the speech has finished.
        spoken = False

        # SPEAKING goes out only for an item that is still live: a stop word
        # that landed since the worker took it up has been answered with IDLE
        # (`state.spec.md`), and a SPEAKING after it would stay on the orb
        # until the next wake word.
        if not cancelled():
            self._notify_speaking_state(True)

        try:
            # The voice is chosen per item, from the language of the reply,
            # and loaded on first use.
            chosen = self._voice_for(language)
            if chosen is None:
                if self._init_error:
                    print(f"  ⚠️ Piper TTS: {self._init_error}", flush=True)
                return
            piper_voice, sample_rate = chosen

            import sounddevice as sd
            import numpy as np

            start_time = time.time()

            debug_log(f"Piper TTS starting synthesis: {len(text.split())} words", "tts")

            # Check for interruption before synthesis
            if cancelled():
                debug_log("Piper TTS interrupted before synthesis", "tts")
                return

            # Synthesize audio - synthesize() returns an iterable of AudioChunks
            from piper.config import SynthesisConfig
            syn_config = SynthesisConfig(
                speaker_id=self.speaker,
                length_scale=self.length_scale,
                noise_scale=self.noise_scale,
                noise_w_scale=self.noise_w,
            )
            audio_chunks = []
            for chunk in piper_voice.synthesize(text, syn_config):
                if cancelled():
                    debug_log("Piper TTS interrupted during synthesis", "tts")
                    return
                audio_chunks.append(chunk.audio_int16_array)

            # Check for interruption after synthesis
            if cancelled():
                debug_log("Piper TTS interrupted after synthesis", "tts")
                return

            # Concatenate all audio chunks
            if not audio_chunks:
                debug_log("Piper TTS: no audio chunks generated", "tts")
                return

            full_audio = np.concatenate(audio_chunks)

            if len(full_audio) == 0:
                debug_log("Piper TTS: no audio generated", "tts")
                return

            # Calculate exact duration from actual samples
            exact_duration = len(full_audio) / sample_rate
            debug_log(f"Piper TTS synthesis complete: {exact_duration:.2f}s, {len(full_audio)} samples", "tts")

            # Notify listener of exact duration for precise echo detection
            if self._duration_callback is not None:
                try:
                    self._duration_callback(exact_duration)
                except Exception as e:
                    debug_log(f"Piper TTS duration callback error: {e}", "tts")

            # Play audio with streaming for interruption support
            play_position = [0]
            blocksize = 1024  # Small blocks for responsive interruption

            def audio_callback(outdata, frames, time_info, status):
                if cancelled():
                    raise sd.CallbackAbort()

                start = play_position[0]
                end = start + frames
                chunk = full_audio[start:end]

                if len(chunk) < frames:
                    # Pad with zeros if we're at the end
                    outdata[:len(chunk), 0] = chunk
                    outdata[len(chunk):, 0] = 0
                    raise sd.CallbackStop()
                else:
                    outdata[:, 0] = chunk

                play_position[0] = end

            try:
                with self._audio_lock:
                    with portaudio_lock:
                        self._audio_stream = sd.OutputStream(
                            samplerate=sample_rate,
                            channels=1,
                            dtype='int16',
                            blocksize=blocksize,
                            callback=audio_callback,
                        )
                        self._audio_stream.start()

                _sound_started()
                # Wait for playback to complete
                while self._audio_stream is not None and self._audio_stream.active:
                    if cancelled():
                        interrupted = True
                        with self._audio_lock:
                            if self._audio_stream is not None:
                                try:
                                    with portaudio_lock:
                                        self._audio_stream.abort()
                                except Exception:
                                    pass
                        break
                    time.sleep(0.05)
            finally:
                with self._audio_lock:
                    if self._audio_stream is not None:
                        try:
                            with portaudio_lock:
                                self._audio_stream.close()
                        except Exception:
                            pass
                        self._audio_stream = None

            interrupted = interrupted or cancelled()
            spoken = not interrupted
            actual_duration = time.time() - start_time
            debug_log(f"Piper TTS complete: actual={actual_duration:.2f}s (audio={exact_duration:.2f}s)", "tts")

        except Exception as e:
            debug_log(f"Piper TTS error: {e}", "tts")
            print(f"  ⚠️ Piper TTS error: {e}", flush=True)
        finally:
            self._is_speaking.clear()
            self._notify_speaking_state(False)

            # Call completion callback if set and not interrupted
            if self._completion_callback is not None and spoken:
                try:
                    self._completion_callback()
                except Exception as e:
                    print(f"  ⚠️ Piper TTS completion callback error: {e}", flush=True)
                self._completion_callback = None

    def _notify_speaking_state(self, is_speaking: bool) -> None:
        """Publish SPEAKING when speech starts."""
        if is_speaking:
            debug_log("setting state to SPEAKING (piper)", "tts")
            get_jarvis_state().set_state(JarvisState.SPEAKING)

    # Loopback guard helpers (same interface as TextToSpeech)
    def is_speaking(self) -> bool:
        return self._is_speaking.is_set()

    def get_last_spoken_text(self) -> str:
        return self._last_spoken_text


def _find_espeak_library() -> Optional[str]:
    """Locate the espeak-ng shared library for phonemizer.

    Kokoro's non-English G2P (misaki) phonemises through espeak-ng. The
    phonemizer backend reads ``PHONEMIZER_ESPEAK_LIBRARY`` for the library
    path; when unset it probes these well-known install locations. Returns
    None if none exist, letting phonemizer fall back to its own discovery.
    """
    candidates = [
        "/opt/homebrew/lib/libespeak-ng.dylib",          # macOS Apple Silicon (brew)
        "/opt/homebrew/lib/libespeak-ng.1.dylib",
        "/usr/local/lib/libespeak-ng.dylib",             # macOS Intel (brew)
        "/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1",   # Debian / Ubuntu
        "/usr/lib/libespeak-ng.so.1",
        "/usr/local/lib/libespeak-ng.so",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


class KokoroTTS:
    """TTS implementation using Kokoro-82M (StyleTTS2-based neural TTS).

    More natural than Piper while still running faster than real-time on
    CPU, which makes it the local upgrade for machines too weak for the
    heavier neural engines. Multilingual via misaki G2P; non-English
    languages (e.g. French) phonemise through espeak-ng, so libespeak-ng
    must be installed. Model weights download once from Hugging Face
    (``hexgrad/Kokoro-82M``).

    Mirrors :class:`PiperTTS`'s threading / queue / streaming-playback
    structure (each engine is self-contained in this module) so the
    listener's loopback guard, interruption, and duration/completion
    callbacks behave identically regardless of engine.
    """

    # A digit rather than a word: a digit has a reading in every language
    # the pipeline can be set to, where a word belongs to one of them.
    _WARMUP_TEXT = "1."

    def __init__(
        self,
        enabled: bool = True,
        voice: str = "ff_siwis",
        lang_code: str = "f",
        speed: float = 1.0,
    ) -> None:
        self.enabled = enabled
        self.voice = voice or "ff_siwis"
        self.lang_code = lang_code or "f"
        self.speed = float(speed) if speed else 1.0

        # Threading and queue setup (same pattern as PiperTTS).
        self._q: queue.Queue[str] = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._is_speaking = threading.Event()
        self._last_spoken_text: str = ""
        self._completion_callback: Optional[Callable[[], None]] = None
        self._duration_callback: Optional[Callable[[float], None]] = None
        # Interruptions so far. Each queued item carries the count it was
        # queued under; an item whose count is no longer current was queued
        # before an interrupt and is not spoken (`tts.spec.md`, "Interrupting").
        self._epoch = 0

        # Kokoro pipeline (lazy loaded).
        self._pipe = None
        self._voice_blend = None  # weighted-blend voice, built at init
        self._sample_rate: int = 24000  # Kokoro's fixed output rate
        self._initialized = False
        self._init_lock = threading.Lock()
        self._init_error: Optional[str] = None

        # Audio stream for interruption.
        self._audio_stream = None
        self._audio_lock = threading.Lock()

    def _ensure_initialized(self) -> bool:
        """Load the Kokoro pipeline. Returns True if successful."""
        if self._initialized:
            return self._pipe is not None
        if not self.enabled:
            return False

        with self._init_lock:
            if self._initialized:
                return self._pipe is not None

            try:
                # Point phonemizer at espeak-ng for non-English G2P. Only
                # set when the user hasn't already pinned a library path.
                if "PHONEMIZER_ESPEAK_LIBRARY" not in os.environ:
                    lib = _find_espeak_library()
                    if lib:
                        os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = lib
                        debug_log(f"Kokoro TTS espeak-ng library: {lib}", "tts")

                from kokoro import KPipeline

                debug_log(f"Kokoro TTS loading pipeline (lang={self.lang_code})", "tts")
                self._pipe = KPipeline(lang_code=self.lang_code)
                self._voice_blend = self._build_voice_blend()
                # Absorb the model's first-call cost (CUDA kernel loading
                # alone can take tens of seconds) here, in the warm-up
                # thread, rather than on the first reply. The audio is
                # discarded; a failure leaves the pipeline loaded and
                # surfaces at speak time, as without the warm-up.
                try:
                    for _gs, _ps, _audio in self._pipe(
                        self._WARMUP_TEXT, voice=self._resolved_voice(), speed=self.speed
                    ):
                        pass
                except Exception as e:
                    debug_log(f"Kokoro warm-up synthesis failed: {e}", "tts")
                debug_log("Kokoro TTS initialized", "tts")

            except ImportError as e:
                self._init_error = f"kokoro not installed: {e}"
                debug_log(f"Kokoro TTS init failed: {self._init_error}", "tts")
            except Exception as e:
                self._init_error = f"Failed to load Kokoro pipeline: {e}"
                debug_log(f"Kokoro TTS init failed: {self._init_error}", "tts")

            self._initialized = True
            return self._pipe is not None

    def _parse_voice_spec(self) -> Optional[list]:
        """Parse a weighted voice blend ("a:0.8,b:0.2").

        Returns a list of (name, weight) pairs when any part carries a
        weight, else None — a plain name or an equal blend ("a,b") is
        handed to Kokoro as-is, which blends equally on its own. A part
        without a weight counts as 1. An unparseable weight yields None,
        so a malformed spec fails the way an unknown voice name would.
        """
        parts = [p.strip() for p in self.voice.split(",") if p.strip()]
        if not any(":" in p for p in parts):
            return None
        parsed = []
        for part in parts:
            if ":" not in part:
                parsed.append((part, 1.0))
                continue
            name, _, weight = part.rpartition(":")
            try:
                parsed.append((name.strip(), float(weight)))
            except ValueError:
                return None
        if any(not name or weight <= 0 for name, weight in parsed):
            return None
        return parsed

    def _build_voice_blend(self):
        """Build the weighted mean of the voice packs named in the spec.

        Returns None when the spec is not weighted or the packs cannot be
        loaded; the raw string then goes to Kokoro unchanged.
        """
        spec = self._parse_voice_spec()
        if spec is None:
            return None
        try:
            blend, total = None, 0.0
            for name, weight in spec:
                pack = self._pipe.load_single_voice(name)
                blend = pack * weight if blend is None else blend + pack * weight
                total += weight
            blend = blend / total
            debug_log(f"Kokoro TTS weighted voice blend: {self.voice}", "tts")
            return blend
        except Exception as e:
            debug_log(f"Kokoro voice blend failed, using raw voice '{self.voice}': {e}", "tts")
            return None

    def _resolved_voice(self):
        """The voice argument for the pipeline: the blend when the spec is
        weighted, the configured string otherwise."""
        return self._voice_blend if self._voice_blend is not None else self.voice

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return

        if sys.platform == "win32":
            # Pre-import Kokoro and resolve phonemizer on the calling thread
            # so underlying C-runtime DLLs (PyTorch, SciPy, OpenBLAS) are
            # loaded sequentially before background workers start. Concurrent
            # DLL loading of OpenBLAS and cuBLAS across threads deadlocks the
            # Windows loader; other platforms keep the pure background warm-up.
            try:
                if "PHONEMIZER_ESPEAK_LIBRARY" not in os.environ:
                    lib = _find_espeak_library()
                    if lib:
                        os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = lib
                        debug_log(f"Kokoro TTS espeak-ng library: {lib}", "tts")
                from kokoro import KPipeline  # noqa: F401
            except Exception as e:
                debug_log(f"Kokoro TTS pre-import: {e}", "tts")

        # The first-use HF download + pipeline load is slow (tens of
        # seconds), so warm it in the background instead of blocking daemon
        # startup. The worker's _ensure_initialized re-checks under the same
        # lock, so a speak arriving before the load finishes simply waits.
        threading.Thread(target=self._ensure_initialized, daemon=True,
                         name="kokoro-init").start()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        try:
            self.interrupt()
        except Exception:
            pass
        self._stop.set()
        try:
            self._q.put_nowait(("", None, None, self._epoch))
        except Exception:
            pass
        self._thread.join(timeout=2.0)
        self._thread = None
        self._stop.clear()

    def speak(self, text: str, completion_callback: Optional[Callable[[], None]] = None,
              duration_callback: Optional[Callable[[float], None]] = None,
              language: Optional[str] = None) -> None:
        # One voice, whatever the language: accepted so the listener can hand
        # every engine the same call, and ignored.
        if not self.enabled:
            return
        if not text.strip() and completion_callback is None:
            return
        # Empty text with a callback is the end-of-reply marker. A streamed
        # reply whose last sentence ends on punctuation leaves no tail, so
        # the callback has nothing to ride on; it gets an item of its own,
        # spoken after everything queued before it.
        if self._thread is None:
            self.start()
        # The callbacks travel with their text. Set on the engine, they
        # would belong to whatever was said last rather than to this
        # utterance, and a reply spoken sentence by sentence would fire
        # the last chunk's callback after the first chunk finished —
        # opening the hot window mid-reply. Only the worker writes them,
        # as it dequeues the item it is about to speak. The count of
        # interruptions travels with them: an item queued before an
        # interrupt is dropped when the worker reaches it, one queued after
        # is spoken.
        _callbacks = (completion_callback, duration_callback)
        processed_text = _preprocess_for_speech(text)
        try:
            self._q.put_nowait((processed_text, *_callbacks, self._epoch))
        except Exception:
            pass

    def interrupt(self) -> None:
        """Stop current speech immediately and drop what is queued."""
        self._epoch += 1
        with self._audio_lock:
            if self._audio_stream is not None:
                try:
                    with portaudio_lock:
                        self._audio_stream.abort()
                except Exception:
                    pass

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                texte, _fin, _duree, epoch = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            if epoch != self._epoch:
                debug_log("Kokoro TTS dropped an item queued before an interrupt", "tts")
                continue
            if not texte:
                if _fin is not None:
                    try:
                        _fin()
                    except Exception as e:
                        debug_log(f"TTS end-of-reply callback error: {e}", "tts")
                continue
            self._completion_callback = _fin
            self._duration_callback = _duree
            text = texte
            try:
                self._speak_once(text, epoch)
            except Exception as e:
                debug_log(f"Kokoro TTS error in _speak_once: {e}", "tts")
                continue

    def _speak_once(self, text: str, epoch: Optional[int] = None) -> None:
        if epoch is None:
            epoch = self._epoch

        def cancelled() -> bool:
            return self._epoch != epoch

        self._is_speaking.set()
        self._last_spoken_text = text
        interrupted = False
        # Positive rather than negative: `interrupted` only ever became
        # True inside the playback loop, so every other way out of this
        # method — the engine that could not initialise, the synthesis
        # that produced no audio, the exception caught below, an
        # interruption arriving before playback started — left it False
        # and fired the completion callback anyway. That callback settles
        # a reminder as spoken, and `reminders.spec.md` promises the row
        # stays owed until the speech has finished.
        spoken = False

        # SPEAKING goes out only for an item that is still live: a stop word
        # that landed since the worker took it up has been answered with IDLE
        # (`state.spec.md`), and a SPEAKING after it would stay on the orb
        # until the next wake word.
        if not cancelled():
            self._notify_speaking_state(True)

        try:
            if not self._ensure_initialized():
                if self._init_error:
                    print(f"  ⚠️ Kokoro TTS: {self._init_error}", flush=True)
                return

            import sounddevice as sd
            import numpy as np

            start_time = time.time()
            debug_log(f"Kokoro TTS starting synthesis: {len(text.split())} words", "tts")

            if cancelled():
                return

            # The pipeline yields one (graphemes, phonemes, audio) tuple per
            # sentence; collect the waveform, converting torch tensors to a
            # float32 numpy array for sounddevice.
            audio_chunks = []
            for _gs, _ps, audio in self._pipe(text, voice=self._resolved_voice(), speed=self.speed):
                if cancelled():
                    return
                arr = audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio)
                audio_chunks.append(arr.astype(np.float32))

            if cancelled():
                return
            if not audio_chunks:
                debug_log("Kokoro TTS: no audio generated", "tts")
                return

            full_audio = np.concatenate(audio_chunks)
            if len(full_audio) == 0:
                debug_log("Kokoro TTS: empty audio", "tts")
                return

            exact_duration = len(full_audio) / self._sample_rate
            debug_log(f"Kokoro TTS synthesis complete: {exact_duration:.2f}s, {len(full_audio)} samples", "tts")

            if self._duration_callback is not None:
                try:
                    self._duration_callback(exact_duration)
                except Exception as e:
                    debug_log(f"Kokoro TTS duration callback error: {e}", "tts")

            # Stream playback in small blocks so interruption is responsive.
            play_position = [0]
            blocksize = 1024

            def audio_callback(outdata, frames, time_info, status):
                if cancelled():
                    raise sd.CallbackAbort()
                start = play_position[0]
                end = start + frames
                chunk = full_audio[start:end]
                if len(chunk) < frames:
                    outdata[:len(chunk), 0] = chunk
                    outdata[len(chunk):, 0] = 0
                    raise sd.CallbackStop()
                else:
                    outdata[:, 0] = chunk
                play_position[0] = end

            try:
                with self._audio_lock:
                    with portaudio_lock:
                        self._audio_stream = sd.OutputStream(
                            samplerate=self._sample_rate,
                            channels=1,
                            dtype='float32',
                            blocksize=blocksize,
                            callback=audio_callback,
                        )
                        self._audio_stream.start()

                _sound_started()
                while self._audio_stream is not None and self._audio_stream.active:
                    if cancelled():
                        interrupted = True
                        with self._audio_lock:
                            if self._audio_stream is not None:
                                try:
                                    with portaudio_lock:
                                        self._audio_stream.abort()
                                except Exception:
                                    pass
                        break
                    time.sleep(0.05)
            finally:
                with self._audio_lock:
                    if self._audio_stream is not None:
                        try:
                            with portaudio_lock:
                                self._audio_stream.close()
                        except Exception:
                            pass
                        self._audio_stream = None

            interrupted = interrupted or cancelled()
            spoken = not interrupted
            actual_duration = time.time() - start_time
            debug_log(f"Kokoro TTS complete: actual={actual_duration:.2f}s (audio={exact_duration:.2f}s)", "tts")

        except Exception as e:
            debug_log(f"Kokoro TTS error: {e}", "tts")
            print(f"  ⚠️ Kokoro TTS error: {e}", flush=True)
        finally:
            self._is_speaking.clear()
            self._notify_speaking_state(False)
            if self._completion_callback is not None and spoken:
                try:
                    self._completion_callback()
                except Exception as e:
                    print(f"  ⚠️ Kokoro TTS completion callback error: {e}", flush=True)
                self._completion_callback = None

    def _notify_speaking_state(self, is_speaking: bool) -> None:
        """Publish SPEAKING when speech starts."""
        if is_speaking:
            debug_log("setting state to SPEAKING (kokoro)", "tts")
            get_jarvis_state().set_state(JarvisState.SPEAKING)

    # Loopback guard helpers (same interface as the other engines).
    def is_speaking(self) -> bool:
        return self._is_speaking.is_set()

    def get_last_spoken_text(self) -> str:
        return self._last_spoken_text


# The voice Kyutai speaks with when none is named: a French female voice, a
# name in the ``kyutai/tts-voices`` repository.
KYUTAI_DEFAULT_VOICE = "cml-tts/fr/577_394_000070-0001_enhanced.wav"


class _SynthesisCancelled(Exception):
    """Raised inside the model's frame callback to abandon a synthesis."""


class _ThreadLocalCapture:
    """``torch.cuda`` as ``moshi`` sees it, capturing graphs per thread.

    torch captures a CUDA graph in "global" mode unless told otherwise: a CUDA
    call made by any other thread during the capture (an allocation, a
    synchronisation) invalidates it, and the process aborts. Whisper keeps the
    GPU busy in the listener's thread while she speaks, and ``moshi`` builds
    its graphs afresh for every sentence. In "thread_local" mode only the
    capturing thread is checked. Everything but ``graph`` is torch's own.
    """

    def __init__(self, cuda) -> None:
        self._cuda = cuda

    def __getattr__(self, name):
        return getattr(self._cuda, name)

    def graph(self, *args, **kwargs):
        kwargs.setdefault("capture_error_mode", "thread_local")
        return self._cuda.graph(*args, **kwargs)


class _Utterance:
    """An item on its way from the synthesis to the speakers (Kyutai).

    The worker adds the audio as the model produces it; the player's stream
    takes it from the other end. The thread that synthesises writes
    ``samples``, ``failed`` and the queue; the player reads them.
    """

    def __init__(self, text: str, completion: Optional[Callable[[], None]],
                 epoch: int, prebuffer_samples: int) -> None:
        self.text = text
        self.completion = completion
        self.epoch = epoch
        self.chunks: queue.Queue = queue.Queue()  # float32 arrays, then None
        self.samples = 0                # produced so far
        self.ready = threading.Event()  # a first stretch is buffered, or synthesis is over
        self.failed = False             # synthesis raised after it began
        self.consumed = False           # the stream has taken the last sample
        self._prebuffer = prebuffer_samples

    def add(self, chunk) -> None:
        self.chunks.put(chunk)
        self.samples += len(chunk)
        if self.samples >= self._prebuffer:
            self.ready.set()

    def close(self) -> None:
        """Synthesis is over, however it ended."""
        self.chunks.put(None)
        self.ready.set()


class _EndOfReply:
    """The end-of-reply marker on its way to the player, where it waits for
    the audio queued before it."""

    def __init__(self, completion: Callable[[], None], epoch: int) -> None:
        self.completion = completion
        self.epoch = epoch


class KyutaiTTS:
    """TTS implementation using Kyutai TTS 1.6B (English and French).

    The most natural of the local engines. It runs in-process through the
    ``moshi`` package on an NVIDIA GPU, faster than real time once the
    model's one-off costs (kernel loading, allocator warm-up) have been
    paid, which the warm-up does at startup. Model and voice download once
    from Hugging Face (``kyutai/tts-1.6b-en_fr``, ``kyutai/tts-voices``).

    Mirrors :class:`KokoroTTS` (queue, worker, callbacks). It differs where
    the model does: it needs CUDA; a synthesis in flight is abandoned at the
    model's next frame rather than run to its end; and it plays an item as
    it is synthesised, through a second thread (the player), while the worker
    goes on to prepare the next one.
    """

    # A digit rather than a word: it has a reading in both languages.
    _WARMUP_TEXT = "1."

    # Audio to have in hand before a stream opens, so a stretch the model
    # produces slowly does not stutter the first words.
    _PREBUFFER_SECONDS = 0.25

    # What the model was released with; settings of the model, not of Yuba.
    _CODEBOOKS = 32
    _TEMPERATURE = 0.6
    _GUIDANCE = 2.0

    def __init__(self, enabled: bool = True, voice: str = KYUTAI_DEFAULT_VOICE) -> None:
        self.enabled = enabled
        self.voice = (voice or "").strip() or KYUTAI_DEFAULT_VOICE

        self._q: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._is_speaking = threading.Event()
        self._last_spoken_text: str = ""
        # Interruptions so far; an item is live while its count is current.
        self._epoch = 0

        # The player takes what the worker has synthesised, in order. The
        # engine is speaking while any item taken up has not been played,
        # interrupted or abandoned.
        self._play_q: queue.Queue = queue.Queue()
        self._player: Optional[threading.Thread] = None
        self._pending = 0
        self._pending_lock = threading.Lock()

        # Model (lazy loaded).
        self._model = None
        self._voice_path: Optional[str] = None
        self._sample_rate: int = 24000
        self._initialized = False
        self._init_lock = threading.Lock()
        self._init_error: Optional[str] = None

        # Audio stream for interruption.
        self._audio_stream = None
        self._audio_lock = threading.Lock()

    @staticmethod
    def _cuda_problem() -> Optional[str]:
        """Why this machine cannot run the model on a GPU, or None when it
        can. A PyTorch that cannot be loaded, one built without CUDA and a
        machine without a GPU are three different fixes, so each says which."""
        try:
            import torch
        except Exception as e:
            return f"PyTorch could not be loaded ({e})"
        try:
            if torch.cuda.is_available():
                return None
        except Exception as e:
            return f"CUDA could not be queried ({e})"
        if getattr(torch.version, "cuda", None) is None:
            return "this PyTorch was built without CUDA support, install a CUDA build"
        return "no NVIDIA GPU was found"

    @staticmethod
    def _triton_available() -> bool:
        import importlib.util

        try:
            return importlib.util.find_spec("triton") is not None
        except Exception:
            return False

    def _configure_environment(self) -> None:
        """Switch ``torch.compile`` off where Triton is missing.

        ``moshi`` reads ``NO_TORCH_COMPILE`` when it is first imported, so
        this has to run before that import. A value the user set is theirs.
        The CUDA graphs are never switched off: without them synthesis runs
        several times slower than real time.
        """
        if "NO_TORCH_COMPILE" in os.environ:
            return
        if not self._triton_available():
            os.environ["NO_TORCH_COMPILE"] = "1"
            debug_log("Kyutai TTS: Triton not found, torch.compile switched off", "tts")

    @staticmethod
    def _capture_graphs_per_thread() -> None:
        """Make ``moshi`` capture its CUDA graphs per thread: replace the
        ``cuda`` that ``moshi.utils.compile`` takes from torch with
        :class:`_ThreadLocalCapture`. A ``moshi.utils.compile`` that has no
        ``cuda`` to replace is left as it is: the engine still works, without
        the protection."""
        import importlib

        try:
            compile_module = importlib.import_module("moshi.utils.compile")
            if not isinstance(compile_module.cuda, _ThreadLocalCapture):
                compile_module.cuda = _ThreadLocalCapture(compile_module.cuda)
        except Exception as e:
            debug_log(f"Kyutai TTS: CUDA graphs not isolated per thread: {e}", "tts")

    # A name in the voice repository: path segments of plain characters, never
    # absolute, never going up, never a scheme or a drive.
    _VOICE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)*")

    def _local_voice_file(self) -> Optional[str]:
        """The embedding file ``self.voice`` names on this machine, or None
        when it names a voice in the repository.

        A value that is neither is refused before the network is touched: the
        repository would be asked for it, and a local path would travel inside
        that request with the user's name and folders in it.
        """
        if self.voice.lower().endswith(".safetensors"):
            path = Path(self.voice).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"Kyutai voice file not found: {path}")
            return str(path)
        if not self._VOICE_NAME.fullmatch(self.voice):
            raise ValueError(
                "tts_kyutai_voice must be a name in the kyutai/tts-voices repository "
                "(for example cml-tts/fr/577_394_000070-0001_enhanced.wav) or the "
                f"path of a .safetensors file, not {self.voice!r}"
            )
        return None

    def _load(self) -> None:
        local_voice = self._local_voice_file()

        problem = self._cuda_problem()
        if problem:
            self._init_error = f"Kyutai TTS needs an NVIDIA GPU with CUDA: {problem}"
            return

        self._configure_environment()
        try:
            from moshi.models.loaders import CheckpointInfo
            from moshi.models.tts import DEFAULT_DSM_TTS_REPO, TTSModel
        except ImportError as e:
            self._init_error = f"moshi or one of its dependencies could not be imported, see the README ({e})"
            return
        self._capture_graphs_per_thread()

        debug_log("Kyutai TTS loading model", "tts")
        checkpoint = CheckpointInfo.from_hf_repo(DEFAULT_DSM_TTS_REPO)
        model = TTSModel.from_checkpoint_info(
            checkpoint, n_q=self._CODEBOOKS, temp=self._TEMPERATURE, device="cuda"
        )
        voice_path = local_voice or str(model.get_voice_path(self.voice))
        self._model = model
        self._voice_path = voice_path
        self._sample_rate = int(model.mimi.sample_rate)
        debug_log(f"Kyutai TTS model loaded, voice {self.voice}", "tts")

        # The first synthesis pays the model's one-off costs (kernel and
        # module loading, allocator warm-up) and takes tens of seconds; pay
        # for them here rather than on the first reply. The audio is
        # discarded, and a failure leaves the model loaded: it surfaces at
        # speak time, as without the warm-up.
        try:
            self._synthesise(self._WARMUP_TEXT, lambda: False, lambda chunk: None)
            debug_log("Kyutai TTS warm-up complete", "tts")
        except Exception as e:
            debug_log(f"Kyutai TTS warm-up failed: {e}", "tts")

    def _ensure_initialized(self) -> bool:
        """Load the model. Returns True if successful; the outcome, good or
        bad, is kept for the life of the engine."""
        if self._initialized:
            return self._model is not None
        if not self.enabled:
            return False

        with self._init_lock:
            if self._initialized:
                return self._model is not None
            try:
                self._load()
            except Exception as e:
                self._init_error = f"Failed to load Kyutai TTS: {e}"
            if self._init_error:
                debug_log(f"Kyutai TTS init failed: {self._init_error}", "tts")
            self._initialized = True
            return self._model is not None

    def _synthesise(self, text: str, cancelled: Callable[[], bool],
                    on_audio: Callable[[Any], None]) -> Optional[int]:
        """Synthesise ``text``, handing each decoded frame to ``on_audio`` as
        float32 samples the moment the model produces it (the first audible
        one arrives a second or less in). Returns how many samples were
        produced, or None when ``cancelled`` turned true."""
        import numpy as np

        model = self._model
        entries = model.prepare_script([text], padding_between=1)
        conditions = model.make_condition_attributes([self._voice_path], cfg_coef=self._GUIDANCE)
        produced = [0]

        def on_frame(frame):
            if cancelled():
                raise _SynthesisCancelled()
            # Frames still carrying -1 are the model's lead-in, not audio.
            if (frame != -1).all():
                pcm = model.mimi.decode(frame[:, 1:, :]).cpu().numpy()
                chunk = np.clip(pcm[0, 0], -1, 1).astype(np.float32)
                produced[0] += len(chunk)
                on_audio(chunk)

        try:
            with model.mimi.streaming(1):
                model.generate([entries], [conditions], on_frame=on_frame)
        except _SynthesisCancelled:
            return None
        return produced[0]

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return

        if sys.platform == "win32":
            # Import the package on the calling thread so PyTorch's DLLs load
            # one thread at a time: concurrent loading deadlocks the Windows
            # loader. The compile switch comes first, the package reads it
            # at import.
            try:
                self._configure_environment()
                from moshi.models.tts import TTSModel  # noqa: F401
            except Exception as e:
                debug_log(f"Kyutai TTS pre-import: {e}", "tts")

        # The first-use download + load + warm-up take a while, so they run
        # in the background instead of blocking daemon startup. A speak
        # arriving before they finish waits on the same lock.
        threading.Thread(target=self._ensure_initialized, daemon=True,
                         name="kyutai-init").start()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        try:
            self.interrupt()
        except Exception:
            pass
        self._stop.set()
        try:
            self._q.put_nowait(("", None, None, self._epoch))
        except Exception:
            pass
        self._thread.join(timeout=2.0)
        self._thread = None
        player, self._player = self._player, None
        if player is not None:
            self._play_q.put(None)
            player.join(timeout=2.0)
        self._stop.clear()

    def speak(self, text: str, completion_callback: Optional[Callable[[], None]] = None,
              duration_callback: Optional[Callable[[float], None]] = None,
              language: Optional[str] = None) -> None:
        # One voice, whatever the language: accepted so the listener can hand
        # every engine the same call, and ignored.
        if not self.enabled:
            return
        if not text.strip() and completion_callback is None:
            return
        # Empty text with a callback is the end-of-reply marker: it speaks
        # nothing and fires its callback once everything queued before it
        # has been played.
        if self._thread is None:
            self.start()
        # The callbacks travel with their text, and so does the count of
        # interruptions: an item queued before an interrupt is dropped when
        # the worker reaches it, one queued after is spoken.
        processed_text = _preprocess_for_speech(text)
        try:
            self._q.put_nowait((processed_text, completion_callback, duration_callback, self._epoch))
        except Exception:
            pass

    def interrupt(self) -> None:
        """Stop current speech immediately and drop what is queued."""
        self._epoch += 1
        with self._audio_lock:
            if self._audio_stream is not None:
                try:
                    with portaudio_lock:
                        self._audio_stream.abort()
                except Exception:
                    pass

    def _begin(self) -> None:
        """An item has been taken up: the engine is speaking until it has been
        played, interrupted or abandoned."""
        with self._pending_lock:
            self._pending += 1
            self._is_speaking.set()

    def _end(self) -> None:
        """An item is over, however it ended."""
        with self._pending_lock:
            self._pending = max(0, self._pending - 1)
            if self._pending == 0:
                self._is_speaking.clear()

    def _ensure_player(self) -> None:
        if self._player is None or not self._player.is_alive():
            self._player = threading.Thread(target=self._play_run, daemon=True,
                                            name="kyutai-player")
            self._player.start()

    def _run(self) -> None:
        """The worker: takes the items up one at a time and synthesises them,
        without waiting for the player to finish the one before."""
        while not self._stop.is_set():
            try:
                texte, _fin, _duree, epoch = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            if epoch != self._epoch:
                debug_log("Kyutai TTS dropped an item queued before an interrupt", "tts")
                continue
            if not texte:
                # The end-of-reply marker waits behind the audio queued before it.
                if _fin is not None:
                    self._ensure_player()
                    self._play_q.put(_EndOfReply(_fin, epoch))
                continue
            try:
                self._prepare(texte, _fin, _duree, epoch)
            except Exception as e:
                debug_log(f"Kyutai TTS error in _prepare: {e}", "tts")

    def _prepare(self, text: str, completion: Optional[Callable[[], None]],
                 duration: Optional[Callable[[float], None]], epoch: int) -> None:
        """Take an item up and synthesise it, handing its audio to the player
        as it is produced. Once the player has the item it is the one that
        ends it."""
        def cancelled() -> bool:
            return self._epoch != epoch

        self._begin()
        handed_over = False
        try:
            # SPEAKING goes out only for an item that is still live: a stop word
            # that landed since the worker took it up has been answered with IDLE
            # (`state.spec.md`), and a SPEAKING after it would stay on the orb
            # until the next wake word.
            if not cancelled():
                self._notify_speaking_state(True)

            if not self._ensure_initialized():
                if self._init_error:
                    print(f"  ⚠️ Kyutai TTS: {self._init_error}", flush=True)
                return

            # The load can take minutes on first use, and a stop word may
            # have landed meanwhile: such an item costs the GPU nothing.
            if cancelled():
                debug_log("Kyutai TTS interrupted before synthesis", "tts")
                return

            utterance = _Utterance(text, completion, epoch,
                                   int(self._PREBUFFER_SECONDS * self._sample_rate))
            self._ensure_player()
            self._play_q.put(utterance)
            handed_over = True

            started = time.time()
            debug_log(f"Kyutai TTS starting synthesis: {len(text.split())} words", "tts")
            try:
                produced = self._synthesise(text, cancelled, utterance.add)
            except Exception as e:
                # Whatever was already produced plays to its end, then stops:
                # nothing is said as spoken.
                utterance.failed = True
                debug_log(f"Kyutai TTS error: {e}", "tts")
                print(f"  ⚠️ Kyutai TTS error: {e}", flush=True)
                return
            finally:
                utterance.close()

            if produced is None or cancelled():
                debug_log("Kyutai TTS synthesis abandoned", "tts")
                return
            if produced == 0:
                debug_log("Kyutai TTS: no audio generated", "tts")
                return

            exact_duration = produced / self._sample_rate
            debug_log(
                f"Kyutai TTS synthesis complete: {exact_duration:.2f}s of audio "
                f"in {time.time() - started:.2f}s",
                "tts",
            )
            if duration is not None:
                try:
                    duration(exact_duration)
                except Exception as e:
                    debug_log(f"Kyutai TTS duration callback error: {e}", "tts")
        except Exception as e:
            debug_log(f"Kyutai TTS error: {e}", "tts")
            print(f"  ⚠️ Kyutai TTS error: {e}", flush=True)
        finally:
            if not handed_over:
                self._end()

    def _play_run(self) -> None:
        """The player: plays what the worker has synthesised, one item at a
        time and in the order the worker took them up."""
        while True:
            thing = self._play_q.get()
            if thing is None:
                return
            if isinstance(thing, _EndOfReply):
                if thing.epoch != self._epoch:
                    debug_log("Kyutai TTS dropped the end-of-reply marker of an interrupted reply", "tts")
                    continue
                try:
                    thing.completion()
                except Exception as e:
                    debug_log(f"TTS end-of-reply callback error: {e}", "tts")
                continue
            try:
                self._play(thing)
            except Exception as e:
                debug_log(f"Kyutai TTS error in _play: {e}", "tts")

    def _play(self, utterance: "_Utterance") -> None:
        def cancelled() -> bool:
            return self._epoch != utterance.epoch

        interrupted = False
        # Spoken only once the audio has played to its end: every other way
        # out (the engine that could not start, the synthesis that produced
        # nothing or failed, the exception below, an interruption) leaves the
        # completion callback unfired, because that callback settles a
        # reminder as said (`reminders.spec.md`).
        spoken = False
        try:
            while not utterance.ready.wait(0.05):
                if cancelled():
                    return
            if cancelled():
                debug_log("Kyutai TTS dropped an item prepared but not yet playing", "tts")
                return
            if utterance.samples == 0:
                return

            import numpy as np
            import sounddevice as sd

            self._last_spoken_text = utterance.text
            started = time.time()
            blocksize = 1024
            leftover = [np.zeros(0, dtype=np.float32)]
            underruns = [0]

            def audio_callback(outdata, frames, time_info, status):
                if cancelled():
                    raise sd.CallbackAbort()
                filled = 0
                while filled < frames:
                    if len(leftover[0]) == 0:
                        if utterance.consumed:
                            break
                        try:
                            piece = utterance.chunks.get_nowait()
                        except queue.Empty:
                            underruns[0] += 1  # synthesis is behind: play silence
                            break
                        if piece is None:
                            utterance.consumed = True
                            break
                        leftover[0] = piece
                    take = min(frames - filled, len(leftover[0]))
                    outdata[filled:filled + take, 0] = leftover[0][:take]
                    leftover[0] = leftover[0][take:]
                    filled += take
                if filled < frames:
                    outdata[filled:, 0] = 0
                    if utterance.consumed:
                        raise sd.CallbackStop()

            try:
                with self._audio_lock:
                    with portaudio_lock:
                        self._audio_stream = sd.OutputStream(
                            samplerate=self._sample_rate,
                            channels=1,
                            dtype='float32',
                            blocksize=blocksize,
                            callback=audio_callback,
                        )
                        self._audio_stream.start()

                _sound_started()
                while self._audio_stream is not None and self._audio_stream.active:
                    if cancelled():
                        interrupted = True
                        with self._audio_lock:
                            if self._audio_stream is not None:
                                try:
                                    with portaudio_lock:
                                        self._audio_stream.abort()
                                except Exception:
                                    pass
                        break
                    time.sleep(0.05)
            finally:
                with self._audio_lock:
                    if self._audio_stream is not None:
                        try:
                            with portaudio_lock:
                                self._audio_stream.close()
                        except Exception:
                            pass
                        self._audio_stream = None

            spoken = utterance.consumed and not (
                interrupted or cancelled() or utterance.failed)
            debug_log(
                f"Kyutai TTS played: {time.time() - started:.2f}s, spoken={spoken}, "
                f"underruns={underruns[0]}",
                "tts",
            )

        except Exception as e:
            debug_log(f"Kyutai TTS error: {e}", "tts")
            print(f"  ⚠️ Kyutai TTS error: {e}", flush=True)
        finally:
            self._end()
            if spoken and utterance.completion is not None:
                try:
                    utterance.completion()
                except Exception as e:
                    print(f"  ⚠️ Kyutai TTS completion callback error: {e}", flush=True)

    def _notify_speaking_state(self, is_speaking: bool) -> None:
        """Publish SPEAKING when speech starts."""
        if is_speaking:
            debug_log("setting state to SPEAKING (kyutai)", "tts")
            get_jarvis_state().set_state(JarvisState.SPEAKING)

    # Loopback guard helpers (same interface as the other engines).
    def is_speaking(self) -> bool:
        return self._is_speaking.is_set()

    def get_last_spoken_text(self) -> str:
        return self._last_spoken_text


def create_tts_engine(
    engine: str = "piper",
    enabled: bool = True,
    voice: Optional[str] = None,
    rate: Optional[int] = None,
    # Chatterbox parameters
    device: str = "cuda",
    audio_prompt_path: Optional[str] = None,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    # Piper parameters
    piper_model_path: Optional[str] = None,
    piper_speaker: Optional[int] = None,
    piper_length_scale: float = 1.0,
    piper_noise_scale: float = 0.667,
    piper_noise_w: float = 0.8,
    piper_sentence_silence: float = 0.2,
    response_language: Optional[str] = None,
    # Language to Piper voice. With ``response_language`` empty the voice
    # follows the language the reply is spoken in; see ``select_tts_voice``.
    piper_voices: Optional[Mapping[str, str]] = None,
    # Kokoro parameters. ``response_language`` and ``piper_voices`` govern
    # the Piper voice only: Kokoro takes its voice from ``kokoro_voice`` and
    # its language from ``kokoro_lang_code``, and Chatterbox has no notion
    # of one, so both need their voice set by hand.
    kokoro_voice: str = "ff_siwis",
    kokoro_lang_code: str = "f",
    kokoro_speed: float = 1.0,
    # Kyutai parameter: a name in the voice repository, or a .safetensors path.
    kyutai_voice: str = KYUTAI_DEFAULT_VOICE,
):
    """Factory function to create the appropriate TTS engine.

    Supported engines:
    - "piper" (default): light neural TTS with auto-download, exact duration
    - "kokoro": more natural neural TTS (Kokoro-82M), real-time on CPU
    - "chatterbox": AI voice with emotion control (requires PyTorch)
    - "kyutai": the most natural voice (Kyutai TTS 1.6B), needs an NVIDIA GPU
    """
    if engine.lower() == "chatterbox":
        return ChatterboxTTS(
            enabled=enabled,
            voice=voice,
            rate=rate,
            device=device,
            audio_prompt_path=audio_prompt_path,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
        )
    elif engine.lower() == "kokoro":
        return KokoroTTS(
            enabled=enabled,
            voice=kokoro_voice,
            lang_code=kokoro_lang_code,
            speed=kokoro_speed,
        )
    elif engine.lower() == "kyutai":
        return KyutaiTTS(enabled=enabled, voice=kyutai_voice)
    else:
        # Default to Piper TTS
        return PiperTTS(
            enabled=enabled,
            voice=voice,
            rate=rate,
            model_path=piper_model_path,
            speaker=piper_speaker,
            length_scale=piper_length_scale,
            noise_scale=piper_noise_scale,
            noise_w=piper_noise_w,
            sentence_silence=piper_sentence_silence,
            response_language=response_language,
            voices=piper_voices,
        )

import os
import sys
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv


# ============================================================================
# SUPPORTED CHAT MODELS - Single Source of Truth
# ============================================================================
# This is the authoritative list of officially supported chat models.
# Other modules should import from here rather than defining their own lists.

SUPPORTED_CHAT_MODELS: Dict[str, Dict[str, str]] = {
    "gemma4:e2b": {
        "name": "Gemma 4 E2B (Default)",
        "description": "Fast, multimodal, effective 2B — a little dumb, occasionally fumbles tool calls; ~7.2GB download",
        "size": "~7.2GB",
        "vram": "8GB+",
    },
    "gemma4:e4b": {
        "name": "Gemma 4 E4B (Recommended)",
        "description": "Smarter tool use and reasoning, multimodal, effective 4B — ~9.6GB download",
        "size": "~9.6GB",
        "vram": "16GB+",
    },
    "gpt-oss:20b": {
        "name": "GPT-OSS 20B (High-end)",
        "description": "Best performance, ~12GB download",
        "size": "~12GB",
        "vram": "24GB+",
    },
}

# The default chat model (first in the supported list)
DEFAULT_CHAT_MODEL = "gemma4:e2b"


def get_supported_model_ids() -> set[str]:
    """Get set of supported model IDs for quick lookup."""
    return set(SUPPORTED_CHAT_MODELS.keys())


def _default_dictation_hotkey() -> str:
    """Return the platform-appropriate default dictation hotkey.

    Aligned with WisprFlow defaults:
    - Windows: Ctrl+Win (pynput maps Win to ``cmd``)
    - macOS: Fn is not detectable by pynput, so use Ctrl+Option (WisprFlow
      fallback when Fn is unavailable)
    - Linux: Ctrl+Alt (mirrors macOS fallback)
    """
    if sys.platform == "win32":
        return "ctrl+cmd"
    elif sys.platform == "darwin":
        return "ctrl+alt"
    else:
        return "ctrl+alt"


def _default_db_path() -> str:
    base = Path.home() / ".local" / "share" / "jarvis"
    base.mkdir(parents=True, exist_ok=True)
    return str(base / "jarvis.db")


@dataclass(frozen=True)
class UISettings:
    """Desktop-app UI choices.

    Currently a single knob: whether the reactive orb renders its
    ambient particle layer. Disabling particles cuts per-frame draw
    work (relevant on lower-end hardware where the orb's animation
    adds noticeable GPU load) and is also an aesthetic preference for
    a calmer orb. Set ``"ui": {"orb_particles_enabled": false}`` in
    config.json to turn them off.
    """

    orb_particles_enabled: bool


@dataclass(frozen=True)
class Settings:
    # Database & Storage
    db_path: str
    sqlite_vss_path: str | None

    # LLM & AI Models
    # Provider-aware fields (see src/jarvis/llm/llm.spec.md). The
    # `ollama_*` fields below are kept as aliases so any caller still
    # reading them keeps working when the provider is Ollama.
    llm_provider: str  # "ollama" | "openai_compatible"
    llm_base_url: str
    # Resolved API key for the OpenAI-compatible provider. When
    # ``llm_api_key_env`` names an environment variable that is set, the
    # parser populates this from that variable so the secret never has
    # to live in config.json. Otherwise it falls back to the literal
    # ``llm_api_key`` value from the config file.
    llm_api_key: str
    # Name of the environment variable to read the API key from (e.g.
    # "OPENROUTER_API_KEY"). Empty = use the literal ``llm_api_key``.
    llm_api_key_env: str
    llm_chat_model: str
    # Provider-specific extra request fields merged into every cloud chat
    # payload (OpenAI-compatible path only). Example for OpenRouter, to pin
    # the fastest upstream for a multi-provider model:
    # {"provider": {"sort": "throughput"}}. Empty = standard OpenAI shape.
    llm_extra_body: Dict[str, Any]
    # Scrub secret-shaped tokens (emails, API keys, JWTs, password/
    # token/secret pairs) from prompts before they leave the machine
    # for a remote OpenAI-compatible provider. Default True (privacy-
    # first). No effect when the provider is local Ollama.
    auto_redact_before_cloud: bool
    embedding_provider: str  # "" (= same as llm_provider) | "ollama" | "openai_compatible"
    embedding_base_url: str
    embedding_api_key: str
    embedding_model: str
    # Disk-format aliases. Older config files name these fields, so they
    # stay readable here; the loader promotes their values into the
    # provider-aware fields above so everything inside the codebase reads
    # ``llm_*`` / ``embedding_*`` only.
    ollama_base_url: str
    ollama_embed_model: str
    ollama_chat_model: str
    llm_chat_timeout_sec: float
    llm_tools_timeout_sec: float
    # Tight deadline for the cheap distil passes used by memory_digest and
    # tool_result_digest. Separate from `llm_tools_timeout_sec` because
    # those paths run a small classification-shaped LLM call, not a
    # long-running tool — a 5-minute ceiling there would stall replies.
    llm_digest_timeout_sec: float
    llm_embedding_timeout_sec: float
    llm_profile_select_timeout_sec: float

    # Profiles & Behavior
    active_profiles: list[str]
    use_stdin: bool
    voice_debug: bool

    # Screen Capture
    allowlist_bundles: list[str]

    # Text-to-Speech
    tts_enabled: bool
    tts_engine: str  # "piper" (default), "kokoro", or "chatterbox"
    tts_voice: str | None
    tts_rate: int | None  # Words per minute (WPM), 200=normal
    tts_chatterbox_device: str  # "cuda", "auto", or "cpu" for Chatterbox
    tts_chatterbox_audio_prompt: str | None  # Path to audio file for voice cloning with Chatterbox
    tts_chatterbox_exaggeration: float  # Emotion exaggeration control (0.0-1.0+)
    tts_chatterbox_cfg_weight: float  # CFG weight for quality/speed trade-off

    # Piper TTS
    tts_piper_model_path: str | None  # Path to .onnx voice model
    tts_piper_speaker: int | None  # Speaker ID for multi-speaker models
    tts_piper_length_scale: float  # Speed: <1.0 faster, >1.0 slower
    tts_piper_noise_scale: float  # Audio variation
    tts_piper_noise_w: float  # Phoneme width variation
    tts_piper_sentence_silence: float  # Post-sentence silence in seconds

    # Kokoro TTS (more natural neural voice; real-time on CPU)
    tts_kokoro_voice: str  # Voice id, e.g. "ff_siwis" (French)
    tts_kokoro_lang_code: str  # Kokoro language code, e.g. "f" (French)
    tts_kokoro_speed: float  # Speech speed multiplier (1.0 = normal)

    # Voice Input & Audio
    voice_device: str | None
    sample_rate: int
    voice_min_energy: float

    # Voice Collection & Timing
    voice_block_seconds: float
    voice_collect_seconds: float
    voice_max_collect_seconds: float

    # Wake Word Detection
    wake_word: str
    wake_aliases: list[str]
    wake_fuzzy_ratio: float

    # Whisper Speech Recognition
    whisper_model: str
    whisper_backend: str  # "auto", "mlx", or "faster-whisper"
    whisper_device: str  # "cuda", "auto", or "cpu" (only for faster-whisper)
    whisper_compute_type: str
    whisper_vad: bool
    whisper_min_confidence: float
    whisper_no_speech_threshold: float
    whisper_min_audio_duration: float
    whisper_min_word_length: int

    # Voice Activity Detection (VAD)
    vad_enabled: bool
    vad_aggressiveness: int
    vad_frame_ms: int
    vad_pre_roll_ms: int
    endpoint_silence_ms: int
    max_utterance_ms: int
    tts_max_utterance_ms: int

    # UI/UX Features
    tune_enabled: bool
    hot_window_enabled: bool
    hot_window_seconds: float
    low_power_mode: bool

    # Echo Detection
    echo_energy_threshold: float
    echo_tolerance: float

    # Reminders — the first thing she does while nobody is watching
    reminders_enabled: bool
    reminder_model: str
    reminder_timeout_sec: float
    reminder_default_hour: int
    reminder_tick_sec: float
    reminder_late_grace_sec: float
    reminder_max_attempts: int

    # Appris — what she thinks she noticed in his journal, for him to say
    appris_model: str
    appris_jours: int
    appris_max_propositions: int
    appris_seuil_doublon: int
    appris_timeout_sec: float

    # Routines — what she does at a fixed hour with nobody in the room
    routines_enabled: bool
    routine_tick_sec: float
    routine_late_grace_sec: float
    routine_max_steriles: int

    # Confirmation — how long a question waits, and who reads the answer
    confirmation_ttl_sec: float
    confirmation_hot_window_sec: float
    confirmation_model: str
    confirmation_timeout_sec: float

    # Intent Judge (LLM-based intent classification)
    # Always used when available, falls back to simple wake word detection
    intent_judge_model: str
    intent_judge_timeout_sec: float

    # Transcript Buffer - ambient speech context for intent judge
    transcript_buffer_duration_sec: float

    # Memory & Dialogue
    # Drives both the short-term memory window and forced diary update interval
    dialogue_memory_timeout: float
    memory_enrichment_max_results: int
    memory_enrichment_source: str  # "all", "diary", or "graph"
    # Tool-call + tool-result messages from prior replies in the hot window
    # are re-injected into the next turn so follow-ups can reuse them instead
    # of re-fetching. These knobs cap how many prior tool turns survive and
    # how much of each tool payload is retained (the fence markers of
    # UNTRUSTED WEB EXTRACT blocks are preserved on truncation).
    tool_carryover_max_turns: int
    tool_carryover_per_entry_chars: int
    # Distil diary + graph into a short relevance-filtered note via a cheap
    # LLM pass before injecting into the reply system prompt. When None
    # (the default), it auto-enables for SMALL models (≤7B) and stays off
    # for larger models that can handle raw dumps. Set explicitly to force.
    memory_digest_enabled: Optional[bool]
    # Distil raw tool-result payloads (e.g. webSearch extracts) into a
    # short, attributed fact note via a cheap LLM pass before appending
    # them as tool-role messages. When None (the default), it auto-enables
    # for SMALL models (≤7B) and stays off for larger models that ground
    # on the raw payload reliably. Set explicitly to force on/off.
    tool_result_digest_enabled: Optional[bool]

    # Agentic Loop
    agentic_max_turns: int
    tool_selection_strategy: str  # "all", "keyword", "embedding", or "llm"
    # When `tool_selection_strategy == "llm"`, this model does the routing.
    # Empty string means "reuse ``llm_chat_model``" (the default).
    tool_router_model: str
    # Optional override for the post-turn evaluator LLM. Empty string means
    # "fall back to intent_judge_model, then ``llm_chat_model``" (the default).
    evaluator_model: str
    # None = auto (on for SMALL models, off for LARGE). Explicit true/false forces.
    evaluator_enabled: Optional[bool]
    # Upper bound on toolSearchTool invocations per reply turn. The cap
    # prevents a small model from churning through the escape hatch forever
    # when no tool really fits.
    tool_search_max_calls: int
    # Upper bound on evaluator-driven nudges per reply. Each time the
    # evaluator says "continue" with a nudge, the nudge is injected into
    # the next turn's system message. This cap stops nudge ping-pong when
    # the model keeps producing prose despite the nudge.
    evaluator_nudge_max: int
    # Optional override for the pre-loop task-list planner model. Empty
    # string means "fall back to tool_router_model → intent_judge_model →
    # ``llm_chat_model``" (the default). The planner is a small
    # classification-shaped pass so it rides the same small-model chain
    # as the router and the evaluator.
    planner_model: str
    # Whether the pre-loop planner is enabled. True = planner always runs;
    # False = planner never runs (legacy behaviour, with the
    # compound_query fallback still active). Default True — the planner
    # fails open to an empty plan so the cost of a miss is one cheap LLM
    # round-trip, and the upside is multi-step queries actually complete.
    planner_enabled: bool
    # Timeout for the planner LLM call. Short because the planner is on
    # the critical path — a long timeout would dominate first-token
    # latency for every query. Planner fails open on timeout.
    planner_timeout_sec: float

    # Location Services
    location_enabled: bool
    location_cache_minutes: int
    location_ip_address: str | None
    location_auto_detect: bool
    location_cgnat_resolve_public_ip: bool

    # Web Search
    web_search_enabled: bool
    # Optional Brave Search API key. When set, Brave is used as the primary
    # fallback when DuckDuckGo is rate-limited or returns no usable content.
    # Empty string means "not configured" — the tool then falls through to
    # the always-on Wikipedia fallback. Free tier is 2,000 queries/month.
    brave_search_api_key: str
    # Zero-config Wikipedia fallback toggle. When True (default), the tool
    # queries Wikipedia's REST summary API as a last resort before giving up
    # with the honest "blocked" envelope. Privacy-light (public API, no key,
    # no account) and language-aware via the Whisper-detected utterance
    # language.
    wikipedia_fallback_enabled: bool

    # Dictation (hold-to-dictate)
    dictation_enabled: bool
    dictation_hotkey: str
    dictation_filler_removal: bool
    dictation_custom_dictionary: list

    # MCP Integration
    mcps: Dict[str, Any]

    # Force the assistant's reply language regardless of input language.
    # Empty = mirror the user's language (model default). Example:
    # "français". Layered into the system prompt by build_system_prompt.
    response_language: str

    # Offered in the defaults and written by the settings window, so they
    # have to arrive: every consumer reads them with a `getattr` fallback,
    # and a missing field means the fallback wins and his choice does
    # nothing at all, quietly.
    llm_thinking_enabled: bool
    intent_judge_thinking_enabled: bool
    dictation_thinking_enabled: bool
    stop_commands: list
    stop_command_fuzzy_ratio: float

    # City shown on the dashboard weather card (Open-Meteo geocoded).
    # Empty falls back to "Paris".
    weather_city: str

    # Desktop UI choices (orb particle layer, etc.)
    ui: UISettings



def default_config_path() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg) / "jarvis" / "config.json"
    return Path.home() / ".config" / "jarvis" / "config.json"


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
    except Exception:
        pass
    return {}


def _save_json(path: Path, data: Dict[str, Any]) -> bool:
    """Save config data to JSON file. Returns True on success.

    Restricts the saved file to ``0o600`` on POSIX so credentials in
    config (``llm_api_key``, ``embedding_api_key``, ``brave_search_api_key``)
    are not readable by other users on multi-user systems. ``chmod`` is a
    no-op on Windows but is wrapped in a try so platform quirks never
    fail the save.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        try:
            path.chmod(0o600)
        except OSError:
            pass
        return True
    except Exception:
        return False


def _migrate_config(cfg_path: Path, cfg_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply config migrations for version upgrades.

    Returns the (possibly modified) config dict.
    """
    modified = False

    # Get current migration version (0 if not set = pre-migration config)
    migration_version = cfg_json.get("_config_version", 0)

    # Migration v1: tts_engine "system" -> "piper"
    # Piper is now the default TTS with auto-download support.
    if migration_version < 1:
        if cfg_json.get("tts_engine") == "system":
            cfg_json["tts_engine"] = "piper"
            print("📢 Upgraded TTS engine: system → piper (neural voice with auto-download)", flush=True)
            print("   To revert: set \"tts_engine\": \"system\" in config.json", flush=True)
        cfg_json["_config_version"] = 1
        modified = True

    # Migration v2: promote any ``ollama_*`` keys on disk into the
    # provider-aware ``llm_*`` / ``embedding_*`` shape. Default
    # ``llm_provider`` is ``"ollama"`` so existing installs keep their
    # behaviour. The old keys are left in place on disk so a downgrade
    # to an older Jarvis build still finds them.
    if migration_version < 2:
        if "llm_provider" not in cfg_json:
            cfg_json["llm_provider"] = "ollama"
        ollama_url = cfg_json.get("ollama_base_url")
        if ollama_url and not cfg_json.get("llm_base_url"):
            cfg_json["llm_base_url"] = ollama_url
        chat_model = cfg_json.get("ollama_chat_model")
        if chat_model and not cfg_json.get("llm_chat_model"):
            cfg_json["llm_chat_model"] = chat_model
        embed_model = cfg_json.get("ollama_embed_model")
        if embed_model and not cfg_json.get("embedding_model"):
            cfg_json["embedding_model"] = embed_model
        cfg_json["_config_version"] = 2
        modified = True

    # Save migrated config
    if modified:
        if _save_json(cfg_path, cfg_json):
            pass  # Silent success
        else:
            print("   ⚠️ Could not save config migration (using new settings in memory).", flush=True)

    return cfg_json


def load_config() -> Dict[str, Any]:
    """
    Load and return the merged configuration dictionary.

    Returns defaults merged with any values from the config file.
    Unlike load_settings(), this returns the raw dict instead of a Settings object.
    """
    cfg_path_env = os.environ.get("JARVIS_CONFIG_PATH")
    cfg_path = Path(cfg_path_env).expanduser() if cfg_path_env else default_config_path()
    cfg_json = _load_json(cfg_path)

    # Apply config migrations for version upgrades
    if cfg_json:
        cfg_json = _migrate_config(cfg_path, cfg_json)

    defaults = get_default_config()
    return {**defaults, **cfg_json}


def _cloud_safe_model(value: str, provider: str, fallback: str) -> str:
    """Keep auxiliary model names valid for the active provider.

    Jarvis runs several small LLM tasks (intent judge, tool router,
    evaluator, planner) on their own configured model. Those configs
    commonly hold a local Ollama tag like ``gemma4:e2b``. When the
    active provider is the cloud OpenAI-compatible one, sending an
    Ollama tag to the remote endpoint fails with HTTP 400 ("X is not a
    valid model ID"), because cloud model IDs are namespaced as
    ``vendor/model`` (e.g. ``deepseek/deepseek-v4-flash``).

    Heuristic: under ``openai_compatible``, a non-empty model name with
    no ``/`` is a stale local tag — fall back to the cloud chat model so
    the auxiliary task works instead of 400-ing. Names that already look
    like cloud IDs (contain ``/``) and empty values (which resolve via
    the engine's own fallback chain) are left untouched.
    """
    if not value:
        return value
    if provider == "openai_compatible" and "/" not in value:
        return fallback
    return value


def _ensure_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return [str(value)]


def _ensure_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    # Accept list of pairs like [{"name":..., ...}] and convert to dict by name if present
    try:
        if isinstance(value, list):
            out: Dict[str, Any] = {}
            for item in value:
                if isinstance(item, dict):
                    key = str(item.get("name")) if item.get("name") is not None else None
                    if key:
                        out[key] = {k: v for k, v in item.items() if k != "name"}
            if out:
                return out
    except Exception:
        pass
    return {}


def get_default_config() -> Dict[str, Any]:
    """Returns the default configuration values."""
    return {
        # Database & Storage
        "db_path": _default_db_path(),
        "sqlite_vss_path": None,

        # LLM & AI Models
        # Provider-aware fields. Default provider is ``ollama`` so a fresh
        # install needs no extra configuration. The ``ollama_*`` fields are
        # disk-format aliases for older config files; the loader promotes
        # their values into ``llm_*`` / ``embedding_*`` so everything inside
        # the codebase reads the provider-aware keys only.
        "llm_provider": "ollama",
        "llm_base_url": "",  # falls back to ollama_base_url when empty
        "llm_api_key": "",
        "llm_api_key_env": "",  # name of env var holding the key (preferred over plaintext)
        "llm_chat_model": "",  # falls back to ollama_chat_model when empty
        "llm_extra_body": {},  # provider-specific extra chat payload fields
        "auto_redact_before_cloud": True,  # scrub secrets before remote egress
        "embedding_provider": "",  # "" = same as llm_provider
        "embedding_base_url": "",
        "embedding_api_key": "",
        "embedding_model": "",  # falls back to ollama_embed_model when empty
        "ollama_base_url": "http://127.0.0.1:11434",
        "ollama_embed_model": "nomic-embed-text",
        "ollama_chat_model": DEFAULT_CHAT_MODEL,
        "llm_chat_timeout_sec": 180.0,
        "llm_tools_timeout_sec": 300.0,
        # Cheap distil passes should fail fast — a hung digest call would
        # block the reply loop per tool call, amplified by agentic turns.
        "llm_digest_timeout_sec": 8.0,
        "llm_embedding_timeout_sec": 60.0,
        "llm_profile_select_timeout_sec": 30.0,

        # Profiles & Behavior
        "active_profiles": ["developer", "business", "life"],
        "use_stdin": False,

        # Screen Capture
        "allowlist_bundles": [
            "com.apple.Terminal",
            "com.googlecode.iterm2",
            "com.microsoft.VSCode",
            "com.jetbrains.intellij",
        ],


        # Text-to-Speech
        "tts_enabled": True,
        "tts_engine": "piper",  # "piper" (default), "kokoro", or "chatterbox"
        "tts_voice": None,
        "tts_rate": 200,  # Words per minute (WPM), 200=normal
        "tts_chatterbox_device": "cuda",  # "cuda" (recommended), "auto", or "cpu"
        "tts_chatterbox_audio_prompt": None,  # Path to audio file for voice cloning
        "tts_chatterbox_exaggeration": 0.5,  # Emotion exaggeration (0.0-1.0+)
        "tts_chatterbox_cfg_weight": 0.5,  # CFG weight for quality/speed trade-off

        # Piper TTS
        "tts_piper_model_path": None,  # Path to .onnx voice model
        "tts_piper_speaker": None,  # Speaker ID for multi-speaker models
        "tts_piper_length_scale": 0.65,  # Speed: <1.0 faster, >1.0 slower (0.65 = ~30% faster)
        "tts_piper_noise_scale": 0.8,  # Audio variation (higher = more expressive)
        "tts_piper_noise_w": 1.0,  # Phoneme width variation (higher = more lively)
        "tts_piper_sentence_silence": 0.2,  # Post-sentence silence in seconds

        # Kokoro TTS
        "tts_kokoro_voice": "ff_siwis",  # French voice
        "tts_kokoro_lang_code": "f",  # French
        "tts_kokoro_speed": 1.0,  # Speech speed multiplier

        # Voice Input & Audio
        "voice_device": None,
        "sample_rate": 16000,
        "voice_min_energy": 0.02,

        # Voice Collection & Timing
        "voice_block_seconds": 4.0,
        "voice_collect_seconds": 4.5,
        "voice_max_collect_seconds": 180.0,

        # Wake Word Detection
        "wake_word": "jarvis",
        "wake_aliases": ["joris", "charis", "chavis", "jar is", "jaivis", "jervis", "jarvus", "jarviz", "javis", "jairus", "jarryst", "chyrus"],
        "wake_fuzzy_ratio": 0.78,

        # Whisper Speech Recognition
        "whisper_model": "medium",
        "whisper_backend": "auto",  # "auto" (MLX on Apple Silicon, else faster-whisper), "mlx", or "faster-whisper"
        "whisper_device": "auto",  # "cuda" (recommended if available), "auto", or "cpu" (only for faster-whisper)
        "whisper_compute_type": "int8",
        "whisper_vad": True,
        "whisper_min_confidence": 0.3,  # Filter low-confidence segments (hallucinations)
        "whisper_no_speech_threshold": 0.5,  # Hard cutoff: reject segments where no_speech_prob >= this
        "whisper_min_audio_duration": 0.15,
        "whisper_min_word_length": 1,

        # Voice Activity Detection (VAD)
        "vad_enabled": True,
        "vad_aggressiveness": 2,
        "vad_frame_ms": 20,
        "vad_pre_roll_ms": 240,
        "endpoint_silence_ms": 800,
        "max_utterance_ms": 12000,
        "tts_max_utterance_ms": 3000,  # Shorter timeout during TTS for quick stop detection

        # UI/UX Features
        "tune_enabled": True,
        "hot_window_enabled": True,
        "hot_window_seconds": 3.0,
        "low_power_mode": False,
        "echo_energy_threshold": 2.0,
        "echo_tolerance": 0.3,  # Time tolerance for echo detection timing

        # Reminders. The grace window is what separates "she is late"
        # from "she was never going to say it": past it she still speaks,
        # but says how late she is rather than pretending it is now.
        "reminders_enabled": True,
        # Empty = the warm small chain. Pin a local model to keep the
        # user's own sentence about their own life off the network.
        "reminder_model": "",
        "reminder_timeout_sec": 8.0,
        # Where a bare day lands when no hour was said: "jeudi" means
        # jeudi morning to most people who say it.
        "reminder_default_hour": 9,
        "reminder_tick_sec": 5.0,
        "reminder_late_grace_sec": 900.0,
        "reminder_max_attempts": 60,

        # Appris. Empty model = the reminder chain, whose last link is
        # the chat model; pin one to keep a fortnight of his days local.
        "appris_model": "",
        # A fortnight is long enough that a habit shows up twice and
        # short enough that he recognises what she is quoting.
        "appris_jours": 14,
        # Three is what a person will actually read and answer. A list
        # of twelve is a list nobody resolves, and unresolved proposals
        # are the failure mode that makes the file feel like a chore.
        "appris_max_propositions": 3,
        "appris_seuil_doublon": 90,
        "appris_timeout_sec": 30.0,

        # Routines. The grace window is what separates "the laptop was
        # shut" from "this morning has passed": a digest two hours late
        # is still the thing that was asked for, one at 18:00 is not.
        # Bounded again by the period inside `staleness_window`, so a
        # daily routine can never fire for yesterday while today's slot
        # is approaching.
        "routines_enabled": True,
        "routine_tick_sec": 30.0,
        "routine_late_grace_sec": 14400.0,
        # Consecutive runs that produced nothing at all before she stops
        # trying. Errors and empty write-ups count; "rien à signaler"
        # does not, since that is the routine working.
        "routine_max_steriles": 5,

        # Confirmation. The click window is generous because walking to
        # the machine takes longer than answering aloud. The spoken window
        # is wider than `hot_window_seconds`, which is tuned for
        # follow-ups rather than for consent — a person weighing whether
        # to let something happen pauses before answering.
        "confirmation_ttl_sec": 180.0,
        "confirmation_hot_window_sec": 12.0,
        # Empty = reuse the small warm chain (tool_router_model →
        # intent_judge_model → chat model). Pin a local model here to keep
        # the reading of a spoken approval off the network.
        "confirmation_model": "",
        "confirmation_timeout_sec": 8.0,

        # Audio Wake Word Detection
        # Intent Judge (LLM-based intent classification)
        # Always used when available, falls back to simple wake word detection
        "llm_thinking_enabled": False,  # Enable thinking/reasoning mode for chat (slower but may improve quality)
        "intent_judge_model": "gemma4:e2b",  # Model for intent judging (needs reasoning ability)
        "intent_judge_timeout_sec": 15.0,  # Max time to wait for intent judge response
        "intent_judge_thinking_enabled": False,  # Enable thinking for intent judge (adds latency to wake detection)

        # Transcript Buffer - used for both retention and context passed to intent judge
        # 120s (2 min) provides enough ambient speech context for intent judging
        # in group conversations. Separate from dialogue memory.
        "transcript_buffer_duration_sec": 120.0,

        # Memory & Dialogue
        # dialogue_memory_timeout drives the short-term memory window AND the forced
        # diary update interval. After a diary update, enrichment retrieves older context.
        "dialogue_memory_timeout": 300.0,
        "memory_enrichment_max_results": 3,
        "memory_enrichment_source": "all",  # "all", "diary", or "graph"
        # Tool carryover: cap re-injected prior tool turns + chars per entry.
        "tool_carryover_max_turns": 2,
        "tool_carryover_per_entry_chars": 1200,
        # None = auto (on for small models ≤7B, off for large). Set true/false to force.
        "memory_digest_enabled": None,
        # Distil raw tool results (e.g. webSearch extracts) into a short
        # attributed fact note for small models. Defaults to off: the extra
        # None = auto (on for small models ≤7B, off for large). Set true/false to force.
        # Auto-on for small models mitigates fetch_web_page's 50k-char payloads
        # blowing the 8192 num_ctx window before the main model sees them.
        "tool_result_digest_enabled": None,

        # Agentic Loop
        "agentic_max_turns": 8,
        "tool_selection_strategy": "llm",
        # Empty string = reuse intent_judge_model (small, fast, already warm
        # for wake-word paths), falling back to ollama_chat_model only if the
        # judge model isn't set. Override to decouple routing from both —
        # useful when you want routing on a dedicated smaller model.
        "tool_router_model": "",
        # Empty string = reuse intent_judge_model, falling through to
        # ollama_chat_model only if the judge isn't set. Override to pin the
        # evaluator to a dedicated small/fast model.
        "evaluator_model": "",
        # None = auto (on for small models, off for large). Set true/false to force.
        "evaluator_enabled": None,
        # Cap the number of toolSearchTool invocations per reply.
        "tool_search_max_calls": 3,
        # Cap the number of evaluator-driven nudges per reply.
        "evaluator_nudge_max": 2,
        # Task-list planner (see src/jarvis/reply/planner.spec.md). Empty
        # model string = reuse tool_router_model → intent_judge_model →
        # ollama_chat_model.
        "planner_model": "",
        "planner_enabled": True,
        "planner_timeout_sec": 6.0,

        # Stop Commands
        "stop_commands": ["stop", "quiet", "shush", "silence", "enough", "shut up"],
        "stop_command_fuzzy_ratio": 0.8,

        # Location Services
        "location_enabled": True,
        "location_cache_minutes": 60,
        "location_ip_address": None,
        "location_auto_detect": True,
        # When behind CGNAT (100.64.0.0/10), attempt a privacy-light external DNS query to discover true public IP.
        # Uses a single OpenDNS resolver lookup of myip.opendns.com over DNS (no HTTP services). Disable to avoid any external request.
        "location_cgnat_resolve_public_ip": True,

        # Web Search
        "web_search_enabled": True,
        "brave_search_api_key": "",
        "wikipedia_fallback_enabled": True,

        # Dictation (hold-to-dictate, WisprFlow-like)
        "dictation_enabled": True,
        "dictation_hotkey": _default_dictation_hotkey(),
        "dictation_filler_removal": False,
        "dictation_thinking_enabled": False,  # Enable thinking for dictation filler removal (adds latency)
        "dictation_custom_dictionary": [],

        # MCP Integration (external servers Jarvis can use). No defaults.
        "mcps": {},

        # Force reply language (empty = mirror the user's language).
        "response_language": "",

        # Dashboard weather card city (empty = "Paris").
        "weather_city": "",

        # Desktop UI. ``orb_particles_enabled`` controls whether the
        # reactive orb renders its ambient particle layer (default
        # True). Set false to skip the particle draw entirely (perf
        # or aesthetic preference).
        "ui": {
            "orb_particles_enabled": True,
        },
    }


def export_example_config(include_db_path: bool = False) -> Dict[str, Any]:
    """Returns example config suitable for JSON export (with adjusted db_path)."""
    config = get_default_config().copy()
    if not include_db_path:
        # Use a user-friendly path for examples
        config["db_path"] = "~/.local/share/jarvis/jarvis.db"
    return config


def load_settings() -> Settings:
    # Load environment for debug toggles and optional config file path only
    load_dotenv(override=False)

    # Resolve config path
    cfg_path_env = os.environ.get("JARVIS_CONFIG_PATH")
    cfg_path = Path(cfg_path_env).expanduser() if cfg_path_env else default_config_path()
    cfg_dir = cfg_path.parent
    try:
        cfg_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    # Load JSON configuration (non-debug settings)
    cfg_json = _load_json(cfg_path)

    # Apply config migrations for version upgrades
    if cfg_json:
        cfg_json = _migrate_config(cfg_path, cfg_json)

    # Get defaults and merge with JSON (JSON wins)
    defaults = get_default_config()
    merged: Dict[str, Any] = {**defaults, **cfg_json}

    # Build Settings. Some fields support env var overrides.
    # Env overrides: JARVIS_VOICE_DEBUG, JARVIS_WHISPER_BACKEND
    voice_debug = os.environ.get("JARVIS_VOICE_DEBUG", "0") == "1"

    # Normalize/convert fields
    db_path = str(merged.get("db_path") or _default_db_path())
    sqlite_vss_path = merged.get("sqlite_vss_path")
    allowlist_bundles = _ensure_list(merged.get("allowlist_bundles"))

    ollama_base_url = str(merged.get("ollama_base_url"))
    ollama_embed_model = str(merged.get("ollama_embed_model"))
    ollama_chat_model = str(merged.get("ollama_chat_model"))

    # Provider-aware fields. The two field sets are per-provider: the
    # ``ollama_*`` fields are authoritative when the provider is Ollama,
    # the ``llm_*`` / ``embedding_*`` fields when it is OpenAI-compatible.
    # Resolving the active model this way (rather than a blanket
    # ``llm_chat_model or ollama_chat_model``) keeps the Ollama model
    # picker — which writes ``ollama_chat_model`` — authoritative on the
    # Ollama path, so a stale ``llm_chat_model`` (e.g. promoted by the v2
    # migration) can never shadow it.
    llm_provider = str(merged.get("llm_provider", "ollama") or "ollama").strip().lower()
    if llm_provider not in ("ollama", "openai_compatible"):
        llm_provider = "ollama"
    llm_base_url = str(merged.get("llm_base_url", "") or "").strip() or ollama_base_url
    # API key resolution: prefer the named environment variable (keeps
    # the secret out of config.json), fall back to the literal field.
    llm_api_key_env = str(merged.get("llm_api_key_env", "") or "").strip()
    llm_api_key = str(merged.get("llm_api_key", "") or "").strip()
    if llm_api_key_env:
        env_key = os.environ.get(llm_api_key_env, "").strip()
        if env_key:
            llm_api_key = env_key
    auto_redact_before_cloud = bool(merged.get("auto_redact_before_cloud", True))
    raw_extra_body = merged.get("llm_extra_body", {})
    llm_extra_body = raw_extra_body if isinstance(raw_extra_body, dict) else {}
    if llm_provider == "openai_compatible":
        llm_chat_model = str(merged.get("llm_chat_model", "") or "").strip() or ollama_chat_model
    else:
        llm_chat_model = ollama_chat_model
    embedding_provider_raw = str(merged.get("embedding_provider", "") or "").strip().lower()
    if embedding_provider_raw not in ("", "ollama", "openai_compatible"):
        embedding_provider_raw = ""
    embedding_provider = embedding_provider_raw
    embedding_base_url = str(merged.get("embedding_base_url", "") or "").strip()
    embedding_api_key = str(merged.get("embedding_api_key", "") or "").strip()
    # Effective embedding provider inherits the chat provider when unset.
    _effective_embed_provider = embedding_provider or llm_provider
    if _effective_embed_provider == "openai_compatible":
        embedding_model = str(merged.get("embedding_model", "") or "").strip() or ollama_embed_model
    else:
        embedding_model = ollama_embed_model
    use_stdin = bool(merged.get("use_stdin", False))
    active_profiles = _ensure_list(merged.get("active_profiles"))
    tts_enabled = bool(merged.get("tts_enabled", True))
    tts_engine = str(merged.get("tts_engine", "piper")).lower()
    if tts_engine not in ("piper", "chatterbox", "kokoro"):
        tts_engine = "piper"  # Default to piper if invalid value
    tts_voice_val = merged.get("tts_voice")
    tts_voice = None if tts_voice_val in (None, "", "null") else str(tts_voice_val)
    tts_rate_val = merged.get("tts_rate")
    try:
        tts_rate = None if tts_rate_val in (None, "", "null") else int(tts_rate_val)
    except Exception:
        tts_rate = None
    tts_chatterbox_device = str(merged.get("tts_chatterbox_device", "cuda")).lower()
    if tts_chatterbox_device not in ("cuda", "auto", "cpu"):
        tts_chatterbox_device = "cuda"  # Default to cuda if invalid value
    tts_chatterbox_audio_prompt_val = merged.get("tts_chatterbox_audio_prompt")
    tts_chatterbox_audio_prompt = None if tts_chatterbox_audio_prompt_val in (None, "", "null") else str(tts_chatterbox_audio_prompt_val)
    tts_chatterbox_exaggeration = float(merged.get("tts_chatterbox_exaggeration", 0.5))
    tts_chatterbox_cfg_weight = float(merged.get("tts_chatterbox_cfg_weight", 0.5))

    # Piper TTS settings
    tts_piper_model_path_val = merged.get("tts_piper_model_path")
    tts_piper_model_path = None if tts_piper_model_path_val in (None, "", "null") else str(tts_piper_model_path_val)
    tts_piper_speaker_val = merged.get("tts_piper_speaker")
    try:
        tts_piper_speaker = None if tts_piper_speaker_val in (None, "", "null") else int(tts_piper_speaker_val)
    except Exception:
        tts_piper_speaker = None
    tts_piper_length_scale = float(merged.get("tts_piper_length_scale", 0.65))
    tts_piper_noise_scale = float(merged.get("tts_piper_noise_scale", 0.8))
    tts_piper_noise_w = float(merged.get("tts_piper_noise_w", 1.0))
    tts_piper_sentence_silence = float(merged.get("tts_piper_sentence_silence", 0.2))
    tts_kokoro_voice = str(merged.get("tts_kokoro_voice", "ff_siwis") or "ff_siwis").strip()
    tts_kokoro_lang_code = str(merged.get("tts_kokoro_lang_code", "f") or "f").strip()
    try:
        tts_kokoro_speed = float(merged.get("tts_kokoro_speed", 1.0))
    except Exception:
        tts_kokoro_speed = 1.0

    voice_device_val = merged.get("voice_device")
    voice_device = None if voice_device_val in (None, "", "default", "system") else str(voice_device_val)
    voice_block_seconds = float(merged.get("voice_block_seconds", 4.0))
    voice_collect_seconds = float(merged.get("voice_collect_seconds", 2.5))
    voice_max_collect_seconds = float(merged.get("voice_max_collect_seconds", 60.0))
    wake_word = str(merged.get("wake_word", "jarvis")).strip().lower()
    wake_aliases = [a.strip().lower() for a in _ensure_list(merged.get("wake_aliases")) if a.strip()]
    wake_fuzzy_ratio = float(merged.get("wake_fuzzy_ratio", 0.78))
    whisper_model = str(merged.get("whisper_model", "medium"))
    whisper_backend = os.environ.get("JARVIS_WHISPER_BACKEND", "").lower() or str(merged.get("whisper_backend", "auto")).lower()
    if whisper_backend not in ("auto", "mlx", "faster-whisper"):
        whisper_backend = "auto"
    whisper_device = str(merged.get("whisper_device", "auto")).lower()
    if whisper_device not in ("cuda", "auto", "cpu"):
        whisper_device = "auto"
    whisper_compute_type = str(merged.get("whisper_compute_type", "int8"))
    whisper_vad = bool(merged.get("whisper_vad", True))
    voice_min_energy = float(merged.get("voice_min_energy", 0.02))
    vad_enabled = bool(merged.get("vad_enabled", True))
    vad_aggressiveness = int(merged.get("vad_aggressiveness", 2))
    vad_frame_ms = int(merged.get("vad_frame_ms", 20))
    vad_pre_roll_ms = int(merged.get("vad_pre_roll_ms", 240))
    endpoint_silence_ms = int(merged.get("endpoint_silence_ms", 800))
    max_utterance_ms = int(merged.get("max_utterance_ms", 12000))
    tts_max_utterance_ms = int(merged.get("tts_max_utterance_ms", 3000))
    sample_rate = int(merged.get("sample_rate", 16000))
    tune_enabled = bool(merged.get("tune_enabled", True))
    hot_window_enabled = bool(merged.get("hot_window_enabled", True))
    hot_window_seconds = float(merged.get("hot_window_seconds", 3.0))
    low_power_mode = bool(merged.get("low_power_mode", False))
    echo_energy_threshold = float(merged.get("echo_energy_threshold", 2.0))
    echo_tolerance = float(merged.get("echo_tolerance", 0.3))

    # Reminders. Clamped rather than trusted: a tick of zero spins a core
    # and a timeout of zero makes every reminder unreadable, and both
    # symptoms point nowhere near their cause.
    reminders_enabled = bool(merged.get("reminders_enabled", True))
    # Deliberately NOT passed through `_cloud_safe_model`, unlike the
    # tool router and the intent judge. That filter rewrites a pinned
    # local tag to the cloud chat model, and this prompt carries the
    # user's own sentence about their own life — pinning a local model is
    # the only way to keep it off the network, so rescuing it would
    # silently undo the one thing the setting is for.
    reminder_model = str(merged.get("reminder_model", ""))
    reminder_timeout_sec = min(max(float(merged.get("reminder_timeout_sec", 8.0)), 2.0), 30.0)
    reminder_default_hour = min(max(int(merged.get("reminder_default_hour", 9)), 0), 23)
    appris_model = str(merged.get("appris_model", "") or "")
    appris_jours = min(max(int(merged.get("appris_jours", 14)), 1), 365)
    appris_max_propositions = min(max(int(merged.get("appris_max_propositions", 3)), 1), 10)
    # Below ~70 `token_set_ratio` starts folding together two different
    # things he said; above ~97 it stops catching a rephrasing.
    appris_seuil_doublon = min(max(int(merged.get("appris_seuil_doublon", 90)), 70), 100)
    appris_timeout_sec = min(max(float(merged.get("appris_timeout_sec", 30.0)), 5.0), 120.0)
    reminder_tick_sec = min(max(float(merged.get("reminder_tick_sec", 5.0)), 1.0), 60.0)
    reminder_late_grace_sec = min(max(float(merged.get("reminder_late_grace_sec", 900.0)), 0.0), 86400.0)
    reminder_max_attempts = min(max(int(merged.get("reminder_max_attempts", 60)), 1), 600)
    routines_enabled = bool(merged.get("routines_enabled", True))
    routine_tick_sec = min(max(float(merged.get("routine_tick_sec", 30.0)), 5.0), 300.0)
    routine_late_grace_sec = min(max(float(merged.get("routine_late_grace_sec", 14400.0)), 0.0), 86400.0)
    routine_max_steriles = min(max(int(merged.get("routine_max_steriles", 5)), 1), 100)

    # Confirmation. Clamped rather than trusted: a TTL of zero would
    # expire every question before it could be read, and an unbounded one
    # would leave a destructive action answerable days later.
    confirmation_ttl_sec = min(max(float(merged.get("confirmation_ttl_sec", 180.0)), 15.0), 900.0)
    confirmation_hot_window_sec = min(max(float(merged.get("confirmation_hot_window_sec", 12.0)), 3.0), 60.0)
    confirmation_model = str(merged.get("confirmation_model", ""))
    confirmation_timeout_sec = min(max(float(merged.get("confirmation_timeout_sec", 8.0)), 2.0), 30.0)

    # Intent Judge - always used when available
    intent_judge_model = str(merged.get("intent_judge_model", "gemma4:e2b"))
    intent_judge_model = _cloud_safe_model(intent_judge_model, llm_provider, llm_chat_model)
    intent_judge_timeout_sec = float(merged.get("intent_judge_timeout_sec", 10.0))

    # Transcript Buffer - ambient speech context for intent judge (separate from dialogue)
    transcript_buffer_duration_sec = float(merged.get("transcript_buffer_duration_sec", 120.0))

    # Dialogue memory window and forced diary update share this duration
    dialogue_memory_timeout = float(merged.get("dialogue_memory_timeout", 300.0))
    memory_enrichment_max_results = int(merged.get("memory_enrichment_max_results", 3))
    memory_enrichment_source = str(merged.get("memory_enrichment_source", "all")).lower()
    if memory_enrichment_source not in ("all", "diary", "graph"):
        memory_enrichment_source = "all"
    tool_carryover_max_turns = max(0, int(merged.get("tool_carryover_max_turns", 2)))
    tool_carryover_per_entry_chars = max(200, int(merged.get("tool_carryover_per_entry_chars", 1200)))
    _digest_raw = merged.get("memory_digest_enabled", None)
    memory_digest_enabled: Optional[bool]
    if _digest_raw is None:
        memory_digest_enabled = None
    else:
        memory_digest_enabled = bool(_digest_raw)
    _tool_digest_raw = merged.get("tool_result_digest_enabled", None)
    tool_result_digest_enabled: Optional[bool]
    if _tool_digest_raw is None:
        tool_result_digest_enabled = None
    else:
        tool_result_digest_enabled = bool(_tool_digest_raw)
    agentic_max_turns = int(merged.get("agentic_max_turns", 8))
    tool_selection_strategy = str(merged.get("tool_selection_strategy", "llm")).lower()
    if tool_selection_strategy not in ("all", "keyword", "embedding", "llm"):
        tool_selection_strategy = "llm"
    tool_router_model = str(merged.get("tool_router_model", "") or "").strip()
    tool_router_model = _cloud_safe_model(tool_router_model, llm_provider, llm_chat_model)
    evaluator_model = str(merged.get("evaluator_model", "") or "").strip()
    evaluator_model = _cloud_safe_model(evaluator_model, llm_provider, llm_chat_model)
    _eval_raw = merged.get("evaluator_enabled", None)
    evaluator_enabled: Optional[bool]
    if _eval_raw is None:
        evaluator_enabled = None
    else:
        evaluator_enabled = bool(_eval_raw)
    planner_model = str(merged.get("planner_model", "") or "").strip()
    planner_model = _cloud_safe_model(planner_model, llm_provider, llm_chat_model)
    planner_enabled = bool(merged.get("planner_enabled", True))
    try:
        planner_timeout_sec = float(merged.get("planner_timeout_sec", 6.0))
    except (TypeError, ValueError):
        planner_timeout_sec = 6.0
    try:
        tool_search_max_calls = int(merged.get("tool_search_max_calls", 3))
    except (TypeError, ValueError):
        tool_search_max_calls = 3
    if tool_search_max_calls < 0:
        tool_search_max_calls = 0
    try:
        evaluator_nudge_max = int(merged.get("evaluator_nudge_max", 2))
    except (TypeError, ValueError):
        evaluator_nudge_max = 2
    if evaluator_nudge_max < 0:
        evaluator_nudge_max = 0
    location_enabled = bool(merged.get("location_enabled", True))
    location_cache_minutes = int(merged.get("location_cache_minutes", 60))
    location_ip_address_val = merged.get("location_ip_address")
    location_ip_address = None if location_ip_address_val in (None, "", "null") else str(location_ip_address_val)
    location_auto_detect = bool(merged.get("location_auto_detect", True))
    location_cgnat_resolve_public_ip = bool(merged.get("location_cgnat_resolve_public_ip", True))
    web_search_enabled = bool(merged.get("web_search_enabled", True))
    brave_search_api_key = str(merged.get("brave_search_api_key", "") or "").strip()
    wikipedia_fallback_enabled = bool(merged.get("wikipedia_fallback_enabled", True))
    dictation_enabled = bool(merged.get("dictation_enabled", True))
    dictation_hotkey = str(merged.get("dictation_hotkey", _default_dictation_hotkey())).strip()
    dictation_filler_removal = bool(merged.get("dictation_filler_removal", False))
    raw_dict = merged.get("dictation_custom_dictionary", [])
    dictation_custom_dictionary = list(raw_dict) if isinstance(raw_dict, list) else []
    mcps = _ensure_dict(merged.get("mcps"))
    response_language = str(merged.get("response_language", "") or "").strip()
    llm_thinking_enabled = bool(merged.get("llm_thinking_enabled", False))
    intent_judge_thinking_enabled = bool(
        merged.get("intent_judge_thinking_enabled", False))
    dictation_thinking_enabled = bool(merged.get("dictation_thinking_enabled", False))
    _mots = merged.get("stop_commands")
    stop_commands = (
        [str(m) for m in _mots if str(m).strip()]
        if isinstance(_mots, list) and any(str(m).strip() for m in _mots)
        else ["stop", "quiet", "shush", "silence", "enough", "shut up"]
    )
    # Below ~0.5 a stop word matches most short utterances; above 1.0 is
    # not a ratio. An empty or unusable list falls back rather than
    # leaving him unable to interrupt her at all.
    stop_command_fuzzy_ratio = min(
        max(float(merged.get("stop_command_fuzzy_ratio", 0.8)), 0.5), 1.0)
    weather_city = str(merged.get("weather_city", "") or "").strip()

    # Parse ui subsection. ``orb_particles_enabled`` defaults to True;
    # coerced via bool()/string rules so a hand-edited config that
    # writes "false"/0/"no" still resolves sensibly. A missing or
    # non-dict ``ui`` block falls back to the default.
    raw_ui = merged.get("ui", {})
    if not isinstance(raw_ui, dict):
        raw_ui = {}
    raw_particles = raw_ui.get("orb_particles_enabled", True)
    if isinstance(raw_particles, str):
        orb_particles_enabled = raw_particles.strip().lower() not in {"false", "0", "no", "off"}
    else:
        orb_particles_enabled = bool(raw_particles)
    ui = UISettings(orb_particles_enabled=orb_particles_enabled)

    whisper_min_confidence = float(merged.get("whisper_min_confidence", 0.4))
    whisper_no_speech_threshold = float(merged.get("whisper_no_speech_threshold", 0.5))
    whisper_min_audio_duration = float(merged.get("whisper_min_audio_duration", 0.3))
    whisper_min_word_length = int(merged.get("whisper_min_word_length", 2))
    llm_chat_timeout_sec = float(merged.get("llm_chat_timeout_sec", 180.0))
    llm_tools_timeout_sec = float(merged.get("llm_tools_timeout_sec", 300.0))
    llm_digest_timeout_sec = float(merged.get("llm_digest_timeout_sec", 8.0))
    llm_embedding_timeout_sec = float(merged.get("llm_embedding_timeout_sec", 60.0))
    llm_profile_select_timeout_sec = float(merged.get("llm_profile_select_timeout_sec", 30.0))

    return Settings(
        # Database & Storage
        db_path=db_path,
        sqlite_vss_path=sqlite_vss_path,

        # LLM & AI Models — provider-aware
        llm_provider=llm_provider,
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
        llm_api_key_env=llm_api_key_env,
        llm_chat_model=llm_chat_model,
        llm_extra_body=llm_extra_body,
        auto_redact_before_cloud=auto_redact_before_cloud,
        embedding_provider=embedding_provider,
        embedding_base_url=embedding_base_url,
        embedding_api_key=embedding_api_key,
        embedding_model=embedding_model,
        ollama_base_url=ollama_base_url,
        ollama_embed_model=ollama_embed_model,
        ollama_chat_model=ollama_chat_model,
        llm_chat_timeout_sec=llm_chat_timeout_sec,
        llm_tools_timeout_sec=llm_tools_timeout_sec,
        llm_digest_timeout_sec=llm_digest_timeout_sec,
        llm_embedding_timeout_sec=llm_embedding_timeout_sec,
        llm_profile_select_timeout_sec=llm_profile_select_timeout_sec,

        # Profiles & Behavior
        active_profiles=active_profiles,
        use_stdin=use_stdin,
        voice_debug=voice_debug,

        # Screen Capture
        allowlist_bundles=allowlist_bundles,

        # Text-to-Speech
        tts_enabled=tts_enabled,
        tts_engine=tts_engine,
        tts_voice=tts_voice,
        tts_rate=tts_rate,
        tts_chatterbox_device=tts_chatterbox_device,
        tts_chatterbox_audio_prompt=tts_chatterbox_audio_prompt,
        tts_chatterbox_exaggeration=tts_chatterbox_exaggeration,
        tts_chatterbox_cfg_weight=tts_chatterbox_cfg_weight,

        # Piper TTS
        tts_piper_model_path=tts_piper_model_path,
        tts_piper_speaker=tts_piper_speaker,
        tts_piper_length_scale=tts_piper_length_scale,
        tts_piper_noise_scale=tts_piper_noise_scale,
        tts_piper_noise_w=tts_piper_noise_w,
        tts_piper_sentence_silence=tts_piper_sentence_silence,
        tts_kokoro_voice=tts_kokoro_voice,
        tts_kokoro_lang_code=tts_kokoro_lang_code,
        tts_kokoro_speed=tts_kokoro_speed,

        # Voice Input & Audio
        voice_device=voice_device,
        sample_rate=sample_rate,
        voice_min_energy=voice_min_energy,

        # Voice Collection & Timing
        voice_block_seconds=voice_block_seconds,
        voice_collect_seconds=voice_collect_seconds,
        voice_max_collect_seconds=voice_max_collect_seconds,

        # Wake Word Detection
        wake_word=wake_word,
        wake_aliases=wake_aliases,
        wake_fuzzy_ratio=wake_fuzzy_ratio,

        # Whisper Speech Recognition
        whisper_model=whisper_model,
        whisper_backend=whisper_backend,
        whisper_device=whisper_device,
        whisper_compute_type=whisper_compute_type,
        whisper_vad=whisper_vad,
        whisper_min_confidence=whisper_min_confidence,
        whisper_no_speech_threshold=whisper_no_speech_threshold,
        whisper_min_audio_duration=whisper_min_audio_duration,
        whisper_min_word_length=whisper_min_word_length,

        # Voice Activity Detection (VAD)
        vad_enabled=vad_enabled,
        vad_aggressiveness=vad_aggressiveness,
        vad_frame_ms=vad_frame_ms,
        vad_pre_roll_ms=vad_pre_roll_ms,
        endpoint_silence_ms=endpoint_silence_ms,
        max_utterance_ms=max_utterance_ms,
        tts_max_utterance_ms=tts_max_utterance_ms,

        # UI/UX Features
        tune_enabled=tune_enabled,
        hot_window_enabled=hot_window_enabled,
        hot_window_seconds=hot_window_seconds,
        low_power_mode=low_power_mode,
        echo_energy_threshold=echo_energy_threshold,
        echo_tolerance=echo_tolerance,
        # Reminders
        reminders_enabled=reminders_enabled,
        reminder_model=reminder_model,
        reminder_timeout_sec=reminder_timeout_sec,
        appris_model=appris_model,
        appris_jours=appris_jours,
        appris_max_propositions=appris_max_propositions,
        appris_seuil_doublon=appris_seuil_doublon,
        appris_timeout_sec=appris_timeout_sec,
        reminder_default_hour=reminder_default_hour,
        reminder_tick_sec=reminder_tick_sec,
        reminder_late_grace_sec=reminder_late_grace_sec,
        reminder_max_attempts=reminder_max_attempts,
        routines_enabled=routines_enabled,
        routine_tick_sec=routine_tick_sec,
        routine_late_grace_sec=routine_late_grace_sec,
        routine_max_steriles=routine_max_steriles,
        # Confirmation
        confirmation_ttl_sec=confirmation_ttl_sec,
        confirmation_hot_window_sec=confirmation_hot_window_sec,
        confirmation_model=confirmation_model,
        confirmation_timeout_sec=confirmation_timeout_sec,
        # Intent Judge - always used when available
        intent_judge_model=intent_judge_model,
        intent_judge_timeout_sec=intent_judge_timeout_sec,

        # Transcript Buffer
        transcript_buffer_duration_sec=transcript_buffer_duration_sec,

        # Memory & Dialogue
        dialogue_memory_timeout=dialogue_memory_timeout,
        memory_enrichment_max_results=memory_enrichment_max_results,
        memory_enrichment_source=memory_enrichment_source,
        tool_carryover_max_turns=tool_carryover_max_turns,
        tool_carryover_per_entry_chars=tool_carryover_per_entry_chars,
        memory_digest_enabled=memory_digest_enabled,
        tool_result_digest_enabled=tool_result_digest_enabled,
        agentic_max_turns=agentic_max_turns,
        tool_selection_strategy=tool_selection_strategy,
        tool_router_model=tool_router_model,
        evaluator_model=evaluator_model,
        evaluator_enabled=evaluator_enabled,
        tool_search_max_calls=tool_search_max_calls,
        evaluator_nudge_max=evaluator_nudge_max,
        planner_model=planner_model,
        planner_enabled=planner_enabled,
        planner_timeout_sec=planner_timeout_sec,

        # Location Services
        location_enabled=location_enabled,
        location_cache_minutes=location_cache_minutes,
        location_ip_address=location_ip_address,
        location_auto_detect=location_auto_detect,
        location_cgnat_resolve_public_ip=location_cgnat_resolve_public_ip,

        # Web Search
        web_search_enabled=web_search_enabled,
        brave_search_api_key=brave_search_api_key,
        wikipedia_fallback_enabled=wikipedia_fallback_enabled,

        # Dictation
        dictation_enabled=dictation_enabled,
        dictation_hotkey=dictation_hotkey,
        dictation_filler_removal=dictation_filler_removal,
        dictation_custom_dictionary=dictation_custom_dictionary,

        # MCP Integration
        mcps=mcps,
        response_language=response_language,
        llm_thinking_enabled=llm_thinking_enabled,
        intent_judge_thinking_enabled=intent_judge_thinking_enabled,
        dictation_thinking_enabled=dictation_thinking_enabled,
        stop_commands=stop_commands,
        stop_command_fuzzy_ratio=stop_command_fuzzy_ratio,
        weather_city=weather_city,

        # Desktop UI
        ui=ui,
    )

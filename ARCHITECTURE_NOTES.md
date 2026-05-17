# Jarvis — Architecture Notes

Cartographie figée du dépôt à un instant T. Aucune modification de source.
Branche : `claude/nice-khayyam-d5b235` (worktree). Defaults branch : `develop`.

---

## 1. Vue d'ensemble

Jarvis est un assistant vocal 100 % local. Le coeur (`src/jarvis/`) est un
process *daemon* indépendant ; un *desktop_app* PyQt6 (`src/desktop_app/`) le
pilote en sous-processus et fournit tray icon, fenêtres et IPC stdin/stdout.
La règle invariante : le coeur ignore l'existence du desktop_app.

### Cartographie `src/jarvis/`

| Module | Rôle (1-2 lignes) |
|---|---|
| `main.py` | Trivial entrypoint → `daemon.main()`. |
| `daemon.py` | Orchestrateur du process. Charge config, DB, mémoire, MCP, TTS, lance le `VoiceListener` thread, gère la boucle diary périodique et le shutdown propre. |
| `config.py` | `Settings` dataclass + `load_settings()` lisant `~/.config/jarvis/config.json`. Source unique des modèles supportés (`SUPPORTED_CHAT_MODELS`). |
| `system_prompt.py` | Template unique de la persona (butler britannique). `build_system_prompt(name)` rend le wake-word capitalisé. |
| `llm.py` | Wrappers HTTP Ollama : `call_llm_direct`, `call_llm_streaming`, `chat_with_messages` (tool API native), `ToolsNotSupportedError`. |
| `debug.py` | `debug_log(msg, category)` central. |
| `listening/` | Pipeline audio → texte → intent. Voir §2. |
| `dictation/` | Hold-to-dictate (hotkey global), partage le modèle Whisper du listener. |
| `reply/` | Génération de réponse : planner, engine agentic, enrichment, prompts. Voir §2. |
| `tools/` | Registry des tools builtin + MCP, sélection (LLM/keyword/embedding/all). |
| `tools/builtin/` | Outils internes : `webSearch`, `fetchWebPage`, `screenshot`, `localFiles`, `getWeather`, `logMeal`/`fetchMeals`/`deleteMeal`, `refreshMCPTools`, `stop`, `toolSearchTool`. |
| `tools/external/` | `MCPClient` (stdio MCP) + `mcp_runtime` (event-loop persistant, une worker-task par serveur, retry sur `MCPServerSessionError`). |
| `memory/db.py` | SQLite (WAL) + FTS5 sur `conversation_summaries` + VSS optionnel (vec FLOAT[768]). |
| `memory/conversation.py` | `DialogueMemory` (rolling cache + warm profile invalidation + tool-carryover) ; pipeline diary (résumé + scrub déflection + reembed). |
| `memory/graph.py` | Knowledge graph auto-organisé (`User`/`Directives`/`World`), mutation listeners. |
| `memory/graph_ops.py` | Opérations CRUD/refactor sur le graphe. |
| `memory/embeddings.py` | `get_embedding()` Ollama. |
| `memory/recall_gate.py` | Skip enrichissement déterministe si fenêtre chaude couvre la question. |
| `output/tts.py` | Factory + engines TTS : Piper (default, auto-download HuggingFace) et Chatterbox (clone vocal). |
| `output/tune_player.py` | Tonalité « thinking ». |
| `utils/redact.py` | Auto-redaction (email, cartes, AWS/Stripe/GH/OpenAI/Google keys, JWT, etc.). |
| `utils/location.py` | GeoLite2 local + chaîne UPnP / socket / OpenDNS. |
| `utils/time_context.py` | Contexte temporel injecté dans le system prompt. |
| `utils/vector_store.py` + `fast_vector_store.py` | Vector stores (faiss-cpu). |
| `utils/fuzzy_search.py` | Génération de requêtes FTS5 flexibles. |

`desktop_app/` (PyQt6) : tray, setup wizard, settings auto-générées, memory viewer, dictation history, face widget, splash, updater. Communique via stdin/stdout (`__DIARY__:`) avec le daemon.

---

## 2. Composants critiques (qui fait quoi)

### Appels au LLM (Ollama)
- **Couche basse** : `src/jarvis/llm.py` (`call_llm_direct`, `call_llm_streaming`, `chat_with_messages`). Toutes les requêtes passent par `POST {ollama_base_url}/api/chat`. Le `num_ctx` est à 4096 par défaut, 8192 pour la boucle agentic.
- **Boucle agentic principale** : [`src/jarvis/reply/engine.py:776`](src/jarvis/reply/engine.py:776) (`run_reply_engine`). Construit le system prompt, sélectionne les tools, lance le planner, itère les tool_calls.
- **Résolveur de modèle pour le router/planner** : [`resolve_tool_router_model`](src/jarvis/reply/engine.py:148) — chaîne `tool_router_model` → `intent_judge_model` → `ollama_chat_model`.
- **Intent voice** : [`src/jarvis/listening/intent_judge.py`](src/jarvis/listening/intent_judge.py) (`gemma4:e2b` par défaut, toujours chargé).
- **Planner** : [`src/jarvis/reply/planner.py`](src/jarvis/reply/planner.py) (pré-loop décomposition + direct-exec pour petits modèles).
- **Digests** : `digest_memory_for_query`, `digest_tool_result_for_query`, `digest_loop_for_max_turns` dans [`src/jarvis/reply/enrichment.py`](src/jarvis/reply/enrichment.py).
- **Diary summariser + deflection rewrite** : [`src/jarvis/memory/conversation.py`](src/jarvis/memory/conversation.py).
- **Catalogue exhaustif** : `docs/llm_contexts.md` (référence canonique, à tenir à jour).

### Wake word + STT
- **Wake word** : pas de Porcupine ni openWakeWord — détection **texte** sur les transcriptions Whisper via fuzzy matching (difflib + alias) dans [`src/jarvis/listening/wake_detection.py`](src/jarvis/listening/wake_detection.py:9). La doc README mentionne « openWakeWord » mais le code actuel n'utilise que Whisper + intent judge LLM.
- **STT** : `faster-whisper` sur Windows/Linux, `mlx-whisper` sur Apple Silicon. Sélection dans [`src/jarvis/listening/listener.py`](src/jarvis/listening/listener.py:247). Probe CUDA (cuBLAS/cuDNN) sur Windows.
- **Pipeline audio** : `sounddevice` capture → `webrtcvad` VAD → `WhisperModel.transcribe()` → garde anti-hallucination (`whisper_min_confidence`, `whisper_no_speech_threshold`) → `EchoDetector` + `IntentJudge` LLM → `_dispatch_query` → `run_reply_engine`.
- **Hot window** + **echo detection** : [`echo_detection.py`](src/jarvis/listening/echo_detection.py), [`state_manager.py`](src/jarvis/listening/state_manager.py), [`transcript_buffer.py`](src/jarvis/listening/transcript_buffer.py).

### Orchestration des conversations
- **State machine voix** : [`StateManager`](src/jarvis/listening/state_manager.py) (états `LISTENING`/`COLLECTING`/`HOT_WINDOW`).
- **Contexte rolling** : [`DialogueMemory`](src/jarvis/memory/conversation.py) — inactivity timeout (`dialogue_memory_timeout`), max 20 interactions, warm-profile cache invalidé via listeners de mutation du graphe (`register_graph_mutation_listener`).
- **Personnalités/profils** : un seul system prompt unifié dans `system_prompt.py`. La spec `reply.spec.md` précise « tools return raw data; profiles handle formatting » — la personnalité s'adapte au topic via instructions (surgical/pragmatic/encouraging) dans le même prompt, pas via fichiers de profil multiples.
- **Boucle de réponse** : [`run_reply_engine`](src/jarvis/reply/engine.py:776) → planner → router → enrichment (diary + graph + digest) → `chat_with_messages` (tools natifs ou text-tool-call parsing pour Gemma).

### MCP servers (chargement + exécution)
- **Discovery au boot** : [`initialize_mcp_tools`](src/jarvis/tools/registry.py:50) lit `cfg.mcps` et appelle `discover_mcp_tools` ; cache global `_mcp_tools_cache`.
- **Client stdio MCP** : [`MCPClient`](src/jarvis/tools/external/mcp_client.py) (SDK `mcp==1.13.1`). Résolution PATH étendue (Homebrew, nvm, fnm, Volta).
- **Runtime persistant** : [`_PersistentMCPRuntime`](src/jarvis/tools/external/mcp_runtime.py:77) — un event-loop asyncio + une worker-task par serveur, files d'attente, retry sur `MCPServerSessionError`, `idle_timeout_sec` opt-in pour stateless. Clé : *sessions stdio long-lived* pour que Chrome (chrome-devtools-mcp) survive aux tool calls.
- **Refresh à la nouvelle conversation** : `refresh_mcp_tools()` dans le hot path de `run_reply_engine`.

### TTS (Piper / Chatterbox)
- **Factory** : [`create_tts_engine`](src/jarvis/output/tts.py) — sélection sur `cfg.tts_engine` (`"piper"` default, `"chatterbox"`).
- **Piper** : auto-download depuis `huggingface.co/rhasspy/piper-voices` (`en_GB-alan-medium` par défaut, ~60MB). Subprocess natif `piper`.
- **Chatterbox** : `chatterbox-tts==0.1.2`, supporte voice cloning (`tts_chatterbox_audio_prompt`).
- **Tracking** : `track_tts_start`/`activate_hot_window` couplés à `EchoDetector` pour ne pas s'écouter parler.

### Mémoire SQLite + FTS5 + auto-redaction
- **DB** : [`Database`](src/jarvis/memory/db.py:93) — SQLite `~/.local/share/jarvis/jarvis.db`, WAL, `synchronous=NORMAL`. Tables : `meals`, `conversation_summaries`. FTS5 virtual table `summaries_fts` (porter tokenizer) avec triggers `ai`/`ad`/`au`. Vecteurs : `vss0` virtual table optionnelle (`embeddings` FLOAT[768]) + `summary_vec`.
- **Auto-redaction** : [`utils/redact.py`](src/jarvis/utils/redact.py) — regex ordonnées (vendor-specific avant génériques) couvrant email, cartes, AWS/Stripe/GH/OpenAI/Google/JWT, headers Authorization, mots-clés password/secret/token/refresh_token/session, OTP. Appelée à chaque entrée user dans `run_reply_engine` ([`engine.py:798`](src/jarvis/reply/engine.py:798)) et avant écriture dans diary/graph.
- **Knowledge graph** : [`memory/graph.py`](src/jarvis/memory/graph.py) — auto-split par topic, branches `User`/`Directives`/`World`, mutation listeners. Migration legacy au boot du daemon.

---

## 3. Extension points propres

### Router LLM hybride (local/cloud)
- **Endroit unique** : [`src/jarvis/llm.py`](src/jarvis/llm.py). Toute la stack passe par `chat_with_messages` / `call_llm_direct` / `call_llm_streaming`. Brancher un router au niveau de ces trois fonctions (ou introduire un `LLMBackend` abstrait dans `llm.py`) propage automatiquement à tous les contextes (`docs/llm_contexts.md`).
- Le `Settings` (`config.py`) attend déjà `ollama_base_url` + `ollama_chat_model` + variants spécialisés (`intent_judge_model`, `tool_router_model`, `planner_model`). Ajouter `chat_model_provider` / `cloud_fallback_*` et router par classe de tâche (intent/router/planner/digest/main) est la voie naturelle.
- Garde-fou privacy : le router doit lire un flag explicite — la promesse 100 % local est un *core selling point* (`CLAUDE.md` : « Data privacy comes first, always »).

### Interface texte parallèle à la voix
- `run_reply_engine` est déjà découplé de l'audio : il prend `text: str` + `dialogue_memory` + `language`. Le `Settings` expose `cfg.use_stdin` (utilisé pour l'attribut `source_app` du diary, [`daemon.py:229`](src/jarvis/daemon.py:229)).
- **Endroit propre** : ajouter un thread serveur (Flask déjà dans `requirements.txt`) ou un mode CLI dans `daemon.py` qui pousse vers `_global_dialogue_memory` et appelle `run_reply_engine` directement. Pas besoin de toucher engine/planner/tools.
- Issue connue : [#35](https://github.com/isair/jarvis/issues/35) — « text chat interface » est sur la roadmap.

### Nouveaux MCP servers custom
- Voie utilisateur : ajouter un objet dans `cfg.mcps` (config.json) — `command`, `args`, `env`, optionnel `idle_timeout_sec`. `initialize_mcp_tools` les découvre au boot, `refresh_mcp_tools` à chaque nouvelle conversation.
- Voie développeur : un serveur MCP custom *en process* (sans subprocess stdio) demanderait d'étendre `MCPClient` ou de bypasser `_PersistentMCPRuntime`. Plus simple : packager le serveur en CLI MCP standard et l'ajouter à `cfg.mcps`.
- Pour étendre **builtin** tools plutôt que MCP : ajouter dans `tools/builtin/`, enregistrer dans `BUILTIN_TOOLS` ([`registry.py:30`](src/jarvis/tools/registry.py:30)), écrire un `*.spec.md` à côté (cf. `web_search.spec.md`, `log_meal.spec.md`).

---

## 4. Tests (`pytest`)

L'environnement local n'a que **Python 3.9** ; le projet utilise PEP 604 (`str | None`) en annotations *évaluées* (dataclass `Settings`) sans `from __future__ import annotations`, ce qui force Python 3.10+.

Résultat de `python3 -m pytest --collect-only -q` :
- **403 tests collectés** (sur ~73 fichiers `test_*.py`).
- **60 erreurs de collection** (toutes : `TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'`, à `config.py:73`).
- Tests *runnable* : aucun ne passe car presque toute la suite importe `jarvis.config` directement ou transitivement (l'erreur cascade).
- Coverage : **non mesurée** (échec de collection).

Pour exécuter la suite localement il faut installer un Python ≥ 3.10 (le projet vise une mamba env, `CLAUDE.md` cite `micromamba activate .mamba_env`).

---

## 5. Dépendances Python clés (`requirements.txt`)

| Catégorie | Paquets |
|---|---|
| LLM HTTP | `requests==2.32.3` (Ollama est externe, pas un paquet pip) |
| STT | `faster-whisper==1.0.3`, `mlx-whisper>=0.4.0` (Apple Silicon), `nvidia-cublas-cu12` + `nvidia-cudnn-cu12` (Windows GPU) |
| Audio | `sounddevice==0.4.7`, `webrtcvad==2.0.10` |
| TTS | `piper-tts>=1.3.0`, `chatterbox-tts==0.1.2`, `pygame>=2.1.0` |
| MCP | `mcp==1.13.1` |
| Web / scraping | `beautifulsoup4`, `lxml`, `html2text`, `playwright>=1.40.0` |
| OCR / images | `pytesseract==0.3.13`, `Pillow==10.4.0` |
| Hotkey | `pynput>=1.7.6` |
| Géo / réseau | `geoip2==4.8.0`, `miniupnpc==2.2.8` |
| Vecteurs | `faiss-cpu>=1.7.4`, `numpy<2.0.0` |
| Texte | `rapidfuzz==3.6.1` |
| Server | `flask>=3.0.0` (présent mais peu utilisé — disponible pour une interface texte) |
| Desktop | `PyQt6>=6.6.0`, `PyQt6-WebEngine`, `psutil`, `pyinstaller>=6.13.0` |
| Tests | `pytest==8.3.2`, `pytest-repeat==0.9.3` |
| Config | `python-dotenv==1.0.1` |

---

## 6. Flux principal voix → réponse (diagramme)

```mermaid
flowchart TD
    Mic[Microphone] -->|sounddevice| AudioQ[Audio queue]
    AudioQ --> VAD[webrtcvad VAD]
    VAD --> Whisper{Whisper backend}
    Whisper -->|Apple Silicon| MLX[mlx-whisper]
    Whisper -->|Win/Linux| FW[faster-whisper +/- CUDA]
    MLX --> Transcript[Transcript + lang]
    FW --> Transcript
    Transcript --> Buffer[TranscriptBuffer rolling 120s]
    Buffer --> Wake{wake word text-match?}
    Wake -->|non + hot window| Echo[EchoDetector early reject]
    Wake -->|oui| Echo
    Echo --> Judge[IntentJudge LLM gemma4 e2b]
    Judge -->|directed=true| Dispatch[_dispatch_query]
    Judge -->|directed=false| Drop[Drop]
    Dispatch --> Redact[utils.redact]
    Redact --> DM[DialogueMemory rolling cache]
    DM --> Planner[reply.planner sub-tasks]
    Planner --> Router[tools.selection LLM/keyword/embedding]
    Router --> Enrich[reply.enrichment diary + graph + digest]
    Enrich --> Engine[reply.engine agentic loop]
    Engine -->|tool_calls| Tools{tool target}
    Tools -->|builtin| Builtin[webSearch / weather / screenshot / ...]
    Tools -->|MCP| MCPRT[mcp_runtime persistent worker]
    MCPRT --> MCPSrv[MCP server subprocess stdio]
    Builtin --> Engine
    MCPSrv --> Engine
    Engine -->|final text| TTS[output.tts Piper or Chatterbox]
    TTS --> Speaker[Speakers]
    Engine --> DM
    DM -.->|periodic 60s + shutdown| Diary[update_diary_from_dialogue_memory]
    Diary --> DB[(SQLite + FTS5 + vss0)]
    Diary --> Graph[Knowledge graph User/Directives/World]
    Graph -.->|mutation| DM
```

---

## 7. Pièges et notes

- **Wake word** : le README mentionne « openWakeWord » dans le subtitle, mais le code ne contient ni Porcupine ni openWakeWord. La détection est texte-only sur les sorties Whisper (`wake_detection.py`). Tout travail audio-level wake-word reste à faire.
- **Spec files** : `CLAUDE.md` impose de chercher les `*.spec.md` avant tout changement. Liste maintenue dans `CLAUDE.md` §Spec File Registry. Le graphe des LLM contexts est dans `docs/llm_contexts.md` (à tenir synchro).
- **British English** + pas d'em dash dans les écrits user-facing (`CLAUDE.md`).
- **Tests** : exigent Python 3.10+. L'env local (3.9) ne peut rien exécuter sans mamba env.
- **Privacy** : redaction systématique à l'entrée, pas de cloud par défaut, GeoLite2 local, MCP stdio uniquement.

---

## UI Reconnaissance (Phase 0 - Orb)

### Desktop app PyQt6 — entry et windows
- **Entry** : `python -m desktop_app` → [src/desktop_app/__main__.py:6](src/desktop_app/__main__.py:6) → `desktop_app.main()` → `app.py::main()` ([app.py:2255](src/desktop_app/app.py:2255)) qui instancie `QApplication(sys.argv)` ([app.py:1250](src/desktop_app/app.py:1250)).
- **Architecture des windows** : tout est `QWidget` ou `QMainWindow` ad hoc, pas de framework custom. La `FaceWindow` ([face_widget.py:1046](src/desktop_app/face_widget.py:1046)) est un `QWidget` floating top-on `WindowStaysOnTopHint`, position right-side, contient un `LowPolyFaceWidget` peint via `QPainter` 30 FPS (`QTimer.start(33)` à [face_widget.py:238](src/desktop_app/face_widget.py:238)). La `SettingsWindow` ([settings_window.py](src/desktop_app/settings_window.py)) est auto-générée depuis config.
- **Memory Viewer est un Flask serveur, pas une fenêtre Qt** ([memory_viewer.py:23](src/desktop_app/memory_viewer.py:23)). Ouvert dans le browser système. Architecture différente, à ne pas confondre avec l'orb cible.
- **PyQt6 version** : `PyQt6>=6.6.0` + `PyQt6-WebEngine>=6.6.0` (requirements.txt). Note : `src/desktop_app/CLAUDE.md` impose l'usage de `themes.py` partagé.

### State bus daemon → UI — DÉJÀ EN PLACE
- Pas de hook à inventer : [face_widget.py:75-150](src/desktop_app/face_widget.py:75) contient `JarvisStateManager(QObject)`, singleton accédé via `get_jarvis_state()`.
- **Enum `JarvisState`** ([face_widget.py:54](src/desktop_app/face_widget.py:54)) : `ASLEEP`, `IDLE`, `LISTENING`, `THINKING`, `SPEAKING`, `DICTATING`, `DICTATION_PROCESSING`. Couvre 5/5 des états demandés (pas d'`ERROR` distinct — à ajouter si l'orb doit le visualiser, ou réutiliser ASLEEP avec un flag séparé).
- **Double mécanisme** : (a) signal Qt `state_changed.emit(value)` pour les consumers in-process, (b) file `/tmp/jarvis_state` ([face_widget.py:70](src/desktop_app/face_widget.py:70)) lue/écrite à chaque set pour cross-process (daemon subprocess ↔ desktop app).
- **Call sites du setter** déjà câblés : listener.py:459/473/2165, reply/engine.py:2139, output/tts.py:594/953, listener.py:1178. L'orb consomme via `get_jarvis_state().state_changed.connect(callback)` + polling `state_manager.state` au timer tick.

### Tap point audio INPUT (mic, avant Whisper)
- **Stream** : `sd.InputStream` ouvert à [listener.py:2045](src/jarvis/listening/listener.py:2045). Paramètres : `samplerate=cfg.sample_rate (default 16000)`, `channels=1`, `dtype="float32"`, `blocksize=frame_samples` (≈480 = 30ms à 16kHz). Fallback native rate si le device refuse 16k.
- **Callback** : `_on_audio(indata, frames, time_info, status)` à [listener.py:1422](src/jarvis/listening/listener.py:1422). `indata` = numpy float32 mono. Le chunk est push dans `_audio_q: queue.Queue(maxsize=64)`. Le callback est pauseable via `_should_stop` / `_dictation_active`.
- **Hook pour RMS 60 FPS** : option (a) recommandée — ajouter un side-effect non-bloquant dans `_on_audio` (sample 1 chunk sur ~2 pour 30→60 FPS, calcul `np.sqrt(np.mean(indata**2))`, push dans une `queue.Queue(maxsize=1)` consommée côté UI). Option (b) inacceptable : ouvrir un second `InputStream` parallèle prend le micro en exclusif sur certains drivers macOS.
- À 16 kHz le frame du callback est ~30ms. Pour 60 FPS animation, faire le RMS à chaque callback suffit largement (l'UI sample 16ms sur sa queue).

### Tap point audio OUTPUT (TTS playback)
- **Piper** (default) : `sd.OutputStream` callback à [tts.py:881](src/jarvis/output/tts.py:881), `audio_callback(outdata, frames, time_info, status)`. `outdata` = buffer envoyé aux speakers (int16 mono à `self._sample_rate`, typiquement 22050 Hz). C'est ÉCRIT par Jarvis, donc on a la donnée brute sortie. **Tap = ajouter un side-effect read-only sur `outdata` avant `return` du callback** : RMS, push dans queue, l'orb consomme.
- **Chatterbox** : passe par `pygame.mixer.music.play()` à [tts.py:546](src/jarvis/output/tts.py:546). pygame ne donne pas de callback par-frame → pas tappable au niveau buffer sans monkey-patch ou wrap. Pour Phase 1, traiter chatterbox comme "non accessible, à wrapper plus tard" (lookup table d'amplitude synthétique pilotée par l'état `SPEAKING` comme fait actuellement le mouth waveform).
- **Path simple Phase 1** : tapper Piper uniquement (default voice path), garder Chatterbox sur une enveloppe synthétique. 95% des users sont sur Piper.

### Deps à ajouter (requirements.txt)
- **Présents déjà** : `PyQt6>=6.6.0`, `PyQt6-WebEngine>=6.6.0`, `numpy<2.0.0`, `scipy>=1.x` (transitif via librosa, version installée 1.17.1).
- **À ajouter** : `moderngl>=5.10` (GL context Python pour shaders custom, indépendant de Qt's OpenGL), `pyrr>=0.10` (matrices 3D si l'orb a une perspective ou rotation — sinon numpy suffit). `scipy.fft` déjà couvert par scipy installé.
- **Optionnel** : `moderngl-window` si on veut un context standalone, mais l'orb sera dans une `QOpenGLWidget` PyQt6 (déjà fourni) → moderngl pur suffit.
- **Pas nécessaire** : `glm` (Python glm wrapper) — redondant avec pyrr/numpy. `pyglet`/`glfw` — Qt fournit déjà le context.

# Yuba sur un PC Windows

Fichier de travail, non suivi par git. Supprime-le quand tu n'en as plus besoin.

## 1. Récupérer le code

```powershell
git clone https://github.com/Hocsman/jarvis.git
cd jarvis
git checkout develop
```

C'est **ton fork**, pas `isair/jarvis` : c'est lui qui porte Yuba, l'orbe,
le noyau mémoire et tout le travail des deux derniers jours.

## 2. Créer l'environnement et lancer

Le script *utilise* micromamba, il ne l'installe pas : sans lui il bascule
sur `python -m venv`, où `webrtcvad` réclame Visual C++ Build Tools. Pose-le
d'abord, puis ouvre un nouveau terminal.

```powershell
winget install --id Mamba.Micromamba -e
```

`pwsh` est PowerShell 7, qui n'est pas livré avec Windows ; `powershell`
suffit.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_windows.ps1
```

Il crée `.mamba_env` en Python 3.12, installe PyAV depuis conda-forge puis
les `requirements.txt`.

⚠️ **Il lance `jarvis.daemon`, pas l'application de bureau.** Tu auras la
voix en console, sans icône de barre des tâches, sans orbe, sans écran de
démarrage. Pour l'app complète, une fois l'environnement créé :

```powershell
$env:PYTHONPATH = "src"
.\.mamba_env\python.exe -m desktop_app
```

## 3. Le GPU, si le PC a une carte NVIDIA

```powershell
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

Détecté tout seul, rien à configurer. MLX est propre à Apple Silicon :
sur Windows c'est `faster-whisper` qui prend le relais, et sur un vrai
GPU `large-v3-turbo` sera nettement plus rapide que sur le MacBook Air —
sans la contention avec Kokoro et le modèle local.

## 4. La clé d'API

```powershell
setx OPENROUTER_API_KEY "sk-or-v1-..."
```

Ouvre un nouveau terminal après : `setx` n'affecte pas la session en
cours.

## 5. La config

Elle vit dans `%USERPROFILE%\.config\jarvis\config.json`. Ni
`default_config_path()` ni `_default_db_path()` n'ont de branche Windows :
ils résolvent `Path.home()` partout. `%LOCALAPPDATA%\Jarvis` ne sert qu'aux
journaux et au verrou d'instance, jamais à la mémoire. **Ne recopie pas celle
du Mac** : elle contient des chemins macOS. Pars d'une config neuve et
remets seulement ce qui compte :

```json
{
  "llm_provider": "openai_compatible",
  "llm_base_url": "https://openrouter.ai/api/v1",
  "llm_api_key_env": "OPENROUTER_API_KEY",
  "llm_chat_model": "deepseek/deepseek-v4-flash",
  "intent_judge_model": "openai/gpt-oss-120b",
  "tool_router_model": "openai/gpt-oss-120b",
  "embedding_provider": "ollama",
  "ollama_chat_model": "gemma4:e2b",
  "whisper_model": "large-v3-turbo",
  "wake_word": "<ton mot d'éveil>",
  "wake_aliases": ["<les variantes que Whisper entend>"],
  "response_language": "français",
  "weather_city": "<ta ville>",
  "planner_enabled": false,
  "auto_redact_before_cloud": true,
  "llm_extra_body": {"provider": {"sort": "throughput"}}
}
```

⚠️ **Ce bloc est un point de départ, pas un inventaire.** La config du
Mac fait foi : ouvre-la et relève ce qui y est vraiment. Une clé que le
code ne connaît pas est conservée dans le fichier et **ignorée en
silence** — ce modèle contenait `fast_model`, recopié d'une vraie config
sans vérifier que quoi que ce soit le lisait. Rien ne l'aurait signalé.

Les clés de modèles que le code reconnaît sont `llm_chat_model`,
`intent_judge_model`, `tool_router_model`, `planner_model`,
`evaluator_model`, `appris_model`, `confirmation_model`,
`reminder_model`, `embedding_model`, `ollama_chat_model`,
`ollama_embed_model` et `whisper_model`. Celles qu'on ne met pas
retombent sur leurs valeurs par défaut.

**À laisser tomber au premier lancement** : tout ce qui commence par
`tts_piper_` (chemins macOS), `location_ip_address` (l'IP d'ici), et
`mcps` (les serveurs MCP se relancent via `npx`, à remettre une fois le
reste vérifié).

**`tts_engine`** : laisse le défaut d'abord. Kokoro est du PyTorch et
devrait tourner, mais je ne l'ai jamais vu sur Windows. Piper est le
repli sûr.

## 6. Sa mémoire — l'étape qu'on oublie

Le dépôt ne contient **pas** ce que Yuba sait de toi. C'est délibéré :
`profil.md`, `regles.md`, le graphe et le journal sont à toi, pas au
code.

Copie :

```
  depuis   ~/.local/share/jarvis/
  vers     %USERPROFILE%\.local\share\jarvis\
```

Ce qui compte là-dedans :

| | |
|---|---|
| `yuba/profil.md`, `yuba/regles.md` | ce qu'elle sait de toi, en clair |
| `yuba/outils.md`, `objectifs.md`, `appris.md` | la porte des outils, les objectifs, les propositions |
| `jarvis.db` | le journal, le graphe, les rappels, le registre |

Sans ça tu auras une Yuba fonctionnelle mais amnésique.

⚠️ **Ces fichiers sont personnels.** Passe-les par une clé USB ou un
canal que tu contrôles, pas par un service tiers.

## 7. Vérifier

```powershell
.\.mamba_env\python.exe -m pytest tests\ -q
```

Autour de 3900 tests, dont 4 sautés : deux qui réclament POSIX (`time.tzset`
pour décaler la zone de l'horloge, et le script d'installation macOS, qui a
besoin du séparateur de PATH et du bit d'exécution d'un vrai Unix), plus deux
sauts de chemin Unix préexistants. Tout le reste doit passer. Un échec est un
vrai signal Windows.

---

## Ce que je n'ai pas vérifié

Tout le travail des deux derniers jours a été mesuré sur le Mac.

- **Kokoro sur Windows** : jamais essayé.
- **Les trois gardes d'écho** ont été calibrées sur *tes* enregistrements,
  avec *ta* clim et *ton* haut-parleur. Un autre micro, une autre pièce,
  d'autres seuils — en particulier `energy_spike_threshold`, qui compare
  l'énergie de ta voix au plancher ambiant.
- **La dictée** est désactivée sur macOS 26+ (incompatibilité `pynput`).
  Sur Windows elle devrait fonctionner, ce qui te rendra le raccourci
  `ctrl+alt`.

Le README le dit aussi : le développement se fait surtout sur macOS et
Windows peut être en retard.

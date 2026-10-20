# 🛠️ Projets — handing a project to Claude Code

Yuba can talk about the user's code, but a small voice model cannot safely change it. Claude Code can. This feature lets her hand a task on one of the user's project folders to Claude Code, which works on its own in that folder, and tell the user what it did once it has finished.

Implemented in [claude_code.py](claude_code.py) (finding the project, running the agent, keeping the job) and [claude_code.py](../tools/builtin/claude_code.py) under `tools/builtin` (the `claudeCode` tool). Wired into the daemon, which delivers the report.

## What leaves the machine

**This is the one feature whose whole point is to send the user's code to a cloud.** Claude Code runs on Anthropic's models: the task, and whatever files of the project it decides to read, go to Anthropic under the user's own Claude Code account. Nothing about this is hidden or optional once the feature is on, so it is off until the user names a folder (`claude_code_projects_dir`), and the README says so beside the setting.

What does not leave: Yuba's own store. The core directory (`yuba/`) is never inside a project handed over: a project whose folder contains it is refused, for the reason `localFiles` refuses it (`src/jarvis/tools/policy.spec.md`, *The file is not reachable by tool*). Claude Code's own edits are confined to the project folder by Claude Code itself.

## The projects

The direct sub-folders of `claude_code_projects_dir`, read off the disk on every call, so a project created this morning is reachable without a restart. Hidden folders (a leading `.`) are not projects.

The user names a project by voice, so the name arrives transcribed: "nikah score" for `nikahscore`, "France carte grise" for `france-carte-grise`. Matching is therefore on a **normalised** form, the name case-folded with every non-word character and underscore removed (`\W` under `re.UNICODE`, so it holds in every script), and runs in two steps:

1. Exactly one folder whose normalised name equals the normalised request: that one.
2. Otherwise the close matches (`difflib`, ratio at least `_MATCH_CUTOFF`, 0.8): exactly one is that one.

Two or more matches is **ambiguous** and nothing runs: the reply names the candidates so the user can say which. No match is **unknown**, and the reply lists the projects that exist. Guessing between two projects would hand the wrong code to an agent that edits it.

A folder that resolves outside the root (a link pointing elsewhere) is not a project.

**When `projet` names nothing, the task is searched.** Small models fold the project into the task ("add a logout button to chantipay") and leave `projet` empty, or put something else in it: measured on `gemma4:e2b` with 20 interleaved calls per schema, asking for the request verbatim left `projet` right 8 times in 20, and searching the arguments as below brings that schema to 16 and the current one from 17 to 18. So when `projet` is empty or matches no folder, `mentioned_project` looks for a project's normalised name (at least `_MIN_MENTION`, 4, characters) inside the normalised text of the call's arguments; a name contained in a longer match gives way to it ("relayer" inside "relayer main"), one match is an answer and two are a question. Only the arguments are searched, never the sentence the user said: the card shows the arguments, and what the user approves must name the project that runs.

## The tool

`claudeCode`, three properties, all French like the other tools of this fork:

| Property | Meaning |
|----------|---------|
| `projet` | The project as the user named it |
| `tache` | What Claude Code must do, as a complete instruction in the user's language. Empty means "how is it going?" |
| `modifier` | `true` when the task changes files, `false` when it only asks a question, an explanation or a review |

**Risk follows the arguments** (`Tool.risk_for`), the way `localFiles` does it:

| Call | Risk | Why |
|------|------|-----|
| no `tache` | `lecture` | Reports on the job; touches nothing |
| `modifier` explicitly false | `lecture` | Claude Code runs with permission to read only |
| anything else, `modifier` absent included | `destructif` | Claude Code edits files in the user's project |

So with the default verdicts a task that changes code costs a click on a card showing the exact project and task, and a question runs straight away. A user who wants to be asked every time writes `- claudeCode` under `## Demande` in `outils.md`; the card then appears for questions too, and the spoken door is open for them because their risk is declared.

Small models do not reliably tell the two apart: on `gemma4:e2b`, no wording of the flag (this boolean, an enum `lecture`/`ecriture`, a predicate name) got a request to explain code filed as a question, in 20 calls each; the model writes a noun from the sentence into whatever the field is called. That is tolerable only because a wrong `modifier` fails safe in both directions: a change filed as a question runs read-only and comes back as an explanation of what it would do; a question filed as a change asks first.

A `tache` whose project cannot be found anywhere starts nothing. The reply says how the last job went (a small model files "how is Claude Code doing?" as a task with no project) and lists the projects so she can ask which one; the call fails, so the tool stays offered on the next turn, when the answer comes. Its risk is still read from `modifier`, because the project may be inside the task.

**One job at a time.** A second task while one is running is refused, and the refusal says which project is being worked on and for how long. Two agents editing at once, perhaps the same project, is not a speed-up.

**A routine never reaches it** (`JAMAIS_EN_ROUTINE`, `src/jarvis/routines/routines.spec.md`): the report is delivered aloud to someone who has to be in the room, and a routine runs while nobody is.

## Running Claude Code

The executable is `claude_code_command` (default `claude`), looked up on `PATH`, then at the native installer's location (`~/.local/bin`), which a process started from the desktop may not have on its `PATH`. Not found, the tool says Claude Code is not installed and nothing runs.

```
claude -p --output-format json --permission-mode <mode> --append-system-prompt <report rule> [--resume <session>]
```

- **The task goes in on standard input, never on the command line.** On Windows an npm install is a `.cmd` file, which `cmd.exe` parses: a task containing `&` or `|` would be run as a second command. The command line carries only fixed flags, a session id that has been checked against `_SESSION_RE`, and the report rule, which contains none of the characters `cmd.exe` treats specially.
- **Mode**: `acceptEdits` for a change, `default` for a question. Headless, nothing can answer a permission prompt, so in `default` every tool that needs one (editing, shell commands) is denied and Claude Code can only read; in `acceptEdits` it can edit files and the shell stays denied.
- **The report rule** (`_REPORT_RULE`) is appended to Claude Code's system prompt: its final message is read aloud by a voice assistant to someone who is not watching, so it ends with two or three plain sentences in the language of the task, no markdown and no code; and since nobody can answer a question mid-run, anything that needs the user goes into that report instead.
- **Follow-ups continue the same session.** The session id of the last job on a project is kept in memory for `_RESUME_WINDOW_SEC` (30 minutes) after it finished, and a new task on that project within the window resumes it, so "and add a test for it" knows what "it" is. Never on disk: a restart starts fresh.
- **Bounded by `claude_code_timeout_sec`** (1800 s, clamped 60 to 7200). Past it the whole process tree is killed (`taskkill /T` on Windows, the process group elsewhere), because killing only `cmd.exe` leaves the agent running.
- **Stopped with the daemon.** A job still running at shutdown is killed: nothing would be left to deliver its report.
- No console window opens on Windows.

The result is the JSON object Claude Code prints: success when the exit code is 0, `is_error` is false and `result` is non-empty. Anything else is a failure, whose reason is `result`, or failing that the tail of standard error. Neither is ever invented: a job with nothing to say reports that it said nothing.

## Delivering the report

The tool returns at once, saying the job has started: the work takes minutes, and the reply engine runs on the listener's thread while holding the query lock, so waiting would leave Yuba deaf for the whole job. The reply she speaks says it has started, never that it is done.

When the job ends, the daemon delivers one message, authored by code around Claude Code's report:

- success: `Claude Code a fini sur <projet>. <report>`
- failure: `Claude Code n'a pas pu finir sur <projet> : <reason>`

The report goes through `scrub_secrets` first: it is spoken aloud and kept in the conversation, and a key Claude Code wrote into a `.env` must not be read out.

- It waits until no query is running, like a reminder: cutting across a reply the user is waiting for is worse than a few seconds late.
- It is added to the shared dialogue memory as an assistant message, so a follow-up ("what did it change?") has it in context.
- It reaches the chat window as a `delegation_done` event (bundled mode through the `on_delegation_done` callback, subprocess mode on the stdout bus), shown as an assistant message.
- It is spoken unless the task came from the chat window, which never speaks.

Asking how it is going (`tache` empty) answers from memory: the job running, with its project and elapsed time; or the last job's outcome and report; or that there is none.

## The ledger

The gate writes the `claudeCode` call like any other. The job adds its own bracket, as a routine run does: a `démarré` row when the process starts and an `ok` or `échec` row when it ends, both under `claudeCode:<projet>`, sharing the job's id as `request_id`, the second carrying the real duration. The call row says the job was launched; only the bracket says whether the code was actually changed, and an unclosed `démarré` is the trace of a job the process took down with it. As everywhere, no output is recorded.

## Settings

| Key | Default | What it decides |
|-----|---------|-----------------|
| `claude_code_projects_dir` | `""` | The folder whose sub-folders can be handed over. Empty = off |
| `claude_code_command` | `claude` | The Claude Code executable, a name on `PATH` or a full path |
| `claude_code_timeout_sec` | `1800.0` | How long a job may run before it is killed |

# 🛠️ Projets — handing a project to Claude Code

Yuba can talk about the user's code, but a small voice model cannot safely change it. Claude Code can. This feature lets her hand a task on one of the user's project folders to Claude Code, which works on its own in that folder, and tell the user what it did once it has finished.

Implemented in [delegation.py](delegation.py) (finding the project, running the agent, keeping the job) and [claude_code.py](../tools/builtin/claude_code.py) (the `claudeCode` tool). Wired into the daemon, which opens the job keeper with the delivery it uses and closes it when it stops.

## What leaves the machine

**This is the one feature whose whole point is to send the user's code to a cloud.** Claude Code runs on Anthropic's models: the task, and whatever files of the project it decides to read, go to Anthropic under the user's own Claude Code account. Nothing about this is hidden or optional once the feature is on, so it is off until the user names a folder (`claude_code_projects_dir`), and the README says so beside the setting. Off, the tool is not even shown to the router or the model (`Tool.offered`); a call that arrives anyway says it is off.

What does not leave: Yuba's own store. A project whose folder holds the core directory (`yuba/`), or lies inside it, is refused, for the reason `localFiles` refuses it (`src/jarvis/tools/policy.spec.md`, *The file is not reachable by tool*). Claude Code's own edits are confined to the project folder by Claude Code itself.

## The projects

The direct sub-folders of `claude_code_projects_dir`, read off the disk on every call, so a project created this morning is reachable without a restart. Hidden folders (a leading `.`) are not projects, and neither is a folder that resolves outside the root (a link pointing elsewhere).

Names are compared in a **normalised** form: case-folded, with every non-word character and underscore removed (`\W` under `re.UNICODE`, so it holds in every script). Three ways to find one:

- **`resolve_project`**, from the `projet` field. The user names a project by voice, so the name arrives transcribed: "nikah score" for `nikahscore`, "France carte grise" for `france-carte-grise`. A folder whose normalised name equals the normalised request is an **exact** match. Otherwise the close matches (`difflib`, ratio at least `_MATCH_CUTOFF`, 0.8) are tried, and one is a match that is not exact.
- **`mentioned_project`**, inside free text. Small models fold the project into the task ("add a logout button to chantipay") and leave `projet` empty, or write the name among other words. A project is mentioned when a run of whole words of one text (up to `_MAX_MENTION_WORDS`, 5) has its normalised name, which must be at least `_MIN_MENTION` (4) characters. Never part of a word ("core" is not in "score"), never assembled across two arguments, and a name inside a longer matching one gives way to it ("relayer" inside "relayer main"). A mention is never exact.
- **`choose_project`**, what a call designates. A non-empty `projet` is resolved, and if it names no folder it is read as free text in case it carries the name among other words. It is never replaced by a folder some other argument mentions: the card shows `projet`, and a folder the task happens to name must not silently take its place. With `projet` empty, the call's other arguments are searched.

In each, two or more matches is **ambiguous** and nothing runs: the reply names the candidates so the user can say which. No match is **unknown**, and the reply lists the projects that exist. Only the call's arguments are ever searched, never the sentence the user said: the card shows the arguments, and what the user approves must name the project that runs.

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
| no `tache` | `lecture` | Reports on the job from memory; touches nothing |
| `modifier` explicitly false (`false`, `0`, as JSON or as a string) | `lecture` | Claude Code is denied every tool that writes or executes |
| anything else, `modifier` absent or unreadable included | `destructif` | Claude Code edits files in the user's project |

So with the default verdicts a task that changes code costs a click on a card showing the exact project and task, and a question runs straight away. **The generated `outils.md` leaves `claudeCode` out** (`Tool.classified_per_call`): the generator classifies a tool from its empty call, which here is the harmless one, so it would write the tool under `## Libre`, and a line there frees every call, a change included. Left out, each call falls back to the default for its own risk. A user who wants to be asked every time writes `- claudeCode` under `## Demande`; the card then appears for questions too, and the spoken door is open for them because their risk is declared.

**A change runs only on a project its `projet` names exactly.** `vet_call` makes the decision. A change on a near miss ("chantipai") or on a project named only inside the task is sent back with the exact name to use ("call the tool again with projet='chantipay'"), so the next card names the folder that will be edited. A question may run on a near miss or a mention: it edits nothing.

Small models do not reliably tell a question from a change; on `gemma4:e2b` a request to explain code is usually filed as a change, whatever the field is called. That is tolerable only because a wrong `modifier` fails safe in both directions: a change filed as a question runs read-only and comes back as an explanation of what it would do; a question filed as a change asks first.

A `tache` whose project cannot be found, because `projet` is empty or names no folder and nothing else names one, starts nothing. The reply says how the last job went (a small model files "how is Claude Code doing?" as a task with no project) and lists the projects so she can ask which one; the call fails, so the tool stays offered on the next turn, when the answer comes.

**One job at a time.** A second task while one is running is refused, and the refusal says which project is being worked on and for how long. Two agents editing at once, perhaps the same project, is not a speed-up. A running job cannot be cancelled except by stopping Yuba.

**A routine never reaches it** (`JAMAIS_EN_ROUTINE`, `src/jarvis/routines/routines.spec.md`): the report is delivered aloud to someone who has to be in the room, and a routine runs while nobody is.

## Running Claude Code

The executable is `claude_code_command` (default `claude`), looked up on `PATH`, then at the native installer's location (`~/.local/bin`), which a process started from the desktop may not have on its `PATH`. Not found, the tool says Claude Code is not installed and nothing runs.

```
claude -p --output-format json --permission-mode <mode> --disallowedTools <tools> --strict-mcp-config --append-system-prompt <report rule> [--resume <session>]
```

- **The task goes in on standard input, never on the command line.** On Windows an npm install is a `.cmd` file, which `cmd.exe` parses: a task containing `&` or `|` would be run as a second command. The command line carries only fixed flags, a session id of the shape `_SESSION_RE` accepts (a UUID, which can never be read as a flag), and the report rule, none of which contains a character `cmd.exe` treats specially.
- **What Claude Code may use.** Headless, nothing can answer a permission prompt, but Claude Code still honours the allow rules in the user's and the project's own settings (a project may allow `git commit` or `docker compose`). Deny rules outrank them, so every job is given some: a question (`--permission-mode default`) is denied `_READ_ONLY_DENIED`, every tool that edits, runs a command or fetches from the web; a change (`--permission-mode acceptEdits`) is denied `_CHANGE_DENIED`, the shell. `--strict-mcp-config` with no configuration loads no MCP server. What the project's committed settings can still do is run hooks, as they would for the user running Claude Code in that folder.
- **The project is never searched for executables.** An npm `.cmd` runs `node` by bare name, and `cmd.exe` looks in the current directory, the project, before `PATH`; the child runs with `NoDefaultCurrentDirectoryInExePath` set, so a repository's own `node.exe` does not run.
- **The report rule** (`_REPORT_RULE`) is appended to Claude Code's system prompt: its final message is read aloud by a voice assistant to someone who is not watching, so it ends with two or three plain sentences in the language of the task, no markdown and no code; and since nobody can answer a question mid-run, anything that needs the user goes into that report instead.
- **Follow-ups continue the same session.** The session id of the last job on a project is kept in memory for `_RESUME_WINDOW_SEC` (30 minutes) after it finished, and a new task on that project within the window resumes it, so "and add a test for it" knows what "it" is. A change resumes only a session whose last job could edit, which a card approved: a question runs without one, and its conversation must not ride into a change the user approved without seeing it. Never on disk, and opening the keeper clears it, so a restart starts fresh, including a daemon restarted inside the desktop's own process.
- **Bounded by `claude_code_timeout_sec`** (1800 s, clamped 60 to 7200). Past it the whole process tree is killed (`taskkill /T` on Windows, the process group elsewhere), because killing only `cmd.exe` leaves the agent running.
- **Stopped with the daemon.** Closing the keeper refuses anything new, so a turn still in flight cannot start a job that outlives the daemon, and kills a job still running: nothing would be left to deliver its report. On Windows the job's process tree is also in a job object that closes with the daemon's process, so a daemon killed rather than stopped takes the job with it. Elsewhere the job runs in its own process group, and a daemon killed outright leaves it to finish or reach its deadline unobserved.
- No console window opens on Windows.

The result is the JSON object Claude Code prints, or the last `result` entry of an array, or failing that the last line of output that parses. Success is an exit code of 0, `is_error` false, a `subtype` of `success` (or none) and a non-empty `result`. Anything else is a failure, whose reason is `result`, or failing that the tail of standard error (`_STDERR_TAIL`, 300 characters). A job past its deadline and a job that could not be launched (the folder gone, the executable unusable) get a reason written by code. None is ever invented: a job with nothing to say reports that it said nothing.

The report is then cleaned once, before anything else sees it: scrubbed of secrets (it is spoken aloud and kept in the conversation, and a key Claude Code wrote into a `.env` must not be read out), and cut at the last sentence that fits in `_REPORT_MAX_CHARS` (600). The rule asks for three sentences; the cap is what holds if it is not obeyed, since a long report would hold the speaker for minutes and ride in every prompt for the conversation's lifetime.

## Delivering the report

The tool returns at once, saying the job has started: the work takes minutes, and the reply engine runs on the listener's thread while holding the query lock, so waiting would leave Yuba deaf for the whole job. The reply she speaks says it has started, never that it is done.

When the job ends, the daemon delivers one message, written by code around Claude Code's report, which is quoted as its words:

- success: `Claude Code a fini sur <projet> : « <report> »`
- failure: `Claude Code n'a pas pu finir sur <projet> : <reason>`

The frame is French, like every sentence this fork's code speaks on its own (a reminder, a confirmation question); the report inside it is in the language of the task.

- It waits until no query is running, like a reminder: cutting across a reply the user is waiting for is worse than a few seconds late. The lock is inspected, never taken, so a query starting in the instant between the check and the delivery can still precede it.
- It is added to the shared dialogue memory as an assistant message, so a follow-up ("what did it change?") has it in context.
- It reaches the chat window as a `delegation_done` event (bundled mode through the `on_delegation_done` callback, subprocess mode on the stdout bus), shown as an assistant message. A window not yet shown, whose first show will seed it from the dialogue memory that already holds the report, does not append it on arrival.
- It reaches the dashboard's transcript on the same event, since a task typed there is never spoken.
- It is spoken unless the task came from a window, which never speaks.
- It is not delivered once Yuba is stopping.

Asking how it is going (`tache` empty) answers from memory: the job running, with its project and elapsed time; or the last job's outcome and report; or that there is none.

The tool's replies are written by code, bounded, and carry project names the user must repeat word for word, so they skip the tool-result digest (`_DIGEST_SKIP_TOOLS`).

## The ledger

The gate writes the `claudeCode` call like any other. The job adds its own bracket, as a routine run does: a `démarré` row when the job is launched and an `ok` or `échec` row when it ends, both under `claudeCode:<projet>`, sharing the job's id as `request_id`, with the verdict read once at launch, the second carrying the real duration. A job whose thread cannot start closes its bracket at once with `échec`. The call row says the job was launched; only the bracket says whether the code was actually changed, and an unclosed `démarré` is the trace of a job the process took down with it. The Activity tab shows bracket rows without the "asked" badge: they share an id to pair, not because anybody was asked. As everywhere, no output is recorded.

## Settings

| Key | Default | What it decides |
|-----|---------|-----------------|
| `claude_code_projects_dir` | `""` | The folder whose sub-folders can be handed over. Empty = off |
| `claude_code_command` | `claude` | The Claude Code executable, a name on `PATH` or a full path. Empty = `claude` |
| `claude_code_timeout_sec` | `1800.0` | How long a job may run before it is killed |

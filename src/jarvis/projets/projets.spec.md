# 🛠️ Projets — handing a project to Claude Code

Yuba can talk about the user's code, but a small voice model cannot safely change it. Claude Code can. This feature lets her hand a task on one of the user's project folders to Claude Code, which works on its own in that folder, and tell the user what it did once it has finished.

Implemented in [delegation.py](delegation.py) (finding the project, running the agent, keeping the job), [claude_code.py](../tools/builtin/claude_code.py) (the `claudeCode` tool), [stop_claude_code.py](../tools/builtin/stop_claude_code.py) (the `stopClaudeCode` tool) and [prompt.py](prompt.py) (what the system prompt says she can do). Wired into the daemon, which opens the job keeper with the delivery it uses and closes it when it stops.

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

**Its description opens with what the user wants done**: "Claude Code does what the user wants done in one of their code projects (fix, add, explain or review its code) and says how it went, and answers when the user asks how Claude Code's work is going: a coding agent ...". The planner reads the first 120 characters of a description; the router reads whole sentences within 200 characters, which for this description is its first sentence alone (279 characters). The opening is for the planner. A small planner restates a delegation as "the user wants a bug fixed in a project"; when no tool in its catalogue holds that phrase, it plans the restatement as a `remember` step, and direct execution writes it into the core as a standing entry about the user. Small changes to those 120 characters move the planner (row 3), so they are kept as measured. The rest of the first sentence is for the router: the progress clause, with "Claude Code's work" rather than "that work", is what gets "où en est Claude Code ?" offered the tool (row 7).

Measured on `gemma4:e2b`, local, with the sentences of `evals/test_claude_code_reaches_the_right_tool.py`. Rows 1 to 6 interleave their two arms call by call. Rows 7 and 8 come from separate router runs, each interleaved with the description that opens "Hand a task on one of the user's code projects to Claude Code" as a control, which was offered 20 in 20 on the progress question and 0 in 20 on the product question in every run.

| What | Setup | This description | Alternative |
|------|-------|------------------|-------------|
| Change requests planned as a `remember` step | Planner only; the three `CHANGES`; 10 rounds of each of the six orders the mandatory tools can take | 81 in 180 | 110 in 180 with the "Hand a task" opening (p = 0.003) |
| Change requests planned as `claudeCode` | The same 180 calls | 140 in 180 | 93 in 180 (p = 3e-7) |
| Change requests and the explain question planned as a `remember` step | Planner only; mandatory tools in the order `stop`, `forget`, `remember`; the three `CHANGES` and the first of `QUESTIONS`; 20 rounds | 5 in 80 | 21 in 80 with the 120 characters ending ", and answ" instead of " and says " (p = 0.001; 14 of the 21 on one sentence) |
| Change requests reaching `remember` | End to end, tools stubbed; the three `CHANGES`; 7 rounds; mandatory tools in the order `stop`, `forget`, `remember` | 2 in 21 | 17 in 21 with the "Hand a task" opening and the order drawn at random per call (p = 6e-6) |
| Change requests reaching `claudeCode` | The same 21 calls | 21 in 21 | 18 in 21 (p = 0.23) |
| Explicit requests to remember reaching `remember` | The same session as the two rows above; the four positives of `evals/test_remember_tool_is_called.py` | 14 in 28 | 14 in 28 (p = 1.0) |
| "où en est Claude Code ?" offered the tool | Router only; 20 calls | 20 in 20 | 10 in 20 with the same clause saying "that work" (one-word contrast, head to head, p = 0.0004); 0 in 20 with "and says how that work is going" |
| "c'est quoi la différence entre Claude Code et Cursor ?" offered the tool | Router only; 20 calls | 0 in 20 | 5, 3 and 0 in 20 for three wordings that put the progress clause in a sentence of its own (p = 0.05, 0.23, 1.0) |

What is left still happens at runtime: about one change request in ten reaches `remember` on `gemma4:e2b`, nothing asks first, and the entry stays in the core until the user strikes it out. Both writes left in row 4 were on the English sentence and were not the restatement the opening addresses: a step planned before the delegation ("remember that Jarvis needs to use the Claude Code tool", "remember that Jarvis is an assistant"). In `evals/test_claude_code_reaches_the_right_tool.py`, `test_a_task_writes_nothing_into_the_core` fails when it happens and is left failing rather than marked expected, and `TestTheRouterOffersIt` holds rows 7 and 8.

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

**One job at a time.** A second task while one is running is refused, and the refusal says which project is being worked on and for how long. Two agents editing at once, perhaps the same project, is not a speed-up. A running job can be stopped early (*Stopping a job*).

**A routine never reaches it** (`JAMAIS_EN_ROUTINE`, `src/jarvis/routines/routines.spec.md`): the report is delivered aloud to someone who has to be in the room, and a routine runs while nobody is.

## What she knows she can do

Asked "can you work on my projects?", she must know the answer is yes. Two things stand in the way:

- **The router reads the question as conversation.** To the tool router's small model, a question about what she can do is small talk, so it offers no tool; and on a catalogue with a few MCP servers it rarely offers `claudeCode` for a request on a project either (numbers below). The chat model is then shown only what every attended turn carries: `stop`, `forget`, `remember` and `toolSearchTool`.
- **The chat model answers from its prior**, which is that an assistant has no access to anyone's files. Nothing in the prompt says otherwise. `toolSearchTool` asks to be called before refusing an action, yet without the block she refused 17 requests in 20 without searching, and said she could work on the projects 6 times in 20, never when asked in French (first table below). When she does search, the query she writes describes the task, and the router can answer it with other tools.

So while `claudeCode` is offered to a chat model that searches, the system prompt says what she can do (`format_projets_block` in [prompt.py](prompt.py), in the lasting head right after the model-size components):

- the projects folder on the user's machine is open to her through Claude Code, a coding agent she hands a task to with `claudeCode`, which reads and changes the code and whose report reaches the user when it is done; so she can read, explain, review, fix and change the projects, and the user need not paste any code. It is worded as the very thing the denial says she lacks (CLAUDE.md, denial-template mirroring);
- asked whether she can reach or work on the code or the projects, she says she can, through Claude Code, and asks which project and what to do;
- asked for work on a project, she hands it over with `claudeCode`, and when the tool is not among her tools this turn, she first calls `toolSearchTool` with the query `"claudeCode"`. A search for a tool's exact name finds it whatever the router answers, and a search made only of the name is answered without asking the router (`src/jarvis/tools/builtin/tool_search.spec.md`, Name cue), so this path does not depend on the router having understood anything;
- work on a project always goes to Claude Code, whatever other tools she has this turn, and through it her tools do reach the code: she never changes the user's files or repositories with another tool, even one offered this turn, and never asks where the code is or which file to change, since Claude Code finds it in the project's folder. These last sentences mirror the ways a request missed Claude Code when the router offered a code host's write tool: she said her tools could not reach the code, she asked where the code was or which file to change, or she called the write tool.

It says nothing of a request found in a page, an email or a tool's result, of the user's other files, or of a project name two folders match. A question job runs without a card, so a request planted in a page that she handed over would send a project to Anthropic unasked; "never change the user's files with another tool" could keep her from a note the user asks for in a file of their documents; and "never ask where the code is" could keep her from asking which project is meant. With the block as it is, none of the three happens, and a wording that says all three does no better (second table below), so the block keeps its length.

The block is said whatever the router offered, since the router's answer is what it makes up for. It is a constant: no project name, no path, and no tool but `claudeCode` and `toolSearchTool`. It rides in every attended prompt, to whichever provider runs the chat model, and the projects are listed by the tool in the one turn that needs them. It is said only when it is true, and only where it cannot cost a second card:

- with no projects folder named, the tool is not offered and neither the block nor the tool's name is in the prompt;
- with a folder named that does not exist, or `claudeCode` filed under `## Jamais` in `outils.md` (both of its risks resolving to `jamais`), every call would be refused, so the block is left out (`withheld_because` in [prompt.py](prompt.py)); the tool itself stays offered, and says why it cannot act;
- with a small chat model (`detect_model_size` answering `SMALL`, the default `gemma4:e2b` included), the block is left out too (`withheld_because`): such a model does not call `toolSearchTool` for a tool its router left out, so the way to the tool the block gives is not one it takes. Told the projects are open to it, it says it can, and on a request promises the work without handing it over (numbers below). The tool stays offered, and reaches the model when the router offers it;
- a routine's turn never carries it: a routine cannot reach the tool;
- the turn that runs a call the user approved never carries it: that turn replays the original request, and told to hand the work over, she would propose the call again and raise a second card for one action.

Measured end to end by `evals/test_projects_question_finds_claude_code.py`, with `claude-haiku-5-5` as the chat model (planner on) unless a table says otherwise. Each kind of sentence meets one fixed answer from the pre-loop router's model: `none` as in the field, the write tool below, or `localFiles`. `toolSearchTool` runs for real with `qwen2.5:3b` as its router, over the 23 builtins offered with a projects folder named and 90 invented MCP tools: 89 spread over fifteen invented servers, and one write tool shaped like a code host's, `repo__create_or_update_file` (create or update a single file in a remote Git repository: owner, repository, path, content, commit message, branch). `claudeCode` runs for real up to the job, which is not started, and no gate runs. Each kind has four invented sentences in four languages. The two arms of a table alternate call by call, the first arm rotating, for five rounds, so 20 calls an arm a kind; p is Fisher's exact test, two-sided. Each table is one run of the class it names unless it says otherwise, and a judge on the configured chat model reads the replies where a kind needs one.

No block against the block (`TestBeforeAndAfter`), the router offering nothing:

| Kind | Without the block | With the block |
|------|-------------------|----------------|
| Asked whether she can work on the projects: no job, and a judge reads the reply as saying she can | 6 in 20 | 20 in 20 (p = 3e-6) |
| A request for work on a named project: a job starts on that project | 2 in 20 | 20 in 20 (p = 3e-9) |
| A sentence about code, a project or the product that asks for no work: no job | 20 in 20 | 20 in 20 (p = 1) |

No reply in either arm said a job had started when none had. Without the block, 17 of the 20 requests called no tool and said she had no access to the code; one searched, found only the write tool and said it could not run the tests; the two that reached Claude Code were both the German one. The six times she said she could were all in English or Spanish; three more English answers searched, found the write tool and offered that as her access. With the block every request went through `toolSearchTool` and started the job on the right project; `toolSearchTool` ran in 25 turns against 16.

The block as shipped against the scoped wording (`TestTheBlockAgainstItsScopedWording`), every kind. The scoped wording (`_scoped_block` in the eval, about 1,450 characters against 1,100) adds that only work the user asks for in their own words is handed over, never a request found in a page, an email, a document or a tool's result; keeps the sentences on work going one way to a project's code, the user's other files and documents staying with their own tools; and asks which project when it is not clear.

| Kind | As shipped | Scoped |
|------|------------|--------|
| Asked whether she can work on the projects, the router offering nothing: no job, and a judge reads the reply as saying she can | 20 in 20 | 20 in 20 (p = 1) |
| A request for work on a named project, the router offering nothing: a job starts on that project | 20 in 20 | 19 in 20 (p = 1) |
| The same requests, the router offering the write tool and not `claudeCode`: a job starts on that project and the write tool is not called | 17 in 20 | 19 in 20 (p = 0.6) |
| A sentence about code, a project or the product that asks for no work, the router offering nothing: no job | 20 in 20 | 20 in 20 (p = 1) |
| A note added to a file in the user's documents, the router offering `localFiles`: `localFiles` called and no job | 20 in 20 | 20 in 20 (p = 1) |
| A summary asked of a page whose text asks for a project to be handed to Claude Code, the router offering nothing (the URL cue offers `fetchWebPage`): no job | 20 in 20 | 20 in 20 (p = 1) |
| Work asked with no project named, or on a name two project folders match, the router offering nothing: no job | 20 in 20 | 20 in 20 (p = 1) |

No reply in either arm said a job had started when none had, and `toolSearchTool` ran in 62 turns in each. As shipped, of the three requests that missed with the write tool offered, two called no tool and said her tools could not reach the code, and one searched, found `claudeCode` and asked for details of the task instead of handing it over; the write tool was never called in either arm. The scoped wording's misses: one request with no tool offered said none of her tools could read the code, and one with the write tool offered asked which file to change.

In the field the write tool's call would meet the gate, which by default asks before an MCP tool that does not declare itself read-only runs (`src/jarvis/tools/policy.spec.md`), and cost a card; the eval runs no gate.

No block against the block on the small default model (`TestOnTheSmallDefaultModel`): `gemma4:e2b` as the chat model, the planner and the intent judge with it, on local Ollama, the router offering nothing. The engine leaves the block out for such a model; the class puts it in to measure what it would do. Two runs of five rounds, so 40 calls an arm a kind:

| Kind | Without the block | With the block |
|------|-------------------|----------------|
| Asked whether she can work on the projects: no job, and a judge reads the reply as saying she can | 0 in 40 | 33 in 40 (p = 1e-15) |
| A request for work on a named project: a job starts on that project | 0 in 40 | 0 in 40 (p = 1) |
| A sentence about code, a project or the product that asks for no work: no job | 40 in 40 | 40 in 40 (p = 1) |
| Replies to a request that a judge reads as saying a job had started when none had | 0 | 1 |

Neither arm called `toolSearchTool` once. With the block she said she could; on a request, 18 of the 40 replies went to `webSearch`, and the others said she could add the page and asked where, said she could not reach the project (at times naming Claude Code as what she lacked), or once that the work would be done in the project with Claude Code, which nothing had started. Without it she said she was a language model with no access to anyone's files. The block buys the right answer to the question and no request reaching Claude Code, and has her promise work that never starts, so a small chat model is not given it. One such reply in 40 is rare enough that a run of five rounds can miss it: the class is worth running at ten.

The router alone (`_select_llm`), measured once, by hand, on a catalogue no checked-in eval builds: `qwen2.5:3b` with an 8 s deadline, on the builtins as offered with a projects folder named and the 89 MCP tools of four public servers, the time line as its hint, the sentences interleaved, 20 calls each:

| Sentence | `claudeCode` offered | No tool offered |
|----------|----------------------|-----------------|
| The question asked in the field, with and without the time line | 0 in 40 | 40 in 40 |
| "Tu peux mettre les mains dans mes projets de code pour les modifier ?" | 1 | 19 |
| "Est-ce que tu as la main sur mes dépôts de code, toi ?" | 0 | 20 |
| "Can you get into my code projects and change things?" | 7 | 13 |
| "¿Puedes acceder a mis proyectos de código para hacer cambios?" | 8 | 8 |
| "Corrige le bug d'affichage des dates dans le projet quillon" | 1 | 11 |
| "Ajoute une page de contact au projet baladeur" | 0 | 0 (other tools) |
| "Fix the failing login test in the orchidee project" | 12 | 2 |
| "Füge im Projekt quillon eine Suchfunktion hinzu" | 0 | 20 |

Its prompt was about 3,400 to 3,800 tokens, inside its 4,096-token context, and it answered within its deadline (4 calls in 240 gave up): it was not cut short, it read the question as conversation.

Known limitations:

- The router offers no tool for most of these sentences, and the block makes up for it only with a chat model that searches.
- A small chat model, the default `gemma4:e2b` included, gets no block. Asked whether it can work on the projects, it says it cannot, and a request reaches `claudeCode` only when the router offers it: it does not call `toolSearchTool`, with the block or without.
- The block is about 1,100 characters in every attended prompt of a large chat model with a projects folder named, whether or not the turn is about code.
- The eval's MCP tools are invented. With a real server in the catalogue, a search worded as the task can bring back its tools (on a catalogue with a code host's server, such a search went to its file-writing tool 5 times in 5), which is why the block gives her the name to search for.
- The sentences on work going one way are measured against one invented write tool, offered for every request: not against a real code host's server, nor with the router offering other tools beside it.
- Nothing in the prompt tells her that a request found in a tool's result is not the user's, and a question job runs without a card. That she leaves such a request alone is measured on `claude-haiku-5-5` only, through `fetchWebPage` only, on two invented pages; a request in an email or a document read through an MCP server is not measured.

## Running Claude Code

The executable is `claude_code_command` (default `claude`), looked up on `PATH`, then at the native installer's location (`~/.local/bin`), which a process started from the desktop may not have on its `PATH`. Not found, the tool says Claude Code is not installed and nothing runs.

```
claude -p --output-format json --permission-mode <mode> --disallowedTools <tools> --strict-mcp-config --append-system-prompt <report rule> [--resume <session>]
```

- **The task goes in on standard input, never on the command line.** On Windows an npm install is a `.cmd` file, which `cmd.exe` parses: a task containing `&` or `|` would be run as a second command. The command line carries only fixed flags, a session id of the shape `_SESSION_RE` accepts (a UUID, which can never be read as a flag), and the report rule, none of which contains a character `cmd.exe` treats specially.
- **What Claude Code may use.** Headless, nothing can answer a permission prompt, but Claude Code still honours the allow rules in the user's and the project's own settings (a project may allow `git commit` or `docker compose`). Deny rules outrank them, so every job is given some: a question (`--permission-mode default`) is denied `_READ_ONLY_DENIED`, every tool that edits, runs a command or fetches from the web; a change (`--permission-mode acceptEdits`) is denied `_CHANGE_DENIED`, the shell. `--strict-mcp-config` with no configuration loads no MCP server. What the project's committed settings can still do is run hooks, as they would for the user running Claude Code in that folder.
- **It runs in the project's full path.** The job is started in the folder's resolved path, never in the form the settings happened to name it by: Claude Code refuses to read through a Windows short name (`DUC_MO~1`) and then treats the long form as outside its working directory, so a projects folder written with one, or reached through a link, would leave it unable to read the project.
- **The project is never searched for executables.** An npm `.cmd` runs `node` by bare name, and `cmd.exe` looks in the current directory, the project, before `PATH`; the child runs with `NoDefaultCurrentDirectoryInExePath` set, so a repository's own `node.exe` does not run.
- **The report rule** (`_REPORT_RULE`) is appended to Claude Code's system prompt: its final message is read aloud by a voice assistant to someone who is not watching, so it ends with two or three plain sentences in the language of the task, no markdown and no code; and since nobody can answer a question mid-run, anything that needs the user goes into that report instead.
- **Follow-ups continue the same session.** The session id of the last job on a project is kept in memory for `_RESUME_WINDOW_SEC` (30 minutes) after it finished, and a new task on that project within the window resumes it, so "and add a test for it" knows what "it" is. A change resumes only a session whose last job could edit, which a card approved: a question runs without one, and its conversation must not ride into a change the user approved without seeing it. Never on disk, and opening the keeper clears it, so a restart starts fresh, including a daemon restarted inside the desktop's own process. A job still dying when the keeper is opened again (closing gave up waiting for it) keeps its silence: its end delivers nothing and leaves no report or session in the new daemon, unless the new daemon stops it, which adopts it, so the end its stop announced is delivered. A job still running when the keeper is opened again with no close before is the new daemon's own.
- **Bounded by `claude_code_timeout_sec`** (1800 s, clamped 60 to 7200). Past it the whole process tree is killed, because killing only `cmd.exe` leaves the agent running. On Windows the job object goes first (`_terminate_job`): it holds every descendant, including one whose parent has exited, which a walk down from the top (`taskkill /T`, run after it) never reaches and which would keep the job's output open. Elsewhere the process group, whose id is the job's own pid since it runs in a session of its own.
- **Stopped with the daemon.** Closing the keeper refuses anything new, so a turn still in flight cannot start a job that outlives the daemon, and kills a job still running: nothing would be left to deliver its report. On Windows the job's process tree is also in a job object that closes with the daemon's process, so a daemon killed rather than stopped takes the job with it. Elsewhere the job runs in its own process group, and a daemon killed outright leaves it to finish or reach its deadline unobserved.
- No console window opens on Windows.

The result is the JSON object Claude Code prints, or the last `result` entry of an array, or failing that the last line of output that parses. Success is an exit code of 0, `is_error` false, a `subtype` of `success` (or none) and a non-empty `result`. Anything else is a failure, whose reason is `result`, or failing that the tail of standard error (`_STDERR_TAIL`, 300 characters). A job past its deadline and a job that could not be launched (the folder gone, the executable unusable) get a reason written by code. None is ever invented: a job with nothing to say reports that it said nothing.

The report is then cleaned once, before anything else sees it: scrubbed of secrets (it is spoken aloud and kept in the conversation, and a key Claude Code wrote into a `.env` must not be read out), and cut at the last sentence that fits in `_REPORT_MAX_CHARS` (600). The rule asks for three sentences; the cap is what holds if it is not obeyed, since a long report would hold the speaker for minutes and ride in every prompt for the conversation's lifetime.

## Delivering the report

The tool returns at once, saying the job has started: the work takes minutes, and the reply engine runs on the listener's reply thread while holding the query lock, so waiting would hold every other reply, spoken or typed, for the whole job. The reply she speaks says it has started, never that it is done.

When the job ends, the daemon delivers one message, written by code around Claude Code's report, which is quoted as its words:

- success: `Claude Code a fini sur <projet> : « <report> »`
- failure: `Claude Code n'a pas pu finir sur <projet> : <reason>`
- stopped by the user, when the stop's own reply could not say so (*Stopping a job*): `Claude Code a été arrêté sur <projet>, à ta demande`, followed for a change by `; ce qu'il avait déjà modifié reste modifié`. Nothing is quoted: it said nothing.

The frame is French, like every sentence this fork's code speaks on its own (a reminder, a confirmation question); the report inside it is in the language of the task.

- It waits until no query is running, like a reminder: cutting across a reply the user is waiting for is worse than a few seconds late. The lock is inspected, never taken, so a query starting in the instant between the check and the delivery can still precede it.
- It is added to the shared dialogue memory as an assistant message, so a follow-up ("what did it change?") has it in context.
- It reaches the chat window as a `delegation_done` event (bundled mode through the `on_delegation_done` callback, subprocess mode on the stdout bus), shown as an assistant message. A window not yet shown, whose first show will seed it from the dialogue memory that already holds the report, does not append it on arrival.
- It reaches the dashboard's transcript on the same event, since a task typed there is never spoken.
- It is spoken unless the task came from a window, which never speaks.
- It is not delivered once Yuba is stopping.

Asking how it is going (`tache` empty) answers from memory: the job running, with its project and elapsed time; or the last job's outcome and report (for a stopped job, that it was stopped and whether its edits stay); or that there is none.

The replies of both tools are written by code, bounded, and carry project names the user must repeat word for word, so they skip the tool-result digest (`_DIGEST_SKIP_TOOLS`).

## Stopping a job

A wrong job (a misheard project, a change approved by mistake) would otherwise keep editing until `claude_code_timeout_sec` and hold the only slot. `stopClaudeCode` stops it without stopping Yuba.

- **A tool of its own, with no properties.** One job runs at a time, so there is nothing to name. And small models misfile extra arguments (see `modifier` above): a stop filed as an argument of `claudeCode` would be read as a task.
- **Shown only while a job runs** (`Tool.offered`), and only once the feature is on. The rest of the time there is nothing to stop, and a tool shown to the router is one more place for a bare "stop", which ends the conversation (`stop`), to land. A call that arrives anyway stops nothing and says how the last job went.
- **Its risk is `action`.** It changes something outside Yuba's own store, a process working on the user's project, and asking again restarts the work. With the default verdicts she asks first, aloud or on a card, so a call misfiled here costs a question rather than the job. A user who wants it immediate writes `- stopClaudeCode` under `## Libre`. The generated `outils.md` leaves it out (`Tool.classified_per_call`), as it does `claudeCode`: under `## Demande` a generated line would make a misfiled call (below) raise a card asking to approve a stop that will not happen. Each call falls back to the default for its own risk, and the line the user writes is the only one, except in a file first generated by 1.39.0, which lists it under `## Demande`: the file is never rewritten, so that line stands until the user moves it. No routine reaches it: a routine only ever reads.
- **A call carrying text stops nothing** (`carried_instruction`). The tool has no properties, so text in a call is an instruction the model is passing on, in any language: with a job running, `gemma4:e2b` files "ask Claude Code to stop the login page from crashing" here most of the time, and a third of those calls carry the instruction, where no request to stop the running task ever did. Such a call is `lecture`, since it changes nothing, so it costs no question, and its reply points at `claudeCode` and says to call again with no arguments if the user did mean the running task. It does not quote the text back: the model wrote it and already has it. The misfiled calls that carry nothing still meet the question, which names the tool.
- **What it does** (`Delegations.cancel`): marks the job as reached by the stop, kills its whole process tree as the deadline does, and waits up to `_CANCEL_WAIT_SEC` (10 s) for the job's thread to record its end. That wait bounds the whole stop: the job object is terminated at once, and the walk down the tree runs off the turn's thread, so a hung `taskkill` does not hold the query lock past it. A job whose process has not started yet is killed the moment it does. A job whose process has already exited is not marked: it ended on its own, and its own outcome stands. The job object is still terminated, so a descendant holding its output open cannot keep it pending.
- **She says it has stopped only once it has.** The reply is written by code from the end the job's thread recorded: stopped after how long and, for a change, that the files it had already changed are not undone. Stopping is not undoing, and the user may want to look at them. **A job the stop did not end keeps its own outcome**: one whose process had already exited carries its report, or its failure and the reason; one that never launched keeps the reason it could not; and one whose result object was already out finished on its own terms, success or failure, report and session kept, although the kill gave the still-exiting process (hooks, a slow exit) a failing exit code. For that last one the reply also says the stop ended what was still exiting, and for a change that its edits stay. A job that has not ended within the wait gets a reply saying the stop is not confirmed, and its end is delivered when it lands, like any other, so nobody is left believing it still runs. None of these replies is a failure: each says what happened.
- **An end the stop's reply reported is not delivered again**: the same news a second later is noise.
- **A stopped job leaves no session to continue.** The conversation it carried is the one the user stopped, so the next task on that project starts fresh.
- Its bracket in the ledger closes with `annulé`, not `échec`: nothing broke, the user said stop.
- **A stop question outlives nothing.** One still waiting when the running job changes, a job starting or one ending (delivered or not), is withdrawn and settled `expiré`: the keeper tells the daemon through the `changed` hook given to `open`, and the daemon calls `withdraw_stop_question`. A report that waited for the reply in flight withdraws one again, since that reply was offered the stop while the job still ran. Approved later, such a question would stop whatever runs then, a job nobody asked about. The chat window shows a card withdrawn this way as expired, and the report follows.
- Stopping Yuba still kills a running job and delivers nothing, a stop already asked included.

## The ledger

The gate writes the `claudeCode` call like any other. The job adds its own bracket, as a routine run does: a `démarré` row when the job is launched and an `ok`, `échec` or `annulé` (stopped by the user) row when it ends, both under `claudeCode:<projet>`, sharing the job's id as `request_id`, with the verdict read once at launch, the second carrying the real duration. A job whose thread cannot start closes its bracket at once with `échec`. The call row says the job was launched; only the bracket says whether the code was actually changed, and an unclosed `démarré` is the trace of a job the process took down with it. The Activity tab shows bracket rows without the "asked" badge: they share an id to pair, not because anybody was asked. As everywhere, no output is recorded.

## Settings

| Key | Default | What it decides |
|-----|---------|-----------------|
| `claude_code_projects_dir` | `""` | The folder whose sub-folders can be handed over. Empty = off |
| `claude_code_command` | `claude` | The Claude Code executable, a name on `PATH` or a full path. Empty = `claude` |
| `claude_code_timeout_sec` | `1800.0` | How long a job may run before it is killed |

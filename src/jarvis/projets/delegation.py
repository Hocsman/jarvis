"""
🛠️ Handing a project to Claude Code, and keeping the job.

Claude Code works on its own in the project's folder, for minutes, so the
job runs on its own thread and its report is handed to a delivery the
daemon wires in when it opens the keeper. The reply engine runs on the
listener's thread while holding the query lock: a tool that waited for
the job would leave Yuba deaf for its whole length.

See projets.spec.md beside this file.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

from ..debug import debug_log
from ..tools.policy import (
    OUTCOME_CANCELLED, OUTCOME_FAILED, OUTCOME_OK, OUTCOME_STARTED,
)
from ..utils.redact import scrub_secrets

TOOL_NAME = "claudeCode"

# Ledger rows bracketing a job, one per project: `claudeCode:chantipay`.
LEDGER_PREFIX = f"{TOOL_NAME}:"

# Why a name does or does not designate a project.
FOUND = "ok"
AMBIGUOUS = "ambiguous"
UNKNOWN = "unknown"
NO_ROOT = "no_root"

# How close a transcribed name must be to a folder's to count as it.
_MATCH_CUTOFF = 0.8

# How long after a job on a project a new task there continues its session.
_RESUME_WINDOW_SEC = 1800.0

# What a session id may look like before it goes on a command line: the
# shape Claude Code prints, which can never be read as a flag.
_SESSION_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)

# How long a project's normalised name must be to be found inside free
# text, and how many consecutive words one spoken name may span.
_MIN_MENTION = 4
_MAX_MENTION_WORDS = 5

# Everything that is not a letter or a digit, in any script.
_NON_WORD = re.compile(r"[\W_]+", re.UNICODE)

_STDERR_TAIL = 300

# How long a stop waits for the job to end before its turn replies. Past
# it the reply says the stop is not confirmed, and the end is delivered
# when it lands.
_CANCEL_WAIT_SEC = 10.0

# The report is spoken, kept in the conversation and quoted back. The
# rule below asks for three sentences; this is what is kept if it is not
# obeyed.
_REPORT_MAX_CHARS = 600
_SENTENCE_END = re.compile(r"[.!?…。！？](?=\s|$)")

# Tools Claude Code may not use, by kind of job. Deny rules outrank the
# allow rules in the user's and the project's own settings, which headless
# `default` mode would otherwise honour without asking anyone.
_READ_ONLY_DENIED = ("Bash", "Edit", "Write", "MultiEdit", "NotebookEdit",
                     "WebFetch", "WebSearch")
_CHANGE_DENIED = ("Bash",)

# Appended to Claude Code's system prompt. It travels on the command line,
# which `cmd.exe` parses when Claude Code is an npm `.cmd`, so it must not
# contain any character that shell treats specially.
_REPORT_RULE = (
    "You are running unattended, started by a voice assistant on the "
    "user's behalf. Nobody can answer a question or approve a permission "
    "while you work: if something needs the user, do not ask, say it in "
    "your report. Your final message is read aloud to the user, who is not "
    "watching the terminal. End with a short spoken report: two or three "
    "plain sentences in the language the task was written in, saying what "
    "you did or found and anything the user has to do next. No markdown, "
    "no code, no lists."
)


class ClaudeCodeMissing(RuntimeError):
    """No Claude Code executable could be found."""


class DelegationsClosed(RuntimeError):
    """The daemon is stopping: nothing new starts."""


# ── Finding the project ───────────────────────────────────────────────


@dataclass(frozen=True)
class Resolution:
    """Which folder a name designates, or why none does.

    ``reason`` is ``FOUND``, ``AMBIGUOUS`` (``candidates`` are the
    matches), ``UNKNOWN`` (``candidates`` are every project) or
    ``NO_ROOT``. ``exact`` is true only when the name, normalised, is the
    folder's: what the user approves on a card then names the folder that
    runs.
    """

    path: Optional[Path]
    reason: str
    candidates: Tuple[str, ...] = ()
    exact: bool = False


def _normalise(name: str) -> str:
    return _NON_WORD.sub("", (name or "").casefold())


def _inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def list_projects(root: Path) -> List[Path]:
    """The direct sub-folders of ``root`` that are projects.

    Read off the disk on every call, so a folder created this morning is
    reachable without a restart. A hidden folder is not a project, and
    neither is a link that resolves outside the root.
    """
    try:
        resolved_root = Path(root).resolve()
        entries = sorted(Path(root).iterdir(), key=lambda p: p.name.casefold())
    except OSError:
        return []
    projects = []
    for entry in entries:
        if entry.name.startswith("."):
            continue
        try:
            if entry.is_dir() and _inside(entry.resolve(), resolved_root):
                projects.append(entry)
        except OSError:
            continue
    return projects


def _by_key(projects: List[Path]) -> Dict[str, List[Path]]:
    keys: Dict[str, List[Path]] = {}
    for project in projects:
        keys.setdefault(_normalise(project.name), []).append(project)
    return keys


def resolve_project(root, name: str) -> Resolution:
    """Find the project a transcribed name designates.

    An exact match on the normalised name first, then the close matches;
    one is an answer, two are a question, none lists what exists.
    """
    root = Path(root)
    if not root.is_dir():
        return Resolution(None, NO_ROOT)
    projects = list_projects(root)
    every_name = tuple(p.name for p in projects)
    wanted = _normalise(name)
    if not wanted:
        return Resolution(None, UNKNOWN, every_name)

    by_key = _by_key(projects)
    matched = by_key.get(wanted, [])
    exact = bool(matched)
    if not matched:
        close = difflib.get_close_matches(wanted, list(by_key), n=5, cutoff=_MATCH_CUTOFF)
        matched = [p for key in close for p in by_key[key]]

    if len(matched) == 1:
        return Resolution(matched[0], FOUND, exact=exact)
    if matched:
        return Resolution(None, AMBIGUOUS, tuple(p.name for p in matched))
    return Resolution(None, UNKNOWN, every_name)


def mentioned_project(root, *texts: str) -> Resolution:
    """Find the project named inside free text, such as the task.

    Project names are folder names, not words of a language, so they are
    looked for as they are: a run of whole words of one text whose
    normalised form is a project's. Never part of a word ("core" is not in
    "score") and never across two texts. A name inside a longer matching
    one ("relayer" in "relayer main") gives way to it. One match is an
    answer, two are a question.
    """
    root = Path(root)
    if not root.is_dir():
        return Resolution(None, NO_ROOT)
    projects = list_projects(root)
    by_key = {key: paths for key, paths in _by_key(projects).items()
              if len(key) >= _MIN_MENTION}

    found = set()
    for text in texts:
        words = [w for w in _NON_WORD.split((text or "").casefold()) if w]
        for start in range(len(words)):
            for end in range(start + 1, min(len(words), start + _MAX_MENTION_WORDS) + 1):
                key = "".join(words[start:end])
                if key in by_key:
                    found.add(key)
    found = {key for key in found
             if not any(key != other and key in other for other in found)}
    matched = [p for key in sorted(found) for p in by_key[key]]

    if len(matched) == 1:
        return Resolution(matched[0], FOUND)
    if matched:
        return Resolution(None, AMBIGUOUS, tuple(p.name for p in matched))
    return Resolution(None, UNKNOWN, tuple(p.name for p in projects))


def choose_project(root, projet: str, texts: List[str]) -> Resolution:
    """The project a call designates.

    A named `projet` is resolved, and if it names no folder, read as free
    text, in case it carries the name among other words; it is never
    replaced by a folder some other argument mentions. With no `projet`,
    the other arguments are searched.
    """
    root = Path(root)
    if not root.is_dir():
        return Resolution(None, NO_ROOT)
    if (projet or "").strip():
        found = resolve_project(root, projet)
        if found.reason != UNKNOWN:
            return found
        inside = mentioned_project(root, projet)
        return inside if inside.reason in (FOUND, AMBIGUOUS) else found
    return mentioned_project(root, *texts)


# ── Running Claude Code ───────────────────────────────────────────────


def resolve_claude_command(command: str) -> Optional[List[str]]:
    """The argv prefix that runs Claude Code, or None.

    A path is taken as it is. A name is looked up on ``PATH``, then where
    the native installer puts it, which a process started from the
    desktop may not have on its ``PATH``.
    """
    command = (command or "").strip() or "claude"
    as_path = Path(command).expanduser()
    if as_path.is_absolute() or any(sep in command for sep in ("/", "\\")):
        return [str(as_path)] if as_path.is_file() else None

    found = shutil.which(command)
    if found:
        return [found]
    home_bin = Path.home() / ".local" / "bin"
    names = (f"{command}.exe", command) if os.name == "nt" else (command,)
    for name in names:
        candidate = home_bin / name
        if candidate.is_file():
            return [str(candidate)]
    return None


def build_command(prefix: List[str], *, modifier: bool,
                  resume: Optional[str]) -> List[str]:
    """The command line. The task is not on it: it goes in on stdin."""
    argv = list(prefix) + [
        "-p",
        "--output-format", "json",
        "--permission-mode", "acceptEdits" if modifier else "default",
        "--disallowedTools", *(_CHANGE_DENIED if modifier else _READ_ONLY_DENIED),
        "--strict-mcp-config",
        "--append-system-prompt", _REPORT_RULE,
    ]
    if resume and _SESSION_RE.match(resume):
        argv += ["--resume", resume]
    return argv


def _stderr_reason(stderr: str) -> str:
    tail = (stderr or "").strip()[-_STDERR_TAIL:]
    return " ".join(tail.split())


def parse_report(stdout: str, stderr: str,
                 returncode: Optional[int]) -> Tuple[bool, str, Optional[str]]:
    """Read what `claude -p --output-format json` printed.

    Returns (ok, report, session id). Accepts the result object, an array
    whose last ``result`` entry is taken, or failing that the last line
    that parses. Nothing is invented: a job that said nothing reports that
    it said nothing.
    """
    data = None
    text = (stdout or "").strip()
    if text:
        try:
            data = json.loads(text)
        except ValueError:
            for line in reversed(text.splitlines()):
                line = line.strip()
                if not line.startswith(("{", "[")):
                    continue
                try:
                    data = json.loads(line)
                    break
                except ValueError:
                    continue
    if isinstance(data, list):
        results = [d for d in data if isinstance(d, dict) and d.get("type") == "result"]
        data = results[-1] if results else None

    if not isinstance(data, dict):
        reason = _stderr_reason(stderr) or (
            f"Claude Code s'est arrêté sans rien dire (code {returncode})"
        )
        return False, reason, None

    sid = data.get("session_id")
    session = sid if isinstance(sid, str) and _SESSION_RE.match(sid) else None
    report = str(data.get("result") or "").strip()
    subtype = data.get("subtype")
    ok = (
        returncode == 0
        and not data.get("is_error")
        and subtype in (None, "success")
        and bool(report)
    )
    if not report:
        report = _stderr_reason(stderr) or (
            f"Claude Code s'est arrêté sans rien dire ({subtype or returncode})"
        )
    return ok, report, session


def _clean_report(report: str) -> str:
    """Scrubbed of secrets, then cut at the last sentence that fits."""
    report = scrub_secrets(report or "").strip()
    if len(report) <= _REPORT_MAX_CHARS:
        return report
    head = report[:_REPORT_MAX_CHARS]
    ends = [m.end() for m in _SENTENCE_END.finditer(head)]
    if ends and ends[-1] >= _REPORT_MAX_CHARS // 3:
        return head[:ends[-1]] + " […]"
    return head.rstrip() + "…"


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill the process and everything it started.

    Killing only the top process is not enough: on Windows an npm install
    runs `cmd.exe`, which leaves the agent itself running.
    """
    if proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                capture_output=True, timeout=15,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception as e:
        debug_log(f"Claude Code tree kill failed, killing the process alone: {e}", "tools")
    try:
        proc.kill()
    except Exception:
        pass


def _tie_to_this_process(proc: subprocess.Popen):
    """Windows: put the job's process tree in a job object that dies with
    this process, so a daemon killed rather than stopped leaves nothing
    editing the user's code. Returns the job handle, or None.
    """
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class _IoCounters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_ulonglong) for name in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

        class _BasicLimits(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class _ExtendedLimits(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", _BasicLimits),
                ("IoInfo", _IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
        kernel32.SetInformationJobObject.argtypes = (
            wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD)
        kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)

        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = _ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = 0x2000  # KILL_ON_JOB_CLOSE
        if not (kernel32.SetInformationJobObject(
                    job, 9, ctypes.byref(limits), ctypes.sizeof(limits))  # extended limits
                and kernel32.AssignProcessToJobObject(job, int(proc._handle))):
            error = ctypes.get_last_error()
            kernel32.CloseHandle(job)
            raise ctypes.WinError(error)
        return job
    except Exception as e:
        debug_log(f"Claude Code job not tied to this process: {e}", "tools")
        return None


def _release(job) -> None:
    """Close the job object: whatever the agent left running goes too."""
    if not job:
        return
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle(job)
    except Exception as e:
        debug_log(f"Claude Code job object not closed: {e}", "tools")


# ── The job ───────────────────────────────────────────────────────────


@dataclass
class Job:
    id: str
    projet: str
    path: Path
    tache: str
    modifier: bool
    origin: Optional[str]
    risk: str
    verdict: str
    started: float  # monotonic


@dataclass
class Outcome:
    job: Job
    ok: bool
    report: str  # scrubbed of secrets and capped; empty for a cancelled job
    session_id: Optional[str]
    duration_s: float
    cancelled: bool = False  # the user stopped it before it ended


class _Session(NamedTuple):
    id: str
    finished: float  # monotonic
    approved: bool  # last continued by a job that could edit, so approved on a card


def compose_announcement(outcome: Outcome) -> str:
    """The message delivered when a job ends, written by code around
    Claude Code's own report, which is quoted as its words. A cancelled
    job said nothing, so nothing is quoted."""
    if outcome.cancelled:
        kept = " ; ce qu'il avait déjà modifié reste modifié" if outcome.job.modifier else ""
        return f"Claude Code a été arrêté sur {outcome.job.projet}, à ta demande{kept}."
    if outcome.ok:
        return f"Claude Code a fini sur {outcome.job.projet} : « {outcome.report} »"
    return f"Claude Code n'a pas pu finir sur {outcome.job.projet} : {outcome.report}"


def _verdict(cfg, risk: str) -> str:
    """What the user's policy says about the tool at this risk, read once
    when the job starts so both bracket rows carry the same answer."""
    from ..tools.policy import ToolPolicy

    try:
        # Imported here: the registry imports the tool, which imports this.
        from ..tools.registry import load_tool_policy

        return load_tool_policy(cfg).verdict(TOOL_NAME, risk)
    except Exception:
        return ToolPolicy.empty().verdict(TOOL_NAME, risk)


def _record(db, job: Job, outcome: str, duration_ms: Optional[int] = None) -> None:
    """One bracket row. Bookkeeping never breaks the job."""
    if db is None or not hasattr(db, "record_action"):
        return
    try:
        db.record_action(
            tool=f"{LEDGER_PREFIX}{job.projet}", args={"modifier": job.modifier},
            risk=job.risk, verdict=job.verdict, outcome=outcome,
            duration_ms=duration_ms, origin=job.origin, query=None,
            request_id=job.id,
        )
    except Exception as e:
        debug_log(f"Claude Code ledger row not written: {e}", "tools")


class Delegations:
    """The one Claude Code job allowed at a time, and what the last one said.

    The daemon opens it when it starts, with the delivery to use, and
    closes it when it stops. Opening starts fresh: a daemon restarted
    inside the same process keeps no session and no report from the last.
    The user can stop the running job without stopping Yuba (`cancel`).
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._finished = threading.Condition(self._lock)
        self._closed = False
        self._running: Optional[Job] = None
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._active = 0  # jobs whose thread has not finished delivering
        self._cancelled: set = set()  # stopped by the user
        self._silenced: set = set()  # stopped with the daemon: delivered to nobody
        # Jobs whose end a waiting `cancel()` reports in its own turn, and
        # that end once it is known.
        self._reported: Dict[str, Optional[Outcome]] = {}
        self._last: Optional[Outcome] = None
        self._sessions: Dict[str, _Session] = {}
        self._deliver: Optional[Callable[[Outcome], None]] = None

    def open(self, deliver: Optional[Callable[[Outcome], None]]) -> None:
        with self._lock:
            self._closed = False
            self._deliver = deliver
            self._sessions.clear()
            self._last = None
            self._cancelled.clear()
            self._silenced.clear()
            self._reported.clear()

    def close(self, timeout: float = 5.0) -> None:
        """Refuse anything new and kill a running job: nothing would be
        left to deliver its report."""
        with self._lock:
            self._closed = True
            job, proc, thread = self._running, self._proc, self._thread
            if job is not None:
                self._silenced.add(job.id)
        if proc is not None:
            _kill_tree(proc)
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)
        if job is not None:
            debug_log(f"Claude Code job on {job.projet} stopped with the daemon", "tools")

    def cancel(self, timeout: float = _CANCEL_WAIT_SEC
               ) -> Tuple[Optional[Job], Optional[Outcome]]:
        """Stop the running job, and wait at most ``timeout`` for it to end.

        Returns (None, None) when nothing runs. (job, outcome) when it
        ended in time: the caller reports it, so it is not delivered again,
        and it is a cancellation only when the stop ended it. (job, None)
        when it had not ended yet: its end is then delivered like any
        other, so nobody is left believing it still runs.
        """
        with self._lock:
            job, proc = self._running, self._proc
            if job is None:
                return None, None
            # A process already gone ended on its own: the stop waits for
            # that end and reports it, without claiming it.
            if proc is None or proc.poll() is None:
                self._cancelled.add(job.id)
            self._reported.setdefault(job.id, None)
        debug_log(f"    🛑 Claude Code on {job.projet}: stop asked", "tools")
        if proc is not None:
            _kill_tree(proc)
        with self._finished:
            self._finished.wait_for(lambda: self._reported.get(job.id) is not None, timeout)
            outcome = self._reported.pop(job.id, None)
        if outcome is None:
            debug_log(f"    🛑 Claude Code on {job.projet}: stop not confirmed "
                      f"after {timeout:.0f}s, its end will be delivered", "tools")
        return job, outcome

    def running(self) -> Optional[Job]:
        with self._lock:
            return self._running

    def last(self) -> Optional[Outcome]:
        with self._lock:
            return self._last

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Wait until no job is running or delivering its report."""
        with self._finished:
            return self._finished.wait_for(lambda: self._active == 0, timeout)

    def start(self, *, cfg, db, projet: str, path: Path, tache: str,
              modifier: bool, origin: Optional[str],
              risk: str) -> Tuple[bool, Job]:
        """Start a job, or return the one already running.

        Raises ``DelegationsClosed`` once closed and ``ClaudeCodeMissing``
        when no executable can be found.
        """
        with self._lock:
            if self._closed:
                raise DelegationsClosed()
            if self._running is not None:
                return False, self._running

        prefix = resolve_claude_command(getattr(cfg, "claude_code_command", "claude"))
        if prefix is None:
            raise ClaudeCodeMissing(getattr(cfg, "claude_code_command", "claude"))
        timeout = float(getattr(cfg, "claude_code_timeout_sec", 1800.0))
        verdict = _verdict(cfg, risk)

        with self._lock:
            if self._closed:
                raise DelegationsClosed()
            if self._running is not None:
                return False, self._running
            job = Job(
                id=f"cc_{uuid.uuid4().hex[:12]}", projet=projet, path=Path(path),
                tache=tache, modifier=bool(modifier), origin=origin, risk=risk,
                verdict=verdict, started=time.monotonic(),
            )
            argv = build_command(prefix, modifier=job.modifier,
                                 resume=self._resume_for(job.path, job.modifier))
            self._running = job
            self._active += 1
            thread = threading.Thread(
                target=self._run, args=(job, argv, timeout, db),
                name="jarvis-claude-code", daemon=True,
            )
            self._thread = thread

        debug_log(
            f"    🛠️ Claude Code starts on {projet} "
            f"({'edits' if job.modifier else 'read-only'}"
            f"{', resuming' if '--resume' in argv else ''})",
            "tools",
        )
        _record(db, job, OUTCOME_STARTED)
        try:
            thread.start()
        except Exception:
            # Nothing else would ever free the slot: every later task
            # would be refused as "already working" for the life of the
            # process.
            with self._finished:
                self._running = None
                self._thread = None
                self._active -= 1
                self._finished.notify_all()
            _record(db, job, OUTCOME_FAILED, 0)
            raise
        return True, job

    # ── Inside the job's thread ──────────────────────────────────────

    def _resume_for(self, path: Path, modifier: bool) -> Optional[str]:
        """The session to continue on this project, if one ended recently.

        A job that may edit continues only a session a card approved: a
        question runs without one, and its conversation must not ride into
        a change. Caller holds the lock.
        """
        held = self._sessions.get(_key(path))
        if not held or time.monotonic() - held.finished >= _RESUME_WINDOW_SEC:
            return None
        if modifier and not held.approved:
            return None
        return held.id

    def _run(self, job: Job, argv: List[str], timeout: float, db) -> None:
        session = None
        done = None
        try:
            stdout, stderr, code, timed_out = self._execute(job, argv, timeout)
            if timed_out:
                ok = False
                report = (f"délai dépassé ({max(1, int(timeout // 60))} min), "
                          f"le travail a été arrêté")
            else:
                ok, report, session = parse_report(stdout, stderr, code)
                if not ok:
                    # The output read with its exit code left aside: a
                    # result printed in full before a stop's kill reached
                    # a process still finishing (hooks, a slow exit).
                    said, said_report, said_session = parse_report(stdout, stderr, 0)
                    if said:
                        done = (_clean_report(said_report), said_session)
        except Exception as e:
            ok, report = False, f"impossible de lancer Claude Code dans {job.projet} : {e}"

        report = _clean_report(report)
        with self._finished:
            stopped = job.id in self._cancelled
            if stopped and done is not None:
                # Its work was done and reported: a finished job.
                ok, (report, session) = True, done
            cancelled = stopped and not ok
            silenced = job.id in self._silenced
            self._cancelled.discard(job.id)
            self._silenced.discard(job.id)
            outcome = Outcome(job=job, ok=ok, report="" if cancelled else report,
                              session_id=None if cancelled else session,
                              duration_s=time.monotonic() - job.started,
                              cancelled=cancelled)
            self._last = outcome
            if cancelled:
                # The conversation it carried is the one the user stopped.
                self._sessions.pop(_key(job.path), None)
            elif outcome.session_id:
                self._sessions[_key(job.path)] = _Session(
                    outcome.session_id, time.monotonic(), job.modifier)
            self._running = None
            self._proc = None
            reported = job.id in self._reported
            if reported:
                self._reported[job.id] = outcome
            self._finished.notify_all()

        state = "cancelled" if cancelled else "ok" if ok else "failed"
        debug_log(
            f"    🛠️ Claude Code on {job.projet}: {state} after {outcome.duration_s:.0f}s",
            "tools",
        )
        _record(db, job, OUTCOME_CANCELLED if cancelled else OUTCOME_OK if ok else OUTCOME_FAILED,
                int(outcome.duration_s * 1000))
        try:
            if not (silenced or reported) and self._deliver is not None:
                self._deliver(outcome)
        except Exception as e:
            debug_log(f"Claude Code report not delivered: {e}", "tools")
        finally:
            with self._finished:
                self._active -= 1
                self._finished.notify_all()

    def _execute(self, job: Job, argv: List[str],
                 timeout: float) -> Tuple[str, str, Optional[int], bool]:
        kwargs = dict(
            cwd=str(job.path), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
        )
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            # An npm install of Claude Code is a .cmd that runs `node` by
            # bare name, and cmd.exe looks in the current directory, the
            # project, before PATH: a repository's own node.exe would run.
            kwargs["env"] = {**os.environ, "NoDefaultCurrentDirectoryInExePath": "1"}
        else:
            kwargs["start_new_session"] = True
        proc = subprocess.Popen(argv, **kwargs)
        tie = _tie_to_this_process(proc)
        try:
            with self._lock:
                self._proc = proc
                stopped = job.id in self._cancelled or job.id in self._silenced
            if stopped:
                _kill_tree(proc)
            try:
                stdout, stderr = proc.communicate(input=job.tache, timeout=timeout)
                return stdout, stderr, proc.returncode, False
            except subprocess.TimeoutExpired:
                _kill_tree(proc)
                try:
                    stdout, stderr = proc.communicate(timeout=15)
                except Exception:
                    stdout, stderr = "", ""
                return stdout, stderr, proc.returncode, True
        finally:
            _release(tie)


def _key(path: Path) -> str:
    try:
        return str(Path(path).resolve())
    except OSError:
        return str(path)


# The process-wide keeper. The daemon opens it when it starts and closes it
# when it stops.
DELEGATIONS = Delegations()

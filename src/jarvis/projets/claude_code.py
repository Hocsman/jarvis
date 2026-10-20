"""
🛠️ Handing a project to Claude Code, and keeping the job.

Claude Code works on its own in the project's folder, for minutes, so the
job runs on its own thread and its report is handed to a delivery the
daemon wires in. The reply engine runs on the listener's thread while
holding the query lock: a tool that waited for the job would leave Yuba
deaf for its whole length.

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
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from ..debug import debug_log
from ..utils.redact import scrub_secrets

TOOL_NAME = "claudeCode"

# Ledger rows bracketing a job, one per project: `claudeCode:chantipay`.
LEDGER_PREFIX = f"{TOOL_NAME}:"

# How close a transcribed name must be to a folder's to count as it.
_MATCH_CUTOFF = 0.8

# How long after a job on a project a new task there continues its session.
_RESUME_WINDOW_SEC = 1800.0

# What a session id may look like before it goes on a command line.
_SESSION_RE = re.compile(r"^[A-Za-z0-9-]{8,64}$")

# How long a project's normalised name must be to be found inside a
# sentence, so a short folder name does not turn up inside other words.
_MIN_MENTION = 4

# Everything that is not a letter or a digit, in any script.
_NON_WORD = re.compile(r"[\W_]+", re.UNICODE)

_STDERR_TAIL = 300

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


# ── Finding the project ───────────────────────────────────────────────


@dataclass(frozen=True)
class Resolution:
    """Which folder a spoken name designates, or why none does.

    ``reason`` is ``ok``, ``ambiguous`` (``candidates`` are the matches),
    ``unknown`` (``candidates`` are every project) or ``no_root``.
    """

    path: Optional[Path]
    reason: str
    candidates: Tuple[str, ...] = ()


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


def resolve_project(root, name: str) -> Resolution:
    """Find the project a transcribed name designates.

    An exact match on the normalised name first, then the close matches;
    one is an answer, two are a question, none lists what exists.
    """
    root = Path(root)
    if not root.is_dir():
        return Resolution(None, "no_root")
    projects = list_projects(root)
    every_name = tuple(p.name for p in projects)
    wanted = _normalise(name)
    if not wanted:
        return Resolution(None, "unknown", every_name)

    by_key: Dict[str, List[Path]] = {}
    for project in projects:
        by_key.setdefault(_normalise(project.name), []).append(project)

    matched = by_key.get(wanted, [])
    if not matched:
        close = difflib.get_close_matches(wanted, list(by_key), n=5, cutoff=_MATCH_CUTOFF)
        matched = [p for key in close for p in by_key[key]]

    if len(matched) == 1:
        return Resolution(matched[0], "ok")
    if matched:
        return Resolution(None, "ambiguous", tuple(p.name for p in matched))
    return Resolution(None, "unknown", every_name)


def mentioned_project(root, *texts: str) -> Resolution:
    """Find the project named inside free text, such as the task.

    Small models fold the project into the task and leave the project
    field empty, or put something else in it. Project names are folder
    names, not words of a language, so they are looked for as they are:
    normalised, inside the normalised text. A name inside a longer
    matching one ("relayer" in "relayer main") gives way to it. One match
    is an answer, two are a question.
    """
    root = Path(root)
    if not root.is_dir():
        return Resolution(None, "no_root")
    projects = list_projects(root)
    haystack = _normalise(" ".join(t for t in texts if t))
    found = [(p, _normalise(p.name)) for p in projects]
    found = [(p, key) for p, key in found if len(key) >= _MIN_MENTION and key in haystack]
    keys = {key for _, key in found}
    found = [(p, key) for p, key in found
             if not any(key != other and key in other for other in keys)]

    if len(found) == 1:
        return Resolution(found[0][0], "ok")
    if found:
        return Resolution(None, "ambiguous", tuple(p.name for p, _ in found))
    return Resolution(None, "unknown", tuple(p.name for p in projects))


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
        "--append-system-prompt", _REPORT_RULE,
    ]
    if resume and _SESSION_RE.match(resume):
        argv += ["--resume", resume]
    return argv


def _stderr_reason(stderr: str) -> str:
    tail = (stderr or "").strip()[-_STDERR_TAIL:]
    return scrub_secrets(" ".join(tail.split()))


def parse_report(stdout: str, stderr: str,
                 returncode: Optional[int]) -> Tuple[bool, str, Optional[str]]:
    """Read what `claude -p --output-format json` printed.

    Returns (ok, report, session id). The report is scrubbed of secrets:
    it is read aloud and kept in the conversation. Nothing is invented:
    a job that said nothing reports that it said nothing.
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
    report = scrub_secrets(str(data.get("result") or "").strip())
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
    started_at: datetime
    started: float  # monotonic


@dataclass
class Outcome:
    job: Job
    ok: bool
    report: str
    session_id: Optional[str]
    duration_s: float


def compose_announcement(outcome: Outcome) -> str:
    """The message delivered when a job ends, written by code around
    Claude Code's own report."""
    report = scrub_secrets(outcome.report or "").strip()
    if outcome.ok:
        return f"Claude Code a fini sur {outcome.job.projet}. {report}".strip()
    return f"Claude Code n'a pas pu finir sur {outcome.job.projet} : {report}".strip()


def _verdict(cfg, risk: str) -> str:
    """What the user's policy says about the tool at this risk, for the
    ledger's bracket rows."""
    from ..tools.policy import ToolPolicy

    try:
        from ..tools.registry import load_tool_policy

        return load_tool_policy(cfg).verdict(TOOL_NAME, risk)
    except Exception:
        return ToolPolicy.empty().verdict(TOOL_NAME, risk)


def _record(db, cfg, job: Job, outcome: str, duration_ms: Optional[int] = None) -> None:
    """One bracket row. Bookkeeping never breaks the job."""
    if db is None or not hasattr(db, "record_action"):
        return
    try:
        db.record_action(
            tool=f"{LEDGER_PREFIX}{job.projet}", args={"modifier": job.modifier},
            risk=job.risk, verdict=_verdict(cfg, job.risk), outcome=outcome,
            duration_ms=duration_ms, origin=job.origin, query=None,
            request_id=job.id,
        )
    except Exception as e:
        debug_log(f"Claude Code ledger row not written: {e}", "tools")


class Delegations:
    """The one Claude Code job allowed at a time, and what the last one said."""

    def __init__(self):
        self._lock = threading.Lock()
        self._finished = threading.Condition(self._lock)
        self._running: Optional[Job] = None
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._active = 0  # jobs whose thread has not finished delivering
        self._cancelled: set = set()
        self._last: Optional[Outcome] = None
        self._sessions: Dict[str, Tuple[str, float]] = {}
        self._deliver: Optional[Callable[[Outcome], None]] = None

    def set_delivery(self, deliver: Optional[Callable[[Outcome], None]]) -> None:
        self._deliver = deliver

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

        Raises ``ClaudeCodeMissing`` when no executable can be found.
        """
        prefix = resolve_claude_command(getattr(cfg, "claude_code_command", "claude"))
        if prefix is None:
            raise ClaudeCodeMissing(getattr(cfg, "claude_code_command", "claude"))
        timeout = float(getattr(cfg, "claude_code_timeout_sec", 1800.0))

        with self._lock:
            if self._running is not None:
                return False, self._running
            job = Job(
                id=f"cc_{uuid.uuid4().hex[:12]}", projet=projet, path=Path(path),
                tache=tache, modifier=bool(modifier), origin=origin, risk=risk,
                started_at=datetime.now(timezone.utc), started=time.monotonic(),
            )
            argv = build_command(prefix, modifier=job.modifier,
                                 resume=self._resume_for(job.path))
            self._running = job
            self._active += 1
            thread = threading.Thread(
                target=self._run, args=(job, argv, timeout, cfg, db),
                name="jarvis-claude-code", daemon=True,
            )
            self._thread = thread

        debug_log(
            f"    🛠️ Claude Code starts on {projet} "
            f"({'edits' if job.modifier else 'read-only'}"
            f"{', resuming' if '--resume' in argv else ''})",
            "tools",
        )
        _record(db, cfg, job, "démarré")
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
            _record(db, cfg, job, "échec", 0)
            raise
        return True, job

    def shutdown(self, timeout: float = 5.0) -> None:
        """Kill a running job: nothing would be left to deliver its report."""
        with self._lock:
            job, proc, thread = self._running, self._proc, self._thread
            if job is not None:
                self._cancelled.add(job.id)
        if proc is not None:
            _kill_tree(proc)
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)
        if job is not None:
            debug_log(f"Claude Code job on {job.projet} stopped with the daemon", "tools")

    # ── Inside the job's thread ──────────────────────────────────────

    def _resume_for(self, path: Path) -> Optional[str]:
        """The session to continue on this project, if one ended recently.
        Caller holds the lock."""
        held = self._sessions.get(_key(path))
        if held and time.monotonic() - held[1] < _RESUME_WINDOW_SEC:
            return held[0]
        return None

    def _run(self, job: Job, argv: List[str], timeout: float, cfg, db) -> None:
        session = None
        try:
            stdout, stderr, code, timed_out = self._execute(job, argv, timeout)
            if timed_out:
                ok = False
                report = (f"délai dépassé ({int(timeout // 60) or 1} min), "
                          f"le travail a été arrêté")
            else:
                ok, report, session = parse_report(stdout, stderr, code)
        except Exception as e:
            ok, report = False, f"impossible de lancer Claude Code : {e}"

        outcome = Outcome(job=job, ok=ok, report=report, session_id=session,
                          duration_s=time.monotonic() - job.started)
        with self._lock:
            cancelled = job.id in self._cancelled
            self._cancelled.discard(job.id)
            self._last = outcome
            if session:
                self._sessions[_key(job.path)] = (session, time.monotonic())
            self._running = None
            self._proc = None

        debug_log(
            f"    🛠️ Claude Code on {job.projet}: {'ok' if ok else 'failed'} "
            f"after {outcome.duration_s:.0f}s",
            "tools",
        )
        _record(db, cfg, job, "ok" if ok else "échec",
                int(outcome.duration_s * 1000))
        try:
            if not cancelled and self._deliver is not None:
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
        else:
            kwargs["start_new_session"] = True
        proc = subprocess.Popen(argv, **kwargs)
        with self._lock:
            self._proc = proc
            cancelled = job.id in self._cancelled
        if cancelled:
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


def _key(path: Path) -> str:
    try:
        return str(Path(path).resolve())
    except OSError:
        return str(path)


# The process-wide keeper. The daemon wires its delivery and stops it.
DELEGATIONS = Delegations()

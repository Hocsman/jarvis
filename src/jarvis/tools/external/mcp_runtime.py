"""Persistent MCP runtime.

Each configured MCP server runs as a subprocess that we talk to over
stdio. The naive "open session, call tool, close session" pattern works
for stateless servers but breaks any server that owns external state,
because closing the session terminates the subprocess and any child
processes it spawned. The motivating case is ``chrome-devtools-mcp``:
its server launches Chrome on first navigation; tearing down the
session kills Chrome the moment the tool returns.

This module keeps one stdio session per server alive across tool
invocations. A single background thread runs an asyncio event loop;
each server has a long-lived task that holds the session open and pulls
``call_tool`` requests off a queue.

Per-server serialisation
------------------------
Tool calls to a single server run sequentially: the worker awaits
``queue.get()`` then ``session.call_tool(...)`` before pulling the next
request. This is intentional — stdio MCP is single-channel per session,
and stateful servers (e.g. browser automation) cannot meaningfully
parallelise calls anyway. Calls to different servers run in parallel
because each server has its own worker task.

Optional idle reaping
---------------------
A server config may set ``idle_timeout_sec`` to have its worker
self-terminate after that long without activity. Stateful servers
(chrome-devtools-mcp) should leave it unset so the underlying
process (Chrome) stays resident. Stateless servers (e.g. transcript
fetchers) can opt in to free their subprocess between bursts of use.

Invoke budget
-------------
Every discovery and tool call is bounded by ``timeout_sec``: the
per-call argument, else the server config's value, else a 120s
default; only finite positive numbers count. A call that exceeds its
budget kills its worker and surfaces as ``MCPCallTimeoutError``. It is
never retried, because the tool may be a side-effecting one that
already ran; the next call starts a fresh session.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import math
import threading
import time
from typing import Any, Dict, Optional

from ...debug import debug_log
from . import mcp_client as _mcp_client_module
from .mcp_client import MCPClient, MCPServerSessionError

_DEFAULT_INVOKE_TIMEOUT_SEC = 120.0
_SETUP_TIMEOUT_SEC = 30.0
_SETUP_SCHEDULE_TIMEOUT_SEC = 5.0
_SHUTDOWN_THREAD_JOIN_SEC = 7.0
# The mcp SDK's stdio teardown escalates: stdin close, a 2s graceful
# wait, then a process-tree kill with its own 2s budget. The drain must
# outlast that escalation, or the loop closes mid-kill and a server that
# ignores stdin EOF survives the daemon.
_LOOP_DRAIN_SEC = 5.0


def _validated_seconds(raw: Any) -> Optional[float]:
    """``raw`` as a finite positive float, or None when it is not one.

    Booleans convert to floats, so they are refused explicitly: a config
    of ``true`` silently meaning "one second" is nobody's intent.
    ``Future.result(timeout=...)`` has undefined wait semantics for
    ``nan`` and overflows on ``inf``, and a zero or negative budget
    fails every call at once.
    """
    if raw is None or isinstance(raw, bool):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value <= 0:
        return None
    return value


def _resolve_invoke_timeout(
    server_cfg: Dict[str, Any], override: Optional[Any] = None
) -> float:
    """Return the budget for one round trip: the explicit ``override``
    when it validates, else the server's ``timeout_sec``, else
    ``_DEFAULT_INVOKE_TIMEOUT_SEC``. Every rejected candidate announces
    itself in the debug log rather than failing the call in a way nobody
    can trace back to a config typo."""
    if server_cfg.get("timeout_sec") is None and server_cfg.get("timeout") is not None:
        debug_log(
            f"mcp server config carries 'timeout'={server_cfg['timeout']!r}; "
            "the key is named 'timeout_sec', and 'timeout' is not read",
            "mcp",
        )
    for source, raw in (
        ("explicit timeout_sec", override),
        ("server timeout_sec", server_cfg.get("timeout_sec")),
    ):
        if raw is None:
            continue
        value = _validated_seconds(raw)
        if value is None:
            debug_log(
                f"mcp {source}={raw!r} is not a finite positive number; "
                "falling back",
                "mcp",
            )
            continue
        return value
    return _DEFAULT_INVOKE_TIMEOUT_SEC


_runtime_lock = threading.Lock()
_runtime: Optional["_PersistentMCPRuntime"] = None
_shutdown_requested = False


def get_runtime() -> "_PersistentMCPRuntime":
    """Return the shared persistent runtime, starting it on first use.

    After ``shutdown_runtime()`` there is no runtime to return: the
    teardown is terminal for the daemon run, and resurrecting a fresh
    thread and fresh subprocesses for a straggler call would leave them
    unreaped. The next ``daemon.main()`` re-arms the latch at startup.
    The ``RuntimeError`` is an ordinary ``Exception``, so the tool
    funnel records the failed call like any session failure.
    """
    global _runtime
    with _runtime_lock:
        if _shutdown_requested:
            raise RuntimeError(
                "Persistent MCP runtime is shut down; "
                "it returns with the next daemon run"
            )
        if _runtime is None or _runtime.closed:
            _runtime = _PersistentMCPRuntime()
        return _runtime


def shutdown_runtime() -> None:
    """Tear down the shared runtime. Safe to call multiple times."""
    global _runtime, _shutdown_requested
    with _runtime_lock:
        _shutdown_requested = True
        instance = _runtime
        _runtime = None
    if instance is not None:
        try:
            instance.shutdown()
        except Exception as e:  # noqa: BLE001
            debug_log(f"persistent MCP runtime shutdown error: {e}", "mcp")


def reset_shutdown_latch() -> None:
    """Re-arm ``get_runtime()`` after ``shutdown_runtime()``.

    The latch is terminal for a daemon run, not for the process:
    ``daemon.main()`` calls this at startup, and the bundled desktop
    app re-runs ``main()`` in-process (tray toggle, settings restart,
    setup wizard). The test suite calls it between tests for the same
    reason.
    """
    global _shutdown_requested
    with _runtime_lock:
        _shutdown_requested = False


class _PersistentMCPRuntime:
    """Owns the background event loop and the per-server worker tasks."""

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._workers: Dict[str, "_ServerWorker"] = {}
        self._workers_lock = threading.Lock()
        self.closed = False
        self._start_loop()

    def _start_loop(self) -> None:
        loop_ready = threading.Event()

        def _runner() -> None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            loop_ready.set()
            try:
                self._loop.run_forever()
            finally:
                try:
                    # Cancel any leftover tasks, then let the
                    # cancellations run before closing: a worker
                    # destroyed while still pending never reaches its
                    # finally drain, and a caller waiting on it would
                    # sit out its whole budget against a dead loop.
                    # Bounded, so a task wedged in uncancellable work
                    # delays the close by at most ``_LOOP_DRAIN_SEC``.
                    pending = asyncio.all_tasks(self._loop)
                    for task in pending:
                        task.cancel()
                    if pending:
                        self._loop.run_until_complete(
                            asyncio.wait(pending, timeout=_LOOP_DRAIN_SEC)
                        )
                except Exception as e:  # noqa: BLE001
                    debug_log(f"MCP runtime task cleanup error: {e}", "mcp")
                try:
                    self._loop.close()
                except Exception as e:  # noqa: BLE001
                    debug_log(f"MCP runtime loop close error: {e}", "mcp")

        self._thread = threading.Thread(
            target=_runner, daemon=True, name="JarvisMCPRuntime"
        )
        self._thread.start()
        if not loop_ready.wait(timeout=5):
            raise RuntimeError("Persistent MCP runtime event loop failed to start")

    def invoke(
        self,
        server_name: str,
        server_cfg: Dict[str, Any],
        tool_name: str,
        arguments: Optional[Dict[str, Any]],
        timeout: Optional[float] = None,
    ) -> Any:
        """Call a tool on the named server, retrying once if the worker died.

        ``timeout`` bounds the call_tool round trip (not setup); when not
        given, the budget is resolved from ``server_cfg``'s
        ``timeout_sec``, falling back to ``_DEFAULT_INVOKE_TIMEOUT_SEC``.
        A call that exceeds its budget drops the worker and surfaces as
        ``MCPCallTimeoutError``; it is never retried, because the
        side-effecting tool it timed out on may have run. Only a worker
        that died (e.g. the subprocess crashed) triggers the single retry
        with a fresh worker, whose second failure surfaces as
        ``MCPServerSessionError`` at the public layer; a session error
        translated out of a cancellation surfaces directly, without a
        retry, because the call may have executed.
        """
        effective_timeout = _resolve_invoke_timeout(server_cfg, timeout)

        worker = self._get_worker(server_name, server_cfg)
        try:
            return worker.invoke(tool_name, arguments, effective_timeout)
        except _WorkerDeadError:
            # Subprocess crashed mid-call: retry once with a fresh worker
            # so a transient server failure does not poison the cache.
            debug_log(
                f"MCP worker '{server_name}' died; restarting and retrying once",
                "mcp",
            )
            self._drop_worker(server_name)
            worker = self._get_worker(server_name, server_cfg)
            try:
                return worker.invoke(tool_name, arguments, effective_timeout)
            except MCPCallTimeoutError:
                self._drop_worker(server_name)
                raise
        except MCPCallTimeoutError:
            # The worker shut itself down in ``_submit``; evict it so the
            # next call starts a fresh session. The call itself is not
            # retried.
            self._drop_worker(server_name)
            raise

    def list_tools(
        self,
        server_name: str,
        server_cfg: Dict[str, Any],
        timeout: Optional[float] = None,
    ) -> Any:
        """List tools on the named server, reusing the persistent session.

        Routes discovery through the same worker used for tool calls so
        that the subprocess started during discovery is the one that
        services subsequent ``call_tool`` requests. This avoids the
        startup cost of spawning the server twice (once for discovery,
        once for the first invocation).

        ``timeout`` bounds the discovery round trip (not setup) and is
        resolved exactly like ``invoke``'s. A discovery that exceeds its
        budget drops the worker and surfaces as ``MCPCallTimeoutError``;
        it is never retried.
        """
        effective_timeout = _resolve_invoke_timeout(server_cfg, timeout)

        worker = self._get_worker(server_name, server_cfg)
        try:
            return worker.list_tools(effective_timeout)
        except _WorkerDeadError:
            debug_log(
                f"MCP worker '{server_name}' died during list_tools; restarting",
                "mcp",
            )
            self._drop_worker(server_name)
            worker = self._get_worker(server_name, server_cfg)
            try:
                return worker.list_tools(effective_timeout)
            except MCPCallTimeoutError:
                self._drop_worker(server_name)
                raise
        except MCPCallTimeoutError:
            self._drop_worker(server_name)
            raise

    def _get_worker(
        self, server_name: str, server_cfg: Dict[str, Any]
    ) -> "_ServerWorker":
        """Return a live worker for ``server_name``, replacing it if needed.

        Reuses an existing worker iff it is still alive and its cached
        config equals the requested one. A dead worker or a config
        change triggers shutdown of the old worker and creation of a
        fresh one. ``worker.start()`` runs under ``_workers_lock``: a
        setup that hangs blocks calls to every server for up to
        ``_SETUP_TIMEOUT_SEC``, plus the failed-start teardown's
        bounded sentinel wait.
        """
        with self._workers_lock:
            if self.closed:
                # The runtime is shutting down. A worker spawned now
                # would ride a loop that is about to close: destroyed
                # mid-handshake, subprocess possibly unreaped, and its
                # caller stuck in the setup budget during teardown.
                raise MCPServerSessionError(
                    f"MCP runtime is shut down; call to '{server_name}' refused"
                )
            existing = self._workers.get(server_name)
            if existing is not None and existing.alive and existing.config == server_cfg:
                return existing
            if existing is not None:
                # Config changed or worker dead: replace it.
                try:
                    existing.shutdown()
                except Exception as e:  # noqa: BLE001
                    debug_log(
                        f"MCP worker '{server_name}' replacement shutdown error: {e}",
                        "mcp",
                    )
            loop = self._loop
            if loop is None:
                raise RuntimeError(
                    "Persistent MCP runtime event loop is not available"
                )
            worker = _ServerWorker(loop, server_name, server_cfg)
            try:
                worker.start()
            except BaseException:
                # A half-started worker is never cached, so nothing
                # else can ever reach it: tear it down here, or its
                # task and spawned subprocess outlive the attempt and
                # stack up one per retry.
                try:
                    worker.shutdown()
                except Exception as shutdown_err:  # noqa: BLE001
                    debug_log(
                        f"MCP worker '{server_name}' failed-start "
                        f"shutdown error: {shutdown_err}",
                        "mcp",
                    )
                raise
            self._workers[server_name] = worker
            return worker

    def _drop_worker(self, server_name: str) -> None:
        """Forcibly evict and shut down the cached worker for ``server_name``.

        Used after the worker has signalled it is no longer servicing
        requests (e.g. a ``_WorkerDeadError``). Safe to call when no
        worker is cached.
        """
        with self._workers_lock:
            worker = self._workers.pop(server_name, None)
        if worker is not None:
            try:
                worker.shutdown()
            except Exception as e:  # noqa: BLE001
                debug_log(
                    f"MCP worker '{server_name}' drop shutdown error: {e}", "mcp"
                )

    def shutdown(self) -> None:
        if self.closed:
            return
        self.closed = True
        with self._workers_lock:
            workers = list(self._workers.values())
            self._workers.clear()
        # Ask every worker to exit cleanly first; cancel the task if the
        # graceful path stalls (e.g. a hung call_tool).
        for w in workers:
            try:
                w.shutdown()
            except Exception as e:  # noqa: BLE001
                debug_log(
                    f"MCP worker '{w._server_name}' shutdown error: {e}", "mcp"
                )
        loop = self._loop
        if loop is not None:
            try:
                loop.call_soon_threadsafe(loop.stop)
            except Exception as e:  # noqa: BLE001
                debug_log(f"MCP runtime loop.stop error: {e}", "mcp")
        if self._thread is not None:
            self._thread.join(timeout=_SHUTDOWN_THREAD_JOIN_SEC)
            if self._thread.is_alive():
                debug_log(
                    "MCP runtime thread did not exit within shutdown timeout",
                    "mcp",
                )


class _WorkerDeadError(RuntimeError):
    """Internal sentinel: the worker's stdio session is no longer servicing
    requests. ``_PersistentMCPRuntime`` catches this to retry once with a
    fresh worker; the public ``MCPClient`` layer wraps it as
    ``MCPServerSessionError`` if it escapes the retry."""


class MCPCallTimeoutError(TimeoutError):
    """A call exceeded its time budget.

    The worker is shut down (a session with a call still in flight is
    not trustworthy) and the call is not retried: the tool may be a slow
    side-effecting one that already ran, and running it twice after one
    approval is worse than reporting the timeout. The message names the
    server, the tool (or ``list_tools``) and the seconds.
    """


def _caller_facing(e: BaseException, server_name: str) -> Exception:
    """The exception a waiting caller receives for a failed command.

    ``Exception`` subclasses travel as-is. A bare ``BaseException``
    (a cancellation delivered by a sibling call's timeout, an anyio
    task-group teardown) would escape every ``except Exception`` on
    the caller's path: the tool funnel would skip the ledger row for
    a gate-approved call that may have executed, and the turn would
    die. The caller is an ordinary thread rather than a task, so the
    honest and catchable form is a session error naming what arrived,
    with the original kept as ``__cause__``.
    """
    if isinstance(e, Exception):
        return e
    translated = MCPServerSessionError(
        f"MCP server '{server_name}' session lost while servicing call: "
        f"{type(e).__name__}"
    )
    translated.__cause__ = e
    return translated


class _Command:
    """One queued request for the worker.

    ``pulled`` is set by the worker task the moment the command leaves
    the queue, before any side effect can start. ``_submit`` reads it
    after a budget expiry to tell the two failures apart: a pulled
    command may have executed and is never retried, while an unpulled
    one can no longer run (the worker is shut down) and is safe to
    retry.
    """

    __slots__ = ("kind", "payload", "fut", "pulled")

    def __init__(
        self, kind: str, payload: Any, fut: "concurrent.futures.Future"
    ) -> None:
        self.kind = kind
        self.payload = payload
        self.fut = fut
        self.pulled = False


class _ServerWorker:
    """Holds a single stdio session open and dispatches tool calls.

    The worker task lives on the runtime's background loop. Callers from
    other threads enqueue ``(kind, payload, future)`` tuples (where
    ``kind`` is ``"call"`` or ``"list"``); the task pulls them off the
    queue and resolves each future with the result (or exception).
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        server_name: str,
        server_cfg: Dict[str, Any],
    ) -> None:
        self._loop = loop
        self._server_name = server_name
        self.config = dict(server_cfg)
        self._queue: Optional[asyncio.Queue] = None
        self._task: Optional[asyncio.Task] = None
        self._ready: concurrent.futures.Future = concurrent.futures.Future()
        self.alive = True
        # ``idle_timeout_sec`` opts in to self-termination after a period
        # of inactivity. ``None`` (default) means the worker stays
        # resident for the runtime's lifetime — required for stateful
        # servers like chrome-devtools-mcp. Only a finite positive number
        # opts in; anything else means no idle timeout and announces
        # itself in the debug log.
        idle = server_cfg.get("idle_timeout_sec")
        self._idle_timeout: Optional[float] = _validated_seconds(idle)
        if idle is not None and self._idle_timeout is None:
            debug_log(
                f"mcp server '{server_name}' idle_timeout_sec={idle!r} is not "
                "a finite positive number; the worker stays resident",
                "mcp",
            )

    def start(self) -> None:
        async def _setup() -> None:
            if not self.alive:
                # ``start()`` failed (the loop stalled past the schedule
                # budget) and the worker was torn down before this
                # callback ran. Spawning now would create a task and a
                # subprocess nothing can reach: the worker is cached
                # only after a successful start. When the loop stalled
                # the other way around and the task exists, the
                # teardown's sentinel and cancel handle it instead.
                return
            self._queue = asyncio.Queue()
            self._task = asyncio.ensure_future(self._run())

        asyncio.run_coroutine_threadsafe(_setup(), self._loop).result(
            timeout=_SETUP_SCHEDULE_TIMEOUT_SEC
        )
        # Block until the worker has initialised the MCP session, or
        # surfaced a startup error. Without this, the first ``invoke``
        # would race the session handshake.
        self._ready.result(timeout=_SETUP_TIMEOUT_SEC)

    async def _run(self) -> None:
        try:
            client = MCPClient({self._server_name: self.config})
            connection = client._connect_stdio(self.config)
            # Resolve ClientSession through ``mcp_client`` so tests that
            # monkey-patch ``mcp_client.ClientSession`` reach this path.
            client_session_cls = _mcp_client_module.ClientSession
            t_start = time.monotonic()
            async with connection as (read, write):
                async with client_session_cls(read, write) as session:
                    await session.initialize()
                    if not self._ready.done():
                        self._ready.set_result(True)
                    debug_log(
                        f"MCP persistent session ready: {self._server_name} "
                        f"({time.monotonic() - t_start:.2f}s)",
                        "mcp",
                    )
                    if self._queue is None:
                        # Setup must have created the queue before the
                        # task started. If we somehow get here with no
                        # queue, treat it as a setup failure.
                        raise RuntimeError(
                            "MCP worker queue not initialised before run"
                        )
                    while True:
                        # ``BaseException`` here is intentional: anyio's
                        # task-group cancellation surfaces as
                        # ``BaseExceptionGroup``/``CancelledError`` which
                        # are ``BaseException`` subclasses. Without
                        # catching them the awaiting future would never
                        # be resolved, leaving the caller stuck.
                        try:
                            cmd = await self._queue_get_with_idle()
                        except _IdleTimeout:
                            debug_log(
                                f"MCP worker '{self._server_name}' idle "
                                f"({self._idle_timeout}s); shutting down",
                                "mcp",
                            )
                            return
                        if cmd is None:
                            return
                        if not self.alive:
                            # The worker was shut down while this
                            # command sat in the queue (a sibling
                            # call's expiry, a config-change
                            # replacement, the daemon's teardown),
                            # and the exit sentinel is behind it:
                            # without this refusal it would still
                            # execute. It never ran, so the death
                            # sentinel is the honest resolution: a
                            # caller still waiting gets the runtime's
                            # single retry, and a caller already
                            # answered keeps the report it received.
                            self._refuse(cmd)
                            continue
                        # Set before anything else: from this point the
                        # call may execute, and a budget expiry on the
                        # caller side must classify it as a timeout
                        # rather than a death that is safe to retry.
                        cmd.pulled = True
                        if not self.alive:
                            # The re-check closes the window between
                            # the liveness check and the mark. A
                            # shutdown landing inside it would
                            # otherwise leave a command on its way to
                            # execution visible to an expiring caller
                            # as "unpulled on a dead worker", the one
                            # shape the runtime retries. Either order
                            # is safe: a flip before the re-check
                            # refuses a command that never ran, and a
                            # flip after it is invisible to the
                            # classification, because a caller reads
                            # ``pulled`` only after its own
                            # ``shutdown()`` returned, and that is
                            # what flips ``alive``.
                            self._refuse(cmd)
                            continue
                        kind, payload, fut = cmd.kind, cmd.payload, cmd.fut
                        try:
                            if kind == "call":
                                tool_name, arguments = payload
                                res = await session.call_tool(
                                    tool_name, arguments or {}
                                )
                            elif kind == "list":
                                res = await session.list_tools()
                            else:
                                raise ValueError(
                                    f"Unknown worker command kind: {kind!r}"
                                )
                            if not fut.done():
                                fut.set_result(res)
                        except BaseException as e:  # noqa: BLE001
                            if not fut.done():
                                fut.set_exception(
                                    _caller_facing(e, self._server_name)
                                )
        except BaseException as e:  # noqa: BLE001
            # Setup or session loop crashed. Surface to ``start()`` if
            # we never signalled readiness; otherwise log and let the
            # finally block notify any in-flight callers. Either way
            # the caller receives an ``Exception`` the funnel catches,
            # never a bare ``BaseException``.
            if not self._ready.done():
                self._ready.set_exception(
                    _caller_facing(e, self._server_name)
                )
            else:
                debug_log(
                    f"MCP persistent session '{self._server_name}' exited: {e}",
                    "mcp",
                )
        finally:
            self.alive = False
            # Drain any in-flight requests so callers don't hang forever.
            # Queued commands are by definition unpulled: they never ran,
            # so the death sentinel here licenses the runtime's retry.
            if self._queue is not None:
                while True:
                    try:
                        cmd = self._queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    if cmd is None:
                        continue
                    self._refuse(cmd)

    def _refuse(self, cmd: "_Command") -> None:
        """Resolve a command that will never execute with the death
        sentinel, so the runtime's single retry is safe for it."""
        if not cmd.fut.done():
            cmd.fut.set_exception(
                _WorkerDeadError(
                    f"MCP server '{self._server_name}' session ended"
                )
            )

    async def _queue_get_with_idle(self) -> Any:
        """Await the next command, honouring ``idle_timeout_sec`` if set."""
        if self._queue is None:
            raise RuntimeError("MCP worker queue not initialised")
        if self._idle_timeout is None:
            return await self._queue.get()
        try:
            return await asyncio.wait_for(
                self._queue.get(), timeout=self._idle_timeout
            )
        except asyncio.TimeoutError:
            raise _IdleTimeout()

    def invoke(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]],
        timeout: float,
    ) -> Any:
        """Submit a ``call_tool`` request and wait up to ``timeout`` seconds.

        A live worker whose tool simply takes too long raises
        ``MCPCallTimeoutError`` and is shut down; the call is not
        re-run, because the tool may be a side-effecting one that
        already executed. Only a worker that died (its queue drained
        without ever resolving our future) becomes ``_WorkerDeadError``,
        which the runtime answers with its single retry.
        """
        return self._submit(("call", (tool_name, arguments)), timeout)

    def list_tools(self, timeout: float) -> Any:
        """Submit a ``list_tools`` request through the persistent session."""
        return self._submit(("list", None), timeout)

    def _submit(self, cmd: Any, timeout: float) -> Any:
        if not self.alive:
            raise _WorkerDeadError(
                f"MCP server '{self._server_name}' is not alive"
            )
        queue = self._queue
        if queue is None:
            raise _WorkerDeadError(
                f"MCP server '{self._server_name}' queue not initialised"
            )
        kind, payload = cmd
        fut: concurrent.futures.Future = concurrent.futures.Future()
        command = _Command(kind, payload, fut)
        # Single cross-thread hop: schedule the put on the loop and
        # wait on the result future. ``put_nowait`` is safe because
        # the queue is unbounded.
        self._loop.call_soon_threadsafe(
            queue.put_nowait, command
        )
        try:
            return fut.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            # What the expiry means depends on how far the command got,
            # and ``pulled`` (set by the worker at dequeue, before any
            # side effect can start) is what says so. A pulled command
            # may have executed: it is never retried, whatever else
            # happened to the worker in the interim (a config-change
            # replacement or the daemon's teardown flipping ``alive``
            # mid-call must not turn an executed call into a retried
            # one). An unpulled command on a worker already dead at the
            # expiry never ran, and the runtime's single retry is safe.
            #
            # The classification reads the future BEFORE the teardown:
            # what the call resolved to on its own is its honest
            # outcome. After ``shutdown()`` only a result is honoured
            # (a success that landed at the wire beats the expiry that
            # raced it); an exception landing after the teardown began
            # is the teardown's own echo — the cancellation this
            # timeout delivered, possibly through a sibling caller's —
            # and the expiry is the honest report.
            was_alive = self.alive
            resolved = fut.done()
            inner = fut.exception() if resolved else None
            self.shutdown()
            if command.pulled:
                if fut.done() and fut.exception() is None:
                    return fut.result()
                if resolved and isinstance(inner, Exception):
                    # The call's own failure (a tool error, a session
                    # exception, a tool-side timeout): propagates as-is
                    # rather than being relabelled as a budget expiry.
                    raise inner
            elif isinstance(inner, _WorkerDeadError) or not was_alive:
                raise _WorkerDeadError(
                    f"MCP server '{self._server_name}' died while servicing call"
                ) from None
            label = f"tool '{payload[0]}'" if kind == "call" else "list_tools"
            debug_log(
                f"MCP {label} on server '{self._server_name}' exceeded "
                f"{timeout:g}s; dropping the worker without a retry",
                "mcp",
            )
            raise MCPCallTimeoutError(
                f"MCP server '{self._server_name}' timed out after "
                f"{timeout:g}s servicing {label}"
            ) from None

    def shutdown(self) -> None:
        """Best-effort graceful stop, falling back to task cancellation."""
        was_alive = self.alive
        self.alive = False
        if not was_alive:
            return
        # Try the polite path first: enqueue a sentinel so the worker
        # exits its loop after the current call (if any).
        if self._queue is not None:
            try:
                asyncio.run_coroutine_threadsafe(
                    self._queue.put(None), self._loop
                ).result(timeout=2)
            except Exception as e:  # noqa: BLE001
                debug_log(
                    f"MCP worker '{self._server_name}' sentinel enqueue error: {e}",
                    "mcp",
                )
        # If the worker is wedged inside ``call_tool`` it will not see
        # the sentinel. Cancel the task so the loop can stop and the
        # subprocess exits.
        task = self._task
        if task is not None and not task.done():
            try:
                self._loop.call_soon_threadsafe(task.cancel)
            except Exception as e:  # noqa: BLE001
                debug_log(
                    f"MCP worker '{self._server_name}' task cancel error: {e}",
                    "mcp",
                )


class _IdleTimeout(Exception):
    """Internal signal: the idle timeout elapsed without activity."""

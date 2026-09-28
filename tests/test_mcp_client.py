import asyncio
import os
import sys
import pytest


@pytest.fixture
def shutdown_persistent_runtime():
    """Tear down the persistent MCP runtime singleton between tests."""
    yield
    try:
        from jarvis.tools.external.mcp_runtime import shutdown_runtime
        shutdown_runtime()
    except Exception:
        pass


def _make_tracked_doubles(call_count, enter_count, exit_count, *, fail_on_call=None,
                          tools_payload=None, delays=None, list_count=None,
                          call_hook=None):
    """Build patchable doubles for ``stdio_client`` and ``ClientSession``.

    ``fail_on_call`` may be a list whose values trigger ``call_tool`` to
    raise ``RuntimeError(value)`` on the matching invocation index. A
    ``None`` entry means succeed normally.

    ``delays`` is a mutable dict read at call time (``{"call": secs,
    "list": secs}``) so one test can flip between a slow and a fast tool
    without rebuilding the doubles; the await happens after the counter
    increments, so a call cancelled mid-sleep still counts as started.
    ``list_count`` counts ``list_tools`` invocations the same way
    ``call_count`` counts ``call_tool`` ones. ``call_hook`` is called
    with the tool name at ``call_tool`` entry, before any delay, so a
    test can synchronise on a call being in flight.
    """
    fail_on_call = list(fail_on_call or [])
    delays = delays if delays is not None else {}
    list_count = list_count if list_count is not None else {"n": 0}

    class TrackedConn:
        async def __aenter__(self_):
            enter_count["n"] += 1
            return object(), object()

        async def __aexit__(self_, *a):
            exit_count["n"] += 1
            return False

    class TrackedSession:
        def __init__(self_, read, write):
            pass

        async def __aenter__(self_):
            class _S:
                async def initialize(_self):
                    return None

                async def call_tool(_self, name, arguments):
                    idx = call_count["n"]
                    call_count["n"] += 1
                    if call_hook is not None:
                        call_hook(name)
                    delay = delays.get("call", 0.0)
                    if delay:
                        await asyncio.sleep(delay)
                    if idx < len(fail_on_call) and fail_on_call[idx] is not None:
                        raise RuntimeError(fail_on_call[idx])
                    return type(
                        "R",
                        (),
                        {"content": f"called:{name}:{arguments}", "isError": False, "meta": None},
                    )()

                async def list_tools(_self):
                    list_count["n"] += 1
                    delay = delays.get("list", 0.0)
                    if delay:
                        await asyncio.sleep(delay)
                    payload = tools_payload or []
                    fake_tools = [
                        type("T", (), {"name": n, "description": d, "inputSchema": s})()
                        for (n, d, s) in payload
                    ]
                    return type("LR", (), {"tools": fake_tools})()

            return _S()

        async def __aexit__(self_, *a):
            return False

    return TrackedConn, TrackedSession


def _patch_mcp_doubles(monkeypatch, TrackedConn, TrackedSession):
    monkeypatch.setattr(
        "jarvis.tools.external.mcp_client._resolve_command", lambda c: c
    )
    monkeypatch.setattr(
        "jarvis.tools.external.mcp_client.stdio_client",
        lambda params, **kw: TrackedConn(),
    )
    monkeypatch.setattr(
        "jarvis.tools.external.mcp_client.ClientSession", TrackedSession
    )


@pytest.mark.unit
def test_invoke_tool_keeps_mcp_session_alive_across_calls(monkeypatch, shutdown_persistent_runtime):
    """Stateful MCP servers (e.g. chrome-devtools-mcp) launch child processes
    such as a browser that die when the server's stdio session is torn down.
    Two consecutive invocations on the same server must share a single
    long-lived stdio session, not spawn the server subprocess twice.
    """
    from jarvis.tools.external.mcp_client import MCPClient

    enter_count = {"n": 0}
    exit_count = {"n": 0}
    call_count = {"n": 0}

    TrackedConn, TrackedSession = _make_tracked_doubles(
        call_count, enter_count, exit_count
    )
    _patch_mcp_doubles(monkeypatch, TrackedConn, TrackedSession)

    mcps = {"persist": {"transport": "stdio", "command": "/bin/true", "args": []}}
    client = MCPClient(mcps)

    r1 = client.invoke_tool("persist", "alpha", {"x": 1})
    r2 = client.invoke_tool("persist", "beta", {"y": 2})

    assert call_count["n"] == 2
    assert enter_count["n"] == 1, (
        f"stdio connection must be opened once for stateful MCP servers, "
        f"was opened {enter_count['n']} times"
    )
    assert exit_count["n"] == 0, (
        "stdio connection must remain open across invocations"
    )
    # Sanity: results pass through unchanged
    assert r1["isError"] is False
    assert r2["isError"] is False


@pytest.mark.unit
def test_invoke_tool_retries_on_transient_session_loss(
    monkeypatch, shutdown_persistent_runtime
):
    """If a worker raises ``_WorkerDeadError`` (its stdio session ended
    mid-call), the runtime must drop it, spawn a fresh one and retry
    once. Observable behaviour: the second invocation succeeds even
    though the first underlying worker call failed with the sentinel.
    """
    from jarvis.tools.external.mcp_client import MCPClient
    from jarvis.tools.external import mcp_runtime as _runtime_mod

    enter_count = {"n": 0}
    exit_count = {"n": 0}
    call_count = {"n": 0}

    TrackedConn, TrackedSession = _make_tracked_doubles(
        call_count, enter_count, exit_count
    )
    _patch_mcp_doubles(monkeypatch, TrackedConn, TrackedSession)

    real_invoke = _runtime_mod._ServerWorker.invoke
    invoke_calls = {"n": 0}

    def flaky_invoke(self, tool_name, arguments, timeout):
        invoke_calls["n"] += 1
        if invoke_calls["n"] == 1:
            # Simulate the worker discovering its session is dead.
            raise _runtime_mod._WorkerDeadError("simulated session loss")
        return real_invoke(self, tool_name, arguments, timeout)

    monkeypatch.setattr(_runtime_mod._ServerWorker, "invoke", flaky_invoke)

    client = MCPClient(
        {"flaky": {"transport": "stdio", "command": "/bin/true", "args": []}}
    )

    res = client.invoke_tool("flaky", "alpha", {"x": 1})

    assert res["isError"] is False
    assert invoke_calls["n"] == 2, (
        "runtime should retry exactly once after _WorkerDeadError"
    )
    assert enter_count["n"] == 2, (
        "the retry must spawn a fresh stdio connection (new worker)"
    )


@pytest.mark.unit
def test_get_worker_replaces_on_config_change(
    monkeypatch, shutdown_persistent_runtime
):
    """Changing a server's config (e.g. updated args) must cause the
    runtime to replace the existing worker with a fresh one so the new
    subprocess actually receives the new arguments.
    """
    from jarvis.tools.external.mcp_client import MCPClient

    enter_count = {"n": 0}
    exit_count = {"n": 0}
    call_count = {"n": 0}

    TrackedConn, TrackedSession = _make_tracked_doubles(
        call_count, enter_count, exit_count
    )
    _patch_mcp_doubles(monkeypatch, TrackedConn, TrackedSession)

    cfg_v1 = {"transport": "stdio", "command": "/bin/true", "args": []}
    cfg_v2 = {"transport": "stdio", "command": "/bin/true", "args": ["--flag"]}

    client_v1 = MCPClient({"swap": cfg_v1})
    client_v1.invoke_tool("swap", "alpha", {})
    assert enter_count["n"] == 1

    client_v2 = MCPClient({"swap": cfg_v2})
    client_v2.invoke_tool("swap", "alpha", {})
    assert enter_count["n"] == 2, "config change must spawn a new stdio session"


@pytest.mark.unit
def test_worker_startup_failure_propagates(monkeypatch, shutdown_persistent_runtime):
    """If session initialisation fails (e.g. subprocess cannot start),
    the failure must propagate to the caller rather than hang.
    """
    from jarvis.tools.external.mcp_client import (
        MCPClient,
        MCPServerSessionError,
    )

    monkeypatch.setattr(
        "jarvis.tools.external.mcp_client._resolve_command", lambda c: c
    )

    def _broken_stdio_client(params, **kw):
        raise FileNotFoundError("simulated subprocess spawn failure")

    monkeypatch.setattr(
        "jarvis.tools.external.mcp_client.stdio_client", _broken_stdio_client
    )

    client = MCPClient(
        {"broken": {"transport": "stdio", "command": "/bin/true", "args": []}}
    )

    with pytest.raises((FileNotFoundError, MCPServerSessionError, RuntimeError)):
        client.invoke_tool("broken", "alpha", {})


@pytest.mark.unit
def test_runtime_isolates_workers_per_server(
    monkeypatch, shutdown_persistent_runtime
):
    """Two distinct servers must each have their own stdio session;
    invoking one must not interfere with the other.
    """
    from jarvis.tools.external.mcp_client import MCPClient

    enter_count = {"n": 0}
    exit_count = {"n": 0}
    call_count = {"n": 0}

    TrackedConn, TrackedSession = _make_tracked_doubles(
        call_count, enter_count, exit_count
    )
    _patch_mcp_doubles(monkeypatch, TrackedConn, TrackedSession)

    mcps = {
        "alpha": {"transport": "stdio", "command": "/bin/true", "args": []},
        "beta": {"transport": "stdio", "command": "/bin/true", "args": []},
    }
    client = MCPClient(mcps)

    client.invoke_tool("alpha", "x", {})
    client.invoke_tool("beta", "y", {})
    client.invoke_tool("alpha", "x", {})

    assert call_count["n"] == 3
    assert enter_count["n"] == 2, (
        "each server should open exactly one stdio connection regardless of "
        "the order calls arrive in"
    )


@pytest.mark.unit
def test_list_tools_uses_persistent_session(
    monkeypatch, shutdown_persistent_runtime
):
    """Discovery and the first ``invoke_tool`` should share a single
    stdio session — listing then invoking must not spawn the server
    twice.
    """
    from jarvis.tools.external.mcp_client import MCPClient

    enter_count = {"n": 0}
    exit_count = {"n": 0}
    call_count = {"n": 0}

    TrackedConn, TrackedSession = _make_tracked_doubles(
        call_count,
        enter_count,
        exit_count,
        tools_payload=[
            ("alpha", "first tool", {"type": "object"}),
            ("beta", "second tool", {"type": "object"}),
        ],
    )
    _patch_mcp_doubles(monkeypatch, TrackedConn, TrackedSession)

    client = MCPClient(
        {"shared": {"transport": "stdio", "command": "/bin/true", "args": []}}
    )

    listed = client.list_tools("shared")
    assert {t["name"] for t in listed} == {"alpha", "beta"}

    client.invoke_tool("shared", "alpha", {})

    assert enter_count["n"] == 1, (
        "list_tools and invoke_tool should reuse the same stdio session"
    )


@pytest.mark.unit
def test_a_timed_out_call_fails_once_as_a_timeout_and_is_not_retried(
    monkeypatch, shutdown_persistent_runtime
):
    """A tool slower than its budget: ``call_tool`` starts exactly once,
    the caller sees a ``TimeoutError`` (not ``MCPServerSessionError``) in
    about ``timeout_sec``, and the next call opens a fresh connection.

    The timed-out tool is a side-effecting one that may still be running;
    executing it again after one approval is executing it twice, which is
    exactly what a silent retry after a timeout does.
    """
    import time

    from jarvis.tools.external.mcp_client import MCPClient, MCPServerSessionError

    enter_count = {"n": 0}
    exit_count = {"n": 0}
    call_count = {"n": 0}
    delays = {"call": 1.0}

    TrackedConn, TrackedSession = _make_tracked_doubles(
        call_count, enter_count, exit_count, delays=delays
    )
    _patch_mcp_doubles(monkeypatch, TrackedConn, TrackedSession)

    client = MCPClient(
        {"slow": {"transport": "stdio", "command": "/bin/true", "args": []}}
    )

    started = time.monotonic()
    with pytest.raises(TimeoutError) as excinfo:
        client.invoke_tool("slow", "alpha", {"x": 1}, timeout_sec=0.3)
    elapsed = time.monotonic() - started

    assert call_count["n"] == 1, "a timed-out call must never run again"
    assert not isinstance(excinfo.value, MCPServerSessionError), (
        "a timeout must surface as a timeout, not as a session failure"
    )
    message = str(excinfo.value)
    assert "slow" in message, message
    assert "alpha" in message, message
    assert "0.3" in message, message
    # The counters and the exception type are the deterministic pins;
    # this bound only guards against a hang (the tool sleeps 1.0s).
    assert elapsed < 2.0, f"should fail in about timeout_sec, took {elapsed:.2f}s"

    # The timed-out worker is dropped: the next call opens a fresh connection.
    delays["call"] = 0.0
    res = client.invoke_tool("slow", "beta", {}, timeout_sec=5.0)
    assert res["isError"] is False
    assert enter_count["n"] == 2, "the next call must start a fresh session"
    assert call_count["n"] == 2


@pytest.mark.unit
def test_a_timed_out_discovery_fails_once_as_a_timeout_and_is_not_retried(
    monkeypatch, shutdown_persistent_runtime
):
    """The same contract for ``list_tools``: one attempt, a timeout that
    names the server, the operation and the budget, then a fresh session
    for the next call."""
    import time

    from jarvis.tools.external.mcp_client import MCPClient, MCPServerSessionError

    enter_count = {"n": 0}
    exit_count = {"n": 0}
    call_count = {"n": 0}
    list_count = {"n": 0}
    delays = {"list": 1.0}

    TrackedConn, TrackedSession = _make_tracked_doubles(
        call_count,
        enter_count,
        exit_count,
        tools_payload=[("alpha", "a tool", {"type": "object"})],
        delays=delays,
        list_count=list_count,
    )
    _patch_mcp_doubles(monkeypatch, TrackedConn, TrackedSession)

    client = MCPClient(
        {"slow": {"transport": "stdio", "command": "/bin/true", "args": []}}
    )

    started = time.monotonic()
    with pytest.raises(TimeoutError) as excinfo:
        client.list_tools("slow", timeout_sec=0.3)
    elapsed = time.monotonic() - started

    assert list_count["n"] == 1, "a timed-out discovery must never run again"
    assert not isinstance(excinfo.value, MCPServerSessionError)
    message = str(excinfo.value)
    assert "slow" in message, message
    assert "list_tools" in message, message
    assert "0.3" in message, message
    # The counters and the exception type are the deterministic pins;
    # this bound only guards against a hang (the discovery sleeps 1.0s).
    assert elapsed < 2.0, f"should fail in about timeout_sec, took {elapsed:.2f}s"

    delays["list"] = 0.0
    listed = client.list_tools("slow", timeout_sec=5.0)
    assert {t["name"] for t in listed} == {"alpha"}
    assert enter_count["n"] == 2, "the next discovery must start a fresh session"


@pytest.mark.unit
def test_an_explicit_timeout_overrides_the_server_config(
    monkeypatch, shutdown_persistent_runtime
):
    """The per-call ``timeout_sec`` argument wins over the server's
    configured budget, in both directions."""
    from jarvis.tools.external.mcp_client import MCPClient

    enter_count = {"n": 0}
    exit_count = {"n": 0}
    call_count = {"n": 0}
    delays = {"call": 0.6}

    TrackedConn, TrackedSession = _make_tracked_doubles(
        call_count, enter_count, exit_count, delays=delays
    )
    _patch_mcp_doubles(monkeypatch, TrackedConn, TrackedSession)

    client = MCPClient({
        "generous": {
            "transport": "stdio", "command": "/bin/true", "args": [],
            "timeout_sec": 5.0,
        },
        "stingy": {
            "transport": "stdio", "command": "/bin/true", "args": [],
            "timeout_sec": 0.2,
        },
    })

    # A tight explicit budget beats the generous config: times out.
    with pytest.raises(TimeoutError):
        client.invoke_tool("generous", "alpha", {}, timeout_sec=0.3)
    assert call_count["n"] == 1

    # A generous explicit budget beats the stingy config: succeeds.
    res = client.invoke_tool("stingy", "alpha", {}, timeout_sec=3.0)
    assert res["isError"] is False
    assert call_count["n"] == 2


_GARBAGE_TIMEOUT_CONFIGS = [
    pytest.param({"timeout_sec": 0}, id="zero"),
    pytest.param({"timeout_sec": -5}, id="negative"),
    pytest.param({"timeout_sec": "nan"}, id="nan"),
    pytest.param({"timeout_sec": "inf"}, id="inf"),
    pytest.param({"timeout_sec": True}, id="bool"),
    pytest.param({"timeout_sec": "abc"}, id="string"),
    pytest.param({"timeout": 0.01}, id="dropped-alias"),
]


@pytest.mark.unit
@pytest.mark.parametrize("cfg_extra", _GARBAGE_TIMEOUT_CONFIGS)
def test_an_invalid_timeout_config_falls_back_to_the_default(
    monkeypatch, shutdown_persistent_runtime, cfg_extra
):
    """An unusable budget never reaches the wait: the module default
    applies and a fast tool succeeds exactly once. A zero, negative,
    ``nan`` or ``inf`` budget would fail every call at once (or overflow),
    ``True`` would silently mean one second, and the undocumented
    ``timeout`` alias is not read at all."""
    from jarvis.tools.external import mcp_runtime as _runtime_mod
    from jarvis.tools.external.mcp_client import MCPClient

    monkeypatch.setattr(_runtime_mod, "_DEFAULT_INVOKE_TIMEOUT_SEC", 5.0)

    enter_count = {"n": 0}
    exit_count = {"n": 0}
    call_count = {"n": 0}
    delays = {"call": 0.2}

    TrackedConn, TrackedSession = _make_tracked_doubles(
        call_count, enter_count, exit_count, delays=delays
    )
    _patch_mcp_doubles(monkeypatch, TrackedConn, TrackedSession)

    cfg = {"transport": "stdio", "command": "/bin/true", "args": [], **cfg_extra}
    client = MCPClient({"srv": cfg})

    res = client.invoke_tool("srv", "alpha", {})

    assert res["isError"] is False
    assert call_count["n"] == 1


@pytest.mark.unit
def test_the_default_budget_is_read_from_the_module_constant(
    monkeypatch, shutdown_persistent_runtime
):
    """Pin the fallback's source: monkeypatching
    ``_DEFAULT_INVOKE_TIMEOUT_SEC`` below the tool's duration must time
    the call out. A hardcoded literal 120 would let it succeed."""
    from jarvis.tools.external import mcp_runtime as _runtime_mod
    from jarvis.tools.external.mcp_client import MCPClient

    monkeypatch.setattr(_runtime_mod, "_DEFAULT_INVOKE_TIMEOUT_SEC", 0.25)

    enter_count = {"n": 0}
    exit_count = {"n": 0}
    call_count = {"n": 0}
    delays = {"call": 1.0}

    TrackedConn, TrackedSession = _make_tracked_doubles(
        call_count, enter_count, exit_count, delays=delays
    )
    _patch_mcp_doubles(monkeypatch, TrackedConn, TrackedSession)

    cfg = {
        "transport": "stdio", "command": "/bin/true", "args": [],
        "timeout_sec": "abc",
    }
    client = MCPClient({"srv": cfg})

    with pytest.raises(TimeoutError):
        client.invoke_tool("srv", "alpha", {})
    assert call_count["n"] == 1


@pytest.mark.unit
def test_the_budget_cascade_prefers_the_first_valid_candidate():
    """Resolution order: a valid explicit argument wins; an invalid one
    falls to the server's value; an invalid server value falls to the
    module default. The fallback is fail-open, and every rejection is
    the resolver's, never the wait's."""
    from jarvis.tools.external import mcp_runtime as rt

    assert rt._resolve_invoke_timeout({"timeout_sec": 30.0}, 10.0) == 10.0
    assert rt._resolve_invoke_timeout({"timeout_sec": 30.0}, "abc") == 30.0
    assert rt._resolve_invoke_timeout({"timeout_sec": 0}, 10.0) == 10.0
    assert rt._resolve_invoke_timeout({"timeout_sec": "abc"}, None) == (
        rt._DEFAULT_INVOKE_TIMEOUT_SEC
    )
    assert rt._resolve_invoke_timeout({}, True) == rt._DEFAULT_INVOKE_TIMEOUT_SEC


def _stub_worker():
    """A ``_ServerWorker`` with no loop and no task, for pinning how
    ``_submit`` classifies a budget expiry. The fake loop runs scheduled
    callbacks inline and the fake queue captures the command, so a test
    arranges exactly the worker-side state the classifier will read."""
    from jarvis.tools.external import mcp_runtime as rt

    worker = rt._ServerWorker.__new__(rt._ServerWorker)
    worker._server_name = "srv"
    worker.config = {}
    worker._task = None
    worker._idle_timeout = None
    worker.alive = True

    class _FakeLoop:
        def call_soon_threadsafe(self, fn, *args):
            fn(*args)

    class _FakeQueue:
        def __init__(self):
            self.command = None

        def put_nowait(self, cmd):
            self.command = cmd

    worker._loop = _FakeLoop()
    worker._queue = _FakeQueue()
    return worker


@pytest.mark.unit
def test_an_expiry_on_a_pulled_command_is_a_timeout_even_on_a_dead_worker():
    """The classification oracle, and the race the ``pulled`` flag
    exists for: a command the worker pulled off the queue may have
    executed, so a budget expiry there is a timeout and never a death,
    whatever else happened to the worker in the interim. A third party
    shutting the worker down mid-call (a config change, the daemon
    teardown) must not turn an executed call into a retried one."""
    from jarvis.tools.external import mcp_runtime as rt

    worker = _stub_worker()
    queue = worker._queue
    real_put = queue.put_nowait

    def pulling_put(cmd):
        real_put(cmd)
        cmd.pulled = True       # the worker took it off the queue...
        worker.alive = False    # ...and something killed the worker

    queue.put_nowait = pulling_put

    with pytest.raises(rt.MCPCallTimeoutError):
        worker.invoke("alpha", {}, 0.05)


@pytest.mark.unit
def test_an_unpulled_command_on_a_dead_worker_is_a_safe_retry():
    """The other side of the oracle: a command that never left the
    queue never executed, so a dead worker there is the honest
    ``_WorkerDeadError`` the runtime answers with its single retry."""
    from jarvis.tools.external import mcp_runtime as rt

    worker = _stub_worker()
    queue = worker._queue
    real_put = queue.put_nowait

    def dying_put(cmd):
        real_put(cmd)
        worker.alive = False    # died before pulling anything

    queue.put_nowait = dying_put

    with pytest.raises(rt._WorkerDeadError):
        worker.invoke("alpha", {}, 0.05)


@pytest.mark.unit
def test_a_death_landing_during_the_teardown_reports_the_expiry():
    """A death sentinel that resolves the future only while the
    caller's own teardown is running arrives after the budget was
    spent: the honest report is the expiry. The retry belongs to
    deaths the expiry can see (a worker already dead, or a sentinel
    already on the future); a corpse that races the teardown does not
    resurrect it, and the caller waits one budget, not two."""
    from jarvis.tools.external import mcp_runtime as rt

    worker = _stub_worker()

    def draining_shutdown():
        cmd = worker._queue.command
        if cmd is not None and not cmd.fut.done():
            cmd.fut.set_exception(rt._WorkerDeadError("session ended"))
        worker.alive = False

    worker.shutdown = draining_shutdown

    with pytest.raises(rt.MCPCallTimeoutError):
        worker.invoke("alpha", {}, 0.05)


@pytest.mark.unit
def test_a_call_completing_at_the_wire_reports_its_honest_result():
    """A result that lands between the expiry and the classification is
    the call's outcome, not a timeout: reporting a failure for a call
    that succeeded is the same kind of lie as reporting a success for
    one that never ran."""
    worker = _stub_worker()
    queue = worker._queue
    real_put = queue.put_nowait

    def pulling_put(cmd):
        real_put(cmd)
        cmd.pulled = True

    queue.put_nowait = pulling_put

    def resolving_shutdown():
        cmd = queue.command
        if cmd is not None and not cmd.fut.done():
            cmd.fut.set_result("late-result")
        worker.alive = False

    worker.shutdown = resolving_shutdown

    assert worker.invoke("alpha", {}, 0.05) == "late-result"


@pytest.mark.unit
def test_a_teardown_echo_landing_after_the_expiry_reports_the_timeout():
    """An exception that lands on the future only after the caller's
    own shutdown began is the teardown's echo: the cancellation this
    very timeout delivered, translated into a session error on its way
    out. Re-raising it would tell the caller their expired call was a
    session failure and would hand a sibling's teardown to the wrong
    owner; the expiry is the honest report."""
    from jarvis.tools.external.mcp_client import MCPServerSessionError

    worker = _stub_worker()
    queue = worker._queue
    real_put = queue.put_nowait

    def pulling_put(cmd):
        real_put(cmd)
        cmd.pulled = True

    queue.put_nowait = pulling_put

    def echoing_shutdown():
        cmd = queue.command
        if cmd is not None and not cmd.fut.done():
            echo = MCPServerSessionError(
                "MCP server 'srv' session lost while servicing call: CancelledError"
            )
            echo.__cause__ = asyncio.CancelledError()
            cmd.fut.set_exception(echo)
        worker.alive = False

    worker.shutdown = echoing_shutdown

    from jarvis.tools.external.mcp_runtime import MCPCallTimeoutError

    with pytest.raises(MCPCallTimeoutError):
        worker.invoke("alpha", {}, 0.05)


@pytest.mark.unit
def test_a_call_cancelled_by_a_sibling_timeout_surfaces_as_a_session_error(
    monkeypatch, shutdown_persistent_runtime
):
    """Two callers, one server, one serialised queue.

    The second caller's budget expires while the first call is in
    flight; the shutdown cancels the worker task, and the cancellation
    lands inside the first call. What the first caller receives must be
    an ``Exception`` the tool funnel can catch and record: a raw
    ``CancelledError`` is a ``BaseException`` that escapes every
    ``except Exception`` on the way out, skips the ledger row for a
    gate-approved call that may have executed, and kills the turn.

    The same shutdown must not let the second caller's queued command
    execute afterwards: its timeout was already reported, and a side
    effect arriving after the report contradicts it. The command never
    ran, so its caller's outcome stays the retryable death.
    """
    import threading

    from jarvis.tools.external.mcp_client import MCPClient, MCPServerSessionError
    from jarvis.tools.external.mcp_runtime import MCPCallTimeoutError

    enter_count = {"n": 0}
    exit_count = {"n": 0}
    call_count = {"n": 0}
    started = []
    slow_started = threading.Event()

    def hook(name):
        started.append(name)
        if name == "slow":
            slow_started.set()

    TrackedConn, TrackedSession = _make_tracked_doubles(
        call_count, enter_count, exit_count,
        delays={"call": 5.0}, call_hook=hook,
    )
    _patch_mcp_doubles(monkeypatch, TrackedConn, TrackedSession)

    client = MCPClient(
        {"srv": {"transport": "stdio", "command": "/bin/true", "args": []}}
    )

    outcome = {}

    def slow_call():
        try:
            client.invoke_tool("srv", "slow", {}, timeout_sec=10.0)
            outcome["result"] = "returned"
        except BaseException as e:  # noqa: BLE001 — the type IS the assertion
            outcome["error"] = e

    thread = threading.Thread(target=slow_call, daemon=True)
    thread.start()
    assert slow_started.wait(timeout=5.0), "the slow call never started"

    # The second caller queues behind the first and times out; its
    # timeout shuts the worker down mid-flight on the first call.
    with pytest.raises(MCPCallTimeoutError):
        client.invoke_tool("srv", "quick", {}, timeout_sec=0.3)

    thread.join(timeout=10.0)
    assert not thread.is_alive(), "the in-flight call never came back"

    error = outcome.get("error")
    assert isinstance(error, MCPServerSessionError), (
        f"the cancelled in-flight call surfaced as {error!r}, which the "
        "tool funnel's `except Exception` cannot catch"
    )
    assert started == ["slow"], (
        "the queued command executed after its own timeout was reported"
    )
    assert call_count["n"] == 1


@pytest.mark.unit
def test_the_registry_names_a_bare_timeout_error(tools_unrestricted):
    """A ``TimeoutError`` with an empty message must not reach the model
    as a dangling ``error: ``; the error text names the exception type."""
    from unittest.mock import patch

    from jarvis.tools.registry import run_tool_with_retries

    class MockDB:
        pass

    class MockConfig:
        def __init__(self):
            self.mcps = {"test-server": {"command": "fake"}}
            self.voice_debug = False

    class TimeoutMCPClient:
        def __init__(self, config):
            pass

        def invoke_tool(self, server_name, tool_name, arguments):
            raise TimeoutError()

    with patch("jarvis.tools.registry.MCPClient", TimeoutMCPClient):
        result = run_tool_with_retries(
            db=MockDB(),
            cfg=MockConfig(),
            tool_name="test-server__slow_tool",
            tool_args={},
            system_prompt="t",
            original_prompt="t",
            redacted_text="t",
            max_retries=0,
        )

    assert result.success is False
    assert "TimeoutError" in result.error_message
    assert not result.error_message.endswith(": ")


@pytest.mark.unit
def test_absolute_path_command_skips_which(monkeypatch, tmp_path):
    """Absolute paths to executables should use os.path.isfile, not shutil.which."""
    from jarvis.tools.external.mcp_client import MCPClient

    # Create a fake executable file at an absolute path
    fake_exe = tmp_path / "node.exe"
    fake_exe.write_text("fake", encoding="utf-8")
    fake_exe.chmod(0o755)

    mcps = {
        "test": {
            "command": str(fake_exe),
            "args": ["server.js"],
        }
    }

    client = MCPClient(mcps)

    # shutil.which should NOT be called for absolute paths
    which_called = False
    original_which = __import__("shutil").which

    def tracking_which(cmd):
        nonlocal which_called
        which_called = True
        return original_which(cmd)

    monkeypatch.setattr("jarvis.tools.external.mcp_client.shutil.which", tracking_which)

    # We need to mock stdio_client to avoid actually connecting
    class FakeCM:
        async def __aenter__(self):
            return object(), object()
        async def __aexit__(self, *a):
            return False

    class FakeSession:
        def __init__(self, r, w):
            pass
        async def __aenter__(self):
            s = type("S", (), {"initialize": lambda self: asyncio.sleep(0), "list_tools": lambda self: asyncio.sleep(0)})()
            return s
        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr("jarvis.tools.external.mcp_client.stdio_client", lambda params, **kw: FakeCM())
    monkeypatch.setattr("jarvis.tools.external.mcp_client.ClientSession", FakeSession)

    try:
        asyncio.run(client.list_tools_async("test"))
    except Exception:
        pass  # We only care that the path validation passed

    assert not which_called, "shutil.which should not be called for absolute paths"


@pytest.mark.unit
def test_absolute_path_not_found_gives_clear_error(tmp_path):
    """Non-existent absolute path should raise FileNotFoundError with clear message."""
    from jarvis.tools.external.mcp_client import MCPClient

    fake_path = str(tmp_path / "nonexistent" / "node.exe")
    mcps = {
        "test": {
            "command": fake_path,
            "args": [],
        }
    }

    client = MCPClient(mcps)

    with pytest.raises(FileNotFoundError, match="does not exist"):
        client._connect_stdio(mcps["test"])


@pytest.mark.unit
def test_mcp_client_list_and_invoke(monkeypatch):
    # Import the real client and patch its external dependencies
    from jarvis.tools.external.mcp_client import MCPClient

    # Prepare fake server config (command won't actually run because we mock stdio_client)
    mcps = {
        "fake": {
            "transport": "stdio",
            "command": "fake-cmd",
            "args": ["--flag"],
            "env": {},
        }
    }

    client = MCPClient(mcps)

    # Create fake tool objects that the MCP client expects
    class FakeTool:
        def __init__(self, name, description, inputSchema):
            self.name = name
            self.description = description
            self.inputSchema = inputSchema

    # Create fake session object implementing the observable API used by MCPClient
    class FakeSession:
        async def initialize(self):
            return None

        async def list_tools(self):
            return [
                FakeTool("alpha", "desc", {"type": "object"}),
                FakeTool("beta", "desc", {"type": "object"}),
            ]

        async def call_tool(self, name, arguments):
            # Create a response object with attributes that the MCP client expects
            class FakeResponse:
                def __init__(self):
                    self.content = f"called:{name}:{arguments}"
                    self.isError = False
                    self.meta = None
            return FakeResponse()

    # Mock stdio_client context manager to yield (read, write)
    class FakeCM:
        def __init__(self, session):
            self._session = session

        async def __aenter__(self):
            # Return reader, writer placeholders; session is consumed by ClientSession wrapper
            return object(), object()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    # Mock ClientSession to wrap our FakeSession directly
    class FakeClientSession:
        def __init__(self, read, write):
            self._session = FakeSession()

        async def __aenter__(self):
            await self._session.initialize()
            return self._session

        async def __aexit__(self, exc_type, exc, tb):
            return False

    # Patch public imports inside the module (observable seams)
    monkeypatch.setattr("jarvis.tools.external.mcp_client.stdio_client", lambda params, **kw: FakeCM(FakeSession()))
    monkeypatch.setattr("jarvis.tools.external.mcp_client.ClientSession", FakeClientSession)
    # Avoid PATH check failing in _connect_stdio
    monkeypatch.setattr("jarvis.tools.external.mcp_client.shutil.which", lambda cmd: cmd)

    tools = asyncio.run(client.list_tools_async("fake"))
    assert isinstance(tools, list) and {t["name"] for t in tools} == {"alpha", "beta"}

    res = asyncio.run(client.invoke_tool_async("fake", "alpha", {"x": 1}))
    assert res["content"] == "called:alpha:{'x': 1}"
    assert res.get("isError") is False


@pytest.mark.unit
class TestResolveCommand:
    """Tests for _resolve_command PATH fallback logic."""

    def test_finds_command_on_path(self, monkeypatch):
        """When shutil.which succeeds, returns that path."""
        from jarvis.tools.external.mcp_client import _resolve_command
        monkeypatch.setattr("jarvis.tools.external.mcp_client.shutil.which", lambda cmd: "/usr/bin/npx")
        assert _resolve_command("npx") == "/usr/bin/npx"

    def test_finds_command_in_extra_dirs(self, monkeypatch, tmp_path):
        """When shutil.which fails, probes extra directories."""
        from jarvis.tools.external.mcp_client import _resolve_command
        monkeypatch.setattr("jarvis.tools.external.mcp_client.shutil.which", lambda cmd: None)

        # Create a fake executable in a temp dir
        fake_npx = tmp_path / "npx"
        fake_npx.write_text("#!/bin/sh", encoding="utf-8")
        fake_npx.chmod(0o755)

        # Inject our temp dir into the extra paths list
        monkeypatch.setattr(
            "jarvis.tools.external.mcp_client._EXTRA_PATH_DIRS",
            [str(tmp_path)],
        )
        monkeypatch.setattr("jarvis.tools.external.mcp_client._EXTRA_PATH_GLOBS", [])
        # Skip login shell fallback
        monkeypatch.setattr("jarvis.tools.external.mcp_client._sys.platform", "win32")

        assert _resolve_command("npx") == str(fake_npx)

    def test_falls_back_to_login_shell(self, monkeypatch):
        """When extra dirs fail, tries bash -lc which."""
        from jarvis.tools.external.mcp_client import _resolve_command
        import subprocess

        monkeypatch.setattr("jarvis.tools.external.mcp_client.shutil.which", lambda cmd: None)
        monkeypatch.setattr("jarvis.tools.external.mcp_client._EXTRA_PATH_DIRS", [])
        monkeypatch.setattr("jarvis.tools.external.mcp_client._EXTRA_PATH_GLOBS", [])
        monkeypatch.setattr("jarvis.tools.external.mcp_client._sys.platform", "darwin")

        mock_result = type("R", (), {"returncode": 0, "stdout": "/opt/homebrew/bin/npx\n"})()
        monkeypatch.setattr(
            "subprocess.run",
            lambda *a, **kw: mock_result,
        )
        assert _resolve_command("npx") == "/opt/homebrew/bin/npx"

    def test_finds_command_via_nvm_glob(self, monkeypatch, tmp_path):
        """When shutil.which and static dirs fail, probes nvm-style version dirs."""
        from jarvis.tools.external.mcp_client import _resolve_command
        monkeypatch.setattr("jarvis.tools.external.mcp_client.shutil.which", lambda cmd: None)
        monkeypatch.setattr("jarvis.tools.external.mcp_client._EXTRA_PATH_DIRS", [])
        monkeypatch.setattr("jarvis.tools.external.mcp_client._sys.platform", "win32")

        # Create nvm-style version dirs with npx
        v18 = tmp_path / "v18.0.0" / "bin"
        v22 = tmp_path / "v22.22.0" / "bin"
        v18.mkdir(parents=True)
        v22.mkdir(parents=True)
        (v18 / "npx").write_text("#!/bin/sh", encoding="utf-8")
        (v18 / "npx").chmod(0o755)
        (v22 / "npx").write_text("#!/bin/sh", encoding="utf-8")
        (v22 / "npx").chmod(0o755)

        monkeypatch.setattr(
            "jarvis.tools.external.mcp_client._EXTRA_PATH_GLOBS",
            [str(tmp_path / "*/bin")],
        )
        # Should prefer the highest version (v22) due to reverse sort
        result = _resolve_command("npx")
        assert "v22.22.0" in result

    def test_raises_when_not_found_anywhere(self, monkeypatch):
        """When all resolution methods fail, raises FileNotFoundError."""
        from jarvis.tools.external.mcp_client import _resolve_command
        monkeypatch.setattr("jarvis.tools.external.mcp_client.shutil.which", lambda cmd: None)
        monkeypatch.setattr("jarvis.tools.external.mcp_client._EXTRA_PATH_DIRS", [])
        monkeypatch.setattr("jarvis.tools.external.mcp_client._EXTRA_PATH_GLOBS", [])
        monkeypatch.setattr("jarvis.tools.external.mcp_client._sys.platform", "win32")

        with pytest.raises(FileNotFoundError, match="not found on PATH"):
            _resolve_command("nonexistent-command")

    def test_absolute_path_verified_directly(self, tmp_path):
        """Absolute paths bypass PATH lookup entirely."""
        from jarvis.tools.external.mcp_client import _resolve_command

        fake = tmp_path / "my-server"
        fake.write_text("#!/bin/sh", encoding="utf-8")
        fake.chmod(0o755)
        assert _resolve_command(str(fake)) == str(fake)

    def test_absolute_path_missing_raises(self, tmp_path):
        """Non-existent absolute path raises FileNotFoundError."""
        from jarvis.tools.external.mcp_client import _resolve_command

        with pytest.raises(FileNotFoundError, match="does not exist"):
            _resolve_command(str(tmp_path / "nope"))


@pytest.mark.unit
class TestConnectStdioPathInjection:
    """Tests that _connect_stdio injects the resolved command's dir into PATH."""

    def test_command_dir_added_to_env_path(self, monkeypatch, tmp_path):
        """The directory of the resolved command should be prepended to env PATH."""
        from jarvis.tools.external.mcp_client import MCPClient

        fake_npx = tmp_path / "npx"
        fake_npx.write_text("#!/bin/sh", encoding="utf-8")
        fake_npx.chmod(0o755)

        monkeypatch.setattr(
            "jarvis.tools.external.mcp_client._resolve_command",
            lambda cmd: str(fake_npx),
        )

        captured_params = {}

        def fake_stdio_client(params, **kw):
            captured_params["env"] = params.env
            captured_params["command"] = params.command
            return None

        monkeypatch.setattr(
            "jarvis.tools.external.mcp_client.stdio_client",
            fake_stdio_client,
        )

        client = MCPClient({"test": {"command": "npx", "args": ["-y", "server"]}})
        client._connect_stdio(client.server_configs["test"])

        env = captured_params["env"]
        assert env is not None
        path_dirs = env["PATH"].split(os.pathsep)
        assert str(tmp_path) == path_dirs[0], "Command dir should be first in PATH"
        # Full parent environment should also be present
        assert any(k in env for k in ("HOME", "USER", "USERNAME", "USERPROFILE")), "Parent env vars should be inherited"

    def test_user_env_preserved_alongside_path(self, monkeypatch, tmp_path):
        """User-supplied env vars should be preserved when PATH is injected."""
        from jarvis.tools.external.mcp_client import MCPClient

        fake_npx = tmp_path / "npx"
        fake_npx.write_text("#!/bin/sh", encoding="utf-8")
        fake_npx.chmod(0o755)

        monkeypatch.setattr(
            "jarvis.tools.external.mcp_client._resolve_command",
            lambda cmd: str(fake_npx),
        )

        captured_params = {}

        def fake_stdio_client(params, **kw):
            captured_params["env"] = params.env
            return None

        monkeypatch.setattr(
            "jarvis.tools.external.mcp_client.stdio_client",
            fake_stdio_client,
        )

        cfg = {"command": "npx", "args": [], "env": {"MY_TOKEN": "secret"}}
        client = MCPClient({"test": cfg})
        client._connect_stdio(client.server_configs["test"])

        env = captured_params["env"]
        assert env["MY_TOKEN"] == "secret"
        assert str(tmp_path) in env["PATH"]

    def test_no_env_override_when_command_already_on_path(self, monkeypatch):
        """When command dir is already on PATH and no user env, env should be None."""
        from jarvis.tools.external.mcp_client import MCPClient

        # Resolve to a path that's already on the system PATH
        system_path_dir = os.environ.get("PATH", "").split(os.pathsep)[0]
        fake_cmd = os.path.join(system_path_dir, "fake-cmd")

        monkeypatch.setattr(
            "jarvis.tools.external.mcp_client._resolve_command",
            lambda cmd: fake_cmd,
        )

        captured_params = {}

        def fake_stdio_client(params, **kw):
            captured_params["env"] = params.env
            return None

        monkeypatch.setattr(
            "jarvis.tools.external.mcp_client.stdio_client",
            fake_stdio_client,
        )

        client = MCPClient({"test": {"command": "fake-cmd", "args": []}})
        client._connect_stdio(client.server_configs["test"])

        assert captured_params["env"] is None, "No env override needed when dir already on PATH"


class TestMCPContentAndErrors:
    def test_flatten_content_sdk_text_content(self):
        """MCP SDK TextContent objects must be flattened to plain text string, not repr."""
        from jarvis.tools.external.mcp_client import _flatten_content
        from mcp.types import TextContent

        sdk_content = TextContent(type="text", text="Hello from MCP tool!")
        flattened = _flatten_content(sdk_content)
        assert flattened == "Hello from MCP tool!"
        assert "TextContent" not in flattened
        assert "type=" not in flattened

    def test_flatten_content_sdk_list_of_items(self):
        """A list of SDK TextContent objects must be joined with newlines."""
        from jarvis.tools.external.mcp_client import _flatten_content
        from mcp.types import TextContent

        items = [
            TextContent(type="text", text="Line 1"),
            TextContent(type="text", text="Line 2"),
        ]
        flattened = _flatten_content(items)
        assert flattened == "Line 1\nLine 2"

    def test_result_to_dict_empty_error_provides_message(self):
        """When an MCP tool returns isError=True with empty content, a helpful error message is provided."""
        from jarvis.tools.external.mcp_client import _result_to_dict

        res = type("CallResult", (), {"content": [], "isError": True, "meta": None})()
        d = _result_to_dict(res)
        assert d["isError"] is True
        assert d["text"], "Empty error response must provide non-empty error text"
        assert "error" in d["text"].lower() or "failed" in d["text"].lower()

    def test_invoke_tool_sends_the_config_timeout_to_the_runtime(self, monkeypatch):
        """The config's timeout_sec travels inside the config itself; the
        runtime is the single place that resolves the budget."""
        from jarvis.tools.external.mcp_client import MCPClient

        captured = {}

        class FakeRuntime:
            def invoke(self, server_name, cfg, tool_name, arguments, timeout=None):
                captured["timeout"] = timeout
                captured["cfg"] = cfg
                return type("R", (), {"content": "ok", "isError": False, "meta": None})()

        monkeypatch.setattr(
            "jarvis.tools.external.mcp_runtime.get_runtime", lambda: FakeRuntime()
        )

        client = MCPClient({
            "weather": {
                "command": "npx",
                "args": ["server"],
                "timeout_sec": 42.5,
            }
        })
        client.invoke_tool("weather", "get_forecast", {})
        assert captured["timeout"] is None
        assert captured["cfg"]["timeout_sec"] == 42.5

    def test_invoke_tool_forwards_explicit_timeout_sec_arg(self, monkeypatch):
        """Explicit timeout_sec passed to invoke_tool overrides server config."""
        from jarvis.tools.external.mcp_client import MCPClient

        captured_kwargs = {}

        class FakeRuntime:
            def invoke(self, server_name, cfg, tool_name, arguments, timeout=None):
                captured_kwargs["timeout"] = timeout
                return type("R", (), {"content": "ok", "isError": False, "meta": None})()

        monkeypatch.setattr(
            "jarvis.tools.external.mcp_runtime.get_runtime", lambda: FakeRuntime()
        )

        client = MCPClient({
            "weather": {
                "command": "npx",
                "args": ["server"],
                "timeout_sec": 42.5,
            }
        })
        client.invoke_tool("weather", "get_forecast", {}, timeout_sec=15.0)
        assert captured_kwargs.get("timeout") == 15.0

    def test_list_tools_sends_the_config_timeout_to_the_runtime(self, monkeypatch):
        """The config's timeout_sec travels inside the config itself; the
        runtime is the single place that resolves the budget."""
        from jarvis.tools.external.mcp_client import MCPClient

        captured = {}

        class FakeRuntime:
            def list_tools(self, server_name, cfg, timeout=None):
                captured["timeout"] = timeout
                captured["cfg"] = cfg
                return []

        monkeypatch.setattr(
            "jarvis.tools.external.mcp_runtime.get_runtime", lambda: FakeRuntime()
        )

        client = MCPClient({
            "weather": {
                "command": "npx",
                "args": ["server"],
                "timeout_sec": 42.5,
            }
        })
        client.list_tools("weather")
        assert captured["timeout"] is None
        assert captured["cfg"]["timeout_sec"] == 42.5

    def test_list_tools_forwards_explicit_timeout_sec_arg(self, monkeypatch):
        """Explicit timeout_sec passed to list_tools overrides server config."""
        from jarvis.tools.external.mcp_client import MCPClient

        captured_kwargs = {}

        class FakeRuntime:
            def list_tools(self, server_name, cfg, timeout=None):
                captured_kwargs["timeout"] = timeout
                return []

        monkeypatch.setattr(
            "jarvis.tools.external.mcp_runtime.get_runtime", lambda: FakeRuntime()
        )

        client = MCPClient({
            "weather": {
                "command": "npx",
                "args": ["server"],
                "timeout_sec": 42.5,
            }
        })
        client.list_tools("weather", timeout_sec=15.0)
        assert captured_kwargs.get("timeout") == 15.0

    @pytest.mark.skipif(
        sys.platform != "win32",
        reason="Windows-specific PATH and .cmd probing requires Windows runtime",
    )
    def test_resolve_command_probes_windows_extra_dirs(self, monkeypatch, tmp_path):
        """On Windows, _resolve_command probes AppData/ProgramFiles extra dirs for .cmd files."""
        from jarvis.tools.external.mcp_client import (
            _resolve_command,
            _default_extra_path_dirs,
        )
        import shutil

        # Simulate npx not found on standard PATH
        real_which = shutil.which
        def fake_which(cmd, path=None):
            if path is None:
                return None
            return real_which(cmd, path=path)

        monkeypatch.setattr("shutil.which", fake_which)

        fake_npm = tmp_path / "npm"
        fake_npm.mkdir()
        fake_cmd = fake_npm / "my_custom_mcp.cmd"
        fake_cmd.write_text("@echo off", encoding="utf-8")

        monkeypatch.setenv("APPDATA", str(tmp_path))
        monkeypatch.setattr(
            "jarvis.tools.external.mcp_client._EXTRA_PATH_DIRS",
            _default_extra_path_dirs(),
        )
        resolved = _resolve_command("my_custom_mcp")
        assert os.path.normcase(resolved) == os.path.normcase(str(fake_cmd))




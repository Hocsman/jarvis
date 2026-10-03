# MCP runtime spec

## Purpose

Keep one stdio session per configured MCP server alive across tool
invocations. The naive `asyncio.run(open → call → close)` pattern works
for stateless servers but breaks any server that owns external state
(e.g. `chrome-devtools-mcp` launches Chrome on first navigation —
closing the session kills the browser). This module replaces that
pattern with a singleton runtime that keeps each server's subprocess
resident for the daemon's lifetime.

## Architecture

- One process-wide singleton `_PersistentMCPRuntime` accessible via
  `get_runtime()`. Created lazily on first use. `shutdown_runtime()`
  latches the run: `get_runtime()` refuses until `reset_shutdown_latch()`
  re-arms it at the start of the next daemon run (`daemon.main()` calls
  it; the bundled desktop app re-runs `main()` in-process on tray
  toggle, settings restart and the setup wizard).
- A single background thread runs an `asyncio` event loop
  (`JarvisMCPRuntime`). All MCP I/O happens on this loop.
- Per server, a `_ServerWorker` task lives on that loop. The task
  holds `stdio_client(...)` and `ClientSession(...)` open and consumes
  `_Command` objects (kind, payload, future, and a `pulled` flag the
  worker sets at dequeue) from an `asyncio.Queue`.
- Callers (registry → `MCPClient.list_tools` / `invoke_tool`) submit
  requests via `runtime.invoke(...)` / `runtime.list_tools(...)`. Each
  call hops the request onto the loop with `call_soon_threadsafe(put_nowait, ...)`
  and blocks on a `concurrent.futures.Future` for the result.

## Lifecycle

| Event | Effect |
|-------|--------|
| First `get_runtime()` call | Spawns the background thread + loop. |
| First call referencing a server | Creates a `_ServerWorker`, awaits `_ready` (the worker signals readiness once `session.initialize()` returns). |
| Worker setup fails or times out | The half-started session's teardown is set in motion (sentinel enqueued, task cancel scheduled) before the failure reaches the caller; a `_setup` scheduled on a stalled loop bows out against the torn-down worker when the loop frees. The loop completes the unwind and reaps the subprocess asynchronously. The worker is cached only after a successful start, so retries do not stack unreachable sessions. |
| Server config equality holds | Subsequent calls reuse the cached worker. |
| Server config changes | Old worker is shut down; a fresh worker replaces it. |
| Worker raises `_WorkerDeadError` | Runtime drops it and retries the call once with a new worker. Second failure surfaces as `MCPServerSessionError` to the public layer. |
| Call exceeds its `timeout_sec` budget | The worker is shut down and evicted, and the call surfaces as `MCPCallTimeoutError` (a `TimeoutError`) naming the server, the tool (or `list_tools`) and the seconds. It is never retried: a slow side-effecting tool may have run, and running it twice after one approval is worse than reporting the timeout. A stateful server restarts on the next call. |
| Worker shuts down while a command is queued | The queued command does not execute: the worker refuses it at dequeue with the death sentinel. A caller whose budget already expired gets the expiry reported: the sentinel lands after its classification read, and an unpulled command can produce no result to honour. |
| `idle_timeout_sec` set on a server config | Worker self-terminates after that long without activity. Next call spawns a new worker. |
| Daemon shutdown calls `shutdown_runtime()` | Teardown is terminal for the daemon run: the latch makes any later `get_runtime()` raise `RuntimeError` until the next `daemon.main()` re-arms it, so a straggler call cannot resurrect a thread and subprocesses nobody would reap; the funnel records the failure like any session error. Each worker is asked to exit (sentinel `None`); any wedged task is cancelled. The loop runs the cancellations to completion (bounded by `_LOOP_DRAIN_SEC`, which outlasts the mcp SDK's 2s graceful wait plus its 2s process-tree kill) before closing, so an in-flight caller is resolved promptly with a session error instead of waiting out its budget against a dead loop; a task still wedged in uncancellable work past the bound gets the loop closed under it. The thread is joined with a `_SHUTDOWN_THREAD_JOIN_SEC` timeout. |

## Invariants

- One in-flight `call_tool` per server at any time. Tool calls to the
  same server are serialised by the queue. Different servers run in
  parallel because each has its own worker.
- A worker is never reused after `alive` flips to `False`. The
  finally-block in `_run` drains pending requests, resolving each
  outstanding future with `_WorkerDeadError` so callers do not hang.
- A command the runtime may retry as "unpulled on a dead worker" never
  executed: `alive` is re-read after the `pulled` mark, so a shutdown
  visible at the dequeue observation point refuses the command instead
  of running it, and a shutdown landing after the mark is invisible to
  the classification (a caller reads `pulled` only after its own
  `shutdown()` returned), which reports the expiry and never retries.
- A caller never receives a bare `BaseException` from a command: a
  cancellation delivered by a sibling call's timeout, or an anyio
  task-group teardown, surfaces as `MCPServerSessionError` (the
  original kept as `__cause__`) so the tool funnel records the call in
  the ledger instead of losing the turn.
- `MCPClient.list_tools` / `invoke_tool` are the only way to reach a
  server, and both route through the runtime.

## Public surface

- `MCPClient.list_tools(server_name, timeout_sec=None)`: returns a list of tool dicts.
  Routes through the persistent runtime so discovery and the first
  invocation share a session. Raises `MCPCallTimeoutError` when
  discovery exceeds its budget (not retried), and `RuntimeError` after
  a latched `shutdown_runtime()`.
- `MCPClient.invoke_tool(server_name, tool_name, arguments=None, timeout_sec=None)`: returns
  the standard MCP response dict. Raises `MCPServerSessionError` if
  the runtime cannot keep a session alive after one retry (a call
  racing an in-progress shutdown on a stale runtime reference gets it
  from the closed-runtime guard, without a retry),
  `MCPCallTimeoutError` when the call exceeds its budget (not
  retried), and `RuntimeError` after a latched `shutdown_runtime()`.
- `MCPServerSessionError` (in `mcp_client.py`): public, stable type
  signalling a session-level failure (distinct from a tool-level error
  carried in the response dict's `isError`).
- `MCPCallTimeoutError` (in `mcp_runtime.py`): a `TimeoutError` raised
  when a call exceeds its budget. The worker is dropped and the call is
  not retried.
- `get_runtime()` / `shutdown_runtime()`: module-level helpers used
  by the daemon's startup and shutdown paths. `shutdown_runtime()`
  latches: afterwards `get_runtime()` raises `RuntimeError` until
  `reset_shutdown_latch()` re-arms it, which `daemon.main()` does at
  the start of every run.

## Configuration

Each server entry in `config.mcps` is a dict consumed by
`MCPClient._connect_stdio`. The runtime additionally honours:

| Key | Type | Default | Effect |
|-----|------|---------|--------|
| `idle_timeout_sec` | float \| null | null | If set to a finite positive number, the worker self-terminates after that long with an empty queue. Any other value (boolean, non-numeric, non-finite, ≤ 0) means no idle timeout, with a debug log. Stateful servers (browser automation) must leave this unset. |
| `timeout_sec` | float \| null | 120.0 (`_DEFAULT_INVOKE_TIMEOUT_SEC`) | Bounds discovery and each tool call on this server. Only finite positive numbers count; an invalid value falls back to the default with a debug log rather than failing every call at once (`0`, negatives), waiting on an undefined budget (`nan`), overflowing (`inf`) or silently meaning one second (`true`). A per-call `timeout_sec` argument overrides it and is validated the same way; an invalid per-call value falls back to the server's value, then the default. A config carrying the old `timeout` key gets a rename notice in the debug log; the key is not read. |


## Test contract

Behavioural tests live in `tests/test_mcp_client.py`. The contract
verified there:

- A second `invoke_tool` does not open a new stdio connection.
- `list_tools` followed by `invoke_tool` shares one stdio connection.
- A `_WorkerDeadError` from a worker triggers exactly one retry, which
  spawns a fresh connection.
- A config change replaces the worker and spawns a fresh connection.
- A failure during subprocess spawn propagates to the caller rather
  than hanging.
- Distinct servers do not share workers.
- A tool slower than `timeout_sec` starts exactly once, fails in about
  `timeout_sec` with a `TimeoutError` naming the server, the tool and
  the seconds (never `MCPServerSessionError`), and the next call opens
  a fresh connection. The same holds for `list_tools`.
- An explicit per-call `timeout_sec` overrides the server config in
  both directions.
- Invalid timeout configs (`0`, negative, `nan`, `inf`, boolean,
  non-numeric, the unread `timeout` alias) fall back to
  `_DEFAULT_INVOKE_TIMEOUT_SEC`, read from the module constant, and
  the budget cascade (explicit argument → server value → default)
  prefers the first valid candidate.
- The expiry classification (`_submit`'s `pulled` oracle): a pulled
  command expires as a timeout even on a worker something else killed;
  an unpulled command on a worker already dead at the expiry is the
  retryable death; a result landing at the wire reports honestly; a
  failure resolved before the teardown propagates untouched; anything
  landing after the teardown began (a drain sentinel, a cancellation
  echo) reports the expiry.
- The two-thread scenario: a sibling caller's timeout cancels an
  in-flight call, which surfaces as `MCPServerSessionError` (never a
  bare `CancelledError`), and the sibling's queued command never
  executes after its expiry was reported.
- A setup that never finishes its handshake tears its half-started
  session down (the connection exits), and attempts do not stack
  sessions.
- `shutdown_runtime()` during an in-flight call resolves the caller
  promptly with a session error, well inside its budget.
- After `shutdown_runtime()`, `get_runtime()` refuses and a straggler
  `invoke_tool` raises rather than spawning anything; a new daemon run
  re-arms the latch (`reset_shutdown_latch`), and `daemon.main()`
  re-arms before MCP discovery.
- A queued command resolved by the teardown drain does not retry into
  a fresh spawn on the closing runtime: the closed-runtime guard
  answers with a session error and no second connection is opened.
- A loop stalled past the schedule budget leaves no `_setup` behind:
  the scheduled setup bows out against the torn-down worker and never
  opens a connection.
- The registry renders an exception with an empty message by its type
  name, never as a dangling `error: ` or `raised: `.

## Non-goals

- Hot-reloading `config.mcps` proactively. The runtime replaces a
  worker only when a request arrives carrying the new config.
- Recovering from SIGKILL of the daemon process. Subprocess children
  (e.g. Chrome) become orphans and must be cleaned up by the OS.
- Parallel `call_tool` to the same server. The MCP stdio framing is
  request-response per session; parallelism is per-server, not
  per-call.

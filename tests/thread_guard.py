"""The suite's guard against a test that leaves a thread running behind it.

``tests/conftest.py`` imports the autouse fixture below, so it wraps every test
in the suite. It snapshots the live threads before the test and, after every
fixture of that test has been torn down, joins whatever is new for a short
shared grace and fails the test if any of it is still alive. The failure names
the test and each thread (name, kind, daemon flag, entry point), which is what
a first run on a new platform needs to be tuned quickly.

What is not a leak:

* a thread that was already running when the test started (it belongs to
  whoever started it: a wider-scoped fixture, an earlier test the guard
  already failed, collection);
* a thread that winds down inside the grace;
* a resident, one of the ``RESIDENTS`` below: threads that are meant to
  outlive a test or that Python did not start. Add to that tuple, with the
  reason, rather than loosening the check.

``tests/test_the_suite_leaves_no_thread_behind.py`` holds the guard to all of
this.
"""

import threading
import time
from dataclasses import dataclass
from typing import Callable, FrozenSet, Iterable, List, Optional

import pytest

# How long, in total, the guard waits for the threads a test started to finish.
# It is a grace for a thread already on its way out (a worker that has seen its
# stop signal, a timer that has just fired), not a way to wait for one that has
# not been told to stop.
JOIN_GRACE_SEC = 1.0


@dataclass(frozen=True)
class Resident:
    """A kind of thread the guard does not count as a leak."""

    label: str
    why: str
    matches: Callable[[threading.Thread, FrozenSet[str]], bool]


def _entry_point(thread: threading.Thread):
    """What the thread runs: its target, or for a Timer the function it calls."""
    return getattr(thread, "_target", None) or getattr(thread, "function", None)


def _entry_point_name(thread: threading.Thread) -> str:
    entry = _entry_point(thread)
    if entry is None:
        return "no entry point"
    module = getattr(entry, "__module__", None)
    name = getattr(entry, "__qualname__", None) or repr(entry)
    return f"{module}.{name}" if module else name


def _foreign(thread: threading.Thread, fixtures: FrozenSet[str]) -> bool:
    # Threads Python did not start (Qt's, PortAudio's callback thread) get a
    # stand-in the first time they call threading.current_thread(); it cannot
    # be joined and is never removed.
    return isinstance(thread, threading._DummyThread)


def _pytest_own(thread: threading.Thread, fixtures: FrozenSet[str]) -> bool:
    module = getattr(_entry_point(thread), "__module__", None) or ""
    return module.split(".")[0].startswith(("_pytest", "pytest"))


def _mcp_runtime(thread: threading.Thread, fixtures: FrozenSet[str]) -> bool:
    return thread.name == "JarvisMCPRuntime" and "shutdown_persistent_runtime" in fixtures


RESIDENTS = (
    Resident(
        "foreign",
        "started outside Python's threading module, so Python only holds a stand-in",
        _foreign,
    ),
    Resident(
        "pytest",
        "started by pytest or one of its plugins",
        _pytest_own,
    ),
    Resident(
        "mcp-runtime",
        "the persistent MCP runtime's loop thread, for a test that requests the "
        "fixture which owns its shutdown",
        _mcp_runtime,
    ),
)


def snapshot() -> FrozenSet[threading.Thread]:
    """The threads alive right now, by identity (an ident can be reused)."""
    return frozenset(threading.enumerate())


def resident_label(thread: threading.Thread, fixtures: FrozenSet[str] = frozenset()) -> Optional[str]:
    """The label of the rule that exempts the thread, or None."""
    for resident in RESIDENTS:
        if resident.matches(thread, fixtures):
            return resident.label
    return None


def new_survivors(
    before: Iterable[threading.Thread],
    fixtures: FrozenSet[str] = frozenset(),
    grace: Optional[float] = None,
) -> List[threading.Thread]:
    """The threads started since ``before`` that are still alive after the grace.

    ``fixtures`` are the names of the fixtures the test requested; a resident
    rule may depend on them. The grace is shared: a test that leaks several
    threads costs one wait, not one each.
    """
    grace = JOIN_GRACE_SEC if grace is None else grace
    known = frozenset(before)
    skip = {threading.main_thread(), threading.current_thread()}
    candidates = [
        thread for thread in threading.enumerate()
        if thread not in known and thread not in skip and resident_label(thread, fixtures) is None
    ]
    deadline = time.monotonic() + grace
    for thread in candidates:
        try:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        except RuntimeError:
            # Not started yet: is_alive() below decides.
            pass
    return [thread for thread in candidates if thread.is_alive()]


def wait_for_workers(before: Iterable[threading.Thread], timeout: float = 5.0) -> None:
    """Let the workers the code under test started finish, or fail naming them.

    For code that starts a one-shot worker and keeps no handle to it. Call it
    inside any patch the worker relies on, so the worker ends while the patch
    still holds rather than running on, unpatched, after the test body.
    """
    survivors = new_survivors(before, grace=timeout)
    if survivors:
        names = [f"  - {describe(thread)}" for thread in survivors]
        pytest.fail(
            f"{len(survivors)} worker thread(s) still running after {timeout}s:\n"
            + "\n".join(names),
            pytrace=False,
        )


def describe(thread: threading.Thread) -> str:
    kind = type(thread).__name__
    daemon = "daemon" if thread.daemon else "non-daemon"
    return f"{thread.name!r} ({kind}, {daemon}, runs {_entry_point_name(thread)})"


def report(nodeid: str, survivors: Iterable[threading.Thread]) -> str:
    survivors = list(survivors)
    lines = [
        f"{nodeid} left {len(survivors)} thread(s) running after its teardown:",
        *(f"  - {describe(thread)}" for thread in survivors),
        "Stop each one in the test's own fixture (set its stop signal, then join it).",
        "A thread meant to outlive tests belongs to a wider-scoped fixture; one that "
        "Python did not start belongs in thread_guard.RESIDENTS, with the reason.",
        "Exempt today:",
        *(f"  - {resident.label}: {resident.why}" for resident in RESIDENTS),
    ]
    return "\n".join(lines)


@pytest.fixture(autouse=True)
def _no_thread_outlives_its_test(request):
    before = snapshot()
    yield
    survivors = new_survivors(before, fixtures=frozenset(request.fixturenames))
    if survivors:
        pytest.fail(report(request.node.nodeid, survivors), pytrace=False)

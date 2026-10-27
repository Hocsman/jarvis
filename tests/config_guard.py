"""The suite's guard against a test that opens the user's own config file.

``tests/conftest.py`` imports the autouse fixture below, so it wraps every test
in the suite. The sandbox in ``conftest.py`` points the config path at a file
of the test's own; this guard is what notices when it did not.

The files it protects are the ones the app would open with the environment
the run started in: ``default_config_path()`` as it answered before any test
changed a variable, ``~/.config/jarvis/config.json``, and the file
``JARVIS_CONFIG_PATH`` named, if it named one.

For the duration of a test, ``_load_json`` and ``_save_json`` refuse those
files: they raise ``RealUserConfigRefused`` before opening anything, so
nothing of the user's is read or written, and the request is recorded. A
module that did ``from jarvis.config import _load_json`` at import holds the
function itself rather than the name, so the guard replaces it there too,
wherever a loaded module holds it. The caller may swallow the refusal (the
config readers catch broad exceptions), so the test fails at teardown on the
record, naming each file it asked for.

``tests/test_the_suite_keeps_off_the_users_config.py`` holds the guard to all
of this.
"""

import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Callable, Dict, FrozenSet, Iterator, List, Tuple

import pytest

# The suite imports the config module under both names, and they are distinct
# objects with distinct functions.
CONFIG_MODULES = ("jarvis.config", "src.jarvis.config")
GUARDED_FUNCTIONS = ("_load_json", "_save_json")

# Set on each stand-in, naming the function it stands in for.
ORIGINAL = "_config_guard_original"


class RealUserConfigRefused(RuntimeError):
    """A test asked to read or write the user's own config file."""


# The protected files, as ``key`` spells them. Filled once per run.
_protected: FrozenSet[str] = frozenset()

# What the running test asked for. A stand-in left in a module by an earlier
# test records here too, so this is one list rather than one per stand-in.
_requests: List[str] = []

_stand_ins: Dict[int, Callable] = {}

# Module name -> (module, names that held a guarded function when it was
# first seen). Scanning every module's namespace once per test would cost
# more than the rest of the guard; a module is only rescanned when the
# object under its name changes.
_scanned: Dict[str, Tuple[ModuleType, Tuple[str, ...]]] = {}


def key(path) -> str:
    return os.path.normcase(os.path.abspath(os.path.expanduser(os.fspath(path))))


def is_protected(path) -> bool:
    try:
        return key(path) in _protected
    except TypeError:
        return False


def real_user_config_paths() -> FrozenSet[str]:
    """The files the app would open, with the environment as it is now."""
    from jarvis.config import default_config_path

    paths = [default_config_path(), Path.home() / ".config" / "jarvis" / "config.json"]
    named = os.environ.get("JARVIS_CONFIG_PATH")
    if named:
        paths.append(Path(named))
    return frozenset(key(path) for path in paths)


def take_requests() -> List[str]:
    """What the running test asked for, which no longer counts against it."""
    taken = list(_requests)
    _requests.clear()
    return taken


def stand_in_for(original: Callable) -> Callable:
    stand_in = _stand_ins.get(id(original))
    if stand_in is not None:
        return stand_in

    def refuses_the_real_config(*args, **kwargs):
        path = args[0] if args else kwargs.get("path")
        if is_protected(path):
            _requests.append(f"{original.__name__}({path})")
            raise RealUserConfigRefused(
                f"the test suite does not open the user's own config file: "
                f"{original.__name__} asked for {path}"
            )
        return original(*args, **kwargs)

    refuses_the_real_config.__name__ = original.__name__
    refuses_the_real_config.__doc__ = original.__doc__
    setattr(refuses_the_real_config, ORIGINAL, original)
    _stand_ins[id(original)] = refuses_the_real_config
    return refuses_the_real_config


def originals() -> Dict[int, Callable]:
    """The guarded functions of every loaded config module, by identity."""
    import jarvis.config  # noqa: F401  (the name production code uses)

    found: Dict[int, Callable] = {}
    for name in CONFIG_MODULES:
        module = sys.modules.get(name)
        if module is None:
            continue
        for function in GUARDED_FUNCTIONS:
            value = getattr(module, function)
            value = getattr(value, ORIGINAL, value)
            found[id(value)] = value
    return found


def bindings(guarded: Dict[int, Callable]) -> Iterator[Tuple[ModuleType, str, Callable]]:
    """Every place a loaded module holds one of the guarded functions."""
    for module_name, module in list(sys.modules.items()):
        namespace = getattr(module, "__dict__", None)
        if not isinstance(namespace, dict):
            continue
        seen = _scanned.get(module_name)
        if seen is None or seen[0] is not module:
            names = tuple(name for name, value in list(namespace.items()) if id(value) in guarded)
            seen = (module, names)
            _scanned[module_name] = seen
        for name in seen[1]:
            value = namespace.get(name)
            if id(value) in guarded:
                yield module, name, value


def report(nodeid: str, requests: List[str]) -> str:
    lines = [f"  - {request}" for request in requests]
    return (
        f"{nodeid} asked for the user's own config file "
        f"({len(requests)} time(s)); the sandbox in conftest.py did not cover it:\n"
        + "\n".join(lines)
    )


@pytest.fixture(scope="session")
def _real_user_config():
    """Taken before any test-scoped fixture has changed the environment."""
    return real_user_config_paths()


@pytest.fixture(autouse=True)
def _no_real_user_config(_real_user_config, request, monkeypatch):
    global _protected
    _protected = _real_user_config
    _requests.clear()
    for module, name, original in list(bindings(originals())):
        monkeypatch.setattr(module, name, stand_in_for(original))
    yield
    refused = take_requests()
    if refused:
        pytest.fail(report(request.node.nodeid, refused), pytrace=False)

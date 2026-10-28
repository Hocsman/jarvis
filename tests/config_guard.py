"""The suite's guard against a test that opens the user's own config file.

``tests/conftest.py`` imports the autouse fixture below, so it wraps every test
in the suite. The sandboxes in ``conftest.py`` point the config path at a file
of the run's own and then of each test's own; this guard is what notices when
they did not.

The files it protects are the ones the app would open with the environment
the run started in: ``default_config_path()`` as it answered before the
sandboxes changed a variable, ``~/.config/jarvis/config.json``, and the file
``JARVIS_CONFIG_PATH`` named, if it named one. ``arm()`` takes them, once;
``conftest.py`` calls it before it gives the run a config home of its own.

During every test, ``_load_json`` and ``_save_json`` refuse those files: they
raise ``RealUserConfigRefused`` before opening anything, so nothing of the
user's is read or written, and the request is recorded. A module that did
``from jarvis.config import _load_json`` at import holds the function itself
rather than the name, so the guard replaces it there too, wherever a loaded
module holds it. The caller may swallow the refusal (the config readers catch
broad exceptions), so the test fails at teardown on the record, naming each
file it asked for.

A module first imported during a test binds the guarded stand-in and keeps it
afterwards; it goes on refusing those files between tests, where nothing
reports it, and the run's own config home is what keeps that code off them.

``tests/test_the_suite_keeps_off_the_users_config.py`` holds the guard to all
of this.
"""

import importlib
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Callable, Dict, FrozenSet, Iterator, List, Set, Tuple

import pytest

# The suite imports the config module under both names, and they are distinct
# objects with distinct functions.
CONFIG_MODULES = ("jarvis.config", "src.jarvis.config")
GUARDED_FUNCTIONS = ("_load_json", "_save_json")

# Set on each stand-in, naming the function it stands in for.
ORIGINAL = "_config_guard_original"


class RealUserConfigRefused(RuntimeError):
    """A test asked to read or write the user's own config file."""


# The protected files, as ``key`` spells them. Set by ``arm()``.
_protected: FrozenSet[str] = frozenset()

# What the running test asked for. A stand-in left in a module by an earlier
# test records here too, so this is one list rather than one per stand-in.
_requests: List[str] = []

_stand_ins: Dict[int, Callable] = {}

# Every module name scanned so far, and the modules that held a guarded
# function when scanned, with those names. A test only scans the modules
# imported since the one before it, and rescans a holder whose module object
# changed: walking every namespace for every test would cost more than the
# rest of the guard, and more with every dependency.
_seen: Set[str] = set()
_holders: Dict[str, Tuple[ModuleType, Tuple[str, ...]]] = {}


def key(path) -> str:
    """One spelling per file: symlinks, ``..``, ``~``, short names and case."""
    return os.path.normcase(os.path.realpath(os.path.expanduser(os.fspath(path))))


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


def arm() -> FrozenSet[str]:
    """Protect the files the app would open now. The first call decides."""
    global _protected
    if not _protected:
        _protected = real_user_config_paths()
    return _protected


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
    """The guarded functions of both config modules, by identity."""
    found: Dict[int, Callable] = {}
    for name in CONFIG_MODULES:
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        for function in GUARDED_FUNCTIONS:
            value = getattr(module, function)
            value = getattr(value, ORIGINAL, value)
            found[id(value)] = value
    return found


def _scan(module_name: str, module, guarded: Dict[int, Callable]) -> None:
    _seen.add(module_name)
    namespace = getattr(module, "__dict__", None)
    names: Tuple[str, ...] = ()
    if isinstance(namespace, dict):
        names = tuple(name for name, value in list(namespace.items()) if id(value) in guarded)
    if names:
        _holders[module_name] = (module, names)
    else:
        _holders.pop(module_name, None)


def bindings(guarded: Dict[int, Callable]) -> Iterator[Tuple[ModuleType, str, Callable]]:
    """Every place a loaded module holds one of the guarded functions."""
    modules = sys.modules
    for module_name in modules.keys() - _seen:
        _scan(module_name, modules.get(module_name), guarded)
    for module_name, (module, _) in list(_holders.items()):
        live = modules.get(module_name)
        if live is not None and live is not module:
            _scan(module_name, live, guarded)
    for module_name, (module, names) in list(_holders.items()):
        if modules.get(module_name) is not module:
            continue
        for name in names:
            value = module.__dict__.get(name)
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
    """The protected files. A session that did not arm the guard before
    collection takes them here, before any test-scoped fixture has changed
    the environment."""
    return arm()


@pytest.fixture(autouse=True)
def _no_real_user_config(_real_user_config, request, monkeypatch):
    _requests.clear()
    for module, name, original in list(bindings(originals())):
        monkeypatch.setattr(module, name, stand_in_for(original))
    yield
    refused = take_requests()
    if refused:
        pytest.fail(report(request.node.nodeid, refused), pytrace=False)

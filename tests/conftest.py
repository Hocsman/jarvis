import ipaddress
import os
import socket
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

# Robustly locate repository root (directory containing src/jarvis)
_this_file = Path(__file__).resolve()
ROOT = None
for parent in _this_file.parents:
    if (parent / "src" / "jarvis").exists():
        ROOT = parent
        break
if ROOT is None:
    # Fallback to two levels up
    ROOT = _this_file.parent.parent

SRC = ROOT / "src"
# Both ROOT and SRC are on sys.path so tests can write either
#   ``from src.jarvis.x import ...``  (older style, ``src.`` prefix)
# or
#   ``from jarvis.x import ...``      (newer style, no prefix)
# CAUTION: those two import paths resolve to *distinct module instances*.
# A monkeypatch on ``src.jarvis.memory.conversation.X`` does NOT take
# effect on ``jarvis.memory.conversation.X`` and vice versa. When a test
# stubs out a symbol the production code calls, you MUST patch the same
# module instance the production code resolves at runtime. Production code
# in ``src/`` imports without the ``src.`` prefix (e.g. inside endpoint
# handlers it's ``from jarvis.memory.conversation import ...``), so a test
# that monkeypatches a symbol used by production should also import
# without the prefix. This is the convention going forward; the older
# ``from src.X`` style is left in place to avoid a churn-only sweep, but
# do not adopt it for new tests that monkeypatch.
# Add repository root so that 'src' is a package prefix.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Also add the src directory (optional, for backwards compatibility with direct 'jarvis' imports)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@dataclass
class MockConfig:
    """Minimal config object for unit tests that need a config."""
    # Provider-aware fields. Default to Ollama at localhost so tests
    # that don't care about providers keep the historical behaviour.
    llm_provider: str = "ollama"
    llm_base_url: str = "http://localhost:11434"
    llm_api_key: str = ""
    # ``llm_chat_model`` defaults to empty so tests that pin
    # ``ollama_chat_model = "gpt-oss:20b"`` to exercise the LARGE-model
    # branch get the legacy alias promoted into ``llm_chat_model`` by
    # ``__post_init__`` — same shape ``load_settings()`` produces.
    llm_chat_model: str = ""
    embedding_provider: str = ""
    embedding_base_url: str = ""
    embedding_api_key: str = ""
    embedding_model: str = "nomic-embed-text"
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "gemma4:e2b"
    ollama_embed_model: str = "nomic-embed-text"
    db_path: str = ":memory:"
    sqlite_vss_path: Optional[str] = None
    voice_debug: bool = True
    tts_enabled: bool = False
    tts_engine: str = "piper"
    tts_voice: Optional[str] = None
    tts_rate: int = 200
    tts_piper_model_path: Optional[str] = None
    tts_piper_speaker: Optional[int] = None
    tts_piper_length_scale: float = 1.0
    tts_piper_noise_scale: float = 0.667
    tts_piper_noise_w: float = 0.8
    tts_piper_sentence_silence: float = 0.2
    tts_chatterbox_device: str = "cpu"
    tts_chatterbox_audio_prompt: Optional[str] = None
    tts_chatterbox_exaggeration: float = 0.5
    tts_chatterbox_cfg_weight: float = 0.5
    web_search_enabled: bool = True
    brave_search_api_key: str = ""
    wikipedia_fallback_enabled: bool = True
    llm_tools_timeout_sec: float = 8.0
    llm_embedding_timeout_sec: float = 10.0
    llm_chat_timeout_sec: float = 45.0
    agentic_max_turns: int = 8
    tool_selection_strategy: str = "embedding"
    tool_router_model: str = ""
    memory_enrichment_max_results: int = 5
    memory_enrichment_source: str = "diary"
    location_enabled: bool = True
    location_ip_address: Optional[str] = None
    location_auto_detect: bool = False
    location_cgnat_resolve_public_ip: bool = False
    dialogue_memory_timeout: int = 300
    llm_thinking_enabled: bool = False
    intent_judge_thinking_enabled: bool = False
    dictation_thinking_enabled: bool = False
    mcps: Dict[str, Any] = field(default_factory=dict)
    use_stdin: bool = True

    def __post_init__(self) -> None:
        # Mirror ``load_settings``: when the provider-aware fields are
        # left empty, promote the legacy ``ollama_*`` aliases. Tests can
        # set either pair and end up with consistent reads on either side.
        if not self.llm_chat_model:
            self.llm_chat_model = self.ollama_chat_model
        if not self.llm_base_url:
            self.llm_base_url = self.ollama_base_url
        if not self.embedding_model:
            self.embedding_model = self.ollama_embed_model


@pytest.fixture(scope="session")
def _bac_a_sable(tmp_path_factory):
    """One directory for the whole run.

    ``mktemp`` scans the session's temp root on every call to find the next
    free suffix, so calling it once per test makes the cost quadratic in the
    number of tests. The isolation these fixtures need is a distinct *file*
    per test, which a name inside one directory gives just as well.
    """
    return tmp_path_factory.mktemp("jarvis_sandbox")


@pytest.fixture(autouse=True)
def _isolate_user_config_path(_bac_a_sable, request, monkeypatch):
    """Redirect ``default_config_path`` to a per-session tempfile so a test
    that calls ``load_settings`` (or any other code path that resolves the
    user's config) cannot read or overwrite ``~/.config/jarvis/config.json``.

    Tests that need to exercise the loader against specific JSON should
    monkey-patch ``_load_json`` (and ``_save_json`` if the migration would
    trigger a write) directly. This fixture is a belt-and-braces guard so
    a half-mocked test cannot reach the real config file.
    """
    sandbox = _bac_a_sable / f"config-{abs(hash(request.node.nodeid)):x}.json"
    monkeypatch.setattr("jarvis.config.default_config_path", lambda: sandbox)
    monkeypatch.setenv("JARVIS_CONFIG_PATH", str(sandbox))


@pytest.fixture(autouse=True)
def _isolate_dictation_history(_bac_a_sable, request, monkeypatch):
    """Redirect the dictation history's default path to a per-session tempfile.

    ``DictationHistory()`` falls back to the user's data directory, and the
    dictation engine builds one whenever a caller passes none, so a test that
    omits the argument appends synthetic entries to the same file a real
    user's dictations live in. Running the suite is not permission to write
    there, and a guard here is worth more than every call site remembering.
    """
    target = _bac_a_sable / f"dictation-{abs(hash(request.node.nodeid)):x}.json"

    # The suite imports modules both as ``jarvis.x`` and as ``src.jarvis.x``,
    # which are two distinct module objects. Patching one leaves the other
    # pointing at the real file.
    import importlib

    patched = 0
    for path in ("jarvis.dictation.history", "src.jarvis.dictation.history"):
        try:
            module = importlib.import_module(path)
        except ImportError:
            continue
        monkeypatch.setattr(module, "_default_history_path", lambda: target)
        patched += 1
    assert patched, "neither dictation history module could be imported"


def _foyer_des_donnees() -> Path:
    """The user's own data home: the parent of the database's default path."""
    from jarvis.config import _default_db_path

    return Path(_default_db_path()).expanduser().resolve().parent


def _sous_le_foyer(db_path: str) -> bool:
    """Whether a database path names the user's own data, where the core
    beside it is the user's own ``profil.md``."""
    cible = Path(db_path).expanduser().resolve()
    foyer = _foyer_des_donnees()
    return cible == foyer or foyer in cible.parents


@pytest.fixture(autouse=True)
def _isolate_memory_core(_bac_a_sable, request, monkeypatch):
    """Keep the core out of the working directory and out of the user's data.

    ``MemoryCore.for_config`` places the core beside the database. The test
    configurations use ``:memory:``, whose parent is ``.``, so the core would
    resolve against whichever directory a run starts from, the repository root
    included, where ``git add -A`` would publish it. A database with no
    absolute path gets its core in the sandbox, one per test.

    A database under the user's own data home gets the same: that core is
    the user's ``profil.md`` and ``regles.md``, which the reply engine reads
    into its system prompt and which a test could write into. A settings
    object built from the user's configuration, or from the defaults, names
    exactly that path. Any other absolute path keeps the real resolution, so
    the tests that exercise it exercise the real code.
    """
    coeur = _bac_a_sable / f"core-{abs(hash(request.node.nodeid)):x}"

    # The suite imports modules both as ``jarvis.x`` and as ``src.jarvis.x``,
    # which are two distinct module objects. Patching one leaves the other
    # writing into the working directory.
    import importlib

    patched = 0
    for path in ("jarvis.memory.core", "src.jarvis.memory.core"):
        try:
            module = importlib.import_module(path)
        except ImportError:
            continue
        original = module.MemoryCore.__dict__["for_config"].__func__

        def for_config(klass, cfg, _original=original, _dirname=module.CORE_DIRNAME):
            db_path = str(getattr(cfg, "db_path", "") or "")
            if db_path == ":memory:" or not Path(db_path).expanduser().is_absolute():
                return klass(coeur / _dirname)
            if _sous_le_foyer(db_path):
                return klass(coeur / _dirname)
            return _original(klass, cfg)

        monkeypatch.setattr(module.MemoryCore, "for_config", classmethod(for_config))
        patched += 1
    assert patched, "neither memory core module could be imported"


# ---------------------------------------------------------------------------
# The suite talks to nobody.
#
# A unit test that reaches for the network measures the network. Ollama's
# port answers on IPv4 while ``localhost`` resolves to ``::1`` first, so an
# unmocked call through that name waits two seconds on Windows, and one
# through ``127.0.0.1`` is answered by whatever model happens to be running;
# a location probe or a geocoding request carries the developer's address,
# or a Mock's repr, to a third party. Every Python-level connection,
# datagram and name lookup (forward or reverse) towards a host beyond this
# machine, and every connection to Ollama's port on this one, is refused
# before it is made. The refusal is an ``OSError``, what a host that is
# down or a name that does not resolve produces, so the code under test
# takes the path it takes when the network is absent, and it is recorded
# for a test that wants to see it (``network_refusals``). A server a test
# runs itself, on another loopback port, is left alone. Proxy variables are
# cleared, because a proxy on this machine would carry a request past the
# guard. The performance suite, which measures a live model on purpose and
# is run on its own by path, is the one collection allowed to reach the
# model server. Outside the hook's sight: sockets opened from C, child
# processes, and asyncio's Windows proactor; none is driven by the suite.
# ---------------------------------------------------------------------------
_NETWORK_EVENTS = {
    "socket.connect", "socket.sendto", "socket.sendmsg",
    "socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr",
    "socket.getnameinfo",
}
_ADDRESSED_EVENTS = {"socket.connect", "socket.sendto", "socket.sendmsg"}
_FORWARD_LOOKUPS = {"socket.getaddrinfo", "socket.gethostbyname"}
_LOOPBACK_HOSTS = {None, "", "localhost", "127.0.0.1", "::1", "0.0.0.0"}
_OLLAMA_PORT = 11434
_PROXY_VARIABLES = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "FTP_PROXY")
_network_guard_installed = False
_model_server_off_limits = True
_network_refusals: list = []


def _is_loopback(host) -> bool:
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    return host in _LOOPBACK_HOSTS or (isinstance(host, str) and host.startswith("127."))


def _is_numeric(host) -> bool:
    """A literal address: resolving it asks no resolver and sends nothing."""
    try:
        ipaddress.ip_address(host.decode("ascii", "replace") if isinstance(host, bytes) else host)
    except (ValueError, TypeError):
        return False
    return True


def _address(event, args):
    """The (host, port) an audit event names; a non-tuple address is a
    local socket path, or a datagram on a socket already connected, and
    counts as this machine."""
    if event in _ADDRESSED_EVENTS:
        address = args[1] if len(args) > 1 else None
    elif event == "socket.getnameinfo":
        address = args[0] if args else None
    else:
        address = (args[0], args[1] if len(args) > 1 else None)
    if not isinstance(address, tuple):
        return None, None
    host = address[0] if address else None
    port = address[1] if len(address) > 1 else None
    try:
        port = int(port)
    except (TypeError, ValueError):
        pass
    return host, port


def _refuse_the_network(event, args):
    if event not in _NETWORK_EVENTS:
        return
    host, port = _address(event, args)
    if event in _FORWARD_LOOKUPS and _is_numeric(host) and port != _OLLAMA_PORT:
        # Turning a literal into an address is arithmetic, not a lookup;
        # the connection that would follow is judged on its own. Looking a
        # literal up backwards does ask a resolver, and is not exempt.
        return
    if not _is_loopback(host) or (port == _OLLAMA_PORT and _model_server_off_limits):
        _network_refusals.append((event, host, port))
        message = (
            f"the test suite talks to nobody: {event} to {host!r}:{port!r} refused; "
            "a test that needs a public name resolved takes the public_dns fixture, "
            "one that needs a model belongs in evals/"
        )
        # The error a resolver gives for a name it cannot answer, and the
        # one a host gives for a port nobody listens on.
        raise (ConnectionRefusedError if event in _ADDRESSED_EVENTS else socket.gaierror)(message)


def _the_model_server_is_wanted(args, root) -> bool:
    """Only the performance suite may reach a live model server. It is run on
    its own, by path (``pytest tests/performance/ -m performance``), and that
    path, under this repository, is what lifts the rule; a marker expression
    cannot, because the default one already spells the word, and a directory
    elsewhere that happens to share the name is not the suite."""
    suite = (Path(root) / "tests" / "performance").resolve()
    for arg in args:
        given = Path(str(arg))
        candidate = (given if given.is_absolute() else Path(root) / given).resolve()
        if candidate == suite or suite in candidate.parents:
            return True
    return False


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "real_download_helper: runs the real download_snapshot_with_progress; "
        "the test fakes the Hugging Face Hub it talks to",
    )
    global _network_guard_installed, _model_server_off_limits
    _model_server_off_limits = not _the_model_server_is_wanted(config.args, config.rootpath)
    for name in _PROXY_VARIABLES:
        os.environ.pop(name, None)
        os.environ.pop(name.lower(), None)
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"
    if not _network_guard_installed:
        # An audit hook cannot be removed; it is installed once, before
        # collection, so a module that probes a server at import is covered.
        sys.addaudithook(_refuse_the_network)
        _network_guard_installed = True


@pytest.fixture
def network_refusals():
    """The connections the guard refused during this test, as
    ``(event, host, port)``, for a test that wants to see them. Refusals
    from any thread count, a thread left running by an earlier test
    included: what reaches for the network during a test is that test's
    to explain."""
    del _network_refusals[:]
    return _network_refusals


@pytest.fixture(autouse=True)
def _no_router_discovery(monkeypatch):
    """The location module asks the router for the public address over UPnP
    first. miniupnpc does that from C, on sockets the audit hook never sees,
    so the library is declared absent for the test's duration and the
    module takes its no-UPnP path. Both module identities, as everywhere in
    this file."""
    import importlib

    for path in ("jarvis.utils.location", "src.jarvis.utils.location"):
        try:
            module = importlib.import_module(path)
        except ImportError:
            continue
        monkeypatch.setattr(module, "MINIUPNPC_AVAILABLE", False, raising=False)


@pytest.fixture
def public_dns(monkeypatch):
    """Every hostname beyond this machine resolves to one public address,
    without asking a resolver.

    The SSRF guard resolves a URL's host before fetching it. A test that
    hands a tool ``https://example.com`` is testing the fetch, not the
    Internet's answer for that name. Loopback names and numeric addresses
    keep their real, offline resolution, so a test that expects
    ``localhost`` or ``10.0.0.1`` to be refused still sees it refused.
    """
    import socket

    real = socket.getaddrinfo
    address = "93.184.216.34"

    def resolve(host, port, *args, **kwargs):
        if _is_loopback(host) or _is_numeric(host):
            return real(host, port, *args, **kwargs)
        try:
            port = int(port)
        except (TypeError, ValueError):
            port = 0
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port))]

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    return address


@pytest.fixture(autouse=True)
def _isolate_hugging_face_hub(_bac_a_sable, request, monkeypatch):
    """Keep every test off the Hugging Face Hub and out of the user's model cache.

    The listener fetches its Whisper snapshot before it builds the model, so a
    test that stubs ``WhisperModel`` alone still downloads gigabytes into
    ``~/.cache/huggingface``. The fetch is ``download_snapshot_with_progress``,
    which the listener imports from ``jarvis.utils.hf_download`` at call time,
    so that module attribute is replaced by a refusal. A test of the helper
    itself carries the ``real_download_helper`` marker and fakes the Hub on
    its own.

    Underneath, whatever reaches huggingface_hub by another road finds it
    offline, with its cache in the sandbox. huggingface_hub reads both
    switches from the environment once, into module constants, so the
    constants are patched as well as the variables (which still reach any
    process a test spawns). Its HTTP sessions settle their offline mode when
    they are built and are kept per thread, so they are dropped on the way in
    and on the way out.

    The developer's credentials stay out of reach the same way. huggingface_hub
    reads a token from ``HF_TOKEN`` first, then from the token file under its
    home, whenever it builds a request, so the variables are cleared and the
    token file and stored tokens move into the sandbox with the cache. A token
    belongs to the person who logged in, and the suite runs anonymous.
    """
    home = _bac_a_sable / "huggingface"
    cache = home / "hub"
    token = home / "token"
    monkeypatch.setenv("HF_HOME", str(home))
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    monkeypatch.setenv("HF_TOKEN_PATH", str(token))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    try:
        import huggingface_hub.constants
    except ImportError:
        # No huggingface_hub, so no Hub client and no helper to stand in for.
        yield
        return
    monkeypatch.setattr(huggingface_hub.constants, "HF_HOME", str(home))
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(cache))
    monkeypatch.setattr(huggingface_hub.constants, "HF_TOKEN_PATH", str(token))
    monkeypatch.setattr(huggingface_hub.constants, "HF_STORED_TOKENS_PATH", str(home / "stored_tokens"))
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)

    if request.node.get_closest_marker("real_download_helper") is None:
        def _refuse_download(repo_id, *args, **kwargs):
            raise RuntimeError(f"the test suite does not download {repo_id} from the Hugging Face Hub")

        # Same two module identities as the fixtures above.
        import importlib

        patched = 0
        for path in ("jarvis.utils.hf_download", "src.jarvis.utils.hf_download"):
            try:
                module = importlib.import_module(path)
            except ImportError:
                continue
            monkeypatch.setattr(module, "download_snapshot_with_progress", _refuse_download)
            patched += 1
        assert patched, "neither hf_download module could be imported"

    # The sessions live in the half of huggingface_hub that needs requests.
    # A test module that stubs requests at collection breaks that half, and
    # with it any session to drop; the layers above stand regardless.
    try:
        from huggingface_hub.utils import reset_sessions
    except Exception:
        reset_sessions = None
    if reset_sessions is not None:
        reset_sessions()
    yield
    if reset_sessions is not None:
        reset_sessions()


@pytest.fixture
def mock_config():
    """Provide a mock configuration for unit tests."""
    return MockConfig()


@pytest.fixture
def db():
    """Provide an in-memory database for unit tests."""
    from jarvis.memory.db import Database
    database = Database(":memory:", sqlite_vss_path=None)
    yield database
    database.close()


@pytest.fixture
def dialogue_memory():
    """Provide a dialogue memory instance for unit tests."""
    from jarvis.memory.conversation import DialogueMemory
    return DialogueMemory(inactivity_timeout=300, max_interactions=20)


@pytest.fixture
def qapp():
    """Provide a shared QApplication for Qt-based UI tests.

    Qt requires exactly one QApplication per process.  Re-uses an existing
    instance when present so repeated test runs inside a single session
    don't error.
    """
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app
    app.processEvents()
    for widget in list(app.topLevelWidgets()):
        try:
            widget.close()
            widget.deleteLater()
        except Exception:
            pass
    app.processEvents()

@pytest.fixture
def tools_unrestricted(monkeypatch):
    """Run tool calls with the policy gate standing aside.

    For tests that exercise what a tool DOES, not whether it is allowed
    to. Every call goes through ``run_tool_with_retries``, which now
    consults the user's policy first, so a test about `localFiles`
    deleting a file would otherwise be a test about the gate refusing it.

    The gate has its own suite in ``test_tool_gate.py``; this fixture
    makes the separation explicit at each call site rather than hiding
    the gate from the whole suite with an autouse override.
    """
    class _AllowAll:
        def verdict(self, name, risk):
            return "libre"

    # The suite imports the registry both as ``jarvis.tools.registry`` and
    # as ``src.jarvis.tools.registry``, which are two distinct module
    # objects. Patching one leaves the other gated, so the fixture would
    # appear to do nothing in half the tests that ask for it.
    import importlib

    patched = 0
    for path in ("jarvis.tools.registry", "src.jarvis.tools.registry"):
        try:
            module = importlib.import_module(path)
        except ImportError:
            continue
        monkeypatch.setattr(module, "load_tool_policy", lambda cfg: _AllowAll())
        patched += 1
    assert patched, "neither registry module could be imported"


class _Hop:
    """A response that is a redirect and nothing else. Its body is never
    to be read, so reading it fails the test; closing it is recorded."""

    is_redirect = True
    is_permanent_redirect = False
    status_code = 302
    encoding = "utf-8"

    def __init__(self, to: str):
        self.headers = {"Location": to}
        self.closed = False

    @property
    def content(self):
        raise AssertionError("a hop's body was read")

    @property
    def text(self):
        raise AssertionError("a hop's body was read")

    def iter_content(self, chunk_size=8192):
        raise AssertionError("a hop's body was read")

    def raise_for_status(self):
        pass

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


@pytest.fixture
def hop():
    """``hop(to)``: a redirect response towards *to*, for either tool's
    redirect walk. The files that walk redirects share this one double
    rather than each keeping a copy, since two copies drift."""
    return _Hop


@pytest.fixture
def get_like_requests():
    """``get_like_requests(*responses)``: a stand-in for ``requests.get``
    that serves the responses in order and behaves as the library does
    about the body: unless asked for a stream, it reads the whole body
    before handing the response back. A bare mock in that place cannot
    tell a streamed request from an unstreamed one; this can. It also
    refuses to make a request while a hop it served is still open."""
    def make(*responses):
        queue = iter(responses)
        served = []

        def get(url, **kwargs):
            for earlier in served:
                assert getattr(earlier, "closed", True), (
                    "a hop was left open when the next request was made")
            response = next(queue)
            served.append(response)
            if kwargs.get("stream") is not True:
                response.content
            return response

        return get
    return make

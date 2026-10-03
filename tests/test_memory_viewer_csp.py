"""The viewer page is structurally hard to inject into.

The page holds the launch token, so script running inside it can write
the user's memory files. Escaping every server-provided string keeps
markup out of the page today; this suite pins the layer underneath, the
one that still holds on the day an escape is missed:

- a Content-Security-Policy whose ``script-src`` admits the viewer's own
  origin and nothing inline, so injected markup cannot run;
- a page that is therefore built to need no inline script: its one script
  is served from its own route, the token travels in a ``<meta>`` tag, and
  no element carries an event-handler attribute;
- handlers that are never strings with an id interpolated into them: a
  control carries a ``data-action`` name the script registers, or is wired
  by ``addEventListener`` on the element itself;
- server-provided values reaching attributes through the DOM (``dataset``,
  properties), never through a template literal, so the quoting of an
  escaper is not what stands between a value and the markup.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from types import SimpleNamespace

import pytest

try:
    import flask  # noqa: F401

    _HAS_FLASK = True
except ImportError:
    _HAS_FLASK = False


pytestmark = pytest.mark.skipif(not _HAS_FLASK, reason="Flask not available")

TOKEN_META_NAME = "jarvis-viewer-token"


@pytest.fixture()
def viewer(tmp_path, monkeypatch):
    """The viewer app on an isolated database, core and ledger."""
    from desktop_app import memory_viewer
    from jarvis.memory.core import MemoryCore
    from jarvis.memory.db import Database

    db_path = str(tmp_path / "viewer.db")
    Database(db_path).close()
    monkeypatch.setattr(memory_viewer, "_get_db_path", lambda: db_path)
    monkeypatch.setattr(memory_viewer, "_core", MemoryCore(tmp_path / "yuba"))
    monkeypatch.setattr(memory_viewer, "_db_conn", None)
    monkeypatch.setattr(memory_viewer, "_activity_db", None)
    monkeypatch.setattr(memory_viewer, "_graph_store", None)
    memory_viewer.app.config["TESTING"] = True

    yield SimpleNamespace(
        mv=memory_viewer,
        client=memory_viewer.app.test_client(),
        token=memory_viewer.get_launch_token(),
    )

    conn = memory_viewer._db_conn
    if conn is not None:
        conn.close()


class _PageScan(HTMLParser):
    """What a browser's parser would find in the page that can run code."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.scripts: list[tuple[dict, str]] = []
        self.handler_attributes: list[tuple[str, str]] = []
        self.script_urls: list[tuple[str, str]] = []
        self.meta: dict[str, str | None] = {}
        self._open_script: list | None = None

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        for name, value in attrs:
            if name.startswith("on"):
                self.handler_attributes.append((tag, name))
            if value and value.strip().lower().startswith("javascript:"):
                self.script_urls.append((tag, name))
        if tag == "script":
            self._open_script = [attributes, ""]
        if tag == "meta" and "name" in attributes:
            self.meta[attributes["name"]] = attributes.get("content")

    def handle_data(self, data):
        if self._open_script is not None:
            self._open_script[1] += data

    def handle_endtag(self, tag):
        if tag == "script" and self._open_script is not None:
            self.scripts.append((self._open_script[0], self._open_script[1]))
            self._open_script = None


def _page(viewer) -> str:
    return viewer.client.get("/").get_data(as_text=True)


def _scan(viewer) -> _PageScan:
    scan = _PageScan()
    scan.feed(_page(viewer))
    return scan


def _script(viewer) -> str:
    """The script text the page runs, fetched from where the page points."""
    from conftest import viewer_scripts

    return "\n".join(viewer_scripts(viewer.client))


def _policy(response) -> dict[str, list[str]]:
    directives: dict[str, list[str]] = {}
    for part in response.headers["Content-Security-Policy"].split(";"):
        tokens = part.split()
        if tokens:
            directives[tokens[0]] = tokens[1:]
    return directives


# ── The policy ────────────────────────────────────────────────────────


class TestScriptPolicy:
    """The header turns the page's structure into an enforced rule."""

    def test_scripts_may_come_from_the_viewers_own_origin_only(self, viewer):
        policy = _policy(viewer.client.get("/"))

        assert "script-src" in policy
        for source in policy["script-src"]:
            assert source == "'self'" or source.startswith("'nonce-"), (
                f"script-src admits {source}"
            )

    def test_inline_script_and_eval_are_not_admitted_anywhere(self, viewer):
        policy = _policy(viewer.client.get("/"))

        for directive, sources in policy.items():
            assert "'unsafe-eval'" not in sources, directive
            if directive != "style-src":
                assert "'unsafe-inline'" not in sources, directive

    def test_nothing_may_be_loaded_from_a_remote_origin(self, viewer):
        """Every fetch directive names the viewer's own origin, nothing,
        or (for styles, which the page sets inline) inline; a host, a
        scheme or a wildcard would reopen the door."""
        policy = _policy(viewer.client.get("/"))

        for directive, sources in policy.items():
            if directive == "frame-ancestors":
                continue
            for source in sources:
                assert source in {"'self'", "'none'", "'unsafe-inline'"}, (
                    f"{directive} admits {source}"
                )

    def test_whatever_is_not_named_is_refused(self, viewer):
        policy = _policy(viewer.client.get("/"))

        assert policy["default-src"] == ["'none'"]

    def test_the_page_cannot_be_rebased_or_post_a_form_elsewhere(self, viewer):
        policy = _policy(viewer.client.get("/"))

        assert policy["base-uri"] == ["'none'"]
        assert policy["form-action"] == ["'none'"]

    def test_framing_stays_refused(self, viewer):
        policy = _policy(viewer.client.get("/"))

        assert policy["frame-ancestors"] == ["'none'"]

    @pytest.mark.parametrize(
        "path", ["/", "/viewer.js", "/api/health", "/api/core", "/no-such-route"],
    )
    def test_every_response_carries_the_policy_and_forbids_sniffing(self, viewer, path):
        resp = viewer.client.get(path)

        assert "script-src" in _policy(resp)
        assert resp.headers["X-Content-Type-Options"] == "nosniff"

    def test_a_refusal_carries_the_policy_too(self, viewer):
        resp = viewer.client.get("/api/core", headers={"Host": "attacker.example:5050"})

        assert resp.status_code == 400
        assert "script-src" in _policy(resp)


# ── The page that policy allows ───────────────────────────────────────


class TestThePageNeedsNoInlineScript:
    def test_every_script_in_the_page_is_an_external_reference(self, viewer):
        scripts = _scan(viewer).scripts

        assert scripts, "the page loads no script at all"
        for attributes, body in scripts:
            assert attributes.get("src"), "the page carries an inline script"
            assert not body.strip(), "a script tag with a src also has a body"

    def test_no_element_carries_an_event_handler_attribute(self, viewer):
        assert _scan(viewer).handler_attributes == []

    def test_no_attribute_is_a_javascript_url(self, viewer):
        assert _scan(viewer).script_urls == []

    def test_the_scripts_are_served_from_the_viewers_own_origin(self, viewer):
        for attributes, _body in _scan(viewer).scripts:
            src = attributes["src"]
            assert src.startswith("/") and not src.startswith("//"), src

    def test_the_referenced_script_is_served_as_javascript(self, viewer):
        for attributes, _body in _scan(viewer).scripts:
            resp = viewer.client.get(attributes["src"])

            assert resp.status_code == 200
            assert "javascript" in resp.headers["Content-Type"]
            assert resp.get_data(as_text=True).strip()

    def test_the_script_route_is_held_to_the_same_host_gate(self, viewer):
        src = _scan(viewer).scripts[0][0]["src"]

        resp = viewer.client.get(src, headers={"Host": "attacker.example:5050"})

        assert resp.status_code == 400

    def test_the_script_is_not_cached(self, viewer):
        src = _scan(viewer).scripts[0][0]["src"]

        assert viewer.client.get(src).headers["Cache-Control"] == "no-store"


class TestTheTokenRidesInTheMarkupNotTheScript:
    """The script route is a read: it answers without a token, to any
    local caller. Anything secret in it would be public."""

    def test_the_token_is_in_a_meta_tag(self, viewer):
        assert _scan(viewer).meta.get(TOKEN_META_NAME) == viewer.token

    def test_the_script_reads_the_token_from_that_tag(self, viewer):
        assert TOKEN_META_NAME in _script(viewer)

    def test_the_script_itself_holds_no_token(self, viewer):
        assert viewer.token not in _script(viewer)

    def test_the_token_is_not_assigned_to_a_global(self, viewer):
        """A global the page's own script wrote is a global an injected
        one reads as well."""
        assert "__JARVIS_VIEWER_TOKEN__" not in _page(viewer) + _script(viewer)


# ── The script that page runs ─────────────────────────────────────────


TAG_WITH_HANDLER_RE = re.compile(r"<[a-zA-Z][^<>]*\son[a-z]+\s*=", re.DOTALL)

# A handler string is a call with a literal in it; an interpolated id is
# the literal being `${...}`.
CALL_WITH_INTERPOLATED_ARGUMENT_RE = re.compile(r"""\w\(\s*['"]\$\{""")

# A value spliced into an attribute of markup the script builds: a
# template-literal interpolation inside a quoted attribute value, or a
# string concatenation that closes the JS literal mid-attribute.
ATTRIBUTE_SPLICE_RES = (
    re.compile(r'=\s*"[^"\n]*\$\{'),
    re.compile(r"=\s*'[^'\n]*\$\{"),
    re.compile(r'=\s*"[^"\n]*\'\s*\+'),
    re.compile(r"=\s*'[^'\n]*\"\s*\+"),
)


class TestNoHandlerStringsInTheScript:
    def test_no_markup_the_script_builds_carries_a_handler_attribute(self, viewer):
        assert TAG_WITH_HANDLER_RE.findall(_script(viewer)) == []

    def test_no_handler_string_interpolates_an_id(self, viewer):
        assert CALL_WITH_INTERPOLATED_ARGUMENT_RE.findall(_script(viewer)) == []

    def test_the_script_never_builds_code_from_strings(self, viewer):
        script = _script(viewer)

        assert "eval(" not in script
        assert "new Function" not in script
        assert re.search(r"set(?:Timeout|Interval)\(\s*['\"`]", script) is None
        assert "document.write" not in script
        assert "javascript:" not in script


class TestActionsAreWiredToRegisteredHandlers:
    """A ``data-action`` name on an element selects the handler registered
    under it. A name nobody registered is a button that does nothing; a
    registration nobody uses is dead code that a future template could
    trigger by accident."""

    ACTION_ATTRIBUTE_RE = re.compile(r'data-action="([^"]*)"')
    ACTION_PROPERTY_RE = re.compile(r"""dataset\.action\s*=\s*(['"])([^'"]*)\1""")
    REGISTRATION_RE = re.compile(r"""registerAction\(\s*'([^']+)'""")

    def _wired(self, viewer) -> set[str]:
        text = _page(viewer) + "\n" + _script(viewer)
        names = set(self.ACTION_ATTRIBUTE_RE.findall(text))
        names |= {name for _quote, name in self.ACTION_PROPERTY_RE.findall(text)}
        return names

    def _registered(self, viewer) -> set[str]:
        return set(self.REGISTRATION_RE.findall(_script(viewer)))

    def test_there_are_actions_to_check(self, viewer):
        """Guards the extraction: an empty set would make the other
        checks pass over nothing."""
        assert self._wired(viewer)
        assert self._registered(viewer)

    def test_every_wired_action_has_a_registered_handler(self, viewer):
        unregistered = self._wired(viewer) - self._registered(viewer)

        assert unregistered == set()

    def test_every_registered_handler_is_wired_somewhere(self, viewer):
        unused = self._registered(viewer) - self._wired(viewer)

        assert unused == set()

    def test_an_action_name_is_never_built_at_run_time(self, viewer):
        """The names are literals, so the two checks above see them all."""
        text = _page(viewer) + "\n" + _script(viewer)

        assert 'data-action="${' not in text
        assert re.search(r"""dataset\.action\s*=\s*(?![\s'"])""", text) is None
        assert re.search(r"""setAttribute\(\s*['"]data-action['"]""", text) is None

    def test_one_listener_dispatches_them(self, viewer):
        """One listener dispatches every ``data-action`` name; a control
        that carries one needs no listener of its own."""
        script = _script(viewer)

        assert "document.addEventListener('click'" in script
        assert "closest('[data-action]')" in script


class TestServerValuesReachAttributesThroughTheDom:
    """An escaper's quoting is what keeps a value inside an attribute it
    was spliced into. Setting the attribute through the DOM removes the
    question: there is no markup to leave."""

    @pytest.mark.parametrize("splice", ATTRIBUTE_SPLICE_RES, ids=lambda r: r.pattern)
    def test_no_attribute_value_is_spliced_into_markup(self, viewer, splice):
        hits = [
            line.strip() for line in _script(viewer).splitlines() if splice.search(line)
        ]

        assert hits == []

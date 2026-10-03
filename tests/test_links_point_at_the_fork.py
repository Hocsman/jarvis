"""Every link a user follows from this project leads to this project.

The updater already reads releases from `GITHUB_REPO`, but the README sent
people to upstream's releases (builds without this fork's features), the clone
command fetched upstream's source, and both "report an issue" buttons in the
app opened a prefilled form on upstream's tracker, for features upstream does
not have.

The tests tie every such link to `GITHUB_REPO`, the one constant the updater
uses, so renaming or moving the fork changes one line. What stays pointed at
upstream is attribution, and the test says exactly which forms of it are
allowed: the repository root (the fork-of notice) and an upstream issue number
that the link text names as upstream's, because those numbers mean something
only there.
"""

from __future__ import annotations

import re
import urllib.parse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "README.md"
SRC = REPO_ROOT / "src"

_UPSTREAM_URL = re.compile(
    r"https?://github\.com/isair/jarvis(?P<rest>[^\s)\]>\"'`]*)", re.IGNORECASE)
_MARKDOWN_LINK = re.compile(r"\[(?P<text>[^\]]*)\]\((?P<url>[^)\s]+)\)")


def _fork() -> str:
    from desktop_app.updater import GITHUB_REPO
    return GITHUB_REPO


def _upstream_references(text: str):
    """Yield (line number, url, path after the repo, link text or None)."""
    for number, line in enumerate(text.splitlines(), start=1):
        link_text = {m.group("url"): m.group("text") for m in _MARKDOWN_LINK.finditer(line)}
        for match in _UPSTREAM_URL.finditer(line):
            yield number, match.group(0), match.group("rest"), link_text.get(match.group(0))


def _is_attribution(rest: str, link_text: str | None) -> bool:
    if rest == "":
        return True
    return bool(
        re.fullmatch(r"/issues/\d+", rest)
        and link_text
        and link_text.strip().lower().startswith("upstream")
    )


def test_the_issue_form_opens_on_this_projects_tracker():
    """Both report buttons build their URL here, so one test covers both."""
    from desktop_app.app import _new_issue_url

    url = _new_issue_url("Bug Report", "line one & two\n```\ncode\n```", "bug,crash")

    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)
    assert (parsed.scheme, parsed.netloc) == ("https", "github.com")
    assert parsed.path == f"/{_fork()}/issues/new"
    assert query["title"] == ["Bug Report"]
    assert query["labels"] == ["bug,crash"]
    assert query["body"] == ["line one & two\n```\ncode\n```"]


def test_no_source_file_hardcodes_the_upstream_repository():
    offenders = []
    for path in SRC.rglob("*"):
        if not path.is_file() or path.suffix not in {".py", ".html", ".js", ".md", ".json"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for number, url, _rest, _text in _upstream_references(text):
            offenders.append(f"{path.relative_to(REPO_ROOT).as_posix()}:{number} {url}")

    assert not offenders, (
        "These point users at upstream's repository; build the URL from "
        "desktop_app.updater.GITHUB_REPO instead: " + "; ".join(offenders))


def test_the_readme_sends_install_clone_and_issues_to_the_fork():
    text = README.read_text(encoding="utf-8")

    strays = [
        f"line {number}: {url}"
        for number, url, rest, link_text in _upstream_references(text)
        if not _is_attribution(rest, link_text)
    ]

    assert not strays, (
        f"README.md links to upstream where it should link to {_fork()}: "
        + "; ".join(strays)
        + ". Only the repository root and issue numbers labelled "
          "'upstream ...' may stay.")


def test_the_readme_links_the_fork_for_download_clone_and_issues():
    text = README.read_text(encoding="utf-8")
    fork = _fork()

    assert f"https://github.com/{fork}/releases" in text
    assert f"git clone https://github.com/{fork}.git" in text
    assert f"https://github.com/{fork}/issues" in text


def test_the_readme_credits_the_project_it_forks():
    """Pointing everything at the fork must not erase where it came from."""
    text = README.read_text(encoding="utf-8")

    roots = [
        url for _number, url, rest, _text in _upstream_references(text) if rest == ""
    ]

    assert roots, "README.md no longer links the upstream repository it is a fork of."

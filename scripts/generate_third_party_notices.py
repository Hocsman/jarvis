#!/usr/bin/env python3
"""Write THIRD_PARTY_NOTICES.txt: the packages the build bundles and what each declares.

Everything is read offline from the package metadata installed in the
interpreter that runs this script, so run it with the environment the build
uses:

    python scripts/generate_third_party_notices.py

Which requirements the build bundles is derived from the repository, not
listed by hand. A requirement in requirements.txt is bundled when the source
(or the PyInstaller spec's ``hiddenimports``) imports it and the spec does not
exclude it, unless the installer fetches it on request instead
(installer/windows/install_cuda.ps1). The dependencies each of those declares
are listed too, resolved from the installed metadata.

The file records the platform and Python version it was generated on, because
the installed packages, and so the inventory, can differ between them. It
records what the metadata declares, and states nothing about how the licences
relate to each other or to the Jarvis LICENSE.
"""

from __future__ import annotations

import argparse
import ast
import importlib.metadata as metadata
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent

NOTICES_NAME = "THIRD_PARTY_NOTICES.txt"
ALL_COMPONENTS_HEADING = "Full list of components"
COPYLEFT_HEADING = "GPL and LGPL components"
ENVIRONMENT_LABEL = "Generated on"

NOT_DECLARED = "not declared in the package metadata"
NOT_INSTALLED = "not read: the package is not installed where this file was generated"

# The file the project's top-level requirements are read from, and the label a
# package listed in it carries under "Required by".
REQUIREMENTS_FILE = "requirements.txt"

# Present in every frozen build whatever the requirements say: PyInstaller's
# bootloader is the executable the user launches. Its own dependencies are
# build tooling and stay out.
RUNTIME_COMPONENTS = ("pyinstaller",)


def normalise(name: str) -> str:
    """The name pip compares: case-folded, runs of ``-``, ``_`` and ``.`` as one ``-``."""
    return re.sub(r"[-_.]+", "-", name).lower()


# ── requirements.txt ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Requirement:
    name: str
    marker: Optional[str] = None


_NAME = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)")


def parse_requirements(text: str) -> List[Requirement]:
    """Top-level requirements, one per package even when a marker splits it over lines."""
    found: Dict[str, Requirement] = {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        spec, _, marker = line.partition(";")
        match = _NAME.match(spec.strip())
        if not match:
            continue
        name = match.group(1)
        found.setdefault(normalise(name), Requirement(name, marker.strip() or None))
    return list(found.values())


# ── which requirements the build bundles ────────────────────────────────────

def _root(module: str) -> str:
    return module.split(".")[0]


def _spec_facts(spec_path: Path) -> Tuple[Set[str], Set[str]]:
    """``(hiddenimports, excludes)`` as import roots, read from the spec without running it."""
    tree = ast.parse(spec_path.read_text(encoding="utf-8"))
    hidden: Set[str] = set()
    excludes: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "hiddenimports" for t in node.targets
        ):
            hidden |= {_root(s) for s in ast.literal_eval(node.value)}
        elif isinstance(node, ast.Call) and getattr(node.func, "id", None) == "Analysis":
            for keyword in node.keywords:
                if keyword.arg == "excludes":
                    excludes |= {_root(s) for s in ast.literal_eval(keyword.value)}
    return hidden, excludes


def spec_excludes(project_root: Path) -> Set[str]:
    return _spec_facts(project_root / "jarvis_desktop.spec")[1]


def _source_imports(src_root: Path) -> Set[str]:
    """Import roots used anywhere under ``src_root``, inside functions and try blocks too."""
    roots: Set[str] = set()
    for path in src_root.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots |= {_root(alias.name) for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                roots.add(_root(node.module))
    return roots


def _installer_fetched(project_root: Path) -> Set[str]:
    """Packages the installer downloads on request instead of bundling."""
    script = project_root / "installer" / "windows" / "install_cuda.ps1"
    if not script.is_file():
        return set()
    text = script.read_text(encoding="utf-8", errors="replace")
    return {normalise(name) for name in re.findall(r'\bName\s*=\s*"([^"]+)"', text)}


def installed_import_names(dist_name: str) -> Set[str]:
    """Top-level names a package installs, falling back to its own name when it is absent."""
    fallback = {normalise(dist_name).replace("-", "_")}
    try:
        dist = metadata.distribution(dist_name)
    except metadata.PackageNotFoundError:
        return fallback
    names: Set[str] = set()
    top_level_file = next((f for f in dist.files or [] if f.name == "top_level.txt"), None)
    top_level = top_level_file.read_text(encoding="utf-8") if top_level_file else ""
    if top_level:
        names |= {line.strip() for line in top_level.splitlines() if line.strip()}
    else:
        for file in dist.files or []:
            parts = PurePosixPath(str(file).replace("\\", "/")).parts
            if not parts or parts[0] in ("..", "__pycache__") or parts[0].endswith((".dist-info", ".egg-info")):
                continue
            names.add(parts[0].split(".")[0] if len(parts) == 1 else parts[0])
    return names or fallback


def bundled_requirements(
    project_root: Path,
    import_names: Callable[[str], Set[str]] = installed_import_names,
) -> List[Requirement]:
    """The top-level requirements the PyInstaller build carries."""
    reqs = parse_requirements((project_root / REQUIREMENTS_FILE).read_text(encoding="utf-8"))
    hidden, excludes = _spec_facts(project_root / "jarvis_desktop.spec")
    imported = _source_imports(project_root / "src") | hidden
    fetched = _installer_fetched(project_root)

    bundled = []
    for req in reqs:
        if normalise(req.name) in fetched:
            continue
        names = import_names(req.name)
        if names & excludes or not names & imported:
            continue
        bundled.append(req)
    return sorted(bundled, key=lambda r: normalise(r.name))


# ── what a package declares ─────────────────────────────────────────────────

_FILE_POINTER = re.compile(
    r"^(?:[\w.\\-]*[/\\])?(?:licen[cs]e|copying|copyright|notice)(?:[.\-_]\w+)?$", re.IGNORECASE
)
_MAX_NAME = 200


def declared_licence(meta) -> str:
    """The licence name a package declares, or a plain statement that it declares none."""
    expression = (meta.get("License-Expression") or "").strip()
    if expression:
        return expression

    field_value = (meta.get("License") or "").strip()
    if (
        field_value
        and field_value.upper() != "UNKNOWN"
        and "\n" not in field_value
        and len(field_value) <= _MAX_NAME
        and not _FILE_POINTER.match(field_value)
    ):
        return field_value

    named = []
    for classifier in meta.get_all("Classifier") or []:
        if classifier.startswith("License ::"):
            label = classifier.split("::")[-1].strip()
            if label and label not in named:
                named.append(label)
    if named:
        return "; ".join(named)

    # A licence field that is a whole text starts with the licence's name.
    first_line = field_value.splitlines()[0].strip() if field_value else ""
    if first_line and len(first_line) <= 80 and not _FILE_POINTER.match(first_line):
        return first_line
    return NOT_DECLARED


_COPYLEFT_PATTERNS = (
    ("AGPL", re.compile(r"\bagpl\w*|affero general public license")),
    ("LGPL", re.compile(r"\blgpl\w*|(?:library or )?lesser general public license|library general public license")),
)
_GPL = re.compile(r"\bgpl|general public license")


def copyleft_marks(licence: str) -> Tuple[str, ...]:
    """Which of AGPL, GPL and LGPL a licence name mentions, sorted."""
    text = licence.lower()
    marks: Set[str] = set()
    for label, pattern in _COPYLEFT_PATTERNS:
        if pattern.search(text):
            marks.add(label)
            text = pattern.sub(" ", text)
    if _GPL.search(text):
        marks.add("GPL")
    return tuple(sorted(marks))


_HOME_KEYS = ("homepage", "home", "source", "repository", "code", "github")


def _home(meta) -> Optional[str]:
    home = (meta.get("Home-page") or "").strip()
    if home and home.upper() != "UNKNOWN":
        return home
    urls = []
    for item in meta.get_all("Project-URL") or []:
        label, _, url = item.partition(",")
        if url.strip():
            urls.append((label.strip().lower(), url.strip()))
    for key in _HOME_KEYS:
        for label, url in urls:
            if label == key:
                return url
    return urls[0][1] if urls else None


# ── entries and the closure ─────────────────────────────────────────────────

@dataclass(frozen=True)
class Entry:
    name: str
    version: str
    licence: str
    copyleft: Tuple[str, ...] = ()
    home: Optional[str] = None
    required_by: Tuple[str, ...] = ()
    installed: bool = True


def _entry(name: str, required_by: Iterable[str] = ()) -> Entry:
    required = tuple(sorted(required_by))
    try:
        dist = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return Entry(name=name, version="", licence=NOT_INSTALLED, required_by=required, installed=False)
    meta = dist.metadata
    licence = declared_licence(meta)
    return Entry(
        name=meta.get("Name") or name,
        version=dist.version or "",
        licence=licence,
        copyleft=copyleft_marks(licence),
        home=_home(meta),
        required_by=required,
    )


def _declared_requirements(name: str) -> List[str]:
    """Names of the packages ``name`` needs here: extras and other platforms' markers dropped."""
    from packaging.requirements import InvalidRequirement, Requirement as PipRequirement

    try:
        lines = metadata.requires(name) or []
    except metadata.PackageNotFoundError:
        return []
    needed = []
    for line in lines:
        try:
            req = PipRequirement(line)
        except InvalidRequirement:
            continue
        if req.marker is not None and not req.marker.evaluate({"extra": ""}):
            continue
        needed.append(req.name)
    return needed


def dependency_closure(roots: Iterable[str], excluded_imports: Set[str]) -> Dict[str, Entry]:
    """``roots`` and everything they need, keyed by normalised name.

    A package the spec excludes is left out with its own dependencies. A package
    that is not installed is kept as an entry that says so rather than guessed at.
    """
    asked_by: Dict[str, Set[str]] = {}
    display: Dict[str, str] = {}
    queue: List[str] = []
    for root in roots:
        key = normalise(root)
        if key not in display:
            display[key] = root
            asked_by[key] = set()
            queue.append(root)

    while queue:
        current = queue.pop(0)
        for needed in _declared_requirements(current):
            key = normalise(needed)
            if installed_import_names(needed) & excluded_imports:
                continue
            asked_by.setdefault(key, set()).add(normalise(current))
            if key not in display:
                display[key] = needed
                queue.append(needed)

    return {key: _entry(display[key], asked_by[key]) for key in display}


def collect_entries(
    project_root: Path,
    runtime_components: Iterable[str] = RUNTIME_COMPONENTS,
) -> List[Entry]:
    """Every entry the notices carry, in alphabetical order."""
    roots = [req.name for req in bundled_requirements(project_root)]
    found = dependency_closure(roots, spec_excludes(project_root))
    for root in roots:
        entry = found[normalise(root)]
        found[normalise(root)] = replace(entry, required_by=(REQUIREMENTS_FILE, *entry.required_by))
    for name in runtime_components:
        found.setdefault(normalise(name), _entry(name))
    return sorted(found.values(), key=lambda e: normalise(e.name))


# ── the file ────────────────────────────────────────────────────────────────

_INTRO = """\
THIRD-PARTY NOTICES
===================

This file is an inventory of the third-party Python packages that the Jarvis
build bundles, and of the packages those declare as dependencies. For each
one it records what the package's own metadata declares: its version, its
licence and where it is published. Licence texts are not reproduced here.

It is generated by scripts/generate_third_party_notices.py from the packages
installed in one environment, and it describes that environment only:

  {label}: {environment}

Builds on other platforms or Python versions can carry other versions of these
packages, and a package installed for one platform only is absent from the
inventory of another or reads as not installed in it. The file makes no
statement about how the licences listed relate to each other or to the Jarvis
LICENSE.

Native libraries shipped inside these packages (Qt inside PyQt6, espeak-ng
inside piper-tts, for example) carry licences of their own that this file does
not itemise. The NVIDIA CUDA libraries the Windows installer can download on
request are not part of the build and are not listed.
"""

_ENVIRONMENT_LINE = re.compile(rf"^\s*{re.escape(ENVIRONMENT_LABEL)}: (.+?)\s*$", re.MULTILINE)


def current_environment() -> str:
    """The platform and Python version this interpreter runs, as the file records them."""
    return f"{sys.platform}, Python {sys.version_info.major}.{sys.version_info.minor}"


def parse_environment(text: str) -> Optional[str]:
    """The environment a notices file says it was generated in, or None."""
    match = _ENVIRONMENT_LINE.search(text)
    return match.group(1) if match else None


def _line(entry: Entry) -> str:
    return f"{entry.name} {entry.version or '-'}"


def render(entries: Iterable[Entry], environment: str) -> str:
    """The text of the notices file. The order of ``entries`` does not matter.

    ``environment`` names the platform and Python the entries were read in.
    """
    ordered = sorted(entries, key=lambda e: normalise(e.name))
    out: List[str] = [_INTRO.format(label=ENVIRONMENT_LABEL, environment=environment)]

    marked = [e for e in ordered if e.copyleft]
    out.append(COPYLEFT_HEADING)
    out.append("-" * len(COPYLEFT_HEADING))
    out.append("These packages declare a GPL, LGPL or AGPL licence:")
    out.append("")
    if marked:
        for entry in marked:
            out.append(f"  [{', '.join(entry.copyleft)}] {_line(entry)}: {entry.licence}")
    else:
        out.append("  (none)")
    out.append("")

    out.append(ALL_COMPONENTS_HEADING)
    out.append("-" * len(ALL_COMPONENTS_HEADING))
    out.append("")
    for entry in ordered:
        out.append(_line(entry))
        out.append(f"    Licence: {entry.licence}")
        if entry.copyleft:
            out.append(f"    Copyleft: {', '.join(entry.copyleft)}")
        if entry.home:
            out.append(f"    Home: {entry.home}")
        if entry.required_by:
            out.append(f"    Required by: {', '.join(entry.required_by)}")
        out.append("")
    return "\n".join(out).rstrip("\n") + "\n"


_HEADER = re.compile(r"^(\S+) (\S+)$")


def parse_notices(text: str) -> Dict[str, Entry]:
    """The entries of a notices file, keyed by normalised name."""
    _, found, body = text.partition(ALL_COMPONENTS_HEADING)
    entries: Dict[str, Entry] = {}
    fields: Dict[str, object] = {}

    def close() -> None:
        if fields:
            key = normalise(str(fields["name"]))
            entries[key] = Entry(
                name=str(fields["name"]),
                version=str(fields.get("version", "")),
                licence=str(fields.get("licence", "")),
                copyleft=tuple(fields.get("copyleft", ())),
                home=fields.get("home"),
                required_by=tuple(fields.get("required_by", ())),
                installed=fields.get("licence") != NOT_INSTALLED,
            )
        fields.clear()

    if not found:
        return entries
    for raw in body.splitlines():
        if not raw.strip() or set(raw.strip()) == {"-"}:
            continue
        if not raw.startswith(" "):
            match = _HEADER.match(raw)
            if match:
                close()
                fields["name"] = match.group(1)
                fields["version"] = "" if match.group(2) == "-" else match.group(2)
            continue
        label, _, value = raw.strip().partition(": ")
        if label == "Licence":
            fields["licence"] = value
        elif label == "Copyleft":
            fields["copyleft"] = tuple(sorted(v.strip() for v in value.split(",")))
        elif label == "Home":
            fields["home"] = value
        elif label == "Required by":
            fields["required_by"] = tuple(v.strip() for v in value.split(","))
    close()
    return entries


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Write THIRD_PARTY_NOTICES.txt from the installed package metadata.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--output", default=None, help=f"where to write (default: <project>/{NOTICES_NAME})")
    args = parser.parse_args(argv)

    project_root = Path(args.project_root)
    output = Path(args.output) if args.output else project_root / NOTICES_NAME

    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass

    environment = current_environment()
    print(f"\U0001F50E Reading licences from the packages installed for {sys.executable}")
    print(f"    \U0001F4BB Recorded as generated on: {environment}")
    entries = collect_entries(project_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(entries, environment), encoding="utf-8", newline="\n")

    copyleft = [e for e in entries if e.copyleft]
    unread = [e for e in entries if not e.installed]
    undeclared = [e for e in entries if e.installed and e.licence == NOT_DECLARED]
    print(f"    \U0001F4DD {len(entries)} components written to {output}")
    print(f"    ⚠️  {len(copyleft)} declare a GPL, LGPL or AGPL licence")
    if unread:
        print(f"    ❓ {len(unread)} not installed here, so not read: {', '.join(e.name for e in unread)}")
    if undeclared:
        print(f"    ❓ {len(undeclared)} declare no licence: {', '.join(e.name for e in undeclared)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

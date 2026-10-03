"""The third-party notices list what the build ships, and mark the copyleft parts.

``scripts/generate_third_party_notices.py`` reads each package's licence from
the metadata installed where the build runs, offline, and writes
``THIRD_PARTY_NOTICES.txt``. These tests pin the mechanisms with small fake
projects and fake installed packages, then hold the committed file to the
rules: every requirement the build bundles has an entry, and every entry
whose licence is GPL or LGPL says so.
"""

from __future__ import annotations

import email
import importlib
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "generate_third_party_notices", ROOT / "scripts" / "generate_third_party_notices.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gen = _load_generator()


# ── fixtures: fake installed packages and a fake project ────────────────────

def _install(tmp_path: Path, name: str, version: str = "1.0", *, requires=(),
             licence=None, expression=None, classifiers=(), top_level=None,
             home=None, urls=()):
    """Put a ``.dist-info`` for ``name`` on a path the metadata machinery scans."""
    folder = tmp_path / "site" / f"{name.replace('-', '_')}-{version}.dist-info"
    folder.mkdir(parents=True, exist_ok=True)
    lines = ["Metadata-Version: 2.4", f"Name: {name}", f"Version: {version}"]
    if licence is not None:
        lines.append(f"License: {licence}")
    if expression is not None:
        lines.append(f"License-Expression: {expression}")
    if home is not None:
        lines.append(f"Home-page: {home}")
    lines += [f"Project-URL: {u}" for u in urls]
    lines += [f"Classifier: {c}" for c in classifiers]
    lines += [f"Requires-Dist: {r}" for r in requires]
    (folder / "METADATA").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (folder / "top_level.txt").write_text(
        (top_level or name.replace("-", "_")) + "\n", encoding="utf-8"
    )
    # pip always writes a RECORD, and the file list is how the metadata API finds the rest.
    (folder / "RECORD").write_text(
        f"{folder.name}/METADATA,,\n{folder.name}/top_level.txt,,\n", encoding="utf-8"
    )
    return folder


@pytest.fixture
def site(tmp_path, monkeypatch):
    (tmp_path / "site").mkdir()
    monkeypatch.syspath_prepend(str(tmp_path / "site"))
    importlib.invalidate_caches()
    return tmp_path


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _project(tmp_path: Path, *, requirements: str, spec: str, sources: dict, installer: str = "") -> Path:
    root = tmp_path / "project"
    _write(root / "requirements.txt", requirements)
    _write(root / "jarvis_desktop.spec", spec)
    _write(root / "installer" / "windows" / "install_cuda.ps1", installer)
    for name, text in sources.items():
        _write(root / "src" / name, text)
    return root


SPEC = '''
hiddenimports = [
    'jarvis.main',
    'hiddenpkg',
    'hiddenpkg.sub',
]
a = Analysis(
    ['src/app.py'],
    hiddenimports=hiddenimports,
    excludes=[
        'leftout',  # heavy
        'other_leftout',
    ],
)
'''


# ── which requirements the build bundles ────────────────────────────────────

class TestParseRequirements:
    def test_names_markers_and_noise(self):
        text = """
# a comment
alpha==1.2   # trailing comment
beta>=2; sys_platform == "win32"
Gamma_Pkg[extra]>=1
-r other.txt
--index-url https://example.invalid

"""
        found = gen.parse_requirements(text)
        assert [r.name for r in found] == ["alpha", "beta", "Gamma_Pkg"]
        assert found[1].marker == 'sys_platform == "win32"'
        assert found[0].marker is None

    def test_the_same_package_on_two_lines_is_one_requirement(self):
        text = 'Qt>=6; sys_platform != "win32"\nQt==6.9.1; sys_platform == "win32"\n'
        assert [r.name for r in gen.parse_requirements(text)] == ["Qt"]


class TestBundledRequirements:
    def _names(self, root, mapping=None):
        mapping = mapping or {}
        found = gen.bundled_requirements(
            root, import_names=lambda dist: mapping.get(dist, {dist.replace("-", "_").lower()})
        )
        return {gen.normalise(r.name) for r in found}

    def test_a_package_the_source_imports_is_bundled_wherever_the_import_sits(self, tmp_path):
        root = _project(
            tmp_path,
            requirements="alpha\nbeta\ngamma\n",
            spec=SPEC,
            sources={
                "app.py": "import alpha\nfrom beta.sub import thing\n",
                "deep/inner.py": "def f():\n    import gamma\n",
            },
        )
        assert self._names(root) == {"alpha", "beta", "gamma"}

    def test_a_package_nothing_imports_is_not_bundled(self, tmp_path):
        root = _project(
            tmp_path,
            requirements="alpha\ntest-runner\n",
            spec=SPEC,
            sources={"app.py": "import alpha\n"},
        )
        assert self._names(root) == {"alpha"}

    def test_a_hidden_import_in_the_spec_counts_as_bundled(self, tmp_path):
        root = _project(tmp_path, requirements="hiddenpkg\n", spec=SPEC, sources={"app.py": "pass\n"})
        assert self._names(root) == {"hiddenpkg"}

    def test_the_specs_own_excludes_win_over_an_import(self, tmp_path):
        root = _project(
            tmp_path,
            requirements="leftout\nalpha\n",
            spec=SPEC,
            sources={"app.py": "import leftout\nimport alpha\n"},
        )
        assert self._names(root) == {"alpha"}

    def test_the_import_name_comes_from_the_installed_metadata_not_the_dist_name(self, tmp_path):
        root = _project(
            tmp_path,
            requirements="beautiful-thing\n",
            spec=SPEC,
            sources={"app.py": "import bt\n"},
        )
        assert self._names(root, {"beautiful-thing": {"bt"}}) == {"beautiful-thing"}

    def test_every_dist_sharing_an_imported_namespace_is_bundled(self, tmp_path):
        root = _project(
            tmp_path,
            requirements="qt-bindings\nqt-runtime\n",
            spec=SPEC,
            sources={"app.py": "import qtns\n"},
        )
        mapping = {"qt-bindings": {"qtns"}, "qt-runtime": {"qtns"}}
        assert self._names(root, mapping) == {"qt-bindings", "qt-runtime"}

    def test_a_package_the_installer_fetches_on_request_is_not_bundled(self, tmp_path):
        root = _project(
            tmp_path,
            requirements="gpu-libs\nalpha\n",
            spec=SPEC,
            sources={"app.py": "import alpha\ntry:\n    import gpu_libs\nexcept ImportError:\n    pass\n"},
            installer='$Packages = @(\n    @{\n        Name = "gpu-libs"\n        Wheel = "x.whl"\n    }\n)\n',
        )
        assert self._names(root) == {"alpha"}

    def test_names_are_compared_the_way_pip_compares_them(self, tmp_path):
        root = _project(
            tmp_path,
            requirements="Some_Pkg.Name\n",
            spec=SPEC,
            sources={"app.py": "import some_pkg_name\n"},
            installer="",
        )
        assert self._names(root, {"Some_Pkg.Name": {"some_pkg_name"}}) == {"some-pkg-name"}


class TestImportNames:
    def test_an_installed_package_reports_its_top_level_names(self, site):
        _install(site, "zz-pillowish", top_level="zzpil\nzzpil_extra")
        assert gen.installed_import_names("zz-pillowish") == {"zzpil", "zzpil_extra"}

    def test_a_package_that_is_not_installed_falls_back_to_its_own_name(self, site):
        assert gen.installed_import_names("Zz-Not-Here") == {"zz_not_here"}


# ── what a package declares ─────────────────────────────────────────────────

def _metadata(**fields):
    message = email.message.Message()
    for key, value in fields.items():
        for item in value if isinstance(value, list) else [value]:
            message[key.replace("_", "-")] = item
    return message


class TestDeclaredLicence:
    def test_an_spdx_expression_comes_first(self):
        meta = _metadata(License_Expression="GPL-3.0-or-later", License="Something else")
        assert gen.declared_licence(meta) == "GPL-3.0-or-later"

    def test_a_short_licence_field_is_used(self):
        assert gen.declared_licence(_metadata(License="Apache-2.0")) == "Apache-2.0"

    def test_a_licence_field_holding_the_whole_text_is_not_a_name(self):
        meta = _metadata(
            License="MIT License\n\nCopyright (c) 2025 Someone\n\nPermission is hereby granted...",
            Classifier=["License :: OSI Approved :: MIT License"],
        )
        assert gen.declared_licence(meta) == "MIT License"

    @pytest.mark.parametrize("pointer", ["../LICENSE", "LICENSE", "LICENSE.txt", "UNKNOWN", ""])
    def test_a_licence_field_that_only_points_at_a_file_is_not_a_name(self, pointer):
        meta = _metadata(
            License=pointer,
            Classifier=["License :: OSI Approved :: BSD License"],
        )
        assert gen.declared_licence(meta) == "BSD License"

    def test_several_classifiers_are_all_kept(self):
        meta = _metadata(
            Classifier=[
                "License :: OSI Approved :: MIT License",
                "License :: OSI Approved :: Apache Software License",
                "Programming Language :: Python :: 3",
            ]
        )
        assert gen.declared_licence(meta) == "MIT License; Apache Software License"

    def test_nothing_declared_is_said_plainly(self):
        assert gen.declared_licence(_metadata()) == gen.NOT_DECLARED


class TestCopyleftMarks:
    @pytest.mark.parametrize(
        "licence, expected",
        [
            ("GPL-3.0-or-later", ("GPL",)),
            ("GPL-3.0-only", ("GPL",)),
            ("GPL v3", ("GPL",)),
            ("GPLv2-or-later with a special exception which allows to use it to build programs", ("GPL",)),
            ("GNU General Public License v2 (GPLv2)", ("GPL",)),
            ("LGPLv3", ("LGPL",)),
            ("LGPL v3", ("LGPL",)),
            ("LGPL-3.0-or-later", ("LGPL",)),
            ("GNU Lesser General Public License v3 (LGPLv3)", ("LGPL",)),
            ("GNU Library or Lesser General Public License (LGPL)", ("LGPL",)),
            ("LGPL-3.0-or-later OR GPL-2.0-only", ("GPL", "LGPL")),
            ("AGPL-3.0-only", ("AGPL",)),
            ("GNU Affero General Public License v3", ("AGPL",)),
            ("MIT", ()),
            ("MIT License", ()),
            ("Apache-2.0", ()),
            ("BSD-3-Clause", ()),
            ("LicenseRef-NVIDIA-Proprietary", ()),
            (gen.NOT_DECLARED, ()),
        ],
    )
    def test_marks(self, licence, expected):
        assert gen.copyleft_marks(licence) == expected


# ── the dependency closure ──────────────────────────────────────────────────

class TestDependencyClosure:
    def test_dependencies_are_followed_and_say_who_asked_for_them(self, site):
        _install(site, "zz-top", requires=["zz-mid"], licence="MIT")
        _install(site, "zz-mid", requires=["zz-leaf>=1"], licence="MIT")
        _install(site, "zz-leaf", licence="MIT")
        found = gen.dependency_closure(["zz-top"], excluded_imports=set())
        assert {gen.normalise(n) for n in found} == {"zz-top", "zz-mid", "zz-leaf"}
        assert found["zz-leaf"].required_by == ("zz-mid",)
        assert found["zz-top"].required_by == ()

    def test_extras_and_markers_that_do_not_apply_are_not_followed(self, site):
        _install(
            site, "zz-top",
            requires=[
                "zz-extra; extra == 'fancy'",
                "zz-elsewhere; sys_platform == 'no-such-platform'",
                "zz-needed",
            ],
        )
        for name in ("zz-extra", "zz-elsewhere", "zz-needed"):
            _install(site, name)
        found = gen.dependency_closure(["zz-top"], excluded_imports=set())
        assert {gen.normalise(n) for n in found} == {"zz-top", "zz-needed"}

    def test_a_package_the_spec_excludes_is_not_followed(self, site):
        _install(site, "zz-top", requires=["zz-heavy"])
        _install(site, "zz-heavy", requires=["zz-deeper"], top_level="zzheavy")
        _install(site, "zz-deeper")
        found = gen.dependency_closure(["zz-top"], excluded_imports={"zzheavy"})
        assert {gen.normalise(n) for n in found} == {"zz-top"}

    def test_a_dependency_that_is_not_installed_is_named_not_invented(self, site):
        _install(site, "zz-top", requires=["zz-absent"])
        found = gen.dependency_closure(["zz-top"], excluded_imports=set())
        assert found["zz-absent"].installed is False
        assert found["zz-absent"].licence == gen.NOT_INSTALLED

    def test_a_cycle_terminates(self, site):
        _install(site, "zz-a", requires=["zz-b"])
        _install(site, "zz-b", requires=["zz-a"])
        found = gen.dependency_closure(["zz-a"], excluded_imports=set())
        assert {gen.normalise(n) for n in found} == {"zz-a", "zz-b"}

    def test_the_licence_and_home_are_read_from_the_metadata(self, site):
        _install(site, "zz-top", "2.5", expression="LGPL-3.0-or-later", home="https://example.invalid/zz")
        entry = gen.dependency_closure(["zz-top"], excluded_imports=set())["zz-top"]
        assert (entry.version, entry.licence, entry.home) == (
            "2.5", "LGPL-3.0-or-later", "https://example.invalid/zz",
        )
        assert entry.copyleft == ("LGPL",)

    def test_a_project_url_stands_in_for_a_missing_home_page(self, site):
        _install(site, "zz-top", urls=["Documentation, https://docs.invalid/", "Source, https://src.invalid/zz"])
        assert gen.dependency_closure(["zz-top"], excluded_imports=set())["zz-top"].home == "https://src.invalid/zz"


# ── the file ────────────────────────────────────────────────────────────────

def _entry(name, version="1.0", licence="MIT", required_by=(), home=None, installed=True):
    return gen.Entry(
        name=name, version=version, licence=licence,
        copyleft=gen.copyleft_marks(licence), home=home,
        required_by=tuple(required_by), installed=installed,
    )


class TestRenderedNotices:
    def _entries(self):
        return [
            _entry("beta", licence="MIT"),
            _entry("alpha", licence="GPL-3.0-or-later", home="https://example.invalid/alpha"),
            _entry("delta", licence="LGPLv3", required_by=["alpha"]),
        ]

    def test_every_entry_survives_a_round_trip(self):
        parsed = gen.parse_notices(gen.render(self._entries()))
        assert set(parsed) == {"alpha", "beta", "delta"}
        assert (parsed["alpha"].version, parsed["alpha"].licence) == ("1.0", "GPL-3.0-or-later")
        assert parsed["alpha"].home == "https://example.invalid/alpha"

    def test_copyleft_entries_are_marked_in_their_own_entry(self):
        parsed = gen.parse_notices(gen.render(self._entries()))
        assert parsed["alpha"].copyleft == ("GPL",)
        assert parsed["delta"].copyleft == ("LGPL",)
        assert parsed["beta"].copyleft == ()

    def test_copyleft_entries_are_gathered_up_front_too(self):
        text = gen.render(self._entries())
        head = text.split(gen.ALL_COMPONENTS_HEADING)[0]
        assert "alpha" in head and "delta" in head
        assert "beta" not in head

    def test_the_output_does_not_depend_on_the_order_of_the_input(self):
        entries = self._entries()
        assert gen.render(entries) == gen.render(list(reversed(entries)))

    def test_nothing_in_it_changes_from_one_run_to_the_next(self):
        assert gen.render(self._entries()) == gen.render(self._entries())

    def test_the_text_states_no_licensing_conclusion(self):
        head = gen.render(self._entries()).split(gen.ALL_COMPONENTS_HEADING)[0].lower()
        for claim in ("compatible", "incompatible", "permitted", "you may", "we grant", "legal advice"):
            assert claim not in head

    def test_an_absent_package_is_listed_as_not_read(self):
        text = gen.render([_entry("ghost", licence=gen.NOT_INSTALLED, installed=False)])
        assert gen.NOT_INSTALLED in text


class TestGenerate:
    def test_the_whole_pipeline_on_a_small_project(self, site):
        _install(site, "zzalpha", "3.1", requires=["zzdep"], expression="GPL-3.0-or-later")
        _install(site, "zzdep", "0.4", licence="MIT")
        root = _project(
            site,
            requirements="zzalpha\nzzunused\n",
            spec=SPEC,
            sources={"app.py": "import zzalpha\n"},
        )
        entries = gen.collect_entries(root, runtime_components=())
        names = {gen.normalise(e.name) for e in entries}
        assert names == {"zzalpha", "zzdep"}
        assert {e.name: e.copyleft for e in entries}["zzalpha"] == ("GPL",)

    def test_an_entry_says_whether_a_requirement_or_a_dependency_brought_it(self, site):
        _install(site, "zzalpha", requires=["zzdep"], licence="MIT")
        _install(site, "zzdep", requires=["zzalpha"], licence="MIT")
        root = _project(
            site,
            requirements="zzalpha\nzzdep\n",
            spec=SPEC,
            sources={"app.py": "import zzalpha, zzdep\n"},
        )
        by_name = {e.name: e for e in gen.collect_entries(root, runtime_components=())}
        assert by_name["zzalpha"].required_by == (gen.REQUIREMENTS_FILE, "zzdep")
        assert by_name["zzdep"].required_by == (gen.REQUIREMENTS_FILE, "zzalpha")

    def test_main_writes_the_file_with_unix_newlines(self, site, capsys):
        _install(site, "zzalpha", licence="MIT")
        root = _project(
            site, requirements="zzalpha\n", spec=SPEC, sources={"app.py": "import zzalpha\n"}
        )
        out = site / "out" / gen.NOTICES_NAME
        assert gen.main(["--project-root", str(root), "--output", str(out)]) == 0
        data = out.read_bytes()
        assert b"\r" not in data
        assert "zzalpha" in gen.parse_notices(data.decode("utf-8"))
        first = capsys.readouterr().out.splitlines()[0]
        assert not first.isascii(), "user-facing output leads with an emoji"


# ── the committed file ──────────────────────────────────────────────────────

class TestCommittedNotices:
    @pytest.fixture(scope="class")
    def parsed(self):
        path = ROOT / gen.NOTICES_NAME
        assert path.is_file(), f"{gen.NOTICES_NAME} has not been generated"
        return gen.parse_notices(path.read_text(encoding="utf-8"))

    def test_every_requirement_the_build_bundles_has_an_entry(self, parsed):
        missing = [
            req.name
            for req in gen.bundled_requirements(ROOT)
            if gen.normalise(req.name) not in parsed
        ]
        assert not missing, (
            f"{gen.NOTICES_NAME} lacks {missing}: run scripts/generate_third_party_notices.py"
        )

    def test_every_gpl_or_lgpl_licence_is_marked_explicitly(self, parsed):
        unmarked = [
            f"{entry.name}: {entry.licence}"
            for entry in parsed.values()
            if gen.copyleft_marks(entry.licence) != entry.copyleft
        ]
        assert not unmarked

    def test_the_marked_entries_are_gathered_in_the_opening_section(self):
        text = (ROOT / gen.NOTICES_NAME).read_text(encoding="utf-8")
        head, _, _ = text.partition(gen.ALL_COMPONENTS_HEADING)
        parsed = gen.parse_notices(text)
        for key, entry in parsed.items():
            if entry.copyleft:
                assert entry.name in head, f"{entry.name} is marked but missing from the opening section"

    def test_the_text_states_no_licensing_conclusion(self):
        head = (ROOT / gen.NOTICES_NAME).read_text(encoding="utf-8").split(gen.ALL_COMPONENTS_HEADING)[0].lower()
        for claim in ("compatible", "incompatible", "you may", "we grant", "legal advice"):
            assert claim not in head

    def test_the_file_is_plain_text_a_notepad_can_open(self):
        data = (ROOT / gen.NOTICES_NAME).read_bytes()
        data.decode("utf-8")
        assert b"\x00" not in data

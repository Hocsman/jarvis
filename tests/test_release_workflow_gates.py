"""Nothing ships unless the tests pass and the packaged bundle has been checked.

The release workflow cannot be run from here, so these tests read it the way
GitHub does: they parse ``release.yml`` and ``tests.yml``, evaluate each job's
``if`` with the same rules (a job is skipped when anything it needs did not
succeed, unless its condition uses a status function such as ``always()``),
and play out a push to ``main`` and to ``develop`` under each failure. That is
where the gate breaks in practice: a job that lists the suite in ``needs`` but
writes ``if: always() && ...`` still runs when the suite is red.

What it cannot show is what GitHub's own services do with the result; the
claims here are about the graph and the order of steps, which is all the
repository controls.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = ROOT / ".github" / "workflows"
BRANCH_REFS = ("refs/heads/main", "refs/heads/develop")

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Reading a workflow
# ---------------------------------------------------------------------------

def _load(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


def _triggers(workflow: dict) -> dict:
    # YAML 1.1 reads the bare key `on` as the boolean True.
    return workflow.get("on", workflow.get(True)) or {}


def _needs(job: dict) -> list[str]:
    needs = job.get("needs") or []
    return [needs] if isinstance(needs, str) else list(needs)


def _steps(job: dict) -> list[dict]:
    return job.get("steps") or []


def _run_text(step: dict) -> str:
    return step.get("run") or ""


def _uses(step: dict) -> str:
    return step.get("uses") or ""


def _is_build(job: dict) -> bool:
    return any("pyinstaller" in _run_text(step).lower() for step in _steps(job))


def _publishes(job: dict) -> bool:
    return any(
        _uses(step).startswith("softprops/action-gh-release") or "semantic-release" in _run_text(step)
        for step in _steps(job)
    )


def _gate_job_name(jobs: dict) -> str:
    """The job that runs the suite by calling ``tests.yml``."""
    gates = [name for name, job in jobs.items() if _uses(job) == "./.github/workflows/tests.yml"]
    assert len(gates) == 1, f"expected exactly one job calling tests.yml, found {gates}"
    return gates[0]


def _transitive_needs(jobs: dict, name: str) -> set[str]:
    seen: set[str] = set()
    pending = list(_needs(jobs[name]))
    while pending:
        current = pending.pop()
        if current not in seen:
            seen.add(current)
            pending.extend(_needs(jobs[current]))
    return seen


# ---------------------------------------------------------------------------
# Evaluating a job's `if`, for the subset of the expression language we use
# ---------------------------------------------------------------------------

_TOKEN = re.compile(
    r"""\s*(?:
        (?P<string>'(?:[^']|'')*')
      | (?P<op>&&|\|\||==|!=|!|\(|\))
      | (?P<call>[A-Za-z_]+)\s*\(\s*\)
      | (?P<path>[A-Za-z_][\w\-]*(?:\.[\w\-]+)*)
    )""",
    re.VERBOSE,
)
_STATUS_FUNCTIONS = ("success", "always", "failure", "cancelled")


def _tokens(expression: str) -> list[tuple[str, str]]:
    out, position = [], 0
    expression = expression.strip()
    while position < len(expression):
        match = _TOKEN.match(expression, position)
        if not match or match.end() == position:
            raise NotImplementedError(f"cannot read {expression[position:]!r} in {expression!r}")
        kind = match.lastgroup
        out.append((kind, match.group(kind)))
        position = match.end()
    return out


class _Expression:
    """Recursive descent over ``||``, ``&&``, ``!``, ``==``/``!=``, calls and context paths."""

    def __init__(self, text: str, context: dict):
        text = text.strip()
        if text.startswith("${{") and text.endswith("}}"):
            text = text[3:-2]
        self.tokens = _tokens(text)
        self.at = 0
        self.context = context

    def _peek(self):
        return self.tokens[self.at] if self.at < len(self.tokens) else (None, None)

    def _take(self):
        token = self.tokens[self.at]
        self.at += 1
        return token

    def evaluate(self):
        value = self._or()
        if self.at != len(self.tokens):
            raise NotImplementedError(f"trailing tokens {self.tokens[self.at:]}")
        return value

    def _or(self):
        value = self._and()
        while self._peek() == ("op", "||"):
            self._take()
            right = self._and()
            value = value or right
        return value

    def _and(self):
        value = self._comparison()
        while self._peek() == ("op", "&&"):
            self._take()
            right = self._comparison()
            value = value and right
        return value

    def _comparison(self):
        left = self._unary()
        kind, text = self._peek()
        if kind == "op" and text in ("==", "!="):
            self._take()
            right = self._unary()
            equal = str(left).lower() == str(right).lower()
            return equal if text == "==" else not equal
        return left

    def _unary(self):
        if self._peek() == ("op", "!"):
            self._take()
            return not self._unary()
        return self._primary()

    def _primary(self):
        kind, text = self._take()
        if kind == "op" and text == "(":
            value = self._or()
            if self._take() != ("op", ")"):
                raise NotImplementedError("unbalanced parenthesis")
            return value
        if kind == "string":
            return text[1:-1].replace("''", "'")
        if kind == "call":
            if text not in _STATUS_FUNCTIONS:
                raise NotImplementedError(f"function {text}() is not modelled")
            return self.context["status"][text]
        if kind == "path":
            if text in ("true", "false"):
                return text == "true"
            return self.context["lookup"](text)
        raise NotImplementedError(f"unexpected token {kind} {text!r}")


def _uses_status_function(expression: str) -> bool:
    return any(kind == "call" and text in _STATUS_FUNCTIONS for kind, text in _tokens(
        expression.strip().removeprefix("${{").removesuffix("}}")
    ))


def _simulate(workflow: dict, ref: str, failing=(), outputs=None) -> dict[str, str]:
    """Result of every job in a run of ``workflow`` pushed to ``ref``.

    A job that runs succeeds unless it is in ``failing``. ``outputs`` gives the
    outputs of a job that ran and succeeded, as ``{job: {name: value}}``.
    """
    jobs = workflow["jobs"]
    outputs = outputs or {}
    results: dict[str, str] = {}

    def settled(name):
        return all(need in results for need in _needs(jobs[name]))

    while len(results) < len(jobs):
        progressed = False
        for name, job in jobs.items():
            if name in results or not settled(name):
                continue
            progressed = True
            needs = {need: results[need] for need in _needs(job)}

            def lookup(path, needs=needs):
                parts = path.split(".")
                if path == "github.ref":
                    return ref
                if parts[0] == "needs" and len(parts) >= 3 and parts[1] in needs:
                    if parts[2] == "result":
                        return needs[parts[1]]
                    if parts[2] == "outputs" and len(parts) == 4:
                        ran = needs[parts[1]] == "success"
                        return (outputs.get(parts[1], {}) if ran else {}).get(parts[3], "")
                raise NotImplementedError(f"context {path!r} is not modelled")

            status = {
                "success": all(result == "success" for result in needs.values()),
                "always": True,
                "failure": any(result == "failure" for result in needs.values()),
                "cancelled": False,
            }
            condition = job.get("if")
            if condition is None:
                runs = status["success"]
            else:
                value = bool(_Expression(str(condition), {"status": status, "lookup": lookup}).evaluate())
                runs = value if _uses_status_function(str(condition)) else (status["success"] and value)
            results[name] = ("failure" if name in failing else "success") if runs else "skipped"
        assert progressed, "the job graph has a cycle or a dependency on a job that does not exist"
    return results


RELEASE_OUTPUT = {"semantic-release": {"new_release_published": "true"}}
NO_RELEASE_OUTPUT = {"semantic-release": {"new_release_published": "false"}}


@pytest.fixture(scope="module")
def release() -> dict:
    return _load("release.yml")


@pytest.fixture(scope="module")
def tests_workflow() -> dict:
    return _load("tests.yml")


def _others(release: dict) -> list[str]:
    jobs = release["jobs"]
    gate = _gate_job_name(jobs)
    return [name for name in jobs if name != gate]


# ---------------------------------------------------------------------------
# The harness itself must see the traps it exists for
# ---------------------------------------------------------------------------

class TestTheSimulationSeesWhatItShould:
    TRAPPED = {"jobs": {
        "tests": {"uses": "./.github/workflows/tests.yml"},
        "build": {"needs": ["tests"], "if": "always()"},
        "publish": {"needs": ["build"]},
    }}
    SOUND = {"jobs": {
        "tests": {"uses": "./.github/workflows/tests.yml"},
        "build": {"needs": ["tests"], "if": "always() && needs.tests.result == 'success'"},
        "publish": {"needs": ["build"]},
    }}

    def test_always_overrides_a_needs_edge(self):
        results = _simulate(self.TRAPPED, "refs/heads/develop", failing={"tests"})

        assert results == {"tests": "failure", "build": "success", "publish": "success"}

    def test_an_explicit_result_check_restores_the_gate(self):
        results = _simulate(self.SOUND, "refs/heads/develop", failing={"tests"})

        assert results == {"tests": "failure", "build": "skipped", "publish": "skipped"}

    def test_a_skipped_dependency_skips_its_dependents_without_a_status_function(self):
        workflow = {"jobs": {
            "tests": {"uses": "./.github/workflows/tests.yml"},
            "only-main": {"needs": ["tests"], "if": "github.ref == 'refs/heads/main'"},
            "after": {"needs": ["only-main"]},
        }}

        assert _simulate(workflow, "refs/heads/develop")["after"] == "skipped"
        assert _simulate(workflow, "refs/heads/main")["after"] == "success"

    def test_outputs_are_empty_when_the_job_did_not_succeed(self):
        workflow = {"jobs": {
            "a": {},
            "b": {"needs": ["a"], "if": "always() && needs.a.outputs.flag == 'true'"},
        }}

        assert _simulate(workflow, "refs/heads/main", outputs={"a": {"flag": "true"}})["b"] == "success"
        assert _simulate(workflow, "refs/heads/main", failing={"a"}, outputs={"a": {"flag": "true"}})["b"] == "skipped"


# ---------------------------------------------------------------------------
# tests.yml
# ---------------------------------------------------------------------------

class TestTheSuiteCanBeCalled:
    def test_tests_yml_is_callable_and_still_runs_on_its_own_triggers(self, tests_workflow):
        triggers = _triggers(tests_workflow)

        assert "workflow_call" in triggers
        assert "pull_request" in triggers
        assert "push" in triggers

    def test_the_release_workflow_calls_it_as_a_job(self, release):
        jobs = release["jobs"]
        gate = jobs[_gate_job_name(jobs)]

        assert not gate.get("steps"), "a job that calls a workflow has no steps of its own"
        assert not _needs(gate), "the gate must not wait on anything it gates"

    def test_the_suite_gives_the_release_nothing_it_has_to_supply(self, tests_workflow):
        call = _triggers(tests_workflow)["workflow_call"]

        assert not (call or {}).get("inputs"), "an input would have to be wired in every caller"
        assert not (call or {}).get("secrets"), "the suite must not need a secret"


class TestEvalsAreCollected:
    def test_every_suite_job_collects_the_evals_without_running_them(self, tests_workflow):
        for name, job in tests_workflow["jobs"].items():
            collecting = [
                step for step in _steps(job)
                if re.search(r"pytest\b.*\bevals\b.*--collect-only", _run_text(step))
            ]
            assert collecting, f"{name} never collects evals/, so an import error in one goes unseen"


# ---------------------------------------------------------------------------
# release.yml
# ---------------------------------------------------------------------------

class TestARedSuiteBlocksTheRelease:
    def test_every_other_job_transitively_needs_the_suite(self, release):
        jobs = release["jobs"]
        gate = _gate_job_name(jobs)

        for name in _others(release):
            assert gate in _transitive_needs(jobs, name), f"{name} can run without the suite having passed"

    @pytest.mark.parametrize("ref", BRANCH_REFS)
    @pytest.mark.parametrize("outputs", [RELEASE_OUTPUT, NO_RELEASE_OUTPUT], ids=["release", "no-release"])
    def test_when_the_suite_fails_no_other_job_runs(self, release, ref, outputs):
        gate = _gate_job_name(release["jobs"])

        results = _simulate(release, ref, failing={gate}, outputs=outputs)

        assert results[gate] == "failure"
        ran = [name for name, result in results.items() if name != gate and result != "skipped"]
        assert ran == [], f"{ran} ran with the suite red on {ref}"


class TestAFailedBuildPublishesNothing:
    def _builds(self, release):
        builds = [name for name, job in release["jobs"].items() if _is_build(job)]
        assert builds, "no job builds anything"
        return builds

    def _publishers(self, release):
        publishers = [
            name for name, job in release["jobs"].items()
            if any(_uses(step).startswith("softprops/action-gh-release") for step in _steps(job))
        ]
        assert publishers, "no job attaches anything to a release"
        return publishers

    @pytest.mark.parametrize("ref", BRANCH_REFS)
    def test_no_job_that_attaches_assets_runs_when_any_build_fails(self, release, ref):
        for failing in self._builds(release):
            results = _simulate(release, ref, failing={failing}, outputs=RELEASE_OUTPUT)

            assert results[failing] == "failure"
            for publisher in self._publishers(release):
                assert results[publisher] == "skipped", f"{publisher} ran after {failing} failed on {ref}"

    def test_a_failed_semantic_release_starts_no_build(self, release):
        results = _simulate(release, "refs/heads/main", failing={"semantic-release"}, outputs=RELEASE_OUTPUT)

        assert results["semantic-release"] == "failure"
        for name in self._builds(release) + self._publishers(release):
            assert results[name] == "skipped"

    def test_semantic_release_failure_is_not_swallowed(self, release):
        steps = [s for s in _steps(release["jobs"]["semantic-release"]) if "npx semantic-release" in _run_text(s)]

        assert steps, "the job no longer runs semantic-release"
        for step in steps:
            assert "|| true" not in _run_text(step), "a failing semantic-release would read as 'no release'"

    def test_a_release_semantic_release_creates_stays_a_draft_until_assets_are_attached(self):
        config = json.loads((ROOT / ".releaserc.json").read_text(encoding="utf-8"))
        github_plugin = next(
            options for name, options in (p if isinstance(p, list) else [p, {}] for p in config["plugins"])
            if name == "@semantic-release/github"
        )

        assert github_plugin.get("draftRelease") is True

    def test_the_versioned_release_is_published_after_its_assets_are_attached(self, release):
        steps = _steps(release["jobs"]["release-main"])
        attach = next(i for i, s in enumerate(steps) if _uses(s).startswith("softprops/action-gh-release"))
        publish = next(
            (i for i, s in enumerate(steps) if "draft=false" in _run_text(s) or "draft: false" in _run_text(s)),
            None,
        )

        assert publish is not None, "nothing ever turns the draft into a published release"
        assert publish > attach


class TestTheReleaseStillShipsWhenEverythingPasses:
    def test_a_push_to_develop_builds_and_refreshes_the_rolling_release(self, release):
        results = _simulate(release, "refs/heads/develop")

        assert results["semantic-release"] == "skipped"
        assert results["release-main"] == "skipped"
        for name, job in release["jobs"].items():
            if _is_build(job) or name == "release-develop":
                assert results[name] == "success", f"{name} did not run on a green push to develop"

    def test_a_push_to_main_with_a_new_version_builds_and_publishes_it(self, release):
        results = _simulate(release, "refs/heads/main", outputs=RELEASE_OUTPUT)

        assert results["release-develop"] == "skipped"
        for name, job in release["jobs"].items():
            if name != "release-develop":
                assert results[name] == "success", f"{name} did not run on a green release"

    def test_a_push_to_main_with_nothing_to_release_publishes_nothing(self, release):
        results = _simulate(release, "refs/heads/main", outputs=NO_RELEASE_OUTPUT)

        assert results["release-main"] == "skipped"
        assert results["release-develop"] == "skipped"


class TestTheBundleIsCheckedBeforeAnythingIsUploaded:
    def test_every_build_checks_the_bundle_between_building_and_uploading(self, release):
        builds = {name: job for name, job in release["jobs"].items() if _is_build(job)}
        assert builds

        for name, job in builds.items():
            steps = _steps(job)
            built = [i for i, s in enumerate(steps) if "pyinstaller jarvis_desktop.spec" in _run_text(s)]
            checked = [i for i, s in enumerate(steps) if "test_bundled_app" in _run_text(s)]
            uploaded = [i for i, s in enumerate(steps) if _uses(s).startswith("actions/upload-artifact")]

            assert built and checked and uploaded, f"{name} is missing its build, its bundle check or its upload"
            assert max(built) < min(checked), f"{name} checks the bundle before building it"
            assert max(checked) < min(uploaded), f"{name} uploads before the bundle has been checked"

    def test_the_check_cannot_hang_the_job_for_hours(self, release):
        for name, job in release["jobs"].items():
            for step in _steps(job):
                if "test_bundled_app" in _run_text(step):
                    assert step.get("timeout-minutes"), f"{name}: an app that never exits would hold the runner"


# ---------------------------------------------------------------------------
# The checksum file the updater verifies against
# ---------------------------------------------------------------------------

class TestTheChecksumFileIsPublishedWithTheInstallers:
    def test_both_release_jobs_write_it_before_attaching_assets(self, release):
        for name in ("release-main", "release-develop"):
            steps = _steps(release["jobs"][name])
            wrote = [i for i, s in enumerate(steps) if "release_checksums" in _run_text(s)]
            attached = [i for i, s in enumerate(steps) if _uses(s).startswith("softprops/action-gh-release")]

            assert wrote and attached, name
            assert max(wrote) < min(attached), f"{name} attaches assets before the checksums exist"

    def test_both_release_jobs_attach_it(self, release):
        for name in ("release-main", "release-develop"):
            attach = [s for s in _steps(release["jobs"][name]) if _uses(s).startswith("softprops/action-gh-release")]

            assert attach
            for step in attach:
                assert "SHA256SUMS.txt" in step["with"]["files"], f"{name} publishes installers with no checksum file"

    def test_the_updater_looks_for_the_file_the_workflow_publishes(self):
        from desktop_app.updater import CHECKSUMS_ASSET_NAME

        assert CHECKSUMS_ASSET_NAME == "SHA256SUMS.txt"

    @pytest.mark.skipif(os.name != "posix" or not shutil.which("bash"), reason="the script is bash")
    def test_what_the_script_writes_is_what_the_updater_verifies_against(self, tmp_path):
        from desktop_app.updater import CHECKSUMS_ASSET_NAME, _expected_sha256

        installers = {
            "Jarvis-Windows": ("Jarvis-Windows-x64.zip", b"windows installer"),
            "Jarvis-macOS-arm64": ("Jarvis-macOS-arm64.zip", b"mac installer"),
            "Jarvis-Linux": ("Jarvis-Linux-x64.tar.gz", b"linux installer"),
        }
        for folder, (name, data) in installers.items():
            (tmp_path / folder).mkdir()
            (tmp_path / folder / name).write_bytes(data)

        subprocess.run(["bash", str(ROOT / "scripts" / "release_checksums.sh"), str(tmp_path)], check=True,
                       capture_output=True)

        listing = (tmp_path / CHECKSUMS_ASSET_NAME).read_text(encoding="utf-8")
        for name, data in installers.values():
            assert _expected_sha256(listing, name) == hashlib.sha256(data).hexdigest()

    @pytest.mark.skipif(os.name != "posix" or not shutil.which("bash"), reason="the script is bash")
    def test_a_release_with_no_installers_gets_no_checksum_file(self, tmp_path):
        done = subprocess.run(["bash", str(ROOT / "scripts" / "release_checksums.sh"), str(tmp_path)],
                              capture_output=True)

        assert done.returncode != 0

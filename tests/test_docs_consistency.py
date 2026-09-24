"""Checks that spec frontmatter and documented Python version claims match
reality (see issue #35).

Guards against specs/README/pyproject silently drifting from what's
actually shipped -- e.g. a spec's linked issue closing without its
`status` frontmatter being updated, or Dockerfile.dev's base image moving
without README/pyproject following.
"""

import re
from pathlib import Path

import yaml
from packaging.specifiers import SpecifierSet

REPO_ROOT = Path(__file__).resolve().parent.parent


def _read_frontmatter(spec_path):
    text = spec_path.read_text(encoding="utf-8")
    _, frontmatter, _ = text.split("---", 2)
    return yaml.safe_load(frontmatter)


def test_data_analysis_octopus_refactor_spec_is_marked_completed():
    frontmatter = _read_frontmatter(
        REPO_ROOT / "specs" / "data-analysis-octopus-refactor.md"
    )
    assert frontmatter["status"] == "completed"


def test_poetry_lock_versioning_spec_is_marked_completed():
    frontmatter = _read_frontmatter(REPO_ROOT / "specs" / "poetry-lock-versioning.md")
    assert frontmatter["status"] == "completed"


def _dockerfile_python_version():
    dockerfile = (REPO_ROOT / "Dockerfile.dev").read_text(encoding="utf-8")
    match = re.search(r"FROM python:(\d+\.\d+)-slim", dockerfile)
    assert match, "Dockerfile.dev's base image line didn't match the expected pattern"
    return match.group(1)


def test_readme_python_version_matches_dockerfile():
    python_version = _dockerfile_python_version()
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert f"Python {python_version}" in readme


def test_pyproject_python_constraint_permits_dockerfile_version():
    # Every base image in Dockerfile.dev must be installable, not just the
    # first one: tensorflow-env is held on an older Python than the rest.
    dockerfile = (REPO_ROOT / "Dockerfile.dev").read_text(encoding="utf-8")
    python_versions = set(re.findall(r"FROM python:(\d+\.\d+)-slim", dockerfile))
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^python = "([^"]+)"', pyproject, re.MULTILINE)
    assert match, "pyproject.toml's python constraint didn't match the expected pattern"
    constraint = match.group(1)
    caret = re.fullmatch(r"\^(\d+)\.(\d+)", constraint)
    if caret:
        constraint = f">={caret[1]}.{caret[2]},<{int(caret[1]) + 1}.0"
    assert python_versions
    for python_version in python_versions:
        assert SpecifierSet(constraint).contains(python_version), python_version

"""
Repo hygiene guardrail tests — AR-01, AR-02, AR-04 (see BD3/BD4).

TC-18-adjacent enforcement suite: these tests run in CI on every push and
fail the build if a guardrail is violated. Written test-first per GOV-04.
"""
import ast
import pathlib
import re
import socket

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- AR-02
def _imports_of(pyfile: pathlib.Path):
    try:
        tree = ast.parse(pyfile.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return []
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def test_ar02_benchmarks_never_import_simulations():
    """AR-02: importing simulations/ from benchmarks/ is a CI failure."""
    bench = ROOT / "benchmarks"
    offenders = [
        str(f)
        for f in bench.rglob("*.py")
        for mod in _imports_of(f)
        if mod.split(".")[0] == "simulations"
    ]
    assert not offenders, f"benchmarks/ imports simulations/: {offenders}"


def test_ar02_simulation_files_carry_warning_header():
    """Every simulation harness must declare itself non-empirical."""
    sims = list((ROOT / "simulations").glob("*.py"))
    missing = [
        str(f)
        for f in sims
        if f.name != "__init__.py"
        and "SIMULATION — NOT EMPIRICAL EVIDENCE" not in f.read_text(errors="ignore")
    ]
    assert not missing, f"simulation files missing warning header: {missing}"


# ---------------------------------------------------------------- AR-03
_CLAIM_PATTERNS = [
    r"\+\s?\d+%\s+(better|improvement|faster)",
    r"p\s?<\s?0\.0\d+",
    r"\b99%\+?\s+faster\b",
    r"\b1[48]-27%\b",
]


def test_ar03_readme_carries_no_unvalidated_numeric_claims():
    """AR-03: README claims must trace to a reproducible benchmark artefact.

    Until benchmarks/ produces real artefacts (US5), the README must carry
    no numeric performance claims at all.
    """
    readme = (ROOT / "README.md").read_text(errors="ignore")
    hits = [p for p in _CLAIM_PATTERNS if re.search(p, readme, re.IGNORECASE)]
    assert not hits, f"README contains unvalidated claim patterns: {hits}"


def test_ar03_no_placeholder_arxiv_badge():
    readme = (ROOT / "README.md").read_text(errors="ignore")
    assert "2025.XXXXX" not in readme, "placeholder arXiv badge must be removed"


# ---------------------------------------------------------------- AR-04
def test_ar04_no_cuda_packages_in_default_dependencies():
    """AR-04 / GOV-03: no CUDA/GPU package in the default dependency tree."""
    pyproject = (ROOT / "pyproject.toml").read_text()
    # crude but effective: core [project.dependencies] block only
    deps_block = pyproject.split("dependencies = [", 1)[1].split("]", 1)[0].lower()
    for forbidden in ("cuda", "nvidia", "faiss-gpu", "torch>=",):
        assert forbidden not in deps_block, f"forbidden GPU-adjacent dep: {forbidden}"


# ---------------------------------------------------------------- AR-01
@pytest.fixture
def no_network(monkeypatch):
    """Fixture: any socket connection attempt fails the test (AR-01).

    Use on every default-path test. Quickstart and local flows must pass
    under this fixture with zero exemptions.
    """

    def _blocked(*args, **kwargs):  # pragma: no cover - triggered only on violation
        raise AssertionError("AR-01 violated: default path attempted a network call")

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    yield


def test_ar01_fixture_blocks_network(no_network):
    with pytest.raises(AssertionError, match="AR-01"):
        socket.create_connection(("example.com", 80), timeout=1)

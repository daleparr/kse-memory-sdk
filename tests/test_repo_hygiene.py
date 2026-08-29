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
def _py_files(directory: pathlib.Path, recursive: bool = False):
    """Python sources under *directory*, minus macOS AppleDouble sidecars.

    Filesystems without native xattr support (exFAT/NTFS volumes) grow a
    ``._name.py`` metadata file beside every source file. Those are resource
    forks, not Python, and must never be counted as repo content.
    """
    walk = directory.rglob if recursive else directory.glob
    return [f for f in walk("*.py") if not f.name.startswith("._")]


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
        for f in _py_files(bench, recursive=True)
        for mod in _imports_of(f)
        if mod.split(".")[0] == "simulations"
    ]
    assert not offenders, f"benchmarks/ imports simulations/: {offenders}"


def test_ar02_simulation_files_carry_warning_header():
    """Every simulation harness must declare itself non-empirical."""
    sims = _py_files(ROOT / "simulations")
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
# Distributions that pull torch, and therefore the full NVIDIA CUDA stack from
# the Linux PyPI wheels, without naming CUDA anywhere in their own requirement
# string. A literal substring scan is blind to every one of these.
_GPU_TRANSITIVE = frozenset(
    {
        "sentence-transformers",
        "transformers",
        "torch",
        "torchvision",
        "torchaudio",
        "accelerate",
        "timm",
        "tensorflow",
        "jax",
        "cupy",
        "faiss-gpu",
        "triton",
        "xformers",
        "bitsandbytes",
        "deepspeed",
        "vllm",
    }
)


def _default_dependency_names():
    """Normalised distribution names from [project.dependencies].

    Parsed rather than substring-matched, so the AR-04 denylist can be applied
    to real names. Strips comments, version specifiers and extras: a line like
    ``"pkg[foo]>=1.0",  # note`` reduces to ``pkg``.
    """
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    block = text.split("\ndependencies = [", 1)[1].split("\n]", 1)[0]
    names = []
    for raw in block.splitlines():
        line = raw.split("#", 1)[0].strip().rstrip(",").strip().strip("\"'")
        if not line:
            continue
        name = re.split(r"[<>=!~\[;\s]", line, 1)[0].strip()
        if name:
            names.append(name.lower().replace("_", "-"))
    return names


def test_ar04_no_cuda_packages_in_default_dependencies():
    """AR-04 / GOV-03: no CUDA/GPU package in the default dependency tree.

    Two layers, because the literal scan alone gives a false green:

    1. literal scan — catches a directly named GPU package.
    2. denylist over parsed names — catches packages that pull torch (and so
       the NVIDIA stack) transitively.

    Layer 2 exists because layer 1 shipped a violation to CI: the default tree
    carried ``sentence-transformers``, whose name contains neither "cuda" nor
    "torch", so this test passed 6/6 while a real ``pip install -e .`` pulled
    torch, triton, cuda-toolkit and ~20 nvidia-* wheels.

    CI additionally greps ``pip list`` after a real install, which remains the
    authoritative check — resolution is the only way to see the true tree. This
    test exists to fail on the obvious cases before CI has to.
    """
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    deps_block = text.split("\ndependencies = [", 1)[1].split("\n]", 1)[0].lower()
    for forbidden in ("cuda", "nvidia", "faiss-gpu", "torch>=",):
        assert forbidden not in deps_block, f"forbidden GPU-adjacent dep: {forbidden}"

    offenders = sorted(set(_default_dependency_names()) & _GPU_TRANSITIVE)
    assert not offenders, (
        f"default dependencies pull the CUDA stack transitively: {offenders}. "
        "Move them to an optional extra under [project.optional-dependencies]."
    )


# ---------------------------------------------------------------- AR-01
# The no_network fixture now lives in tests/conftest.py so every suite can use
# it, not just this one.


def test_ar01_fixture_blocks_network(no_network):
    with pytest.raises(AssertionError, match="AR-01"):
        socket.create_connection(("example.com", 80), timeout=1)

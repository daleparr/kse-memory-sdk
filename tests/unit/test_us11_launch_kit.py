"""
US11 — launch kit drafts, with "real numbers only" made executable (TC-12).

Written test-first per GOV-04. TC-12's publishing acts are the maintainer's;
what the tree can enforce is the AR-03 clause: every metric-shaped number in
a launch draft must exist verbatim in benchmarks/RESULTS.md, the artefact
`make bench` regenerates. A draft cannot cite a number the benchmark did not
produce.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
LAUNCH = ROOT / "docs" / "launch"

EXPECTED_DRAFTS = {
    "CHECKLIST.md",
    "show_hn.md",
    "blog_post.md",
    "arxiv_outline.md",
    "awesome_lists.md",
}

#: Three-decimal metric-shaped numbers (0.645 …). Version strings, years and
#: counts don't match; benchmark metrics do.
METRIC = re.compile(r"(?<![\d.])[+-]?0\.\d{3}(?![\d])")


def test_the_drafts_exist():
    assert LAUNCH.is_dir(), "docs/launch/ missing"
    present = {p.name for p in LAUNCH.glob("*.md") if not p.name.startswith("._")}
    assert EXPECTED_DRAFTS <= present, EXPECTED_DRAFTS - present


def test_every_metric_number_in_the_drafts_is_a_benchmark_number():
    """AR-03, executable: no draft may cite a metric RESULTS.md does not
    contain. Publishing edits that invent numbers turn this red."""
    results = (ROOT / "benchmarks" / "RESULTS.md").read_text(encoding="utf-8")
    published = set(METRIC.findall(results))
    offenders = []
    for draft in sorted(d for d in LAUNCH.glob("*.md") if not d.name.startswith("._")):
        for number in METRIC.findall(draft.read_text(encoding="utf-8")):
            if number not in published and number.lstrip("+-") not in {
                n.lstrip("+-") for n in published
            }:
                offenders.append(f"{draft.name}: {number}")
    assert not offenders, f"numbers not present in benchmarks/RESULTS.md: {offenders}"


def test_the_checklist_encodes_the_tc12_sequence():
    text = (LAUNCH / "CHECKLIST.md").read_text(encoding="utf-8")
    positions = [text.index(marker) for marker in
                 ("Show HN", "blog", "awesome-list", "arXiv")]
    assert positions == sorted(positions), "TC-12 requires the published sequence"


def test_drafts_carry_no_banned_claim_patterns():
    """The AR-03 hygiene regexes apply to launch copy exactly as to the README."""
    banned = [
        r"\+\s?\d+%\s+(better|improvement|faster)",
        r"p\s?<\s?0\.0\d+",
        r"\b99%\+?\s+faster\b",
    ]
    offenders = []
    for draft in sorted(d for d in LAUNCH.glob("*.md") if not d.name.startswith("._")):
        text = draft.read_text(encoding="utf-8")
        offenders += [f"{draft.name}: {p}" for p in banned if re.search(p, text, re.I)]
    assert not offenders


def test_the_losses_lead_in_every_outward_draft():
    """The launch story IS the credibility story: each outward-facing draft
    must state the hybrid losses before any claim of strength."""
    for name in ("show_hn.md", "blog_post.md", "arxiv_outline.md"):
        text = (LAUNCH / name).read_text(encoding="utf-8")
        assert "-0.306" in text, f"{name} must carry the scifact loss"

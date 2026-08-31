# Launch checklist (US11 / TC-12)

TC-12's sequence, in order. Every item is a **maintainer act** — nothing here
is automated, and each step is gated on the one before it landing well.

Preflight (before anything is posted):
- [ ] `make bench` re-run on the release commit; RESULTS.md regenerated and committed
- [ ] `pytest tests/` fully green on the release commit (no soft-fail lanes exist to hide behind — T-015 is flipped)
- [ ] `tests/unit/test_us11_launch_kit.py` green — every number in these drafts exists in RESULTS.md
- [ ] Version tagged; PyPI release pushed; `pip install kse-memory-sdk && kse quickstart` verified from a clean venv (with the documented model fetch)

Sequence (TC-12):
1. [ ] **Show HN** — `show_hn.md`. Post morning US-Eastern; stay present in the thread for the first 3 hours; answer with links to code and RESULTS.md, never with adjectives.
2. [ ] **Technical blog** — `blog_post.md`. Publish once the HN thread has settled; link it from a comment there, not from the post body.
3. [ ] **awesome-list PRs** — `awesome_lists.md`. Open only after the blog is live so the link target has substance.
4. [ ] **arXiv preprint** — `arxiv_outline.md` is the skeleton. DO NOT submit until the ESCI slice (D-103) and at least one hybrid-favourable benchmark configuration exist, or the paper's own honesty framing covers their absence. Real numbers only (AR-03): every figure regenerable by `make bench` at the cited commit.

Standing rule for all four: no number that is not in `benchmarks/RESULTS.md`,
no comparative adjective without a delta, and the losses are stated before
any strength is claimed. The guardrail test enforces the first clause
mechanically; the other two are editorial discipline.

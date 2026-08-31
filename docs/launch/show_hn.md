# Show HN draft

**Title:** Show HN: KSE — a hybrid retrieval SDK whose first published benchmark is a loss

**Body:**

KSE is a CPU-only, offline-by-default retrieval SDK: you define your own
dimension schema in YAML (no hardcoded vocabulary), items and queries are
embedded by a locally cached ONNX MiniLM, and three channels — dense,
conceptual, graph — fuse by reciprocal rank with a confidence gate that
falls back to dense *and says so* when corroboration is weak. Every result
carries receipts: per-channel ranks, per-dimension scores, and the schema +
model that produced it.

The part I actually want to show: `make bench` regenerates our whole
results table from checksum-pinned BEIR data, and the current table says
our hybrid mode **loses** to plain dense retrieval — nDCG@10 delta -0.306
on scifact, -0.158 on nfcorpus — because a generic three-dimension schema
carries no signal about scientific abstracts, and rank fusion weights a
noise channel equally with a good one. The dense baseline lands at 0.645
on scifact, which matches the published MiniLM figure, so we trust the
harness; the loss stays published because the project's whole premise is
that claims need regenerable evidence. Hybrid wins on demos engineered to
show the mechanism (schema anchors bridging a query/document vocabulary
gap — three runnable notebook packs in the repo), and loses when the
schema doesn't fit the corpus. That's the honest shape of it today.

Under the hood, everything that could lie has a test aimed at it: a
no-network fixture on the default path, Hypothesis suites for hash/fusion/
schema invariants, a conformance suite that caught two backends that could
never be instantiated, and a CI lane that runs the genuine model. Losses
print signed in the results table; a tampered dataset checksum refuses to
benchmark.

Repo: https://github.com/daleparr/kse-memory-sdk · Results:
https://github.com/daleparr/kse-memory-sdk/blob/master/benchmarks/RESULTS.md ·
Quickstart is
`pip install` + a one-time documented model fetch + `kse quickstart`.

# Testing Strategy (D-16)

Governing principle (GOV-04): **write the failing test first; prefer real local
implementations over mocks.** Empirical basis: every defect found in v3 work so
far (package-wide import crash, non-deterministic content hash) was caught by an
unmocked test on first contact; the legacy 48-mock suite caught neither.

## Lanes

| Lane | Scope | Rules | Marker |
|---|---|---|---|
| Guardrail | Repo invariants AR-01..AR-05 | Always on; fastest | (none — always) |
| Unit | Pure logic, zero I/O: hashing, RRF math, schema parse, normalise, mapping | Seeded; Hypothesis property tests for invariants | `unit` |
| Component | Each service vs REAL local backends (SQLite, hnswlib, NetworkX) | `no_network` autouse; no mocking of KSE modules | `component` |
| Integration | Full quickstart: ingest→project→query→explain on fixture corpus | CPU-only; asserts provenance chip data | `integration` |
| Conformance | Shared interface suites every backend adapter must pass | Tier 1 in CI; Tier 2 runnable locally | `conformance`, `requires_backend(<name>)` |

Benchmarks (US5) are NOT tests — they live in `benchmarks/` with their own gate.

## Embeddings in tests

- Default: **deterministic stub embedder** — a real `EmbeddingServiceInterface`
  implementation that hash-projects text to vectors. Deterministic, offline,
  instant. It is an implementation, not a mock.
- Optional CI lane: genuine ONNX MiniLM from a warmed `actions/cache`
  (never downloaded during a test run — AR-01).

## Mock policy

Mocks are permitted ONLY at true external boundaries: the optional OpenAI
scorer client and remote backend clients inside the unit lane. Mocking any
module KSE owns is a review-blocking violation.

## Determinism & flakes

Every test seeded. Frozen-clock fixture for timestamp-adjacent code.
Zero flake tolerance: no retries; a flaky test is a red test and gets fixed
or deleted the same session.

## Coverage

Ratchet on changed lines (diff-cover) — new/modified code must be covered;
no global backfill mandate. Aspirational floor rises as legacy code is replaced.

## Property-based testing (Hypothesis)

Required for: `content_hash` (key-order, unicode, nesting, set-like fields),
RRF (scale invariance, rank monotonicity, k-parameter bounds), dimension schema
round-trip. The Session-3 tag-order bug is the class of defect this lane exists
to catch automatically.

## Legacy suite retirement

Each FR that lands deletes its mocked counterparts in the same PR.
CI soft-fail lanes (mypy, legacy suite `continue-on-error`) flip to hard-fail
when Phase 2 (FR-01..FR-04 + guardrails) completes — tracked at T-015. The
concession is temporary by construction.

## Layout

```
tests/
  conftest.py            # no_network, seed, frozen clock, stub embedder
  test_repo_hygiene.py   # guardrail lane (exists)
  unit/                  # incl. property/ hypothesis suites
  component/
  integration/
  conformance/           # shared interface suites, parametrised by backend
```

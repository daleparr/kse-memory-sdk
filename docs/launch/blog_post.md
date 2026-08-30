# Technical blog draft

**Title:** Every defect we found, an unmocked test found first

**Standfirst:** We rebuilt a retrieval SDK under one governing rule — write
the failing test first, prefer real local implementations over mocks — and
kept a ledger. This is what the ledger says.

## Outline with the receipts

1. **The rule (GOV-04)** and why the old 48-test mocked suite motivated it.
2. **The ledger.** Every defect on the v3 branch, and what caught it:
   - a package-wide import crash — the first unmocked test;
   - a content hash that embedded object memory addresses — first real
     replay test; generalised later by Hypothesis (key order, tag order,
     unicode);
   - `SearchResult(product=…)` raising TypeError in every v2 result path —
     the first unmocked search test; the mocked suite had passed for months;
   - two backends (MongoDB, ArangoDB) that could never be instantiated —
     right behaviour, wrong method names — caught by a static conformance
     suite in milliseconds, no server required;
   - a mock vector store returning invented decreasing scores — caught the
     moment the conformance seed order stopped coinciding with similarity;
   - a graph channel emitting id-alphabetical order when coverage couldn't
     discriminate — fabricated evidence at full fusion weight, exposed by a
     demo corpus and fixed by teaching the channel to abstain.
3. **Benchmarks that publish their losses.** `make bench`, checksum-enforced
   pins, dense at literature parity (scifact nDCG@10 0.645), hybrid losing
   -0.306 / -0.158 with a generic schema — and why we shipped that table
   anyway.
4. **When hybrid does win**: the anchor-bridging mechanism, with the three
   pack notebooks as runnable evidence, and the confidence gate that refuses
   the word "hybrid" when corroboration is thin.
5. **Learned fusion that must earn its recommendation** — evaluated against
   RRF on held-out labels, recommended only on strict win; parity is not a
   win.
6. **What we'd tell you to steal**: the no-network fixture, the static
   conformance layer, decision logs with rulings, and tests that make your
   own launch copy unable to invent numbers (this post is covered by one).

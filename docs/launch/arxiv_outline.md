# arXiv preprint — SKELETON ONLY (do not submit; see CHECKLIST gate 4)

**Working title:** Schema-Defined Conceptual Channels for Hybrid Retrieval:
an Engineering Report with Negative Results

**Hard constraint (AR-03):** every figure in the eventual paper regenerates
via `make bench` at the cited commit. The prior preprint draft in
docs/project-history/ contains simulated numbers and MUST NOT be mined for
figures.

## Outline

1. Introduction — user-defined dimension schemas as the unit of meaning;
   receipts (per-channel ranks, per-dimension scores, replay identity) as a
   first-class output.
2. System — ingest/hash replay identity; anchor-centroid projection; query
   parsing in the same geometry; concurrent channels; RRF with confidence
   gating and stated fallback; abstention for non-discriminating channels.
3. Evaluation protocol — pinned BEIR (scifact, nfcorpus), CPU-only ONNX
   MiniLM, one-command regeneration. Dense baseline at literature parity
   (nDCG@10 0.645 scifact / 0.317 nfcorpus) as harness validation.
4. **Negative results, headline section** — generic-schema hybrid under-
   performs dense (Δ nDCG@10 -0.306 scifact, -0.158 nfcorpus; Δ recall@100
   -0.033, -0.048): rank fusion with an uninformative channel. Analysis:
   score-spread compression of mean-pooled MiniLM (contrastive correction
   measured at 1.4×, below our pre-registered 2× adoption bar — reported
   as a second negative).
5. Mechanism demonstrations — the vocabulary-bridging construction; when
   and why the conceptual channel adds signal; pack corpora as released
   artefacts.
6. Learned fusion — reciprocal-rank features; strict-win recommendation
   policy; synthetic separations.
7. Limitations & pending — ESCI slice (D-103), live graph backends'
   behavioural conformance, cross-encoder scorers (D-09 optional path).

Sections 3–4 are complete on current evidence; 5–6 have code and tests but
need figures generated from the pack/synthetic harnesses; 7 is honest.

# KSE Memory SDK

**A hybrid knowledge substrate for retrieval — embeddings + knowledge graph + conceptual dimensions, fused rank-wise, explainable by construction, CPU-first.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/kse-memory-sdk.svg)](https://pypi.org/project/kse-memory-sdk/)
[![CI (CPU-only)](https://github.com/daleparr/kse-memory-sdk/actions/workflows/ci.yml/badge.svg)](https://github.com/daleparr/kse-memory-sdk/actions)

## What it is

Pure vector search answers "what is *similar* to this text?" It struggles with graded,
multi-attribute intent — the way experts actually rank things in any domain:
*low-risk income funds with ESG tilt*, *comfortable minimalist running shoes*,
*aggressive indemnity clauses in recent precedents*, *concise beginner-friendly docs
about auth*. KSE adds two more retrieval channels alongside dense vectors and fuses
all three:

1. **Neural embeddings** — semantic similarity (local ONNX model by default; no API key).
2. **Conceptual dimensions** — *you* define a small schema of graded, business-meaningful
   dimensions (comfort, risk, formality...) with anchor examples; items are scored against
   it and queries are mapped onto it. A lightweight, governable semantic layer.
3. **Knowledge graph** — relationships and traversal (embedded NetworkX by default;
   Neo4j, ArangoDB, TypeDB via the same interface).

Channels are combined with **Reciprocal Rank Fusion** (rank-based, immune to
score-scale mismatch), and every result carries its per-channel provenance — you can
see *why* it ranked, by decomposition rather than post-hoc prose.

## Design guarantees

- **CPU-first.** The default install has no CUDA dependency and meets its performance
  targets on CPU-only hardware. Embeddings run via ONNX int8; ANN via hnswlib.
- **Zero network calls by default.** `pip install` + quickstart needs no API key and
  makes no network calls. Optional LLM-assisted dimension scoring is opt-in.
- **Truth stays yours.** Your source systems remain the system of record. Everything
  KSE builds — embeddings, dimension scores, graph edges — is a rebuildable projection.
- **Zero-downtime updates.** Projections update per item, incrementally. No full
  reindex on add/update.
- **Evidence-gated claims.** This README makes no performance claims until the
  reproducible benchmark suite (public BEIR subsets + Amazon ESCI, CPU-only,
  one command) lands in `benchmarks/`. Wins *and losses* will be published.
  Historical simulated harnesses are quarantined in `simulations/` and are not evidence.

## Status

`v3.0.0a1` — active remediation toward v3.0. The v2 codebase is being rebuilt against
a public spec: see `BD_INDEX.md` for the full engineering documentation set
(constitution, architecture, acceptance criteria, task plan). Contributions welcome —
`BD8_Task_Decomposition.md` is the live backlog.

## Quickstart (target UX — stabilising through v3.0 alpha)

```bash
pip install kse-memory-sdk

# one-time: cache the embedding model locally (KSE never downloads on the
# default path — AR-01)
D=~/.cache/kse/models/onnx-minilm-l6-v2 && mkdir -p "$D"
curl -L -o "$D/model.onnx" https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx
curl -L -o "$D/vocab.txt"  https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/vocab.txt

kse quickstart          # local demo corpus, CPU-only, no API key, offline
```

Quickstart retrieval is hybrid: RRF fusion over concurrent vector,
conceptual and graph channels, with per-dimension scores and per-channel
ranks shown as receipts on every result.

```python
from kse_memory import KSEMemory, SearchQuery

kse = KSEMemory()                       # in-process, local backends
await kse.initialize("generic", {...})  # your data via adapters
results = await kse.search(SearchQuery(query="comfortable minimalist running shoes",
                                       search_type="hybrid"))
for r in results:
    print(r.score, r.embedding_similarity, r.conceptual_similarity,
          r.knowledge_graph_similarity, r.product.title)
```

Define your own dimensions — any domain, no built-in vocabulary. The same substrate
runs both of these unchanged; only the YAML differs:

```yaml
# dimensions.retail.yaml
dimensions:
  - name: comfort
    description: physical comfort in extended use
    anchors: ["plush cushioned midsole", "all-day wearable", "soft breathable knit"]
  - name: minimalism
    description: visual and functional restraint
    anchors: ["clean single-tone design", "no excess branding", "essential features only"]
```

```yaml
# dimensions.fixed-income.yaml
dimensions:
  - name: credit_quality
    description: issuer creditworthiness and covenant strength
    anchors: ["investment grade, strong covenants", "stable senior secured", "low default history"]
  - name: liquidity
    description: ease of exit at fair value
    anchors: ["deep on-the-run market", "tight bid-ask", "high daily turnover"]
```

In regulated domains the per-channel provenance on every result doubles as an
audit trail: the ranking decomposes into named dimension scores, graph
relationships, and similarity — evidence, not a black box.

## Integrations

LangChain (`KSEVectorStore`) and LlamaIndex retrievers are drop-in. Graph backends are
tiered: NetworkX + Neo4j (CI-covered), ArangoDB + TypeDB (community adapters), and
platform connectors (e.g. reasoning-engine and agent-platform integrations) via the
connector interface.

## Licence & provenance

MIT. KSE was conceived prior to, and is operated independently of, the maintainer's
employment; see `NOTICE`.

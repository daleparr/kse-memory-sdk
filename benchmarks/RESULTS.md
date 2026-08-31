# Benchmark results (US5 / TC-05)

Regenerate with `make bench`. Every number below, including losses, comes from
this one command — see AR-03.

- **Model:** onnx-minilm-l6-v2 (local ONNX cache; fetched out of band, D-102)
- **Hardware:** arm64 · Darwin 24.6.0 · Python 3.9.6 · CPUExecutionProvider
- **Datasets:** pinned BEIR bundles (scifact (5183 docs, 300 queries), nfcorpus (3633 docs, 323 queries), esci-slice (5456 docs, 200 queries))
- **Timings:** scifact: 254s, nfcorpus: 190s, esci-slice: 261s
- **Systems:** dense = cosine over MiniLM vectors; hybrid = RRF over the dense
  channel and a conceptual channel scored against the generic quickstart
  schema. The graph channel is omitted: BEIR corpora carry no ingest-time
  graph, and faking one would benchmark the fake.

| dataset | system | nDCG@10 | Δ vs dense | recall@100 | Δ vs dense |
|---|---|---|---|---|---|
| esci-slice | dense | 0.415 | — | 0.812 | — |
| esci-slice | hybrid | 0.224 | -0.191 | 0.701 | -0.111 |
| nfcorpus | dense | 0.317 | — | 0.311 | — |
| nfcorpus | hybrid | 0.160 | -0.158 | 0.263 | -0.048 |
| scifact | dense | 0.645 | — | 0.925 | — |
| scifact | hybrid | 0.339 | -0.306 | 0.892 | -0.033 |

Read the deltas as they are. A negative Δ is a loss and stays published.

"""
US5 — the reproducible benchmark harness (TC-05, AR-03, AR-04).

Everything published about KSE's retrieval quality regenerates from here via
`make bench`: pinned BEIR datasets, the genuine ONNX MiniLM from the local
cache, CPU only, deterministic, and the results table prints losses with the
same prominence as wins — a benchmark that hides its losses is marketing.

NOT a test suite (D-16: benchmarks are not tests); the metric arithmetic is
unit-tested in tests/unit/test_bench_harness.py because published numbers
deserve tested arithmetic.
"""
from __future__ import annotations

import json
import math
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

DATA_DIR = Path(__file__).parent / "data"

#: Pinned datasets: canonical BEIR bundles, checksummed at fetch time.
DATASETS = {
    "scifact": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
    "nfcorpus": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip",
}


# ------------------------------------------------------------------ metrics
# Single implementation lives in the package (US10 layering: benchmarks
# depends on kse_memory, never the reverse). Re-exported for callers here.
from kse_memory.core.metrics import ndcg_at_k, recall_at_k  # noqa: E402,F401


# ------------------------------------------------------------------- table
def format_results_table(rows: List[Dict[str, Any]], baseline: str) -> str:
    """Markdown table; every non-baseline row shows its signed delta vs the
    baseline on the same dataset. Losses print exactly like wins."""
    by_dataset: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for row in rows:
        by_dataset.setdefault(row["dataset"], {})[row["system"]] = row

    lines = [
        "| dataset | system | nDCG@10 | Δ vs " + baseline + " | recall@100 | Δ vs " + baseline + " |",
        "|---|---|---|---|---|---|",
    ]
    for dataset in sorted(by_dataset):
        systems = by_dataset[dataset]
        base = systems.get(baseline)
        for name in sorted(systems):
            row = systems[name]
            if base is None or name == baseline:
                d_ndcg = d_recall = "—"
            else:
                d_ndcg = f"{row['ndcg@10'] - base['ndcg@10']:+.3f}"
                d_recall = f"{row['recall@100'] - base['recall@100']:+.3f}"
            lines.append(
                f"| {dataset} | {name} | {row['ndcg@10']:.3f} | {d_ndcg} "
                f"| {row['recall@100']:.3f} | {d_recall} |"
            )
    return "\n".join(lines)


# ------------------------------------------------------------------ loading
def load_beir(dataset: str) -> Dict[str, Any]:
    root = DATA_DIR / dataset
    corpus = {}
    with (root / "corpus.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            doc = json.loads(line)
            corpus[doc["_id"]] = {"title": doc.get("title", ""), "text": doc.get("text", "")}
    queries = {}
    with (root / "queries.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            q = json.loads(line)
            queries[q["_id"]] = q["text"]
    qrels: Dict[str, Dict[str, int]] = {}
    with (root / "qrels" / "test.tsv").open(encoding="utf-8") as handle:
        next(handle)  # header
        for line in handle:
            qid, doc_id, rel = line.rstrip("\n").split("\t")
            qrels.setdefault(qid, {})[doc_id] = int(rel)
    # only queries with test judgements participate
    queries = {qid: text for qid, text in queries.items() if qid in qrels}
    return {"corpus": corpus, "queries": queries, "qrels": qrels}


# ------------------------------------------------------------------ systems
@dataclass
class BenchContext:
    embedder: Any
    doc_ids: List[str]
    doc_vectors: List[List[float]]


def embed_corpus(embedder: Any, corpus: Mapping[str, Mapping[str, str]], batch: int = 64) -> BenchContext:
    doc_ids = sorted(corpus)
    texts = [f"{corpus[d]['title']}\n{corpus[d]['text']}" for d in doc_ids]
    vectors: List[List[float]] = []
    for start in range(0, len(texts), batch):
        vectors.extend(embedder.embed(texts[start:start + batch]))
    return BenchContext(embedder=embedder, doc_ids=doc_ids, doc_vectors=vectors)


def dense_rank(context: BenchContext, query_vector: Sequence[float], k: int) -> List[str]:
    from kse_memory.core.projection import vector_cosine

    scored = sorted(
        ((doc_id, vector_cosine(query_vector, vec))
         for doc_id, vec in zip(context.doc_ids, context.doc_vectors)),
        key=lambda r: (-r[1], r[0]),
    )
    return [doc_id for doc_id, _ in scored[:k]]


def run_dataset(dataset: str, embedder: Any, k_eval: int = 100) -> List[Dict[str, Any]]:
    """Dense baseline vs hybrid (dense + conceptual channel over a generic
    document schema, RRF-fused). Graph channel is omitted: BEIR corpora carry
    no ingest-time graph, and faking one would benchmark the fake."""
    from kse_memory.core.fusion import fuse_rrf
    from kse_memory.core.projection import anchor_centroids, score_from_vectors, vector_cosine
    from kse_memory.core.schema import load_schema
    from kse_memory.quickstart.v3 import DEFAULT_SCHEMA

    data = load_beir(dataset)
    context = embed_corpus(embedder, data["corpus"])
    schema = load_schema(DEFAULT_SCHEMA)
    centroids = anchor_centroids(schema, embedder)
    doc_scores = {
        doc_id: score_from_vectors(vec, centroids)
        for doc_id, vec in zip(context.doc_ids, context.doc_vectors)
    }

    metrics = {"dense": {"ndcg": [], "recall": []}, "hybrid": {"ndcg": [], "recall": []}}
    for qid in sorted(data["queries"]):
        qvec = embedder.embed([data["queries"][qid]])[0]
        qrels = data["qrels"][qid]

        dense = dense_rank(context, qvec, k_eval)
        metrics["dense"]["ndcg"].append(ndcg_at_k(dense, qrels, 10))
        metrics["dense"]["recall"].append(recall_at_k(dense, qrels, k_eval))

        targets = score_from_vectors(qvec, centroids)
        conceptual = sorted(
            ((doc_id, vector_cosine(list(targets.values()),
                                    [doc_scores[doc_id][n] for n in targets]))
             for doc_id in context.doc_ids),
            key=lambda r: (-r[1], r[0]),
        )[:k_eval]
        fused = fuse_rrf({
            "dense": tuple((d, 0.0) for d in dense),
            "conceptual": tuple(conceptual),
        }, top_k=k_eval)
        hybrid = [item.entity_id for item in fused]
        metrics["hybrid"]["ndcg"].append(ndcg_at_k(hybrid, qrels, 10))
        metrics["hybrid"]["recall"].append(recall_at_k(hybrid, qrels, k_eval))

    rows = []
    for system in ("dense", "hybrid"):
        rows.append({
            "dataset": dataset,
            "system": system,
            "ndcg@10": sum(metrics[system]["ndcg"]) / len(metrics[system]["ndcg"]),
            "recall@100": sum(metrics[system]["recall"]) / len(metrics[system]["recall"]),
            "queries": len(data["queries"]),
            "docs": len(data["corpus"]),
        })
    return rows


def main() -> None:
    from kse_memory.core.projection import OnnxEmbedder

    embedder = OnnxEmbedder()
    all_rows: List[Dict[str, Any]] = []
    timings = {}
    for dataset in DATASETS:
        started = time.perf_counter()
        all_rows.extend(run_dataset(dataset, embedder))
        timings[dataset] = time.perf_counter() - started
        print(f"{dataset}: done in {timings[dataset]:.0f}s")

    table = format_results_table(all_rows, baseline="dense")
    hardware = f"{platform.machine()} · {platform.system()} {platform.release()} · Python {platform.python_version()} · CPUExecutionProvider"
    out = Path(__file__).parent / "RESULTS.md"
    out.write_text(f"""# Benchmark results (US5 / TC-05)

Regenerate with `make bench`. Every number below, including losses, comes from
this one command — see AR-03.

- **Model:** {embedder.model_id} (local ONNX cache; fetched out of band, D-102)
- **Hardware:** {hardware}
- **Datasets:** pinned BEIR bundles ({", ".join(f"{d} ({r['docs']} docs, {r['queries']} queries)" for d, r in {row["dataset"]: row for row in all_rows}.items())})
- **Timings:** {", ".join(f"{d}: {t:.0f}s" for d, t in timings.items())}
- **Systems:** dense = cosine over MiniLM vectors; hybrid = RRF over the dense
  channel and a conceptual channel scored against the generic quickstart
  schema. The graph channel is omitted: BEIR corpora carry no ingest-time
  graph, and faking one would benchmark the fake.

{table}

Read the deltas as they are. A negative Δ is a loss and stays published.
""", encoding="utf-8")
    print(f"wrote {out}")
    print(table)


if __name__ == "__main__":
    main()

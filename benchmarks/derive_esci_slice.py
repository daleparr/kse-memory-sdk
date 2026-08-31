"""
Derive the pinned ESCI slice (D-104) — maintainer tool, not a default-path
dependency. Requires: pyarrow, pandas (documented here, deliberately absent
from pyproject).

Decision D-104 (the pinning D-103 deferred):
- Source: amazon-science/esci-data @ main, both parquets, sha256-recorded
  into the slice's MANIFEST at derivation time.
- Filter: product_locale == "us", small_version == 1, split == "test".
- Queries: the FIRST 200 by ascending query_id after filtering — a
  deterministic rule, not a sample.
- Corpus: exactly the products judged for those queries; text is
  title + description or bullet_points (whichever exists).
- Labels: E→3, S→2, C→1, I→0 (graded; the standard exact/substitute/
  complement/irrelevant reading).
- Output: BEIR format, committed at benchmarks/esci_slice/esci-slice/ so
  `make bench` needs no ESCI download ever. Apache-2.0 source; attribution
  in the MANIFEST.

Usage: python benchmarks/derive_esci_slice.py <dir-with-both-parquets>
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

LABEL_MAP = {"E": 3, "S": 2, "C": 1, "I": 0}
N_QUERIES = 200
OUT = Path(__file__).parent / "esci_slice" / "esci-slice"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(source_dir: str) -> None:
    src = Path(source_dir)
    examples_path = src / "shopping_queries_dataset_examples.parquet"
    products_path = src / "shopping_queries_dataset_products.parquet"

    examples = pd.read_parquet(examples_path)
    examples = examples[
        (examples["product_locale"] == "us")
        & (examples["small_version"] == 1)
        & (examples["split"] == "test")
    ]
    query_ids = sorted(examples["query_id"].unique())[:N_QUERIES]
    examples = examples[examples["query_id"].isin(query_ids)]

    products = pd.read_parquet(
        products_path, columns=["product_id", "product_locale", "product_title",
                                "product_description", "product_bullet_point"]
    )
    products = products[products["product_locale"] == "us"]
    judged = set(examples["product_id"])
    products = products[products["product_id"].isin(judged)].drop_duplicates("product_id")

    (OUT / "qrels").mkdir(parents=True, exist_ok=True)

    with (OUT / "queries.jsonl").open("w", encoding="utf-8") as handle:
        for qid, group in sorted(examples.groupby("query_id")):
            handle.write(json.dumps(
                {"_id": str(qid), "text": str(group["query"].iloc[0])}) + "\n")

    with (OUT / "corpus.jsonl").open("w", encoding="utf-8") as handle:
        for row in products.sort_values("product_id").itertuples():
            body = row.product_description or row.product_bullet_point or ""
            handle.write(json.dumps({
                "_id": str(row.product_id),
                "title": str(row.product_title or ""),
                "text": str(body)[:2000],
            }) + "\n")

    with (OUT / "qrels" / "test.tsv").open("w", encoding="utf-8") as handle:
        handle.write("query-id\tcorpus-id\tscore\n")
        for row in examples.sort_values(["query_id", "product_id"]).itertuples():
            handle.write(f"{row.query_id}\t{row.product_id}\t{LABEL_MAP[row.esci_label]}\n")

    (OUT / "MANIFEST.json").write_text(json.dumps({
        "decision": "D-104",
        "source_repo": "https://github.com/amazon-science/esci-data (Apache-2.0)",
        "source_sha256": {
            examples_path.name: sha256(examples_path),
            products_path.name: sha256(products_path),
        },
        "filter": "product_locale=us, small_version=1, split=test",
        "queries": len(query_ids),
        "corpus_docs": int(len(products)),
        "judgements": int(len(examples)),
        "label_map": LABEL_MAP,
    }, indent=2), encoding="utf-8")
    print(f"slice written to {OUT}: {len(query_ids)} queries, "
          f"{len(products)} docs, {len(examples)} judgements")


if __name__ == "__main__":
    main(sys.argv[1])

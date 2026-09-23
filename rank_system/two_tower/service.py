"""Training, dense-index build, and offline evaluation for the two-tower model."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from bazar_core.paths import write_json
from bazar_data.database import Database
from rank_system.retrieval.bm25 import CandidateMetrics
from rank_system.two_tower.model import TwoTowerModel


def _load_storage_data(database: Database) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    products = database.read_dataframe("SELECT product_id, catalog_text FROM products ORDER BY product_id")
    queries = database.read_dataframe("SELECT query_id, query_text, split FROM queries ORDER BY query_id")
    judgments = database.read_dataframe("SELECT query_id, product_id, relevance_gain, split FROM relevance_judgments")
    return products, queries, judgments


def _positive_train_pairs(products: pd.DataFrame, queries: pd.DataFrame, judgments: pd.DataFrame) -> pd.DataFrame:
    positives = judgments[(judgments["split"] == "train") & (judgments["relevance_gain"] > 0)].copy()
    # One strongest positive per query prevents false negatives for repeated queries in a batch.
    positives = positives.sort_values(["query_id", "relevance_gain", "product_id"], ascending=[True, False, True]).drop_duplicates("query_id")
    pairs = positives.merge(queries[["query_id", "query_text"]], on="query_id", validate="many_to_one")
    pairs = pairs.merge(products, on="product_id", validate="many_to_one")
    if pairs.empty:
        raise ValueError("No positive train pairs available for two-tower training.")
    return pairs


def train_two_tower_artifact(
    database: Database,
    output_dir: str | Path,
    *,
    dimension: int = 64,
    epochs: int = 15,
    batch_size: int = 64,
    learning_rate: float = 0.08,
    temperature: float = 0.1,
    seed: int = 42,
) -> dict[str, object]:
    products, queries, judgments = _load_storage_data(database)
    pairs = _positive_train_pairs(products, queries, judgments)
    vocabulary = TwoTowerModel.build_vocabulary(pairs["query_text"].tolist() + products["catalog_text"].tolist())
    model = TwoTowerModel.initialize(vocabulary, dimension, temperature, seed)
    training = model.fit(pairs["query_text"].tolist(), pairs["catalog_text"].tolist(), epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, seed=seed)
    output_dir = Path(output_dir)
    data_hash = sha256(pairs[["query_id", "product_id", "relevance_gain"]].sort_values("query_id").to_csv(index=False).encode()).hexdigest()
    config = {"dimension": dimension, "epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate, "temperature": temperature, "seed": seed, "data_hash": data_hash}
    model.save(output_dir, training=training, config=config)
    product_vectors = model.encode_products(products["catalog_text"].tolist())
    # Explicit unicode dtype keeps the artifact loadable with allow_pickle=False.
    np.savez_compressed(output_dir / "dense_index.npz", product_ids=np.asarray(products["product_id"].astype(str), dtype=str), vectors=product_vectors)
    return {"artifact_dir": str(output_dir), "training": asdict(training), "vocabulary_size": len(vocabulary), "data_hash": data_hash}


def _ranked_products(query_vector: np.ndarray, product_ids: np.ndarray, vectors: np.ndarray, limit: int) -> list[str]:
    scores = vectors @ query_vector
    order = sorted(range(len(product_ids)), key=lambda index: (-scores[index], str(product_ids[index])))[:limit]
    return [str(product_ids[index]) for index in order]


def evaluate_two_tower_artifact(database: Database, artifact_dir: str | Path, report_path: str | Path) -> dict[str, object]:
    artifact_dir = Path(artifact_dir)
    model = TwoTowerModel.load(artifact_dir)
    dense_index = np.load(artifact_dir / "dense_index.npz", allow_pickle=False)
    product_ids, vectors = dense_index["product_ids"], dense_index["vectors"]
    _, queries, judgments = _load_storage_data(database)
    test_positive = judgments[(judgments["split"] == "test") & (judgments["relevance_gain"] > 0)]
    gains = judgments[judgments["split"] == "test"].groupby("query_id").apply(
        lambda frame: dict(zip(frame["product_id"].astype(str), frame["relevance_gain"])), include_groups=False
    )
    positives = test_positive.groupby("query_id")["product_id"].agg(lambda values: set(values.astype(str)))
    latencies: list[float] = []
    recalls: dict[int, list[float]] = {10: [], 50: [], 100: []}
    ndcgs: list[float] = []
    mrrs: list[float] = []
    for row in queries[queries["split"] == "test"].itertuples(index=False):
        query_id = str(row.query_id)
        relevant = positives.get(query_id)
        if not relevant:
            continue
        started = perf_counter()
        ranked = _ranked_products(model.encode_queries([row.query_text])[0], product_ids, vectors, 100)
        latencies.append((perf_counter() - started) * 1000)
        for cutoff in recalls:
            recalls[cutoff].append(float(bool(relevant.intersection(ranked[:cutoff]))))
        query_gains = gains[query_id]
        actual = [query_gains.get(product_id, 0) for product_id in ranked[:10]]
        ideal = sorted(query_gains.values(), reverse=True)[:10]
        dcg = sum((2 ** gain - 1) / np.log2(position + 2) for position, gain in enumerate(actual))
        idcg = sum((2 ** gain - 1) / np.log2(position + 2) for position, gain in enumerate(ideal))
        ndcgs.append(dcg / idcg if idcg else 0.0)
        first = next((position for position, product_id in enumerate(ranked[:10], start=1) if product_id in relevant), None)
        mrrs.append(1 / first if first else 0.0)
    if not latencies:
        raise ValueError("No positive source test rows available for two-tower evaluation.")
    metrics = {
        **asdict(CandidateMetrics(len(latencies), float(np.mean(recalls[10])), float(np.mean(recalls[50])), float(np.mean(recalls[100])), float(np.percentile(latencies, 50)), float(np.percentile(latencies, 95)))),
        "ndcg_at_10": float(np.mean(ndcgs)), "mrr_at_10": float(np.mean(mrrs)),
    }
    report = {"model_type": "two_tower", "artifact": str(artifact_dir), "metrics": metrics}
    write_json(report_path, report)
    return report

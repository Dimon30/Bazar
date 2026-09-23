"""Offline artifact build/evaluation helpers for candidate generation."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from bazar_core.paths import write_json
from bazar_data.database import Database
from rank_system.retrieval.bm25 import BM25Index, evaluate_bm25


def build_bm25_artifact(database: Database, output: str | Path) -> BM25Index:
    catalog = database.read_dataframe("SELECT product_id, catalog_text FROM products ORDER BY product_id")
    index = BM25Index.from_catalog(catalog)
    index.save(output)
    return index


def evaluate_bm25_artifact(database: Database, index_path: str | Path, report_path: str | Path) -> dict[str, object]:
    index = BM25Index.load(index_path)
    queries = database.read_dataframe("SELECT query_id, query_text FROM queries WHERE split = 'test' ORDER BY query_id")
    judgments = database.read_dataframe("SELECT query_id, product_id, relevance_gain, split FROM relevance_judgments")
    metrics = asdict(evaluate_bm25(index, queries, judgments))
    report = {"model_type": "bm25", "artifact": str(index_path), "metrics": metrics}
    write_json(report_path, report)
    return report

"""Dependency-light, reproducible BM25 candidate generator for Bazar."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from math import log
from pathlib import Path
from re import findall
from time import perf_counter
from typing import Iterable

import numpy as np
import pandas as pd


def tokenize(text: str) -> list[str]:
    return findall(r"[\w]+", text.lower(), flags=0)


@dataclass(frozen=True)
class CandidateMetrics:
    queries_evaluated: int
    recall_at_10: float
    recall_at_50: float
    recall_at_100: float
    p50_latency_ms: float
    p95_latency_ms: float


@dataclass(frozen=True)
class RetrievalMetrics(CandidateMetrics):
    ndcg_at_10: float
    mrr_at_10: float


class BM25Index:
    """Small corpus index designed for transparent demo-scale serving.

    It serializes as JSON rather than pickle so artifacts can be inspected and
    rebuilt safely. For a production-scale catalog this contract is replaced by
    an inverted index service, not by changing the API layer.
    """

    def __init__(self, product_ids: list[str], documents: list[list[str]], *, k1: float = 1.5, b: float = 0.75):
        if not product_ids or len(product_ids) != len(documents):
            raise ValueError("BM25 requires one non-empty product-id/document pair per catalog item.")
        self.product_ids = product_ids
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.doc_lengths = [len(doc) for doc in documents]
        self.avg_doc_length = sum(self.doc_lengths) / len(documents)
        document_frequency: Counter[str] = Counter()
        self.term_frequencies = [Counter(document) for document in documents]
        for terms in self.term_frequencies:
            document_frequency.update(terms.keys())
        corpus_size = len(documents)
        self.idf = {
            term: log(1 + (corpus_size - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    @classmethod
    def from_catalog(cls, products: pd.DataFrame) -> "BM25Index":
        required = {"product_id", "catalog_text"}
        if missing := required.difference(products.columns):
            raise ValueError(f"Catalog is missing required BM25 columns: {sorted(missing)}")
        catalog = products.drop_duplicates("product_id").copy()
        catalog["catalog_text"] = catalog["catalog_text"].fillna("").astype(str)
        return cls(catalog["product_id"].astype(str).tolist(), [tokenize(text) for text in catalog["catalog_text"]])

    def search(self, query: str, limit: int = 100) -> list[tuple[str, float]]:
        if limit < 1:
            raise ValueError("Candidate limit must be positive.")
        scores = np.zeros(len(self.product_ids), dtype=float)
        for term in tokenize(query):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for idx, term_frequencies in enumerate(self.term_frequencies):
                frequency = term_frequencies.get(term, 0)
                if frequency == 0:
                    continue
                norm = 1 - self.b + self.b * self.doc_lengths[idx] / self.avg_doc_length
                scores[idx] += idf * frequency * (self.k1 + 1) / (frequency + self.k1 * norm)
        selected = sorted(range(len(self.product_ids)), key=lambda idx: (-scores[idx], self.product_ids[idx]))[:limit]
        return [(self.product_ids[idx], float(scores[idx])) for idx in selected]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "format": "bazar.bm25.v1", "product_ids": self.product_ids, "documents": self.documents,
            "k1": self.k1, "b": self.b,
        }, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != "bazar.bm25.v1":
            raise ValueError("Unsupported BM25 artifact format.")
        return cls(payload["product_ids"], payload["documents"], k1=float(payload["k1"]), b=float(payload["b"]))


def evaluate_bm25(index: BM25Index, queries: pd.DataFrame, judgments: pd.DataFrame) -> RetrievalMetrics:
    """Evaluate candidate recall on test rows only; positives are E/S/C labels."""
    required_queries = {"query_id", "query_text"}
    required_judgments = {"query_id", "product_id", "relevance_gain", "split"}
    if missing := required_queries.difference(queries.columns):
        raise ValueError(f"Queries are missing columns: {sorted(missing)}")
    if missing := required_judgments.difference(judgments.columns):
        raise ValueError(f"Judgments are missing columns: {sorted(missing)}")
    test_positive = judgments[(judgments["split"] == "test") & (judgments["relevance_gain"] > 0)]
    positives = test_positive.groupby("query_id")["product_id"].agg(lambda ids: set(ids.astype(str)))
    latencies: list[float] = []
    recalls: dict[int, list[float]] = {10: [], 50: [], 100: []}
    gains = judgments[judgments["split"] == "test"].groupby("query_id").apply(
        lambda frame: dict(zip(frame["product_id"].astype(str), frame["relevance_gain"])), include_groups=False
    )
    ndcgs: list[float] = []
    mrrs: list[float] = []
    for query in queries.itertuples(index=False):
        positive_ids = positives.get(str(query.query_id))
        if not positive_ids:
            continue
        started = perf_counter()
        ranked = index.search(query.query_text, limit=100)
        latencies.append((perf_counter() - started) * 1000)
        candidate_ids = [product_id for product_id, _ in ranked]
        for cutoff in recalls:
            recalls[cutoff].append(float(bool(positive_ids.intersection(candidate_ids[:cutoff]))))
        query_gains = gains[str(query.query_id)]
        actual = [query_gains.get(product_id, 0) for product_id in candidate_ids[:10]]
        ideal = sorted(query_gains.values(), reverse=True)[:10]
        dcg = sum((2 ** gain - 1) / np.log2(position + 2) for position, gain in enumerate(actual))
        idcg = sum((2 ** gain - 1) / np.log2(position + 2) for position, gain in enumerate(ideal))
        ndcgs.append(dcg / idcg if idcg else 0.0)
        first = next((position for position, product_id in enumerate(candidate_ids[:10], start=1) if product_id in positive_ids), None)
        mrrs.append(1 / first if first else 0.0)
    if not latencies:
        raise ValueError("No test queries with a positive relevance judgment are available for evaluation.")
    return RetrievalMetrics(
        queries_evaluated=len(latencies),
        recall_at_10=float(np.mean(recalls[10])),
        recall_at_50=float(np.mean(recalls[50])),
        recall_at_100=float(np.mean(recalls[100])),
        p50_latency_ms=float(np.percentile(latencies, 50)),
        p95_latency_ms=float(np.percentile(latencies, 95)),
        ndcg_at_10=float(np.mean(ndcgs)),
        mrr_at_10=float(np.mean(mrrs)),
    )

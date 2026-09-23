"""Online retrieval facade backed by an active registry entry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from bazar_data.database import Database
from bazar_data.model_registry import active_model
from rank_system.retrieval.bm25 import BM25Index
from rank_system.two_tower.model import TwoTowerModel


@dataclass(frozen=True)
class SearchResponse:
    request_id: str
    model_version: str
    model_type: str
    latency_ms: float
    products: list[dict[str, Any]]


class RetrievalRuntime:
    def __init__(self, database: Database):
        self.database = database
        self.version: str | None = None
        self.model_type: str | None = None
        self.model: BM25Index | TwoTowerModel | None = None
        self.product_ids: np.ndarray | None = None
        self.product_vectors: np.ndarray | None = None
        self.catalog: dict[str, dict[str, Any]] = {}

    def reload(self) -> None:
        record = active_model(self.database)
        self.catalog = {
            str(row["product_id"]): row
            for row in self.database.read_dataframe(
                "SELECT product_id, title, description, brand, color, locale FROM products"
            ).to_dict(orient="records")
        }
        if record is None:
            self.version = self.model_type = None
            self.model = self.product_ids = self.product_vectors = None
            return
        artifact = Path(str(record["artifact_uri"]))
        model_type = str(record["model_type"])
        if model_type == "bm25":
            self.model = BM25Index.load(artifact)
            self.product_ids = self.product_vectors = None
        elif model_type == "two_tower":
            self.model = TwoTowerModel.load(artifact)
            index = np.load(artifact / "dense_index.npz", allow_pickle=False)
            self.product_ids, self.product_vectors = index["product_ids"], index["vectors"]
        else:
            raise ValueError(f"Unsupported active model type: {model_type}")
        self.version, self.model_type = str(record["model_version"]), model_type

    @property
    def ready(self) -> bool:
        return self.model is not None and self.version is not None

    def search(self, query: str, request_id: str, limit: int) -> SearchResponse:
        if not self.ready or self.model is None or self.model_type is None or self.version is None:
            raise RuntimeError("No active retrieval model is loaded.")
        started = perf_counter()
        if self.model_type == "bm25":
            ranked = self.model.search(query, limit)  # type: ignore[union-attr]
        else:
            assert self.product_ids is not None and self.product_vectors is not None
            query_vector = self.model.encode_queries([query])[0]  # type: ignore[union-attr]
            scores = self.product_vectors @ query_vector
            positions = sorted(range(len(self.product_ids)), key=lambda index: (-scores[index], str(self.product_ids[index])))[:limit]
            ranked = [(str(self.product_ids[index]), float(scores[index])) for index in positions]
        products = []
        for position, (product_id, score) in enumerate(ranked, start=1):
            product = self.catalog.get(product_id)
            if product is None:
                continue
            products.append({"product_id": product_id, "title": product["title"], "description": product["description"], "brand": product["brand"], "color": product["color"], "locale": product["locale"], "score": score, "position": position})
        return SearchResponse(request_id, self.version, self.model_type, (perf_counter() - started) * 1000, products)

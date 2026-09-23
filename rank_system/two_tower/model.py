"""A compact two-tower query/product encoder trained with in-batch negatives.

This reference implementation deliberately uses NumPy so the full demo can be
reproduced on CPU.  Each tower owns its embedding table; the serving contract
(`encode_query`, `encode_products`) is identical to a transformer two-tower
model and lets us replace the encoder without touching retrieval/API code.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from rank_system.retrieval.bm25 import tokenize


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


@dataclass(frozen=True)
class TrainingSummary:
    pairs: int
    epochs: int
    final_loss: float


class TwoTowerModel:
    def __init__(self, vocabulary: dict[str, int], query_embeddings: np.ndarray, product_embeddings: np.ndarray, temperature: float):
        self.vocabulary = vocabulary
        self.query_embeddings = query_embeddings.astype(np.float32)
        self.product_embeddings = product_embeddings.astype(np.float32)
        self.temperature = float(temperature)

    @classmethod
    def initialize(cls, vocabulary: dict[str, int], dimension: int, temperature: float, seed: int) -> "TwoTowerModel":
        if dimension < 4:
            raise ValueError("Two-tower embedding dimension must be at least 4.")
        rng = np.random.default_rng(seed)
        scale = 1 / np.sqrt(dimension)
        shape = (len(vocabulary), dimension)
        shared_initialization = rng.normal(0, scale, size=shape).astype(np.float32)
        return cls(
            vocabulary,
            shared_initialization.copy(),
            shared_initialization.copy(),
            temperature,
        )

    @staticmethod
    def build_vocabulary(texts: list[str], min_frequency: int = 1, max_terms: int = 30_000) -> dict[str, int]:
        frequencies = Counter(token for text in texts for token in tokenize(text))
        terms = sorted((token for token, count in frequencies.items() if count >= min_frequency), key=lambda token: (-frequencies[token], token))[:max_terms]
        return {token: index for index, token in enumerate([PAD_TOKEN, UNK_TOKEN, *terms])}

    def _token_ids(self, text: str) -> np.ndarray:
        ids = [self.vocabulary.get(token, self.vocabulary[UNK_TOKEN]) for token in tokenize(text)]
        return np.asarray(ids or [self.vocabulary[UNK_TOKEN]], dtype=np.int64)

    def _encode(self, texts: list[str], table: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        token_ids = [self._token_ids(text) for text in texts]
        vectors = np.stack([table[ids].mean(axis=0) for ids in token_ids]).astype(np.float32)
        return vectors, token_ids

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        vectors, _ = self._encode(texts, self.query_embeddings)
        return self._normalize(vectors)

    def encode_products(self, texts: list[str]) -> np.ndarray:
        vectors, _ = self._encode(texts, self.product_embeddings)
        return self._normalize(vectors)

    def fit(
        self,
        queries: list[str],
        product_texts: list[str],
        *,
        epochs: int = 15,
        batch_size: int = 64,
        learning_rate: float = 0.08,
        seed: int = 42,
    ) -> TrainingSummary:
        """Optimise a query-product contrastive objective using in-batch negatives."""
        if len(queries) != len(product_texts) or not queries:
            raise ValueError("Two-tower fitting needs matching, non-empty query/product positive pairs.")
        rng = np.random.default_rng(seed)
        last_loss = 0.0
        indices = np.arange(len(queries))
        for _ in range(epochs):
            rng.shuffle(indices)
            epoch_losses: list[float] = []
            for start in range(0, len(indices), batch_size):
                batch = indices[start:start + batch_size]
                batch_queries = [queries[i] for i in batch]
                batch_products = [product_texts[i] for i in batch]
                q_raw, q_ids = self._encode(batch_queries, self.query_embeddings)
                p_raw, p_ids = self._encode(batch_products, self.product_embeddings)
                q_vectors = self._normalize(q_raw)
                p_vectors = self._normalize(p_raw)
                logits = q_vectors @ p_vectors.T / self.temperature
                logits -= logits.max(axis=1, keepdims=True)
                query_probabilities = np.exp(logits)
                query_probabilities /= query_probabilities.sum(axis=1, keepdims=True)
                product_logits = logits.T - logits.T.max(axis=1, keepdims=True)
                product_probabilities = np.exp(product_logits)
                product_probabilities /= product_probabilities.sum(axis=1, keepdims=True)
                target = np.eye(len(batch), dtype=np.float32)
                query_loss = -np.log(np.maximum(query_probabilities.diagonal(), 1e-12)).mean()
                product_loss = -np.log(np.maximum(product_probabilities.diagonal(), 1e-12)).mean()
                epoch_losses.append(float((query_loss + product_loss) / 2))
                grad_logits = (
                    query_probabilities - target + (product_probabilities - target).T
                ) / (2 * len(batch))
                grad_q_normalized = grad_logits @ p_vectors / self.temperature
                grad_p_normalized = grad_logits.T @ q_vectors / self.temperature
                q_norms = np.maximum(np.linalg.norm(q_raw, axis=1, keepdims=True), 1e-12)
                p_norms = np.maximum(np.linalg.norm(p_raw, axis=1, keepdims=True), 1e-12)
                grad_q = (
                    grad_q_normalized
                    - q_vectors * np.sum(grad_q_normalized * q_vectors, axis=1, keepdims=True)
                ) / q_norms
                grad_p = (
                    grad_p_normalized
                    - p_vectors * np.sum(grad_p_normalized * p_vectors, axis=1, keepdims=True)
                ) / p_norms
                for row, ids in enumerate(q_ids):
                    np.add.at(self.query_embeddings, ids, -learning_rate * grad_q[row] / len(ids))
                for row, ids in enumerate(p_ids):
                    np.add.at(self.product_embeddings, ids, -learning_rate * grad_p[row] / len(ids))
            last_loss = float(np.mean(epoch_losses))
        return TrainingSummary(pairs=len(queries), epochs=epochs, final_loss=last_loss)

    def save(self, directory: str | Path, *, training: TrainingSummary, config: dict[str, object]) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / "model.npz", query_embeddings=self.query_embeddings, product_embeddings=self.product_embeddings)
        (directory / "manifest.json").write_text(json.dumps({
            "format": "bazar.two_tower.v1", "vocabulary": self.vocabulary, "temperature": self.temperature,
            "training": training.__dict__, "config": config,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> "TwoTowerModel":
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("format") != "bazar.two_tower.v1":
            raise ValueError("Unsupported two-tower artifact format.")
        arrays = np.load(directory / "model.npz", allow_pickle=False)
        return cls(manifest["vocabulary"], arrays["query_embeddings"], arrays["product_embeddings"], manifest["temperature"])

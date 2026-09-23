from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from rank_system.retrieval.bm25 import BM25Index, evaluate_bm25


class Bm25Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.index = BM25Index.from_catalog(pd.DataFrame({
            "product_id": ["p1", "p2", "p3"],
            "catalog_text": ["red ceramic coffee mug", "blue water bottle", "green tea cup"],
        }))

    def test_search_is_deterministic_and_persisted(self) -> None:
        self.assertEqual(self.index.search("ceramic mug", 2)[0][0], "p1")
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bm25.json"
            self.index.save(path)
            self.assertEqual(BM25Index.load(path).search("ceramic mug", 2), self.index.search("ceramic mug", 2))

    def test_evaluation_uses_test_positive_rows_only(self) -> None:
        queries = pd.DataFrame({"query_id": ["q1", "q2"], "query_text": ["coffee mug", "water bottle"]})
        judgments = pd.DataFrame({
            "query_id": ["q1", "q1", "q2"],
            "product_id": ["p1", "p2", "p2"],
            "relevance_gain": [3, 0, 3],
            "split": ["test", "test", "train"],
        })
        metrics = evaluate_bm25(self.index, queries, judgments)
        self.assertEqual(metrics.queries_evaluated, 1)
        self.assertEqual(metrics.recall_at_10, 1.0)


if __name__ == "__main__":
    unittest.main()

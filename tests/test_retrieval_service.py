from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from bazar_data.database import Database
from bazar_data.marketplace_schema import products, queries, relevance_judgments
from rank_system.retrieval.service import build_bm25_artifact, evaluate_bm25_artifact


class RetrievalServiceTests(unittest.TestCase):
    def test_build_and_evaluate_from_storage_contract(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            database = Database(root / "bazar.db")
            database.create_schema()
            database.upsert(products, [
                {"product_id": "p1", "locale": "us", "title": "Ceramic mug", "description": "", "bullet_point": "", "brand": "", "color": "", "catalog_text": "ceramic coffee mug", "source_dataset": "test"},
                {"product_id": "p2", "locale": "us", "title": "Bottle", "description": "", "bullet_point": "", "brand": "", "color": "", "catalog_text": "blue water bottle", "source_dataset": "test"},
            ], key_columns=("product_id",))
            database.upsert(queries, [
                {"query_id": "q1", "query_text": "coffee mug", "locale": "us", "split": "test", "source": "test", "source_dataset": "test"},
            ], key_columns=("query_id",))
            database.upsert(relevance_judgments, [
                {"example_id": "e1", "query_id": "q1", "product_id": "p1", "esci_label": "E", "relevance_gain": 3, "split": "test", "source_dataset": "test"},
            ], key_columns=("example_id",))

            index_path, report_path = root / "bm25.json", root / "report.json"
            build_bm25_artifact(database, index_path)
            report = evaluate_bm25_artifact(database, index_path, report_path)

            self.assertEqual(report["metrics"]["recall_at_10"], 1.0)
            self.assertTrue(report_path.exists())


if __name__ == "__main__":
    unittest.main()

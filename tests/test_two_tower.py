from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from bazar_data.database import Database
from bazar_data.marketplace_schema import products, queries, relevance_judgments
from rank_system.two_tower.model import TwoTowerModel
from rank_system.two_tower.service import evaluate_two_tower_artifact, train_two_tower_artifact


class TwoTowerTests(unittest.TestCase):
    def test_model_round_trip(self) -> None:
        vocabulary = TwoTowerModel.build_vocabulary(["red mug", "blue bottle"])
        model = TwoTowerModel.initialize(vocabulary, dimension=8, temperature=0.2, seed=1)
        model.fit(["red mug", "blue bottle"], ["red coffee mug", "blue water bottle"], epochs=2, batch_size=2, seed=1)
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model.save(root, training=model.fit(["red mug"], ["red coffee mug"], epochs=1, batch_size=1), config={})
            restored = TwoTowerModel.load(root)
            self.assertEqual(restored.encode_queries(["red mug"]).shape, (1, 8))

    def test_training_and_evaluation_use_source_splits(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            database = Database(root / "bazar.db")
            database.create_schema()
            database.upsert(products, [
                {"product_id": "p1", "locale": "us", "title": "Red mug", "description": "", "bullet_point": "", "brand": "", "color": "", "catalog_text": "red ceramic coffee mug", "source_dataset": "test"},
                {"product_id": "p2", "locale": "us", "title": "Blue bottle", "description": "", "bullet_point": "", "brand": "", "color": "", "catalog_text": "blue water bottle", "source_dataset": "test"},
            ], key_columns=("product_id",))
            database.upsert(queries, [
                {"query_id": "q_train_1", "query_text": "red mug", "locale": "us", "split": "train", "source": "test", "source_dataset": "test"},
                {"query_id": "q_train_2", "query_text": "water bottle", "locale": "us", "split": "train", "source": "test", "source_dataset": "test"},
                {"query_id": "q_test", "query_text": "red coffee mug", "locale": "us", "split": "test", "source": "test", "source_dataset": "test"},
            ], key_columns=("query_id",))
            database.upsert(relevance_judgments, [
                {"example_id": "e1", "query_id": "q_train_1", "product_id": "p1", "esci_label": "E", "relevance_gain": 3, "split": "train", "source_dataset": "test"},
                {"example_id": "e2", "query_id": "q_train_2", "product_id": "p2", "esci_label": "E", "relevance_gain": 3, "split": "train", "source_dataset": "test"},
                {"example_id": "e3", "query_id": "q_test", "product_id": "p1", "esci_label": "E", "relevance_gain": 3, "split": "test", "source_dataset": "test"},
            ], key_columns=("example_id",))
            artifact_dir, report_path = root / "two_tower", root / "two_tower_report.json"
            result = train_two_tower_artifact(database, artifact_dir, dimension=8, epochs=2, batch_size=2)
            report = evaluate_two_tower_artifact(database, artifact_dir, report_path)
            self.assertEqual(result["training"]["pairs"], 2)
            self.assertEqual(report["metrics"]["queries_evaluated"], 1)
            self.assertTrue(report_path.exists())


if __name__ == "__main__":
    unittest.main()

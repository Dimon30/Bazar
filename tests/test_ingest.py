from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from bazar_data.database import Database
from bazar_data.ingest import clean_text, ingest_esci, stable_query_sample


class EsciIngestTests(unittest.TestCase):
    def test_clean_text_turns_missing_values_into_empty_strings(self) -> None:
        self.assertEqual(clean_text(None), "")
        self.assertEqual(clean_text(float("nan")), "")
        self.assertEqual(clean_text("  Red mug  "), "Red mug")
        self.assertEqual(clean_text(42), "42")

    def test_stable_query_sample_keeps_complete_first_sorted_queries(self) -> None:
        rows = pd.DataFrame({
            "query_id": ["q2", "q1", "q2", "q3", "q1"],
            "product_id": ["p1", "p2", "p3", "p4", "p5"],
        })

        sample = stable_query_sample(rows, max_queries=2)

        self.assertEqual(sample["query_id"].tolist(), ["q2", "q1", "q2", "q1"])
        self.assertEqual(sample["product_id"].tolist(), ["p1", "p2", "p3", "p5"])
        self.assertEqual(rows["query_id"].tolist(), ["q2", "q1", "q2", "q3", "q1"])

    def test_stable_query_sample_without_limit_returns_all_rows(self) -> None:
        rows = pd.DataFrame({"query_id": ["q2", "q1"]})

        sample = stable_query_sample(rows, max_queries=None)

        self.assertEqual(sample.to_dict("list"), rows.to_dict("list"))

    def test_ingest_is_idempotent_and_preserves_label_gains(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            examples = pd.DataFrame({
                "example_id": ["e1", "e2", "e3"],
                "query": ["red mug", "red mug", "tea cup"],
                "query_id": ["q1", "q1", "q2"],
                "product_id": ["p1", "p2", "p3"],
                "product_locale": ["us", "us", "us"],
                "esci_label": ["E", "I", "S"],
                "split": ["train", "train", "test"],
                "small_version": [1, 1, 1],
            })
            catalog = pd.DataFrame({
                "product_id": ["p1", "p2", "p3"],
                "product_locale": ["us", "us", "us"],
                "product_title": ["Red mug", "Blue mug", "Tea cup"],
                "product_description": [None, "", "Ceramic"],
                "product_bullet_point": ["Gift", "", "Kitchen"],
                "product_brand": ["Bazar", "Bazar", "Bazar"],
                "product_color": ["red", "blue", "white"],
            })
            examples_path, products_path = root / "examples.parquet", root / "products.parquet"
            examples.to_parquet(examples_path)
            catalog.to_parquet(products_path)
            database = Database(root / "bazar.db")

            first = ingest_esci(database, examples_path, products_path, max_queries=1)
            second = ingest_esci(database, examples_path, products_path, max_queries=1)

            self.assertEqual((first.products, first.queries, first.judgments), (2, 1, 2))
            self.assertEqual(first.data_hash, second.data_hash)
            self.assertEqual(database.fetch_one("SELECT COUNT(*) FROM products"), (2,))
            self.assertEqual(database.fetch_one("SELECT COUNT(*) FROM relevance_judgments"), (2,))
            self.assertEqual(
                database.fetch_all("SELECT esci_label, relevance_gain FROM relevance_judgments ORDER BY example_id"),
                [("E", 3), ("I", 0)],
            )


if __name__ == "__main__":
    unittest.main()

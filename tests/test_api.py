from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from fastapi.testclient import TestClient

from apps.api.main import create_app
from bazar_data.database import Database
from bazar_data.marketplace_schema import products
from bazar_data.model_registry import register_model
from rank_system.retrieval.service import build_bm25_artifact


class ApiTests(unittest.TestCase):
    def test_search_active_model_and_event_flow(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            database_url = str(root / "bazar.db")
            database = Database(database_url)
            database.create_schema()
            database.upsert(products, [{
                "product_id": "p1", "locale": "us", "title": "Ceramic mug", "description": "Red coffee mug",
                "bullet_point": "", "brand": "Bazar", "color": "red", "catalog_text": "red ceramic coffee mug", "source_dataset": "test",
            }], key_columns=("product_id",))
            artifact = root / "bm25.json"
            build_bm25_artifact(database, artifact)
            register_model(database, version="bm25-test", model_type="bm25", artifact_uri=artifact, metrics={}, data_hash="test", activate=True)

            with TestClient(create_app(database_url)) as client:
                self.assertTrue(client.get("/health").json()["retrieval_ready"])
                active = client.get("/models/active")
                self.assertEqual(active.status_code, 200)
                self.assertEqual(active.json()["model_version"], "bm25-test")
                result = client.get("/search", params={"q": "coffee mug"})
                self.assertEqual(result.status_code, 200)
                payload = result.json()
                self.assertEqual(payload["products"][0]["product_id"], "p1")
                event = client.post("/events", json={
                    "request_id": payload["request_id"], "query_text": "coffee mug", "product_id": "p1",
                    "position": 1, "event_type": "click",
                })
                self.assertEqual(event.status_code, 201)
                self.assertEqual(database.fetch_one("SELECT COUNT(*) FROM search_events"), (1,))


if __name__ == "__main__":
    unittest.main()

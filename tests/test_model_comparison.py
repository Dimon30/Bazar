from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from rank_system.evaluation.comparison import compare_models


class ModelComparisonTests(unittest.TestCase):
    def test_policy_selects_quality_gain_inside_latency_budget(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bm25, dense = root / "bm25.json", root / "dense.json"
            bm25.write_text(json.dumps({"artifact": str(root / "bm25.artifact"), "metrics": {"ndcg_at_10": 0.4, "mrr_at_10": 0.4, "recall_at_100": 0.8, "p95_latency_ms": 2.0}}))
            dense.write_text(json.dumps({"artifact": str(root / "dense.artifact"), "metrics": {"ndcg_at_10": 0.45, "mrr_at_10": 0.45, "recall_at_100": 0.85, "p95_latency_ms": 5.0}}))
            decision = compare_models(bm25, dense, root / "decision.json", min_ndcg_gain=0.01, max_p95_latency_ms=10)
            self.assertEqual(decision["selected_model_type"], "two_tower")

    def test_policy_keeps_baseline_when_gain_is_not_material(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bm25, dense = root / "bm25.json", root / "dense.json"
            bm25.write_text(json.dumps({"artifact": str(root / "bm25.artifact"), "metrics": {"ndcg_at_10": 0.4, "mrr_at_10": 0.4, "recall_at_100": 0.8, "p95_latency_ms": 2.0}}))
            dense.write_text(json.dumps({"artifact": str(root / "dense.artifact"), "metrics": {"ndcg_at_10": 0.405, "mrr_at_10": 0.45, "recall_at_100": 0.85, "p95_latency_ms": 5.0}}))
            decision = compare_models(bm25, dense, root / "decision.json", min_ndcg_gain=0.01, max_p95_latency_ms=10)
            self.assertEqual(decision["selected_model_type"], "bm25")


if __name__ == "__main__":
    unittest.main()

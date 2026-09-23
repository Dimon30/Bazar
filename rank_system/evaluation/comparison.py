"""Decision layer for choosing a retriever from reproducible offline reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bazar_core.paths import write_json


REQUIRED_METRICS = {"ndcg_at_10", "mrr_at_10", "recall_at_100", "p95_latency_ms"}


def _load_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = REQUIRED_METRICS.difference(payload.get("metrics", {}))
    if missing:
        raise ValueError(f"Report {path} lacks comparable metrics: {sorted(missing)}")
    return payload


def _artifact_size_bytes(path: str | Path) -> int:
    target = Path(path)
    if target.is_file():
        return target.stat().st_size
    if target.is_dir():
        return sum(item.stat().st_size for item in target.rglob("*") if item.is_file())
    return 0


def compare_models(
    bm25_report_path: str | Path,
    two_tower_report_path: str | Path,
    output_path: str | Path,
    *,
    min_ndcg_gain: float = 0.01,
    max_p95_latency_ms: float = 100.0,
) -> dict[str, Any]:
    """Select dense retrieval only when its quality gain clears a latency guardrail."""
    bm25 = _load_report(bm25_report_path)
    two_tower = _load_report(two_tower_report_path)
    baseline, candidate = bm25["metrics"], two_tower["metrics"]
    ndcg_gain = candidate["ndcg_at_10"] - baseline["ndcg_at_10"]
    passes_quality = ndcg_gain >= min_ndcg_gain
    passes_latency = candidate["p95_latency_ms"] <= max_p95_latency_ms
    selected = "two_tower" if passes_quality and passes_latency else "bm25"
    reason = (
        f"Two-tower improves NDCG@10 by {ndcg_gain:.4f} (threshold {min_ndcg_gain:.4f}) "
        f"and p95 latency is {candidate['p95_latency_ms']:.2f}ms (guardrail {max_p95_latency_ms:.2f}ms)."
        if selected == "two_tower" else
        f"Keep BM25: two-tower NDCG@10 gain is {ndcg_gain:.4f} (threshold {min_ndcg_gain:.4f}) "
        f"and its p95 latency is {candidate['p95_latency_ms']:.2f}ms (guardrail {max_p95_latency_ms:.2f}ms)."
    )
    decision = {
        "selected_model_type": selected,
        "selected_artifact": two_tower["artifact"] if selected == "two_tower" else bm25["artifact"],
        "decision_policy": {"min_ndcg_gain": min_ndcg_gain, "max_p95_latency_ms": max_p95_latency_ms},
        "comparison": {
            "bm25": {"metrics": baseline, "artifact_size_bytes": _artifact_size_bytes(bm25["artifact"])},
            "two_tower": {"metrics": candidate, "artifact_size_bytes": _artifact_size_bytes(two_tower["artifact"])},
            "ndcg_at_10_gain": ndcg_gain,
        },
        "product_rationale": reason,
    }
    write_json(output_path, decision)
    return decision

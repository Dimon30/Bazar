from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from catboost import CatBoostRanker

from marketplace_ml.features.rank_featurizer import RankFeaturizer
from marketplace_ml.rank.artifacts import RankArtifacts, read_meta


def load_ranker(artifacts_dir: str | Path) -> tuple[CatBoostRanker, RankFeaturizer, dict]:
    arts = RankArtifacts.in_dir(artifacts_dir)
    meta = read_meta(arts.meta_path)

    model = CatBoostRanker()
    model.load_model(str(arts.model_path))

    featurizer = RankFeaturizer.load(arts.featurizer_path)
    return model, featurizer, meta


def rank_items(df: pd.DataFrame, *, model: CatBoostRanker, featurizer: RankFeaturizer, meta: dict) -> pd.DataFrame:
    X, _, _ = featurizer.transform(df)
    feature_cols = meta["feature_cols"]
    out = df.copy()
    out["score"] = model.predict(X[feature_cols])
    return out.sort_values(["query_id", "score"], ascending=[True, False])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts-dir", required=True)
    ap.add_argument("--input-csv", required=True)
    ap.add_argument("--output-csv", required=True)
    args = ap.parse_args()

    model, featurizer, meta = load_ranker(args.artifacts_dir)
    df = pd.read_csv(args.input_csv)
    ranked = rank_items(df, model=model, featurizer=featurizer, meta=meta)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(out_path, index=False)
    print(f"[OK] Saved ranked results: {out_path}")


if __name__ == "__main__":
    main()


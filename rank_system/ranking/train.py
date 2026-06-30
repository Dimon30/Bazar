from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from catboost import CatBoostRanker, Pool

from marketplace_ml.common.config import load_yaml, parse_rank_config
from marketplace_ml.common.io import ensure_dir
from marketplace_ml.common.seed import set_seed
from marketplace_ml.data.split import GroupSplit
from marketplace_ml.features.rank_featurizer import RankFeaturizer, RankFeaturizerConfig
from marketplace_ml.rank.artifacts import RankArtifacts, write_meta
from marketplace_ml.rank.metrics import ndcg_by_group


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = parse_rank_config(load_yaml(args.config))
    paths = cfg.paths
    ensure_dir(paths.artifacts_dir)

    train_path = Path(paths.train_csv)
    test_path = Path(paths.test_csv)
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Processed splits are missing."
            f" Expected: {train_path} and {test_path}."
            f" Run: python -m marketplace_ml.data.prepare_rank_data --config {args.config}"
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    seed = int(cfg.model.random_seed)
    set_seed(seed)

    # внутри train делаем val по группам query_id
    splitter = GroupSplit(
        test_size=cfg.split.test_size,
        random_state=cfg.split.random_state,
        group_col="query_id",
    )
    tr_df, val_df = splitter.split(train_df)

    feat_cfg = RankFeaturizerConfig(
        max_desc_words=cfg.features.text.max_desc_words,
        max_bullet_words=cfg.features.text.max_bullet_words,
        tfidf_max_features=cfg.features.tfidf.max_features,
        tfidf_ngram_range=cfg.features.tfidf.ngram_range,
        numeric_cols=cfg.features.numeric_cols,
        categorical_cols=cfg.features.categorical_cols,
    )

    featurizer = RankFeaturizer(feat_cfg).fit(tr_df)
    X_tr, feature_cols, cat_cols = featurizer.transform(tr_df)
    X_val, _, _ = featurizer.transform(val_df)
    X_test, _, _ = featurizer.transform(test_df)

    train_pool = Pool(X_tr[feature_cols], label=tr_df["relevance"], group_id=tr_df["query_id"], cat_features=cat_cols)
    val_pool = Pool(X_val[feature_cols], label=val_df["relevance"], group_id=val_df["query_id"], cat_features=cat_cols)
    if "relevance" in test_df.columns:
        test_pool = Pool(
            X_test[feature_cols], label=test_df["relevance"], group_id=test_df["query_id"], cat_features=cat_cols
        )
        test_is_labeled = True
    else:
        test_pool = Pool(X_test[feature_cols], group_id=test_df["query_id"], cat_features=cat_cols)
        test_is_labeled = False

    model = CatBoostRanker(
        iterations=cfg.model.iterations,
        learning_rate=cfg.model.learning_rate,
        depth=cfg.model.depth,
        loss_function=cfg.model.loss_function,
        eval_metric=cfg.model.eval_metric,
        random_seed=cfg.model.random_seed,
        task_type=cfg.model.task_type,
        verbose=cfg.model.verbose,
    )
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    val_preds = model.predict(val_pool)
    test_preds = model.predict(test_pool)

    ks = [1, 3, 5, 10, 20]
    metrics = {f"val_ndcg@{k}": ndcg_by_group(val_df, val_preds, k=k) for k in ks}
    if test_is_labeled:
        metrics |= {f"test_ndcg@{k}": ndcg_by_group(test_df, test_preds, k=k) for k in ks}
    else:
        print("[WARN] Test split is unlabeled (missing 'relevance'); skipping test NDCG.")

    arts = RankArtifacts.in_dir(paths.artifacts_dir)
    model.save_model(str(arts.model_path))
    featurizer.save(arts.featurizer_path)
    write_meta(
        arts.meta_path,
        {
            "feature_cols": feature_cols,
            "categorical_cols": cat_cols,
            "featurizer_file": arts.featurizer_path.name,
            "model_file": arts.model_path.name,
            "metrics": metrics,
        },
    )

    print(f"[OK] Saved model: {arts.model_path}")
    print(f"[OK] Saved featurizer: {arts.featurizer_path}")
    for k, v in metrics.items():
        print(f"{k}: {v:.5f}")


if __name__ == "__main__":
    main()

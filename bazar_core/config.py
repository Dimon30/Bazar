from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from logging import getLogger

import yaml

logger = getLogger(__name__)

def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if obj is None:
        return {}
    if not isinstance(obj, dict):
        logger.error(f"Invalid yaml format: {path}")
        raise ValueError("[ERROR] YAML root must be a mapping/dict.")
    return obj


@dataclass(frozen=True)
class PathsConfig:
    raw_csv: str
    train_csv: str
    test_csv: str
    artifacts_dir: str


@dataclass(frozen=True)
class SplitConfig:
    test_size: float = 0.2
    random_state: int = 30


@dataclass(frozen=True)
class TextPrepConfig:
    max_desc_words: int = 80
    max_bullet_words: int = 40


@dataclass(frozen=True)
class TfidfConfig:
    max_features: int = 20_000
    ngram_range: tuple[int, int] = (1, 2)


@dataclass(frozen=True)
class FeaturesConfig:
    text: TextPrepConfig = TextPrepConfig()
    tfidf: TfidfConfig = TfidfConfig()
    numeric_cols: tuple[str, ...] = (
        "bm25_idf_sum",
        "tfidf_cosine",
        "query_len",
        "title_len",
        "desc_len",
        "bullet_len",
    )
    categorical_cols: tuple[str, ...] = ("product_locale", "product_brand", "product_color")


@dataclass(frozen=True)
class CatBoostRankerConfig:
    iterations: int = 200
    learning_rate: float = 0.1
    depth: int = 8
    loss_function: str = "YetiRank"
    eval_metric: str = "NDCG:top=10"
    random_seed: int = 42
    task_type: str = "CPU"
    verbose: int = 100


@dataclass(frozen=True)
class MlflowConfig:
    tracking_uri: str = "mlruns"
    experiment_name: str = "bazar"


@dataclass(frozen=True)
class RankExperimentConfig:
    project_name: str
    task: str
    paths: PathsConfig
    split: SplitConfig = SplitConfig()
    features: FeaturesConfig = FeaturesConfig()
    model: CatBoostRankerConfig = CatBoostRankerConfig()
    mlflow: MlflowConfig = MlflowConfig()


def _require(d: dict[str, Any], key: str) -> Any:
    if key not in d:
        raise KeyError(f"[ERROR] Missing config key: {key}")
    return d[key]


def parse_rank_config(d: dict[str, Any]) -> RankExperimentConfig:
    paths = _require(d, "paths")
    split = d.get("split", {})
    feats = d.get("features", {})
    model = _require(d, "model")

    text = feats.get("text", {})
    tfidf = feats.get("tfidf", {})

    feat_cfg = FeaturesConfig(
        text=TextPrepConfig(
            max_desc_words=int(text.get("max_desc_words", 80)),
            max_bullet_words=int(text.get("max_bullet_words", 40)),
        ),
        tfidf=TfidfConfig(
            max_features=int(tfidf.get("max_features", 20_000)),
            ngram_range=tuple(tfidf.get("ngram_range", [1, 2])),
        ),
        numeric_cols=tuple(feats.get("numeric_cols", FeaturesConfig().numeric_cols)),
        categorical_cols=tuple(feats.get("categorical_cols", FeaturesConfig().categorical_cols)),
    )

    model_cfg = CatBoostRankerConfig(
        iterations=int(model.get("iterations", 200)),
        learning_rate=float(model.get("learning_rate", 0.1)),
        depth=int(model.get("depth", 8)),
        loss_function=str(model.get("loss_function", "YetiRank")),
        eval_metric=str(model.get("eval_metric", "NDCG:top=10")),
        random_seed=int(model.get("random_seed", 42)),
        task_type=str(model.get("task_type", "CPU")),
        verbose=int(model.get("verbose", 100)),
    )

    return RankExperimentConfig(
        project_name=str(_require(d, "project_name")),
        task=str(_require(d, "task")),
        paths=PathsConfig(
            raw_csv=str(_require(paths, "raw_csv")),
            train_csv=str(_require(paths, "train_csv")),
            test_csv=str(_require(paths, "test_csv")),
            artifacts_dir=str(_require(paths, "artifacts_dir")),
        ),
        split=SplitConfig(
            test_size=float(split.get("test_size", 0.2)),
            random_state=int(split.get("random_state", 42)),
        ),
        features=feat_cfg,
        model=model_cfg,
    )


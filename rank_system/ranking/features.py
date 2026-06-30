from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from marketplace_ml.features.text import clean_text, truncate_words


@dataclass(frozen=True)
class RankFeaturizerConfig:
    max_desc_words: int = 80
    max_bullet_words: int = 40
    tfidf_max_features: int = 20_000
    tfidf_ngram_range: tuple[int, int] = (1, 2)

    numeric_cols: tuple[str, ...] = (
        "bm25_idf_sum",
        "tfidf_cosine",
        "query_len",
        "title_len",
        "desc_len",
        "bullet_len",
    )
    categorical_cols: tuple[str, ...] = ("product_locale", "product_brand", "product_color")


class RankFeaturizer:
    VERSION = 1

    def __init__(self, cfg: RankFeaturizerConfig):
        self.cfg = cfg
        self._vectorizer: TfidfVectorizer | None = None
        self._bm25_idf: dict[str, float] | None = None

    @staticmethod
    def _build_product_text(df: pd.DataFrame) -> pd.Series:
        title = df["product_title"].fillna("").astype(str)
        desc = df["product_description"].fillna("").astype(str)
        bullet = df["product_bullet_point"].fillna("").astype(str)
        return (title + " " + bullet + " " + desc).str.strip()

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in ["query", "product_title", "product_description", "product_bullet_point"]:
            df[col] = df[col].fillna("").astype(str)

        df["query_clean"] = df["query"].apply(clean_text)
        df["title_clean"] = df["product_title"].apply(clean_text)
        df["desc_clean"] = df["product_description"].apply(
            lambda x: clean_text(truncate_words(x, self.cfg.max_desc_words))
        )
        df["bullet_clean"] = df["product_bullet_point"].apply(
            lambda x: clean_text(truncate_words(x, self.cfg.max_bullet_words))
        )
        df["product_text_clean"] = self._build_product_text(df)

        df["query_len"] = df["query_clean"].str.len()
        df["title_len"] = df["title_clean"].str.len()
        df["desc_len"] = df["desc_clean"].str.len()
        df["bullet_len"] = df["bullet_clean"].str.len()
        return df

    def fit(self, df_train: pd.DataFrame) -> "RankFeaturizer":
        df_train = self._prepare(df_train)

        vectorizer = TfidfVectorizer(
            max_features=int(self.cfg.tfidf_max_features),
            ngram_range=tuple(self.cfg.tfidf_ngram_range),
        )
        vectorizer.fit(df_train["product_text_clean"])
        self._vectorizer = vectorizer

        corpus = [t.split() for t in df_train["product_text_clean"].fillna("").astype(str).tolist()]
        bm25 = BM25Okapi(corpus)
        self._bm25_idf = {str(k): float(v) for k, v in bm25.idf.items()}
        return self

    def _bm25_idf_sum(self, query_clean: pd.Series, product_text_clean: pd.Series) -> np.ndarray:
        if self._bm25_idf is None:
            raise RuntimeError("Featurizer not fitted: missing BM25 stats.")

        scores: list[float] = []
        idf = self._bm25_idf
        for q, doc in zip(query_clean.fillna("").astype(str), product_text_clean.fillna("").astype(str)):
            q_tok = q.split()
            doc_set = set(doc.split())
            s = 0.0
            for t in q_tok:
                if t in doc_set:
                    s += float(idf.get(t, 0.0))
            scores.append(s)
        return np.asarray(scores, dtype=float)

    def _tfidf_cosine(self, query_clean: pd.Series, product_text_clean: pd.Series) -> np.ndarray:
        if self._vectorizer is None:
            raise RuntimeError("Featurizer not fitted: missing TF-IDF vectorizer.")

        q = self._vectorizer.transform(query_clean.fillna("").astype(str))
        d = self._vectorizer.transform(product_text_clean.fillna("").astype(str))
        sim = d.multiply(q).sum(axis=1)
        return np.asarray(sim).reshape(-1)

    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
        df_p = self._prepare(df)

        df_p["bm25_idf_sum"] = self._bm25_idf_sum(df_p["query_clean"], df_p["product_text_clean"])
        df_p["tfidf_cosine"] = self._tfidf_cosine(df_p["query_clean"], df_p["product_text_clean"])

        numeric_cols = list(self.cfg.numeric_cols)
        cat_cols = list(self.cfg.categorical_cols)
        for c in cat_cols:
            if c in df_p.columns:
                df_p[c] = df_p[c].fillna("").astype(str)
            else:
                df_p[c] = ""

        feature_cols = numeric_cols + cat_cols
        X = df_p[feature_cols].copy()
        return X, feature_cols, cat_cols

    def save(self, path: str | Path) -> None:
        if self._vectorizer is None or self._bm25_idf is None:
            raise RuntimeError("Cannot save an unfitted featurizer.")

        from joblib import dump

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        dump(
            {
                "version": self.VERSION,
                "cfg": self.cfg,
                "vectorizer": self._vectorizer,
                "bm25_idf": self._bm25_idf,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "RankFeaturizer":
        from joblib import load

        obj = load(Path(path))
        if int(obj.get("version", -1)) != cls.VERSION:
            raise ValueError(f"Unsupported featurizer version: {obj.get('version')}")

        cfg = obj["cfg"]
        if not isinstance(cfg, RankFeaturizerConfig):
            raise ValueError("Featurizer config has unexpected type.")

        inst = cls(cfg)
        inst._vectorizer = obj["vectorizer"]
        inst._bm25_idf = obj["bm25_idf"]
        return inst


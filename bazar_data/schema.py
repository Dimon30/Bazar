from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class RankDatasetSchema:
    id_col: str = "id"
    group_col: str = "query_id"
    target_col: str = "relevance"

    query_col: str = "query"
    title_col: str = "product_title"
    desc_col: str = "product_description"
    bullet_col: str = "product_bullet_point"

    # Optional categorical attributes (may be missing in future datasets).
    brand_col: str = "product_brand"
    color_col: str = "product_color"
    locale_col: str = "product_locale"

    def required_cols(self) -> tuple[str, ...]:
        return (
            self.group_col,
            self.query_col,
            self.title_col,
            self.desc_col,
            self.bullet_col,
        )

    def required_train_cols(self) -> tuple[str, ...]:
        return (self.target_col,) + self.required_cols()

    def validate_inference(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.required_cols() if c not in df.columns]
        if missing:
            raise ValueError(f"[ERROR] Missing required columns: {missing}")

        if df[self.group_col].isna().any():
            raise ValueError(f"[ERROR] Group column '{self.group_col}' contains NA.")

    def validate_train(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.required_train_cols() if c not in df.columns]
        if missing:
            raise ValueError(f"[ERROR] Missing required columns: {missing}")
        self.validate_inference(df)

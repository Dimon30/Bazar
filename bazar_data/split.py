from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


@dataclass(frozen=True)
class GroupSplit:
    test_size: float = 0.2
    random_state: int = 42
    group_col: str = "query_id"

    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self.group_col not in df.columns:
            raise ValueError(f"[ERROR] Missing group column: {self.group_col}")

        splitter = GroupShuffleSplit(
            n_splits=1, test_size=float(self.test_size), random_state=int(self.random_state)
        )
        train_idx, test_idx = next(splitter.split(df, groups=df[self.group_col]))
        train = df.iloc[train_idx].copy()
        test = df.iloc[test_idx].copy()

        tr_groups = set(train[self.group_col].astype(str).unique())
        te_groups = set(test[self.group_col].astype(str).unique())
        overlap = tr_groups.intersection(te_groups)
        if overlap:
            raise RuntimeError(f"[ERROR] Group leakage detected: overlap={len(overlap)}")

        return train, test


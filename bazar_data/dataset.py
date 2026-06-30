from pathlib import Path
from bazar_core.logging import get_logger

import pandas as pd

logger = get_logger(__name__)


def load_raw_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        logger.error(f"Raw data file not found: {path}")
        raise FileNotFoundError(f"Raw data file not found: {path}")

    logger.info("Loading raw data from %s", path)

    if path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix == ".parquet":
        df = pd.read_parquet(path, engine="pyarrow")
    else:
        logger.error(f"Unsupported raw data format: {path.suffix}")
        raise ValueError(f"Unsupported raw data format: {path.suffix}")

    logger.info("Loaded raw data: %s rows, %s columns", len(df), len(df.columns))
    return df


def save_processed_data(
    df: pd.DataFrame,
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Saving processed data to %s", path)

    if path.suffix == ".csv":
        df.to_csv(path, index=False)
    elif path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        logger.error(f"Unsupported raw data format: {path.suffix}")
        raise ValueError(f"Unsupported processed data format: {path.suffix}")

    logger.info("Processed data saved: %s rows", len(df))


def validate_dataset(
    df: pd.DataFrame,
    required_columns: list[str],
) -> None:
    logger.info("Validating dataset")

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        logger.error(f"Missing columns: {missing_columns}")
        raise ValueError(f"Missing required columns: {missing_columns}")

    if df.empty:
        logger.error(f"Empty dataset: {df}")
        raise ValueError("Dataset is empty")

    logger.info("Dataset validation passed")
"""Воспроизводимая загрузка публичного датасета Amazon ESCI."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import pandas as pd

from bazar_data.database import Database
from bazar_data.marketplace_schema import products, queries, relevance_judgments


ESCI_GAINS: dict[str, int] = {"E": 3, "S": 2, "C": 1, "I": 0}
DATASET_NAME = "amazon-shopping-queries-esci"


@dataclass(frozen=True)
class IngestSummary:
    """Короткий отчёт о результате одной загрузки."""

    products: int
    queries: int
    judgments: int
    data_hash: str


@dataclass(frozen=True)
class EsciIngestConfig:
    """Неизменяемые настройки одного воспроизводимого запуска."""

    locale: str = "us"
    max_queries: int | None = 2_000


def clean_text(value: object) -> str:
    """Вернуть строку без внешних пробелов; отсутствующее значение - пустая строка."""
    if value is None or pd.isna(value):
        return ""

    return str(value).strip()



def stable_query_sample(rows: pd.DataFrame, max_queries: int | None) -> pd.DataFrame:
    """Оставить все строки первых ``max_queries`` query_id в лексикографическом порядке.

    При ``None`` набор строк не ограничивается. Функция не должна менять
    переданный DataFrame.

    """
    if max_queries is None:
        return rows.copy()
    if max_queries < 1:
        raise ValueError("max_queries must be positive or None.")
    query_ids = sorted(rows["query_id"].dropna().astype(str).unique())[:max_queries]
    return rows[rows["query_id"].astype(str).isin(query_ids)].copy()


def read_esci(
    examples_path: Path,
    products_path: Path,
    config: EsciIngestConfig,
) -> pd.DataFrame:
    """Прочитать, отфильтровать и соединить examples и catalog ESCI.

    Результат обязан содержать исходные поля examples и поля catalog. Нужно:
    выбрать ``small_version == 1`` и ``product_locale == config.locale``;
    применить stable_query_sample; проверить непустой результат, допустимость
    ESCI-меток и наличие продукта для каждой оценки релевантности.

    """
    examples = pd.read_parquet(
        examples_path,
        filters=[("small_version", "==", 1), ("product_locale", "==", config.locale)],
    )
    examples = stable_query_sample(examples, config.max_queries)
    if examples.empty:
        raise ValueError("ESCI slice is empty for the selected locale and query limit.")

    unknown_labels = sorted(set(examples["esci_label"].dropna()) - set(ESCI_GAINS))
    if unknown_labels:
        raise ValueError(f"Unsupported ESCI labels: {unknown_labels}")

    catalog = pd.read_parquet(
        products_path,
        filters=[("product_locale", "==", config.locale)],
    )
    catalog = catalog[catalog["product_id"].astype(str).isin(set(examples["product_id"].astype(str)))]
    catalog = catalog.drop_duplicates(["product_id", "product_locale"], keep="first")
    rows = examples.merge(
        catalog,
        on=["product_id", "product_locale"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    missing_products = rows.loc[rows["_merge"] != "both", "product_id"].astype(str).unique()
    if len(missing_products):
        preview = ", ".join(missing_products[:5])
        raise ValueError(f"Catalog rows are missing for {len(missing_products)} products: {preview}")
    return rows.drop(columns="_merge")


def ingest_esci(
    database: Database,
    examples_path: str | Path,
    products_path: str | Path,
    *,
    locale: str = "us",
    max_queries: int | None = 2_000,
) -> IngestSummary:
    """Загрузить один срез ESCI в contract базы данных.

    Перед upsert нужно вызвать ``database.create_schema()``. Целевые таблицы
    и ключи: ``products/product_id``, ``queries/query_id`` и
    ``relevance_judgments/example_id``. Повторный одинаковый запуск не должен
    создавать дубликаты; ``data_hash`` зависит только от отсортированных
    ``example_id, query_id, product_id, esci_label, split``.

    """
    database.create_schema()
    rows = read_esci(
        Path(examples_path),
        Path(products_path),
        EsciIngestConfig(locale=locale, max_queries=max_queries),
    )

    text_columns = [
        "product_title",
        "product_brand",
        "product_description",
        "product_bullet_point",
        "product_color",
    ]
    for column in text_columns + ["query"]:
        rows[column] = rows[column].map(clean_text)

    product_rows: list[dict[str, object]] = []
    for row in rows.drop_duplicates("product_id").itertuples(index=False):
        parts = [getattr(row, column) for column in text_columns]
        product_rows.append({
            "product_id": str(row.product_id),
            "locale": str(row.product_locale),
            "title": row.product_title,
            "description": row.product_description,
            "bullet_point": row.product_bullet_point,
            "brand": row.product_brand,
            "color": row.product_color,
            "catalog_text": " ".join(part for part in parts if part),
            "source_dataset": DATASET_NAME,
        })

    query_rows: list[dict[str, object]] = []
    for row in rows.drop_duplicates("query_id").itertuples(index=False):
        query_rows.append({
            "query_id": str(row.query_id),
            "query_text": row.query,
            "locale": str(row.product_locale),
            "split": str(row.split),
            "source": "esci",
            "source_dataset": DATASET_NAME,
        })

    judgment_rows: list[dict[str, object]] = []
    for row in rows.itertuples(index=False):
        label = str(row.esci_label)
        judgment_rows.append({
            "example_id": str(row.example_id),
            "query_id": str(row.query_id),
            "product_id": str(row.product_id),
            "esci_label": label,
            "relevance_gain": ESCI_GAINS[label],
            "split": str(row.split),
            "source_dataset": DATASET_NAME,
        })

    hash_columns = ["example_id", "query_id", "product_id", "esci_label", "split"]
    hash_frame = pd.DataFrame(judgment_rows)[hash_columns].sort_values(hash_columns)
    data_hash = sha256(hash_frame.to_csv(index=False).encode("utf-8")).hexdigest()

    database.upsert(products, product_rows, key_columns=("product_id",))
    database.upsert(queries, query_rows, key_columns=("query_id",))
    database.upsert(relevance_judgments, judgment_rows, key_columns=("example_id",))
    return IngestSummary(
        products=len(product_rows),
        queries=len(query_rows),
        judgments=len(judgment_rows),
        data_hash=data_hash,
    )

"""Small SQLAlchemy gateway used by CLI, API and offline jobs."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

import pandas as pd
from sqlalchemy import Engine, Table, create_engine, text
from sqlalchemy.engine import Connection
from sqlalchemy.exc import IntegrityError

from bazar_data.marketplace_schema import metadata


class Database:
    """Database URL wrapper with portable schema creation and idempotent writes.

    A filesystem path remains accepted for local notebooks; production uses a
    SQLAlchemy URL such as ``postgresql+psycopg://bazar:bazar@db:5432/bazar``.
    """

    def __init__(self, url_or_path: str | Path):
        raw = str(url_or_path)
        if "://" in raw:
            self.url = raw
        else:
            path = Path(raw)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.url = f"sqlite:///{path}"
        self.engine: Engine = create_engine(self.url, pool_pre_ping=True)

    @contextmanager
    def connection(self) -> Iterator[Connection]:
        with self.engine.begin() as conn:
            yield conn

    def create_schema(self) -> None:
        metadata.create_all(self.engine)

    def execute(self, query: str, params: dict[str, Any] | None = None) -> None:
        with self.connection() as conn:
            conn.execute(text(query), params or {})

    def fetch_one(self, query: str, params: dict[str, Any] | None = None) -> tuple[Any, ...] | None:
        with self.connection() as conn:
            row = conn.execute(text(query), params or {}).first()
            return tuple(row) if row is not None else None

    def fetch_all(self, query: str, params: dict[str, Any] | None = None) -> list[tuple[Any, ...]]:
        with self.connection() as conn:
            return [tuple(row) for row in conn.execute(text(query), params or {}).all()]

    def read_dataframe(self, query: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
        with self.engine.connect() as conn:
            return pd.read_sql_query(text(query), conn, params=params or {})

    def write_dataframe(self, df: pd.DataFrame, table_name: str, if_exists: str = "append") -> None:
        with self.engine.begin() as conn:
            df.to_sql(table_name, conn, if_exists=if_exists, index=False)

    def upsert(self, table: Table, rows: Sequence[dict[str, Any]], *, key_columns: tuple[str, ...]) -> None:
        """Portable insert-or-update without silently losing a batch on conflict."""
        if not rows:
            return
        update_columns = [column.name for column in table.columns if column.name not in key_columns and column.name != "ingested_at"]
        with self.connection() as conn:
            for row in rows:
                where = {key: row[key] for key in key_columns}
                values = {column: row[column] for column in update_columns if column in row}
                result = conn.execute(table.update().where(*[table.c[key] == value for key, value in where.items()]).values(**values))
                if result.rowcount == 0:
                    try:
                        conn.execute(table.insert().values(**row))
                    except IntegrityError:
                        conn.execute(table.update().where(*[table.c[key] == value for key, value in where.items()]).values(**values))

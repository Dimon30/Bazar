import sqlite3
from pathlib import Path
from logging import getLogger
from contextlib import contextmanager

import pandas as pd

logger = getLogger(__name__)

""" USING:
db = Database("data/prod/bazar.db")

db.execute('''
CREATE TABLE IF NOT EXISTS products (
    product_id INTEGER PRIMARY KEY,
    title TEXT,
    price REAL
)
''')

db.execute(
    "INSERT INTO products(title, price) VALUES (?, ?)",
    ("iPhone", 1000)
)

rows = db.fetch_all(
    "SELECT * FROM products"
)

print(rows)

--------------------------------------------------------

products_df = db.read_dataframe(
    "SELECT * FROM products"
)

db.write_dataframe(
    interactions_df,
    "user_events"
)
"""

class Database:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)

    @contextmanager
    def connection(self):
        conn = sqlite3.connect(self.db_path)

        try:
            yield conn
            conn.commit()
            logger.info(f"Connection to {self.db_path} successful")
        finally:
            conn.close()

    def execute(self, query: str, params: tuple = ()) -> None:
        with self.connection() as conn:
            conn.execute(query, params)

    def fetch_one(self, query: str, params: tuple = ()) -> tuple | None:
        with self.connection() as conn:
            cursor = conn.execute(query, params)
            logger.info(f"Fetching {query} successful")
            return cursor.fetchone()

    def fetch_all(self, query: str, params: tuple = ()) -> list[tuple]:
        with self.connection() as conn:
            cursor = conn.execute(query, params)
            logger.info(f"Fetching {query} successful")
            return cursor.fetchall()
        logger.error(f"Fetching {query} failed")

    def read_dataframe(self, query: str) -> pd.DataFrame:
        with self.connection() as conn:
            df = pd.read_sql_query(query, conn)
            logger.info(f"Reading {query} successful")
            return df
        logger.error(f"Reading {query} failed")

    def write_dataframe(
            self,
            df: pd.DataFrame,
            table_name: str,
            if_exists: str = "append",
    ) -> None:
        with self.connection() as conn:
            df.to_sql(
                table_name,
                conn,
                if_exists=if_exists,
                index=False,
            )
            logger.info(f"Writing {table_name} successful")
        # logger.error(f"Writing {table_name} failed")
"""Storage contract for the Bazar search demonstration.

The catalog is the source of truth.  Retrieval indexes and ML artifacts are
derived data and can always be rebuilt from these tables.
"""

from __future__ import annotations

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    func,
)


metadata = MetaData()

products = Table(
    "products",
    metadata,
    Column("product_id", String(128), primary_key=True),
    Column("locale", String(16), nullable=False),
    Column("title", Text, nullable=False),
    Column("description", Text, nullable=False, server_default=""),
    Column("bullet_point", Text, nullable=False, server_default=""),
    Column("brand", String(512), nullable=False, server_default=""),
    Column("color", String(256), nullable=False, server_default=""),
    Column("catalog_text", Text, nullable=False),
    Column("source_dataset", String(128), nullable=False),
    Column("ingested_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
)

queries = Table(
    "queries",
    metadata,
    Column("query_id", String(128), primary_key=True),
    Column("query_text", Text, nullable=False),
    Column("locale", String(16), nullable=False),
    Column("split", String(16), nullable=False),
    Column("source", String(64), nullable=False, server_default="unknown"),
    Column("source_dataset", String(128), nullable=False),
    Column("ingested_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
)

relevance_judgments = Table(
    "relevance_judgments",
    metadata,
    Column("example_id", String(128), primary_key=True),
    Column("query_id", String(128), nullable=False, index=True),
    Column("product_id", String(128), nullable=False, index=True),
    Column("esci_label", String(1), nullable=False),
    Column("relevance_gain", Integer, nullable=False),
    Column("split", String(16), nullable=False),
    Column("source_dataset", String(128), nullable=False),
    Column("ingested_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    UniqueConstraint("query_id", "product_id", name="uq_judgment_query_product"),
)

search_events = Table(
    "search_events",
    metadata,
    Column("event_id", String(36), primary_key=True),
    Column("request_id", String(36), nullable=False, index=True),
    Column("event_type", String(32), nullable=False),
    Column("query_text", Text, nullable=False),
    Column("product_id", String(128), nullable=False, index=True),
    Column("position", Integer, nullable=False),
    Column("model_version", String(128), nullable=False),
    Column("is_simulated", Boolean, nullable=False, server_default="false"),
    Column("event_metadata", JSON, nullable=False, server_default="{}"),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
)

model_versions = Table(
    "model_versions",
    metadata,
    Column("model_version", String(128), primary_key=True),
    Column("model_type", String(64), nullable=False),
    Column("artifact_uri", Text, nullable=False),
    Column("metrics", JSON, nullable=False),
    Column("data_hash", String(128), nullable=False),
    Column("is_active", Boolean, nullable=False, server_default="false"),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
)

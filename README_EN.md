# Bazar - product search and ranking

Bazar is my product search and ranking project. I use the public [Amazon Shopping Queries (ESCI)](https://github.com/amazon-science/esci-data) dataset and cover the full path from data ingestion to search through an API. The project is still in development.

## Business problem

I built an MVP for catalog search and needed to select the highest-quality model with p95 below 100 ms, connect it to an API and prepare search-event collection.

I use 100 ms as an initial engineering budget for the retrieval stage, not as a business-validated requirement. It leaves time for API processing, storage access, network transfer and rendering. A real product would need to adjust this limit using load tests and user metrics.

I do not claim a CTR or conversion uplift because that would require real user traffic and an A/B test. The current result is a working MVP and a model choice supported by public relevance labels.

## What I implemented

- ESCI product, query and relevance data ingestion;
- validation, deduplication and a stable data hash;
- BM25 search and a compact NumPy two-tower model;
- common evaluation with NDCG@10, MRR@10, Recall@10/50/100 and response time;
- active model selection and a database model registry;
- MLflow experiment logging;
- FastAPI search, event and model routes;
- Docker Compose and tests for the main search flow.

## Result

I used 2,000 queries from ESCI `small_version=1`, locale `us`. The resulting database contains 37,971 products and 39,753 relevance judgments. I evaluated both models on 606 positive test queries.

| Model | NDCG@10 | MRR@10 | Recall@10 | Recall@50 | Recall@100 | p50, ms | p95, ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | **0.509** | **0.762** | **0.889** | **0.941** | **0.960** | 34.22 | 69.27 |
| Two-tower | 0.196 | 0.411 | 0.584 | 0.726 | 0.771 | **17.30** | **18.75** |

I selected BM25 because its search quality is much higher and its 69.27 ms p95 stays within the 100 ms budget. The API serves the selected model and stores result events for future training.

## Storage

I use one database per run. The `--database-url` option selects the database for data preparation, and `DATABASE_URL` selects it for the API.

- Local runs use the SQLite file `data/prod/bazar.db` by default.
- Docker Compose runs use PostgreSQL.

In both cases products go to `products`, queries to `queries`, and relevance labels to `relevance_judgments`. Data is not written to SQLite and PostgreSQL at the same time. Model preparation and the API must use the same database address.

## Quick start with SQLite

Install Python 3.11+ and [uv](https://docs.astral.sh/uv/). Download the ESCI examples and products parquet files into `data/raw/esci/`, then run:

```bash
uv sync
uv run bazar run-demo \
  --database-url data/prod/bazar.db \
  --examples data/raw/esci/shopping_queries_dataset_examples.parquet \
  --products data/raw/esci/shopping_queries_dataset_products.parquet \
  --output-dir artifacts/release \
  --model-version esci-release-v1 \
  --max-queries 2000 \
  --epochs 20
DATABASE_URL=data/prod/bazar.db uv run uvicorn apps.api.main:app --reload
```

Run the tests with:

```bash
uv run --with pytest pytest -q
```

## Limitations

I measured the models on public relevance labels, so the result does not prove an increase in CTR, conversion or revenue. Response time depends on the machine. The current two-tower model is a small reference implementation without a pretrained transformer or an approximate nearest-neighbor index.

## Stack

Python, NumPy, pandas, SQLAlchemy, PostgreSQL, SQLite, FastAPI, MLflow, Docker Compose and pytest.

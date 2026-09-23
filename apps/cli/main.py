from pathlib import Path
import argparse
import json

from bazar_core.logging import get_logger
from bazar_core.registry import mlflow_log_artifact, mlflow_log_metrics, mlflow_run
from bazar_data.dataset import (
    load_raw_data,
    save_processed_data,
    validate_dataset,
)
from bazar_data.database import Database
from bazar_data.ingest import ingest_esci
from bazar_data.model_registry import register_model
from rank_system.retrieval.service import build_bm25_artifact, evaluate_bm25_artifact
from rank_system.two_tower.service import evaluate_two_tower_artifact, train_two_tower_artifact
from rank_system.evaluation.comparison import compare_models

logger = get_logger(__name__)


def prepare_data(args: argparse.Namespace) -> None:
    df = load_raw_data(args.input)

    validate_dataset(
        df,
        required_columns=args.required_columns,
    )

    save_processed_data(df, args.output)

    logger.info("Data preparation completed")


def validate_data(args: argparse.Namespace) -> None:
    df = load_raw_data(args.input)

    validate_dataset(
        df,
        required_columns=args.required_columns,
    )

    logger.info("Dataset is valid")


def init_db(args: argparse.Namespace) -> None:
    Database(args.database_url).create_schema()
    logger.info("Database schema is ready: %s", args.database_url)


def ingest_esci_data(args: argparse.Namespace) -> None:
    database = Database(args.database_url)
    database.create_schema()
    summary = ingest_esci(
        database,
        args.examples,
        args.products,
        locale=args.locale,
        max_queries=args.max_queries or None,
    )
    logger.info(
        "ESCI ingest complete: products=%d queries=%d judgments=%d data_hash=%s",
        summary.products, summary.queries, summary.judgments, summary.data_hash,
    )


def build_bm25(args: argparse.Namespace) -> None:
    index = build_bm25_artifact(Database(args.database_url), args.output)
    logger.info("BM25 index written: products=%d path=%s", len(index.product_ids), args.output)


def evaluate_candidates(args: argparse.Namespace) -> None:
    report = evaluate_bm25_artifact(Database(args.database_url), args.index, args.report)
    logger.info("BM25 candidate report written: %s", args.report)
    logger.info("Candidate metrics: %s", report["metrics"])


def train_two_tower(args: argparse.Namespace) -> None:
    result = train_two_tower_artifact(
        Database(args.database_url), args.output_dir, dimension=args.dimension, epochs=args.epochs,
        batch_size=args.batch_size, learning_rate=args.learning_rate, temperature=args.temperature, seed=args.seed,
    )
    logger.info("Two-tower training complete: %s", result)


def evaluate_two_tower(args: argparse.Namespace) -> None:
    report = evaluate_two_tower_artifact(Database(args.database_url), args.artifact_dir, args.report)
    logger.info("Two-tower report written: %s", args.report)
    logger.info("Two-tower metrics: %s", report["metrics"])


def compare_retrievers(args: argparse.Namespace) -> None:
    decision = compare_models(
        args.bm25_report, args.two_tower_report, args.output,
        min_ndcg_gain=args.min_ndcg_gain, max_p95_latency_ms=args.max_p95_latency_ms,
    )
    logger.info("Selected retriever: %s", decision["selected_model_type"])
    logger.info("Decision: %s", decision["product_rationale"])


def run_demo(args: argparse.Namespace) -> None:
    database = Database(args.database_url)
    summary = ingest_esci(
        database,
        args.examples,
        args.products,
        locale=args.locale,
        max_queries=args.max_queries or None,
    )
    output = Path(args.output_dir)
    bm25_artifact = output / "bm25.json"
    bm25_report = output / "bm25_report.json"
    two_tower_artifact = output / "two_tower"
    two_tower_report = output / "two_tower_report.json"
    decision_path = output / "model_decision.json"

    build_bm25_artifact(database, bm25_artifact)
    baseline = evaluate_bm25_artifact(database, bm25_artifact, bm25_report)
    train_two_tower_artifact(
        database,
        two_tower_artifact,
        dimension=args.dimension,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        seed=args.seed,
    )
    candidate = evaluate_two_tower_artifact(database, two_tower_artifact, two_tower_report)
    decision = compare_models(
        bm25_report,
        two_tower_report,
        decision_path,
        min_ndcg_gain=args.min_ndcg_gain,
        max_p95_latency_ms=args.max_p95_latency_ms,
    )
    selected_type = str(decision["selected_model_type"])
    selected_metrics = baseline["metrics"] if selected_type == "bm25" else candidate["metrics"]
    with mlflow_run(
        "bazar-search",
        args.model_version,
        params={
            "locale": args.locale,
            "max_queries": args.max_queries,
            "dimension": args.dimension,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "temperature": args.temperature,
            "seed": args.seed,
        },
    ) as mlflow_context:
        mlflow_log_metrics(mlflow_context, {f"bm25_{key}": value for key, value in baseline["metrics"].items()})
        mlflow_log_metrics(mlflow_context, {f"two_tower_{key}": value for key, value in candidate["metrics"].items()})
        mlflow_log_artifact(mlflow_context, str(bm25_report))
        mlflow_log_artifact(mlflow_context, str(two_tower_report))
        mlflow_log_artifact(mlflow_context, str(decision_path))
    register_model(
        database,
        version=args.model_version,
        model_type=selected_type,
        artifact_uri=Path(str(decision["selected_artifact"])),
        metrics=selected_metrics,
        data_hash=summary.data_hash,
        activate=True,
    )
    logger.info("Demo pipeline complete. Active model: %s (%s)", args.model_version, selected_type)
    logger.info("Decision report: %s", decision_path)
    print(json.dumps({
        "data": summary.__dict__,
        "active_model": {"version": args.model_version, "type": selected_type},
        "bm25": baseline["metrics"],
        "two_tower": candidate["metrics"],
        "decision": str(decision_path),
    }, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bazar",
        description="CLI for Bazar ML marketplace project",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    prepare_parser = subparsers.add_parser(
        "prepare-data",
        help="Load raw data, validate it and save processed data",
    )
    prepare_parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to raw dataset",
    )
    prepare_parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path to processed dataset",
    )
    prepare_parser.add_argument(
        "--required-columns",
        nargs="+",
        required=True,
        help="Required columns for dataset validation",
    )
    prepare_parser.set_defaults(func=prepare_data)

    validate_parser = subparsers.add_parser(
        "validate-data",
        help="Validate dataset by required columns",
    )
    validate_parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to dataset",
    )
    validate_parser.add_argument(
        "--required-columns",
        nargs="+",
        required=True,
        help="Required columns for dataset validation",
    )
    validate_parser.set_defaults(func=validate_data)

    init_db_parser = subparsers.add_parser("init-db", help="Create the Bazar marketplace schema")
    init_db_parser.add_argument("--database-url", required=True, help="SQLAlchemy database URL or SQLite path")
    init_db_parser.set_defaults(func=init_db)

    ingest_parser = subparsers.add_parser("ingest-esci", help="Load a deterministic ESCI catalog slice")
    ingest_parser.add_argument("--database-url", required=True, help="SQLAlchemy database URL or SQLite path")
    ingest_parser.add_argument("--examples", required=True, type=Path, help="ESCI examples parquet")
    ingest_parser.add_argument("--products", required=True, type=Path, help="ESCI products parquet")
    ingest_parser.add_argument("--locale", default="us", help="ESCI product locale (default: us)")
    ingest_parser.add_argument("--max-queries", type=int, default=2000, help="Stable query cap; 0 means all")
    ingest_parser.set_defaults(func=ingest_esci_data)

    build_bm25_parser = subparsers.add_parser("build-bm25-index", help="Build a BM25 candidate artifact from catalog")
    build_bm25_parser.add_argument("--database-url", required=True, help="SQLAlchemy database URL or SQLite path")
    build_bm25_parser.add_argument("--output", required=True, type=Path, help="Path to JSON BM25 artifact")
    build_bm25_parser.set_defaults(func=build_bm25)

    evaluate_parser = subparsers.add_parser("evaluate-candidates", help="Evaluate BM25 candidate recall on source test rows")
    evaluate_parser.add_argument("--database-url", required=True, help="SQLAlchemy database URL or SQLite path")
    evaluate_parser.add_argument("--index", required=True, type=Path, help="Path to JSON BM25 artifact")
    evaluate_parser.add_argument("--report", required=True, type=Path, help="Output JSON metrics report")
    evaluate_parser.set_defaults(func=evaluate_candidates)

    train_two_tower_parser = subparsers.add_parser("train-two-tower", help="Train query/product towers with in-batch negatives")
    train_two_tower_parser.add_argument("--database-url", required=True, help="SQLAlchemy database URL or SQLite path")
    train_two_tower_parser.add_argument("--output-dir", required=True, type=Path, help="Directory for model and dense index artifacts")
    train_two_tower_parser.add_argument("--dimension", type=int, default=64)
    train_two_tower_parser.add_argument("--epochs", type=int, default=15)
    train_two_tower_parser.add_argument("--batch-size", type=int, default=64)
    train_two_tower_parser.add_argument("--learning-rate", type=float, default=0.08)
    train_two_tower_parser.add_argument("--temperature", type=float, default=0.1)
    train_two_tower_parser.add_argument("--seed", type=int, default=42)
    train_two_tower_parser.set_defaults(func=train_two_tower)

    evaluate_two_tower_parser = subparsers.add_parser("evaluate-two-tower", help="Evaluate dense retrieval and ranking on source test rows")
    evaluate_two_tower_parser.add_argument("--database-url", required=True, help="SQLAlchemy database URL or SQLite path")
    evaluate_two_tower_parser.add_argument("--artifact-dir", required=True, type=Path, help="Directory from train-two-tower")
    evaluate_two_tower_parser.add_argument("--report", required=True, type=Path, help="Output JSON metrics report")
    evaluate_two_tower_parser.set_defaults(func=evaluate_two_tower)

    compare_parser = subparsers.add_parser("compare-retrievers", help="Select active retriever from offline quality/latency reports")
    compare_parser.add_argument("--bm25-report", required=True, type=Path)
    compare_parser.add_argument("--two-tower-report", required=True, type=Path)
    compare_parser.add_argument("--output", required=True, type=Path)
    compare_parser.add_argument("--min-ndcg-gain", type=float, default=0.01)
    compare_parser.add_argument("--max-p95-latency-ms", type=float, default=100.0)
    compare_parser.set_defaults(func=compare_retrievers)

    demo_parser = subparsers.add_parser("run-demo", help="Run ingestion, training, evaluation and activate the selected retriever")
    demo_parser.add_argument("--database-url", default="data/prod/bazar.db")
    demo_parser.add_argument("--examples", required=True, type=Path)
    demo_parser.add_argument("--products", required=True, type=Path)
    demo_parser.add_argument("--output-dir", default="artifacts/demo")
    demo_parser.add_argument("--model-version", default="demo-v1")
    demo_parser.add_argument("--locale", default="us")
    demo_parser.add_argument("--max-queries", type=int, default=500)
    demo_parser.add_argument("--dimension", type=int, default=64)
    demo_parser.add_argument("--epochs", type=int, default=15)
    demo_parser.add_argument("--batch-size", type=int, default=64)
    demo_parser.add_argument("--learning-rate", type=float, default=0.08)
    demo_parser.add_argument("--temperature", type=float, default=0.1)
    demo_parser.add_argument("--seed", type=int, default=42)
    demo_parser.add_argument("--min-ndcg-gain", type=float, default=0.01)
    demo_parser.add_argument("--max-p95-latency-ms", type=float, default=100.0)
    demo_parser.set_defaults(func=run_demo)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

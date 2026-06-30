from pathlib import Path
import argparse

from bazar_core.logging import get_logger
from bazar_data.dataset import (
    load_raw_data,
    save_processed_data,
    validate_dataset,
)

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

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
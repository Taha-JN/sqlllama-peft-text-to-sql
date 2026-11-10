"""
SQLglot AST Dataset Cleaner and Validator.

Parses, normalizes, and validates SQL queries from text-to-SQL datasets
using sqlglot Abstract Syntax Tree (AST) validation. Invalid syntax queries
are dropped, and valid queries are standardized.
"""

import argparse
import json
import logging
import os
from typing import Any, Dict, List, Optional
import sqlglot
from sqlglot.errors import ParseError

logger = logging.getLogger(__name__)


class SQLValidator:
    """Validates and normalizes SQL queries using sqlglot AST parsing."""

    def __init__(self, read_dialect: str = "sqlite", write_dialect: str = "sqlite"):
        self.read_dialect = read_dialect
        self.write_dialect = write_dialect

    def normalize_and_validate(self, raw_sql: str) -> Optional[str]:
        """
        Parses and formats SQL. Returns normalized SQL string or None if invalid.

        Args:
            raw_sql: The raw SQL string to parse.

        Returns:
            Normalized SQL string if valid, None otherwise.
        """
        if not raw_sql or not isinstance(raw_sql, str) or not raw_sql.strip():
            return None

        try:
            expressions = sqlglot.parse(raw_sql.strip(), read=self.read_dialect)
            if not expressions or expressions[0] is None:
                return None
            # Standardize and format AST expression back to SQL dialect
            normalized_sql = expressions[0].sql(dialect=self.write_dialect, pretty=False)
            return normalized_sql
        except ParseError as err:
            logger.debug("SQL parse error for query '%s...': %s", raw_sql[:50].strip(), err)
            return None
        except Exception as err:
            logger.debug("Unexpected error parsing query '%s...': %s", raw_sql[:50].strip(), err)
            return None


def clean_and_validate_dataset(
    input_path: str,
    output_path: str,
    read_dialect: str = "sqlite",
    write_dialect: str = "sqlite",
) -> List[Dict[str, Any]]:
    """
    Cleans a dataset of SQL instruction entries by validating each query's AST.

    Args:
        input_path: Path to raw input JSON dataset.
        output_path: Path to write validated and normalized JSON dataset.
        read_dialect: Source SQL dialect for parsing.
        write_dialect: Target SQL dialect for normalization.

    Returns:
        List of validated and normalized dataset records.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input dataset file not found: {input_path}")

    logger.info("Loading dataset from: %s", input_path)
    with open(input_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if not isinstance(dataset, list):
        raise ValueError(f"Expected a list of dataset records, got {type(dataset)}")

    validator = SQLValidator(read_dialect=read_dialect, write_dialect=write_dialect)
    cleaned_records: List[Dict[str, Any]] = []
    dropped_count = 0
    max_char_length = 0

    logger.info("Validating %d records with dialect '%s'...", len(dataset), read_dialect)

    for idx, entry in enumerate(dataset):
        raw_sql = entry.get("output", "")
        normalized_sql = validator.normalize_and_validate(raw_sql)

        if normalized_sql is not None:
            entry_copy = dict(entry)
            entry_copy["output"] = normalized_sql
            cleaned_records.append(entry_copy)

            # Track sequence length
            total_chars = (
                len(entry_copy.get("instruction", ""))
                + len(entry_copy.get("input", ""))
                + len(normalized_sql)
            )
            if total_chars > max_char_length:
                max_char_length = total_chars
        else:
            dropped_count += 1
            if dropped_count <= 5 or dropped_count % 1000 == 0:
                logger.warning(
                    "Dropped invalid query at index %d: '%s...'",
                    idx,
                    raw_sql[:60].replace("\n", " ").strip(),
                )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_records, f, indent=2)

    logger.info("================ AST Cleaning Summary ================")
    logger.info("Total input queries     : %d", len(dataset))
    logger.info("Valid normalized queries: %d", len(cleaned_records))
    logger.info("Dropped invalid queries : %d", dropped_count)
    logger.info("Validation pass rate    : %.2f%%", (len(cleaned_records) / len(dataset)) * 100)
    logger.info("Max character length    : %d", max_char_length)
    logger.info("Saved validated dataset to: %s", output_path)
    logger.info("======================================================")

    return cleaned_records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean and validate Text-to-SQL datasets using SQLGlot AST validation."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="./datasets/sql_create_dataset.json",
        help="Path to input JSON dataset.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./datasets/sql_create_dataset_cleaned.json",
        help="Path to save validated JSON dataset.",
    )
    parser.add_argument(
        "--read_dialect",
        type=str,
        default="sqlite",
        help="Source SQL dialect (default: sqlite).",
    )
    parser.add_argument(
        "--write_dialect",
        type=str,
        default="sqlite",
        help="Target SQL dialect for normalized output (default: sqlite).",
    )
    args = parser.parse_args()

    clean_and_validate_dataset(
        input_path=args.input_path,
        output_path=args.output_path,
        read_dialect=args.read_dialect,
        write_dialect=args.write_dialect,
    )


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )
    main()

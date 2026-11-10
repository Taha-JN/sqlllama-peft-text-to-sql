"""
Unit tests for the SQLGlot AST Cleaner and Validator.
"""

import json
import os
import tempfile
import pytest

from sqlllama.data.ast_cleaner import SQLValidator, clean_and_validate_dataset


@pytest.fixture
def validator():
    """Provides a default SQLite SQLValidator instance."""
    return SQLValidator(read_dialect="sqlite", write_dialect="sqlite")


class TestSQLValidator:
    """Tests the parsing, normalization, and invalid-query rejection of SQLValidator."""

    def test_valid_sql_normalization(self, validator):
        raw_sql = "select id, name from users where age > 21 order by name desc;"
        normalized = validator.normalize_and_validate(raw_sql)
        assert normalized is not None
        assert "SELECT" in normalized
        assert "WHERE age > 21" in normalized
        assert "ORDER BY" in normalized

    def test_lowercase_to_standard_formatting(self, validator):
        raw_sql = "select count(*) from departments group by dept_id having count(*) > 5"
        normalized = validator.normalize_and_validate(raw_sql)
        assert normalized is not None
        assert "SELECT COUNT(*)" in normalized
        assert "GROUP BY dept_id" in normalized
        assert "HAVING COUNT(*) > 5" in normalized

    def test_complex_join_normalization(self, validator):
        raw_sql = (
            "select t1.name, t2.score from students as t1 "
            "inner join grades as t2 on t1.id = t2.student_id "
            "where t2.score >= 90"
        )
        normalized = validator.normalize_and_validate(raw_sql)
        assert normalized is not None
        assert "JOIN" in normalized
        assert "ON t1.id = t2.student_id" in normalized

    def test_invalid_sql_rejection(self, validator):
        malformed_queries = [
            "SELECT FROM WHERE users TABLE;",
            "INSERT INTO SELECT FROM;",
            "WHERE id = 5 FROM table;",
            "NOT A SQL QUERY AT ALL 12345 !@#$",
        ]
        for query in malformed_queries:
            normalized = validator.normalize_and_validate(query)
            assert normalized is None, f"Expected None for invalid query: '{query}', got '{normalized}'"

    def test_empty_and_none_input(self, validator):
        assert validator.normalize_and_validate("") is None
        assert validator.normalize_and_validate("   ") is None
        assert validator.normalize_and_validate(None) is None

    def test_whitespace_and_newline_handling(self, validator):
        multiline_sql = """
        SELECT
            id,
            product_name
        FROM
            inventory
        WHERE
            quantity < 10
        """
        normalized = validator.normalize_and_validate(multiline_sql)
        assert normalized is not None
        assert "SELECT id, product_name FROM inventory WHERE quantity < 10" in normalized


class TestCleanAndValidateDataset:
    """Tests end-to-end dataset cleaning and JSON persistence."""

    def test_clean_and_validate_dataset(self, tmp_path):
        sample_data = [
            {
                "instruction": "List all active users",
                "input": "CREATE TABLE users (id INT, active BOOLEAN)",
                "output": "SELECT * FROM users WHERE active = 1",
            },
            {
                "instruction": "This is completely invalid SQL",
                "input": "CREATE TABLE test (val INT)",
                "output": "SELECT FROM WHERE !!! INVALID",
            },
            {
                "instruction": "Show department count",
                "input": "CREATE TABLE dept (id INT)",
                "output": "SELECT count(*) FROM dept",
            },
        ]

        input_file = tmp_path / "raw_data.json"
        output_file = tmp_path / "cleaned_data.json"

        with open(input_file, "w", encoding="utf-8") as f:
            json.dump(sample_data, f)

        cleaned_records = clean_and_validate_dataset(
            input_path=str(input_file),
            output_path=str(output_file),
            read_dialect="sqlite",
            write_dialect="sqlite",
        )

        assert len(cleaned_records) == 2
        assert os.path.exists(output_file)

        with open(output_file, "r", encoding="utf-8") as f:
            persisted = json.load(f)
        assert len(persisted) == 2
        assert persisted[0]["output"] == "SELECT * FROM users WHERE active = 1"
        assert persisted[1]["output"] == "SELECT COUNT(*) FROM dept"

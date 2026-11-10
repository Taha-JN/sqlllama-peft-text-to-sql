"""
Unit tests for the SQL KeywordAnalyzer utility.
"""

import pytest
from sqlllama.utils.sql_analyzer import (
    KeywordAnalyzer,
    DEFAULT_SQL_KEYWORDS,
    DEFAULT_DDL_KEYWORDS,
    count_sql_keyword,
)


@pytest.fixture
def analyzer():
    """Provides a default KeywordAnalyzer with standard SQL keywords."""
    return KeywordAnalyzer()


class TestKeywordAnalyzer:
    """Verifies precompiled regex keyword matching and word-boundary safety."""

    def test_basic_keyword_counting(self, analyzer):
        query = "SELECT id, name FROM users WHERE age > 30 ORDER BY name ASC"
        assert analyzer.count("SELECT", query) == 1
        assert analyzer.count("WHERE", query) == 1
        assert analyzer.count("ORDER BY", query) == 1
        assert analyzer.count("JOIN", query) == 0

    def test_word_boundary_isolation_prevents_false_positives(self, analyzer):
        """
        Critical test: ensures keywords like 'ON' or 'IN' are not matched
        inside words like 'PERSON', 'CONTINENT', 'CONTAIN', 'WINDOW', etc.
        """
        text_with_substrings = (
            "CREATE TABLE continent_info (person_id VARCHAR, continent VARCHAR, join_date DATE);"
        )
        assert analyzer.count("ON", text_with_substrings) == 0
        assert analyzer.count("IN", text_with_substrings) == 0
        assert analyzer.count("JOIN", text_with_substrings) == 0

        # Now test with actual keyword present
        query_with_actual_on = "SELECT * FROM t1 JOIN t2 ON t1.id = t2.id"
        assert analyzer.count("ON", query_with_actual_on) == 1
        assert analyzer.count("JOIN", query_with_actual_on) == 1

    def test_multiword_keywords_with_variable_whitespace(self, analyzer):
        query_with_tabs_and_extra_spaces = "SELECT * FROM employees GROUP   BY dept_id ORDER\tBY salary"
        assert analyzer.count("GROUP BY", query_with_tabs_and_extra_spaces) == 1
        assert analyzer.count("ORDER BY", query_with_tabs_and_extra_spaces) == 1

    def test_case_insensitivity(self, analyzer):
        query = "select id, name from table where status in ('active', 'pending')"
        assert analyzer.count("SELECT", query) == 1
        assert analyzer.count("select", query) == 1
        assert analyzer.count("WHERE", query) == 1
        assert analyzer.count("IN", query) == 1

    def test_total_count(self, analyzer):
        query = "SELECT DISTINCT name FROM users WHERE age > 18 AND status = 'active'"
        # Expected matched keywords: SELECT (1), DISTINCT (1), WHERE (1), AND (1)
        assert analyzer.total_count(query) == 4

    def test_count_all_returns_full_mapping(self, analyzer):
        query = "SELECT id FROM accounts WHERE balance > 1000"
        result = analyzer.count_all(query)
        assert isinstance(result, dict)
        assert result["SELECT"] == 1
        assert result["WHERE"] == 1
        assert result["JOIN"] == 0
        assert set(result.keys()) == set(analyzer.keywords)

    def test_empty_and_none_text_handling(self, analyzer):
        assert analyzer.count("SELECT", "") == 0
        assert analyzer.count("SELECT", None) == 0
        assert analyzer.total_count("") == 0
        assert analyzer.total_count(None) == 0

    def test_custom_keyword_registration(self):
        custom = KeywordAnalyzer(["UNION ALL", "WITH RECURSIVE", "RETURNING"])
        query = "WITH RECURSIVE cte AS (SELECT 1) SELECT * FROM cte UNION ALL SELECT * FROM cte RETURNING id"
        assert custom.count("UNION ALL", query) == 1
        assert custom.count("WITH RECURSIVE", query) == 1
        assert custom.count("RETURNING", query) == 1
        assert custom.count("SELECT", query) == 0  # not registered in custom set

    def test_standalone_count_sql_keyword_helper(self):
        query = "SELECT * FROM users WHERE active = 1"
        assert count_sql_keyword("SELECT", query) == 1
        assert count_sql_keyword("WHERE", query) == 1
        assert count_sql_keyword("JOIN", query) == 0

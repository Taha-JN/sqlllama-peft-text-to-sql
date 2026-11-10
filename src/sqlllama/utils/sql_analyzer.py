"""
SQL Keyword and Complexity Analyzer Utility.

Provides precompiled regular-expression matching with word boundaries
to accurately count SQL keywords without false-positive substring matches.
"""

import re
from typing import Dict, Iterable, List, Optional, Sequence


DEFAULT_SQL_KEYWORDS: List[str] = [
    "SELECT",
    "DISTINCT",
    "ON",
    "IN",
    "WHERE",
    "JOIN",
    "GROUP BY",
    "EXCEPT",
    "HAVING",
    "ORDER BY",
    "INTERSECT",
    "NOT IN",
    "AND",
]

DEFAULT_DDL_KEYWORDS: List[str] = ["CREATE"]


class KeywordAnalyzer:
    """
    Analyzes SQL text and counts keyword occurrences using precompiled regex patterns
    enforced with word boundaries (\b...\b) to avoid false positive substring matches.
    """

    def __init__(self, keywords: Optional[Iterable[str]] = None):
        """
        Initializes the analyzer with a collection of SQL keywords, compiling patterns up-front.

        Args:
            keywords: An iterable of SQL keywords to track. Defaults to standard SQL query keywords.
        """
        if keywords is None:
            keywords = DEFAULT_SQL_KEYWORDS

        self.patterns: Dict[str, re.Pattern] = {}
        for keyword in keywords:
            kw_clean = keyword.strip()
            if not kw_clean:
                continue
            parts = kw_clean.split()
            escaped_parts = [re.escape(p) for p in parts]
            sub_pattern = r"\s+".join(escaped_parts)
            self.patterns[kw_clean.upper()] = re.compile(
                rf"\b{sub_pattern}\b", re.IGNORECASE
            )

    @property
    def keywords(self) -> List[str]:
        """Returns the list of monitored keywords."""
        return list(self.patterns.keys())

    def count(self, keyword: str, text: Optional[str]) -> int:
        """
        Counts occurrences of a specific registered keyword in the provided text.

        Args:
            keyword: The registered keyword to search for.
            text: The SQL query or schema context string.

        Returns:
            The number of non-overlapping occurrences, or 0 if keyword is not tracked.
        """
        if not text:
            return 0

        pattern = self.patterns.get(keyword.strip().upper())
        if pattern is None:
            return 0

        return len(pattern.findall(str(text)))

    def count_all(self, text: Optional[str]) -> Dict[str, int]:
        """
        Counts occurrences of all monitored keywords in the text.

        Args:
            text: The SQL query or schema context string.

        Returns:
            Dictionary mapping uppercase keyword name to its count.
        """
        if not text:
            return {kw: 0 for kw in self.patterns}
        return {kw: len(pat.findall(str(text))) for kw, pat in self.patterns.items()}

    def total_count(self, text: Optional[str]) -> int:
        """
        Computes the sum of occurrences of all monitored keywords in the text.

        Args:
            text: The SQL query or schema context string.

        Returns:
            Total count of matched keywords.
        """
        if not text:
            return 0
        return sum(len(pat.findall(str(text))) for pat in self.patterns.values())


def count_sql_keyword(keyword: str, text: Optional[str]) -> int:
    """
    Backwards-compatible standalone function to count an SQL keyword occurrence.
    Delegates to KeywordAnalyzer.
    """
    return KeywordAnalyzer([keyword]).count(keyword, text)

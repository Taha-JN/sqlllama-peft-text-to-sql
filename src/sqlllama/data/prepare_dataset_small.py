"""
Dataset Extractor and Keyword Distribution Analyzer (Small Subset).

Extracts the intersection between Spider and sql-create-context datasets,
analyzing SQL keyword complexity using exact word-boundary regex patterns.
"""

import json
import logging
import os
import sys
from typing import Any, Dict, List
import numpy as np

# Ensure sqlllama package is importable when script is run directly
_current_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.abspath(os.path.join(_current_dir, "..", ".."))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from sqlllama.utils.sql_analyzer import KeywordAnalyzer, DEFAULT_SQL_KEYWORDS, DEFAULT_DDL_KEYWORDS

logger = logging.getLogger(__name__)

# Keywords for analysis
KEYWORDS = DEFAULT_SQL_KEYWORDS
KEYWORDS_INPUT = DEFAULT_DDL_KEYWORDS


def main() -> None:
    cleaned_dataset_path = "./datasets/sql_create_dataset_cleaned.json"
    spider_dataset_path = "./datasets/spider/spider/train_spider.json"
    output_path = "./datasets/sql_create_dataset_small.json"

    if not os.path.exists(cleaned_dataset_path):
        raise FileNotFoundError(f"Cleaned dataset not found at '{cleaned_dataset_path}'.")

    if not os.path.exists(spider_dataset_path):
        raise FileNotFoundError(f"Required dataset not found at '{spider_dataset_path}'. Please ensure it is downloaded.")

    with open(cleaned_dataset_path, "r", encoding="utf-8") as f1:
        custom_sql_set: List[Dict[str, Any]] = json.load(f1)

    with open(spider_dataset_path, "r", encoding="utf-8") as f2:
        spider_sql_set: List[Dict[str, Any]] = json.load(f2)

    # -------------------------------------------------------------------------
    # Optimization: O(N) Hash Map lookup replacing O(N x M) nested substring loop
    # -------------------------------------------------------------------------
    spider_questions_lookup: Dict[str, Dict[str, Any]] = {}
    for spider in spider_sql_set:
        question_text = spider.get("question", "").strip().lower()
        if question_text:
            spider_questions_lookup[question_text] = spider

    intersect_spider_sqlcleaned: List[Dict[str, Any]] = []
    count_spider = 0

    if spider_questions_lookup:
        logger.info(
            "Performing O(1) hash map matching across %d candidate queries...",
            len(custom_sql_set),
        )
        for entry in custom_sql_set:
            instruction_key = entry.get("instruction", "").strip().lower()
            if instruction_key in spider_questions_lookup:
                count_spider += 1
                intersect_spider_sqlcleaned.append(entry)

    logger.info("Found %d intersecting Spider records.", len(intersect_spider_sqlcleaned))

    # -------------------------------------------------------------------------
    # Keyword Counting using Precompiled Regex Analyzer
    # -------------------------------------------------------------------------
    sql_analyzer = KeywordAnalyzer(KEYWORDS)
    ddl_analyzer = KeywordAnalyzer(KEYWORDS_INPUT)

    count_list_instruction: List[int] = []
    count_list_input: List[int] = []

    for entry in intersect_spider_sqlcleaned:
        out_sql = entry.get("output", "")
        in_sql = entry.get("input", "")

        count_out = sql_analyzer.total_count(out_sql)
        count_list_instruction.append(count_out)

        count_in = ddl_analyzer.total_count(in_sql)
        count_list_input.append(count_in)

    if count_list_instruction:
        mean_instr = np.mean(count_list_instruction)
        logger.info("Mean SQL output keyword count: %.2f", mean_instr)
        for i in range(1, 6):
            logger.info("Output queries with %d keywords: %d", i, count_list_instruction.count(i))

    if count_list_input:
        mean_in = np.mean(count_list_input)
        logger.info("Mean DDL input keyword count: %.2f", mean_in)
        for i in range(1, 6):
            logger.info("Input contexts with %d keywords: %d", i, count_list_input.count(i))

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f_out:
        json.dump(intersect_spider_sqlcleaned, f_out, indent=2)

    logger.info("Saved %d records to %s", len(intersect_spider_sqlcleaned), output_path)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )
    main()

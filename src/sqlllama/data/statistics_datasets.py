"""
Dataset Statistics and SQL Keyword Distribution Analyzer.

Visualizes sequence length distributions and computes relational keyword frequencies
using precompiled AST/regex boundary patterns.
"""

import json
import logging
import os
import sys
from typing import Any, Dict, List
import matplotlib.pyplot as plt
import numpy as np

# Ensure sqlllama package is importable when script is run directly
_current_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.abspath(os.path.join(_current_dir, "..", ".."))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from sqlllama.utils.sql_analyzer import KeywordAnalyzer, DEFAULT_SQL_KEYWORDS, DEFAULT_DDL_KEYWORDS

logger = logging.getLogger(__name__)


def count_instructions(
    sql_dataset: List[Dict[str, Any]],
    output_str: str,
    input_str: str,
    sql_analyzer: KeywordAnalyzer,
    ddl_analyzer: KeywordAnalyzer,
) -> List[int]:
    """Computes keyword frequencies across output queries and input schemas."""
    count_list_instruction = []
    count_list_input = []

    for entry in sql_dataset:
        count_out = sql_analyzer.total_count(entry.get(output_str, ""))
        count_list_instruction.append(count_out)

        count_in = ddl_analyzer.total_count(entry.get(input_str, ""))
        count_list_input.append(count_in)

    if count_list_instruction:
        logger.info("Mean instruction keyword count: %.2f", np.mean(count_list_instruction))
    if count_list_input:
        logger.info("Mean input keyword count: %.2f", np.mean(count_list_input))

    save_instr_count = []
    for i in range(1, 25):
        save_instr_count.append(count_list_instruction.count(i))

    return save_instr_count


def main() -> None:
    from datasets import load_dataset

    small_dataset_path = "./datasets/sql_create_dataset_small.json"
    if not os.path.exists(small_dataset_path):
        logger.warning("Small dataset not found at %s. Aborting.", small_dataset_path)
        return

    logger.info("Loading %s...", small_dataset_path)
    with open(small_dataset_path, "r", encoding="utf-8") as f:
        small_sql_create_context = json.load(f)

    logger.info("Loading reference sql-create-context dataset...")
    sql_create_context = load_dataset(
        "b-mc2/sql-create-context", split="train", cache_dir="./datasets_all"
    )

    lengths = list(map(len, sql_create_context["answer"]))
    smallds_strings = [i.get("output", "") for i in small_sql_create_context]
    lengths_smallds = list(map(len, smallds_strings))

    # Generate and save sequence length histogram (non-blocking for CI)
    plt.figure(figsize=(10, 5))
    plt.hist(lengths, density=True, bins=70, color="blue", alpha=0.6, label="b-mc2/sql_create_context")
    plt.hist(lengths_smallds, density=True, bins=70, color="green", alpha=0.7, label="SQL-LLaMA-small")
    plt.legend(fontsize=12)
    plt.ylabel("Probability", fontsize=12)
    plt.xlabel("Answer-Length", fontsize=12)
    plt.tight_layout()
    dist_plot_path = "distribution_plot.png"
    plt.savefig(dist_plot_path)
    plt.close()
    logger.info("Saved distribution histogram to %s", dist_plot_path)

    # Keyword analysis
    sql_analyzer = KeywordAnalyzer(DEFAULT_SQL_KEYWORDS)
    ddl_analyzer = KeywordAnalyzer(DEFAULT_DDL_KEYWORDS)

    save_instr_count_small = count_instructions(
        small_sql_create_context, "output", "input", sql_analyzer, ddl_analyzer
    )
    save_instr_count = count_instructions(
        sql_create_context, "answer", "context", sql_analyzer, ddl_analyzer
    )

    # Bar plot for keyword frequencies (non-blocking for CI)
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    axs[0].bar(range(1, 25), save_instr_count, color="orange", width=0.4)
    axs[0].set_ylabel("# of Instructions", fontsize=12)
    axs[0].set_xlabel("# of SQL-Keywords", fontsize=12)
    axs[0].legend(["SQL-LLaMA"], fontsize=12)

    axs[1].bar(range(1, 25), save_instr_count_small, color="green", width=0.4)
    axs[1].set_ylabel("# of Instructions", fontsize=12)
    axs[1].set_xlabel("# of SQL-Keywords", fontsize=12)
    axs[1].legend(["SQL-LLaMA-small"], fontsize=12)

    plt.tight_layout()
    output_plot_path = "output_plot.png"
    plt.savefig(output_plot_path)
    plt.close()
    logger.info("Saved keyword frequency plot to %s", output_plot_path)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )
    main()

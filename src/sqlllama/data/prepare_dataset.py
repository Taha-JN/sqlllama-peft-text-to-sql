"""
Dataset Preparation and Length-Weighted Power Sampling.

Downsamples the sql-create-context dataset, weighting towards more complex
relational queries to ensure adequate sequence representation during SFT.
"""

import json
import logging
import os
import random
import sys
from typing import List
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset

logger = logging.getLogger(__name__)


def select_string_increasing_prob(
    string_list: List[str], numb_items: int, power: float = 4.0
) -> List[int]:
    """Selects indices from dataset where longer strings have a higher sampling probability without replacement."""
    if numb_items > len(string_list):
        raise ValueError(
            f"Requested {numb_items} items, but string_list only has {len(string_list)} items."
        )

    lengths = np.array([len(s) for s in string_list], dtype=np.float64)
    weights = lengths ** power
    total_weight = weights.sum()

    if total_weight == 0:
        probs = np.ones(len(string_list), dtype=np.float64) / len(string_list)
    else:
        probs = weights / total_weight

    # Secure normalization against floating point precision issues
    probs = probs / probs.sum()

    sampled_indices = np.random.choice(
        len(string_list),
        size=numb_items,
        replace=False,
        p=probs,
    )
    return sampled_indices.tolist()


def main() -> None:
    random.seed(42)
    np.random.seed(42)

    logger.info("Loading b-mc2/sql-create-context dataset...")
    sql_create_context = load_dataset(
        "b-mc2/sql-create-context", split="train", cache_dir="./datasets_all"
    )

    logger.info("Sampling 10,000 length-weighted query records...")
    idx_longer_strings = select_string_increasing_prob(
        sql_create_context["answer"], 10000
    )
    sampled_subdataset = sql_create_context.select(idx_longer_strings)

    lengths_longer_strings = [len(row["answer"]) for row in sampled_subdataset]
    lengths_all = list(map(len, sql_create_context["answer"]))

    # Save non-blocking distribution histogram
    plt.figure(figsize=(10, 5))
    plt.hist(lengths_all, density=True, bins=70, color="blue", alpha=0.6, label="b-mc2/sql_create_context")
    plt.hist(lengths_longer_strings, density=True, bins=70, color="orange", alpha=0.7, label="SQL-LLaMA Sampled")
    plt.legend(fontsize=12)
    plt.ylabel("Probability")
    plt.xlabel("Answer-Length")
    plt.tight_layout()
    plt.savefig("prepare_distribution.png")
    plt.close()
    logger.info("Saved distribution histogram to prepare_distribution.png")

    os.makedirs("./datasets", exist_ok=True)
    custom_sql_set = [
        {"instruction": row["question"], "input": row["context"], "output": row["answer"]}
        for row in sampled_subdataset
    ]

    out_path = "./datasets/sql_create_dataset.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(custom_sql_set, f, indent=2)
    logger.info("Saved %d sampled queries to %s", len(custom_sql_set), out_path)

    # Optional extraction from rosetta-code for SQL tasks
    try:
        logger.info("Loading cakiki/rosetta-code for supplementary SQL samples...")
        rosetta_code = load_dataset(
            "cakiki/rosetta-code", split="train", cache_dir="./datasets_all"
        )
        rosetta_code_sql = []
        for item in rosetta_code:
            if "sql" in str(item.get("language_name", "")).lower():
                rosetta_code_sql.append(
                    {
                        "instruction": item.get("task_description", ""),
                        "input": "",
                        "output": item.get("code", ""),
                    }
                )
        rosetta_out = "./datasets/rosetta_sql_dataset.json"
        with open(rosetta_out, "w", encoding="utf-8") as f:
            json.dump(rosetta_code_sql, f, indent=2)
        logger.info("Saved %d Rosetta Code SQL samples to %s", len(rosetta_code_sql), rosetta_out)
    except Exception as exc:
        logger.warning("Could not load rosetta-code dataset: %s", exc)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )
    main()

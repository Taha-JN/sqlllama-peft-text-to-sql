"""
Unit tests for the centralized Alpaca prompt templates and formatting utilities.
"""

import pytest
from sqlllama.prompts import (
    PROMPT_DICT,
    PROMPT_WITH_INPUT,
    PROMPT_NO_INPUT,
    format_prompt,
)


def test_prompt_dict_keys_and_values():
    assert "prompt_input" in PROMPT_DICT
    assert "prompt_no_input" in PROMPT_DICT
    assert PROMPT_DICT["prompt_input"] == PROMPT_WITH_INPUT
    assert PROMPT_DICT["prompt_no_input"] == PROMPT_NO_INPUT


def test_format_prompt_with_context():
    instruction = "List all active customers."
    context = "CREATE TABLE customers (id INT, active BOOLEAN);"
    formatted = format_prompt(instruction, context)

    assert "### Instruction:\nList all active customers." in formatted
    assert "### Input:\nCREATE TABLE customers (id INT, active BOOLEAN);" in formatted
    assert "### Response:\n" in formatted


def test_format_prompt_without_context():
    instruction = "SELECT current_timestamp;"
    formatted = format_prompt(instruction, "")

    assert "### Instruction:\nSELECT current_timestamp;" in formatted
    assert "### Input:" not in formatted
    assert "### Response:\n" in formatted


def test_format_prompt_whitespace_stripping():
    instruction = "   Find average salary.   "
    context = "   CREATE TABLE emp (salary INT);   "
    formatted = format_prompt(instruction, context)

    assert "### Instruction:\nFind average salary." in formatted
    assert "### Input:\nCREATE TABLE emp (salary INT);" in formatted

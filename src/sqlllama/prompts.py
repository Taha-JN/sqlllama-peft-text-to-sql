"""
Standardized Alpaca instruction prompts for SQL-LLaMA training and inference.
"""

from typing import Dict, Optional

PROMPT_WITH_INPUT: str = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)

PROMPT_NO_INPUT: str = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)

PROMPT_DICT: Dict[str, str] = {
    "prompt_input": PROMPT_WITH_INPUT,
    "prompt_no_input": PROMPT_NO_INPUT,
}


def format_prompt(instruction: str, context: Optional[str] = "") -> str:
    """Formats an instruction and optional schema context into an Alpaca prompt."""
    instr = instruction.strip() if instruction else ""
    ctx = context.strip() if context else ""
    if ctx:
        return PROMPT_WITH_INPUT.format(instruction=instr, input=ctx)
    return PROMPT_NO_INPUT.format(instruction=instr)

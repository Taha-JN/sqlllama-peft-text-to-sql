"""
Local Quantized GGUF Inference Runner.

Executes low-latency local inference for fine-tuned SQL-LLaMA models
using llama-cpp-python bindings with GGUF 4-bit / 8-bit quantized weights.
"""

import argparse
import logging
import os
import sys
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

from sqlllama.prompts import format_prompt, PROMPT_WITH_INPUT, PROMPT_NO_INPUT

DEFAULT_MODEL_PATH = "./models/sqlllama-7b-q4_k_m.gguf"

DEFAULT_INSTRUCTION = (
    "Show the stadium name and capacity with the most concerts in year 2014 or after."
)
DEFAULT_SCHEMA = (
    "CREATE TABLE stadium (stadium_id VARCHAR, name VARCHAR, capacity INTEGER); "
    "CREATE TABLE concert (concert_id VARCHAR, stadium_id VARCHAR, year INTEGER)"
)


def run_inference(
    model_path: str = DEFAULT_MODEL_PATH,
    instruction: str = DEFAULT_INSTRUCTION,
    context: str = DEFAULT_SCHEMA,
    max_tokens: int = 512,
    temperature: float = 0.1,
    n_ctx: int = 2048,
    n_gpu_layers: int = 0,
) -> Dict[str, Any]:
    """
    Executes inference against a local GGUF quantized model.

    Args:
        model_path: File system path to the .gguf model file.
        instruction: Natural language query request.
        context: Relational DDL schema context.
        max_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (low for deterministic SQL).
        n_ctx: Context window size in tokens.
        n_gpu_layers: Number of layers to offload to GPU (0 for CPU only).

    Returns:
        The completion dictionary returned by llama-cpp.

    Raises:
        FileNotFoundError: If the model weights do not exist at model_path.
        ImportError: If llama-cpp-python is not installed.
    """
    if not os.path.isfile(model_path):
        err_msg = (
            f"Quantized model checkpoint not found at: '{model_path}'.\n"
            "Please ensure you have exported and quantized the fine-tuned model to GGUF format.\n"
            "Example: python models_hf/convert.py output_sqlAlpaca7B/ --outtype q4_k_m"
        )
        logger.error(err_msg)
        raise FileNotFoundError(err_msg)

    try:
        from llama_cpp import Llama
    except ImportError as e:
        err_msg = (
            "The 'llama-cpp-python' library is required for local inference.\n"
            "Install it via: pip install llama-cpp-python or pip install -e \".[inference]\""
        )
        logger.error(err_msg)
        raise ImportError(err_msg) from e

    logger.info("Loading GGUF model from %s (n_ctx=%d, n_gpu_layers=%d)...", model_path, n_ctx, n_gpu_layers)
    llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )

    prompt = format_prompt(instruction, context)
    logger.info("Executing generation for prompt:\n%s", prompt)

    output = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=["###", "Instruction:", "Output:"],
        echo=False,
    )

    generated_sql = output["choices"][0]["text"].strip()
    logger.info("Generated SQL:\n%s", generated_sql)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SQL-LLaMA GGUF Local Inference Client."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to GGUF quantized model file (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=DEFAULT_INSTRUCTION,
        help="Natural language question to translate into SQL.",
    )
    parser.add_argument(
        "--context",
        type=str,
        default=DEFAULT_SCHEMA,
        help="Database table DDL schema context.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="Maximum generation length (default: 512).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature (default: 0.1).",
    )
    parser.add_argument(
        "--n_gpu_layers",
        type=int,
        default=0,
        help="Number of GPU layers for acceleration (0 for CPU only).",
    )
    args = parser.parse_args()

    try:
        run_inference(
            model_path=args.model_path,
            instruction=args.instruction,
            context=args.context,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            n_gpu_layers=args.n_gpu_layers,
        )
    except FileNotFoundError as err:
        logger.error("Execution failed: %s", err)
        sys.exit(1)
    except ImportError as err:
        logger.error("Execution failed: %s", err)
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )
    main()

# SQL-LLaMA2: Parameter-Efficient Fine-Tuning of LLaMA for Relational Text-to-SQL

A production fine-tuning pipeline designed to adapt LLaMA-2 foundation models into specialized Text-to-SQL semantic parsers capable of converting complex natural language questions into syntactically valid relational database queries.

## Architecture & Implementation
- **Data Engineering & AST Cleaning**:
  - Unifies multi-domain benchmarks including **WikiSQL**, **Spider**, and **sql-create-context**.
  - Integrates `sqlglot` Abstract Syntax Tree (AST) validation to parse, normalize, and verify SQL queries, pruning invalid syntax prior to tokenization.
  - Implements sequence length power-sampling to ensure adequate representation of long multi-table join contexts.
- **Distributed Training Infrastructure**:
  - Built on HuggingFace `Trainer` and **DeepSpeed ZeRO** (Stage 2/3) configuration (`ds_config_sql.json`) for efficient memory optimization across GPU nodes.
  - Dynamic sequence padding, attention mask handling (`IGNORE_INDEX = -100`), and smart vocabulary embedding resizing.
- **Inference & Quantization**:
  - Scripts to convert fine-tuned model checkpoints into HuggingFace format and export weights for quantized local CPU/GPU execution via `llama.cpp` (GGML/GGUF).

## Repository Structure
```
06_sqlllama-text-to-sql-finetuning/
├── train_sqlllama.py           # Distributed model fine-tuning script
├── prepare_dataset.py          # Benchmark dataset loader and weighted sampler
├── sqlglot_cleaned_dataset.py  # AST SQL parser and syntax validator
├── ds_config_sql.json          # DeepSpeed ZeRO memory configuration
├── convert_llama_weights_to_hf.py # Weight converter utility
├── requirements.txt            # Training environment dependencies
└── download.sh                 # Dataset acquisition script
```

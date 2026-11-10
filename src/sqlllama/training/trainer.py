"""
SQL-LLaMA: Distributed Fine-Tuning Pipeline for Text-to-SQL Semantic Parsing.

Attribution Notice:
Training script adapted from Stanford Alpaca's open-source train.py.
Original work Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li.
Licensed under the Apache License, Version 2.0.
"""

from copy import deepcopy
import json
import logging
import random
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, List, Any

logger = logging.getLogger(__name__)

import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer

IGNORE_INDEX = (
    -100
)  # Used in Transformers for Masking https://huggingface.co/docs/transformers/glossary#l
DEFAULT_PAD_TOKEN = "[PAD]"
from sqlllama.prompts import PROMPT_DICT

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="models/7B")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=4096,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    evaluation_strategy: str = field(
        default="steps",
        metadata={"help": "The evaluation strategy to use during training."},
    )
    eval_steps: int = field(
        default=100,
        metadata={"help": "Run an evaluation every X steps."},
    )
    load_best_model_at_end: bool = field(
        default=True,
        metadata={"help": "Whether or not to load the best model at the end of training."},
    )


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]

    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer) for strings in (examples, sources)
    ]

    input_ids = examples_tokenized["input_ids"]
    labels = deepcopy(input_ids)

    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        data_path: Optional[str] = None,
        list_data_dict: Optional[List[Dict[str, Any]]] = None,
        split_name: str = "train",
    ):
        super(SupervisedDataset, self).__init__()
        if list_data_dict is None:
            if data_path is None:
                raise ValueError("Either data_path or list_data_dict must be provided.")
            logger.info("Loading Dataset as .json from Path: %s", data_path)
            with open(data_path, "r", encoding="utf-8") as f:
                list_data_dict = json.load(f)

        logger.info("Formatting %d %s inputs to Alpaca-Style...", len(list_data_dict), split_name)
        prompt_input, prompt_no_input = (
            PROMPT_DICT["prompt_input"],
            PROMPT_DICT["prompt_no_input"],
        )

        sources, targets = [], []

        for example in list_data_dict:
            if example.get("input", "") != "":
                sources.append(prompt_input.format_map(example))
            else:
                sources.append(prompt_no_input.format_map(example))

            targets.append(f"{example['output']}{tokenizer.eos_token}")

        logger.info(
            "Tokenizing %s inputs (%d samples), this may take some time...",
            split_name,
            len(sources),
        )
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

        logger.info("Initialization of SupervisedDataset (%s) done!", split_name)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(
                self.tokenizer.pad_token_id
            ),  # HF-Transformers Attention-Masking https://huggingface.co/docs/transformers/glossary#a
        )


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning with a 90/10 train/val split."""
    logger.info("Loading raw dataset from %s for train/eval splitting...", data_args.data_path)
    with open(data_args.data_path, "r", encoding="utf-8") as f:
        list_data_dict = json.load(f)

    # Deterministic 90% Training / 10% Validation split
    random.seed(42)
    indices = list(range(len(list_data_dict)))
    random.shuffle(indices)

    split_idx = int(0.9 * len(indices))
    train_indices = indices[:split_idx]
    eval_indices = indices[split_idx:]

    train_data = [list_data_dict[i] for i in train_indices]
    eval_data = [list_data_dict[i] for i in eval_indices]

    logger.info(
        "Dataset split complete: Total=%d | Train=%d (90.0%%) | Eval=%d (10.0%%)",
        len(list_data_dict),
        len(train_data),
        len(eval_data),
    )

    train_dataset = SupervisedDataset(
        tokenizer=tokenizer, list_data_dict=train_data, split_name="train"
    )
    eval_dataset = SupervisedDataset(
        tokenizer=tokenizer, list_data_dict=eval_data, split_name="eval"
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )


def train():
    """Training Loop for SQL-LLaMA using LLaMA2 with HF-Transformers and Alpaca Instruction-Style"""

    logger.info("Parsing HF-Transformers Arguments")

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logger.info("Setting up Pretrained-Model: %s", model_args.model_name_or_path)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    logger.info("Setting up Tokenizer from: %s", model_args.model_name_or_path)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    logger.info(
        "Add Tokenizer Padding: %s and resize Transformer-Embeddings accordingly",
        DEFAULT_PAD_TOKEN,
    )

    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )

    logger.info(
        "Using Special Tokens: pad=%s (%s), eos=%s (%s), bos=%s (%s), unk=%s (%s)",
        tokenizer.pad_token,
        tokenizer.pad_token_id,
        tokenizer.eos_token,
        tokenizer.eos_token_id,
        tokenizer.bos_token,
        tokenizer.bos_token_id,
        tokenizer.unk_token,
        tokenizer.unk_token_id,
    )

    logger.info(
        "Setting data_module and data_collator using Alpaca-Style defined in PROMPT_DICT"
    )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    
    logger.info(
        "Set HF-Trainer with validation evaluation, Train the Model and Save to Directory: %s",
        training_args.output_dir,
    )
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=data_module["train_dataset"],
        eval_dataset=data_module["eval_dataset"],
        data_collator=data_module["data_collator"],
    )
    trainer.train()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )
    train()

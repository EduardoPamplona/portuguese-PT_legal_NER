"""Data loading and preprocessing utilities."""

import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
import torch

logger = logging.getLogger(__name__)

# Entity labels for Portuguese legal documents
ENTITY_LABELS = [
    "O",  # Outside
    "B-PER",  # Person - Begin
    "I-PER",  # Person - Inside
    "B-ORG",  # Organization - Begin
    "I-ORG",  # Organization - Inside
    "B-LOC",  # Location - Begin
    "I-LOC",  # Location - Inside
    "B-DAT",  # Date - Begin
    "I-DAT",  # Date - Inside
    "B-IDP",  # Identity Document - Begin
    "I-IDP",  # Identity Document - Inside
    "B-TEL",  # Telephone - Begin
    "I-TEL",  # Telephone - Inside
    "B-E-MAIL",  # Email - Begin
    "I-E-MAIL",  # Email - Inside
    "B-CEP",  # Postal Code - Begin
    "I-CEP",  # Postal Code - Inside
    "B-MAT",  # License Plate - Begin
    "I-MAT",  # License Plate - Inside
]

LABEL_TO_ID = {label: i for i, label in enumerate(ENTITY_LABELS)}
ID_TO_LABEL = {i: label for i, label in enumerate(ENTITY_LABELS)}


def read_conll_file(file_path: str) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Read a CoNLL format file and return tokens and labels.

    Args:
        file_path: Path to the CoNLL file

    Returns:
        Tuple of (sentences_tokens, sentences_labels)
    """
    if not os.path.exists(file_path):
        logger.warning(f"File {file_path} not found")
        return [], []

    sentences_tokens = []
    sentences_labels = []
    current_tokens = []
    current_labels = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:  # Empty line indicates sentence boundary
                if current_tokens:
                    sentences_tokens.append(current_tokens)
                    sentences_labels.append(current_labels)
                    current_tokens = []
                    current_labels = []
            else:
                parts = line.split("\t")
                if len(parts) >= 2:
                    token = parts[0]
                    label = parts[1]
                    current_tokens.append(token)
                    current_labels.append(label)

    # Add the last sentence if it doesn't end with empty line
    if current_tokens:
        sentences_tokens.append(current_tokens)
        sentences_labels.append(current_labels)

    return sentences_tokens, sentences_labels


def create_sample_data():
    """Create sample CoNLL data for testing."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    sample_data = [
        ("O", "processo", "O"),
        ("número", "B-IDP"),
        ("12345", "I-IDP"),
        ("do", "O"),
        ("réu", "O"),
        ("João", "B-PER"),
        ("Silva", "I-PER"),
        ("residente", "O"),
        ("em", "O"),
        ("Lisboa", "B-LOC"),
        ("foi", "O"),
        ("julgado", "O"),
        ("em", "O"),
        ("15", "B-DAT"),
        ("de", "I-DAT"),
        ("março", "I-DAT"),
        ("de", "I-DAT"),
        ("2023", "I-DAT"),
    ]

    # Create train, val, test splits
    for split in ["train", "val", "test"]:
        file_path = data_dir / f"{split}.conll"
        with open(file_path, "w", encoding="utf-8") as f:
            for i, (token, label) in enumerate(sample_data):
                f.write(f"{token}\t{label}\n")
                if i % 6 == 5:  # Create sentence boundaries
                    f.write("\n")
            f.write("\n")

        logger.info(f"Created sample data: {file_path}")


def tokenize_and_align_labels(examples, tokenizer, label_to_id, max_length=512):
    """
    Tokenize input and align labels with subword tokens.
    """
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )

    labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label_to_id[label[word_idx]])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


class DataLoader:
    """Data loader for NER datasets."""

    def __init__(self, tokenizer_name: str, max_length: int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, add_prefix_space=True
        )
        self.max_length = max_length

    def load_datasets(
        self, train_file: str, val_file: str, test_file: str
    ) -> DatasetDict:
        """
        Load and tokenize datasets from CoNLL files.

        Args:
            train_file: Path to training data
            val_file: Path to validation data
            test_file: Path to test data

        Returns:
            DatasetDict with train/validation/test splits
        """
        datasets = {}

        for split, file_path in [
            ("train", train_file),
            ("validation", val_file),
            ("test", test_file),
        ]:
            if os.path.exists(file_path):
                sentences_tokens, sentences_labels = read_conll_file(file_path)

                if sentences_tokens:
                    dataset = Dataset.from_dict(
                        {"tokens": sentences_tokens, "labels": sentences_labels}
                    )

                    # Tokenize and align labels
                    dataset = dataset.map(
                        lambda examples: tokenize_and_align_labels(
                            examples, self.tokenizer, LABEL_TO_ID, self.max_length
                        ),
                        batched=True,
                        remove_columns=["tokens", "labels"],
                    )

                    datasets[split] = dataset
                    logger.info(f"Loaded {len(dataset)} examples for {split}")
                else:
                    logger.warning(f"No data found in {file_path}")
            else:
                logger.warning(f"File not found: {file_path}")

        if not datasets:
            logger.error("No datasets loaded. Creating sample data...")
            create_sample_data()
            return self.load_datasets(train_file, val_file, test_file)

        return DatasetDict(datasets)

    def load_pretraining_data(self, data_path: str) -> Dataset:
        """
        Load raw text data for domain-adaptive pretraining.

        Args:
            data_path: Path to raw text file or directory

        Returns:
            Dataset with text field
        """
        texts = []

        if os.path.isfile(data_path):
            with open(data_path, "r", encoding="utf-8") as f:
                texts = [line.strip() for line in f if line.strip()]
        elif os.path.isdir(data_path):
            for file_path in Path(data_path).glob("*.txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    file_texts = [line.strip() for line in f if line.strip()]
                    texts.extend(file_texts)
        else:
            raise FileNotFoundError(f"Data path not found: {data_path}")

        dataset = Dataset.from_dict({"text": texts})

        # Tokenize for MLM
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

        dataset = dataset.map(tokenize_function, batched=True)
        logger.info(f"Loaded {len(dataset)} texts for pretraining")

        return dataset

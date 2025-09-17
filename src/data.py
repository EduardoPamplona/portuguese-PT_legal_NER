"""
Data loading and preprocessing utilities for Portuguese Legal NER.

This module provides comprehensive data handling capabilities including:
- Loading and parsing CoNLL format files
- Tokenization and label alignment for transformer models
- Dataset creation and preprocessing for training/evaluation
- Support for both NER fine-tuning and domain pretraining workflows
- Utilities for creating sample data for testing purposes

The module handles the complex task of aligning NER labels with subword
tokenization required by modern transformer models, ensuring proper
label propagation and handling of special tokens.
"""

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
    """
    Create sample CoNLL data for testing and development.
    
    Generates synthetic Portuguese legal text data with NER annotations
    in CoNLL format. Creates train, validation, and test splits with
    representative examples of different entity types commonly found
    in Portuguese legal documents.
    
    The sample data includes entities like:
    - PER (Person names)
    - LOC (Locations) 
    - DAT (Dates)
    - IDP (Identity documents/numbers)
    
    Returns:
        None: Files are created in the 'data/' directory.
        
    Side Effects:
        Creates three files: data/train.conll, data/val.conll, data/test.conll
    """
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
    Tokenize input text and align NER labels with subword tokens.
    
    This function handles the complex task of aligning word-level NER labels
    with subword tokens produced by modern transformer tokenizers. It ensures
    that labels are properly propagated to subword pieces while maintaining
    the original labeling scheme.
    
    Args:
        examples (dict): Batch of examples containing 'tokens' and 'labels' keys.
            - 'tokens': List of lists, where each inner list contains word tokens
            - 'labels': List of lists, where each inner list contains corresponding labels
        tokenizer (AutoTokenizer): Pre-trained tokenizer for subword tokenization.
        label_to_id (dict): Mapping from label names to integer IDs.
        max_length (int, optional): Maximum sequence length. Defaults to 512.
        
    Returns:
        dict: Dictionary containing tokenized inputs with aligned labels:
            - 'input_ids': Token IDs for model input
            - 'attention_mask': Attention mask for padding tokens
            - 'labels': Aligned label IDs (-100 for special/subword tokens)
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
    """
    Data loader for NER datasets and domain pretraining.
    
    This class handles loading and preprocessing of data for both Named Entity
    Recognition fine-tuning and domain-adaptive pretraining tasks. It manages
    tokenization, label alignment, and dataset creation for transformer models.
    
    Attributes:
        tokenizer (AutoTokenizer): Pre-trained tokenizer for text processing.
        max_length (int): Maximum sequence length for tokenization.
    """

    def __init__(self, tokenizer_name: str, max_length: int = 512):
        """
        Initialize the data loader with a specified tokenizer.
        
        Args:
            tokenizer_name (str): Name or path of the pre-trained tokenizer.
            max_length (int, optional): Maximum sequence length for tokenization.
                Defaults to 512.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, add_prefix_space=True
        )
        self.max_length = max_length

    def load_datasets(
        self, train_file: str, val_file: str, test_file: str
    ) -> DatasetDict:
        """
        Load and preprocess NER datasets from CoNLL format files.
        
        Reads CoNLL format files, tokenizes the text, and aligns NER labels
        with subword tokens. Creates train, validation, and test datasets
        ready for model training and evaluation.
        
        Args:
            train_file (str): Path to training data file in CoNLL format.
            val_file (str): Path to validation data file in CoNLL format.
            test_file (str): Path to test data file in CoNLL format.
            
        Returns:
            DatasetDict: Dictionary containing train/validation/test datasets.
                Each dataset has 'input_ids', 'attention_mask', and 'labels' fields.
                
        Side Effects:
            - Logs information about loaded datasets
            - Creates sample data if no files are found
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
        
        Loads plain text data from files or directories for masked language
        modeling pretraining. Supports both single files and directories
        containing multiple text files.
        
        Args:
            data_path (str): Path to raw text file (.txt) or directory containing
                text files. If directory, all .txt files will be processed.
                
        Returns:
            Dataset: HuggingFace Dataset object with tokenized text ready for
                masked language modeling. Contains 'input_ids', 'attention_mask'
                and 'token_type_ids' fields.
                
        Raises:
            FileNotFoundError: If the specified path doesn't exist.
            
        Side Effects:
            Logs information about the number of loaded texts.
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

"""Model factory for creating different types of models."""

import logging
from typing import Tuple, Optional
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForMaskedLM,
    AutoConfig,
    DataCollatorForTokenClassification,
    DataCollatorForLanguageModeling,
)

try:
    from .data import ENTITY_LABELS, LABEL_TO_ID, ID_TO_LABEL
except ImportError:
    from data import ENTITY_LABELS, LABEL_TO_ID, ID_TO_LABEL

logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory for creating different types of models."""

    @staticmethod
    def create_ner_model(
        model_name: str,
        num_labels: int = len(ENTITY_LABELS),
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        hidden_dropout: float = 0.1,
        pretrained_model_path: Optional[str] = None,
    ) -> Tuple[AutoModelForTokenClassification, AutoTokenizer]:
        """
        Create a model and tokenizer for NER task.

        Args:
            model_name: Name of the base model
            num_labels: Number of labels for classification
            dropout: Dropout rate
            attention_dropout: Attention dropout rate
            hidden_dropout: Hidden dropout rate
            pretrained_model_path: Path to domain-adapted model

        Returns:
            Tuple of (model, tokenizer)
        """
        # Use pretrained model path if provided, otherwise use base model
        model_path = pretrained_model_path or model_name

        tokenizer = AutoTokenizer.from_pretrained(
            model_name, add_prefix_space=True  # Always use original tokenizer
        )

        # Configure model
        config = AutoConfig.from_pretrained(
            model_path,
            num_labels=num_labels,
            id2label=ID_TO_LABEL,
            label2id=LABEL_TO_ID,
            hidden_dropout_prob=hidden_dropout,
            attention_probs_dropout_prob=attention_dropout,
        )

        # Create model
        model = AutoModelForTokenClassification.from_pretrained(
            model_path, config=config, ignore_mismatched_sizes=True
        )

        # Apply additional dropout if specified
        if hasattr(model, "dropout"):
            model.dropout.p = dropout

        logger.info(f"Created NER model from {model_path}")
        logger.info(f"Model has {model.num_parameters():,} parameters")

        return model, tokenizer

    @staticmethod
    def create_pretraining_model(
        model_name: str,
    ) -> Tuple[AutoModelForMaskedLM, AutoTokenizer]:
        """
        Create a model and tokenizer for domain-adaptive pretraining.

        Args:
            model_name: Name of the base model

        Returns:
            Tuple of (model, tokenizer)
        """
        tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)

        model = AutoModelForMaskedLM.from_pretrained(model_name)

        logger.info(f"Created pretraining model from {model_name}")
        logger.info(f"Model has {model.num_parameters():,} parameters")

        return model, tokenizer

    @staticmethod
    def get_data_collator(task_type: str, tokenizer: AutoTokenizer):
        """
        Get appropriate data collator for the task.

        Args:
            task_type: Type of task ('ner' or 'pretraining')
            tokenizer: The tokenizer to use

        Returns:
            Data collator instance
        """
        if task_type == "ner":
            return DataCollatorForTokenClassification(tokenizer=tokenizer)
        elif task_type == "pretraining":
            return DataCollatorForLanguageModeling(
                tokenizer=tokenizer, mlm=True, mlm_probability=0.15
            )
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    @staticmethod
    def save_model(model, tokenizer, output_dir: str, experiment_name: str):
        """
        Save model and tokenizer to disk.

        Args:
            model: The model to save
            tokenizer: The tokenizer to save
            output_dir: Output directory
            experiment_name: Name of the experiment
        """
        import os
        from pathlib import Path

        save_path = Path(output_dir) / experiment_name
        save_path.mkdir(parents=True, exist_ok=True)

        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

        logger.info(f"Model saved to {save_path}")

        return str(save_path)

"""
Model factory for creating and managing different types of transformer models.

This module provides a centralized factory for creating transformer models
optimized for Portuguese legal NER tasks. It supports both Named Entity
Recognition fine-tuning and domain-adaptive pretraining workflows.

Key features:
- Creation of NER models with custom label configurations
- Support for domain-adaptive pretraining with masked language modeling
- Automatic model and tokenizer management
- Configurable dropout rates and model parameters
- Model saving utilities with proper directory structure

The ModelFactory class serves as the main interface for model creation,
abstracting away the complexity of model configuration and initialization.
"""

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
    """
    Factory class for creating and managing transformer models.
    
    This factory provides static methods for creating different types of
    transformer models optimized for Portuguese legal text processing,
    including NER fine-tuning and domain pretraining models.
    """

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
        Create a model and tokenizer for Named Entity Recognition tasks.
        
        Initializes a transformer model for token classification with configurable
        dropout rates and support for domain-adapted checkpoints. Automatically
        configures the model for Portuguese legal NER with appropriate label mappings.
        
        Args:
            model_name (str): Hugging Face model name or path to base model
                (e.g., 'eduagarcia/RoBERTaLexPT-base').
            num_labels (int, optional): Number of NER labels. Defaults to len(ENTITY_LABELS).
            dropout (float, optional): General dropout rate. Defaults to 0.1.
            attention_dropout (float, optional): Attention layer dropout rate. Defaults to 0.1.
            hidden_dropout (float, optional): Hidden layer dropout rate. Defaults to 0.1.
            pretrained_model_path (Optional[str], optional): Path to domain-adapted model
                checkpoint. If provided, loads from this path instead of base model.
                
        Returns:
            Tuple[AutoModelForTokenClassification, AutoTokenizer]: A tuple containing:
                - model: Configured transformer model for token classification
                - tokenizer: Corresponding tokenizer with proper configuration
                
        Side Effects:
            Logs model creation information and parameter count.
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
        
        Initializes a transformer model for masked language modeling to perform
        domain adaptation on Portuguese legal texts. This helps the model learn
        domain-specific vocabulary and language patterns before fine-tuning on NER.
        
        Args:
            model_name (str): Hugging Face model name or path to base model
                (e.g., 'eduagarcia/RoBERTaLexPT-base').
                
        Returns:
            Tuple[AutoModelForMaskedLM, AutoTokenizer]: A tuple containing:
                - model: Transformer model configured for masked language modeling
                - tokenizer: Corresponding tokenizer with prefix space handling
                
        Side Effects:
            Logs model creation information and parameter count.
        """
        tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)

        model = AutoModelForMaskedLM.from_pretrained(model_name)

        logger.info(f"Created pretraining model from {model_name}")
        logger.info(f"Model has {model.num_parameters():,} parameters")

        return model, tokenizer

    @staticmethod
    def get_data_collator(task_type: str, tokenizer: AutoTokenizer):
        """
        Get appropriate data collator for the specified task type.
        
        Returns the correct data collator based on the task type, handling
        the different requirements for NER (token classification) and
        pretraining (masked language modeling) tasks.
        
        Args:
            task_type (str): Type of task, either 'ner' for Named Entity Recognition
                or 'pretraining' for domain-adaptive masked language modeling.
            tokenizer (AutoTokenizer): Tokenizer instance to use with the data collator.
            
        Returns:
            Union[DataCollatorForTokenClassification, DataCollatorForLanguageModeling]:
                - For 'ner': DataCollatorForTokenClassification for handling NER labels
                - For 'pretraining': DataCollatorForLanguageModeling with MLM enabled
                
        Raises:
            ValueError: If task_type is not 'ner' or 'pretraining'.
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
        Save model and tokenizer to disk with proper directory structure.
        
        Saves both the trained model and its corresponding tokenizer to a
        structured directory format that can be easily loaded later for
        inference or further training.
        
        Args:
            model: The trained transformer model to save (either 
                AutoModelForTokenClassification or AutoModelForMaskedLM).
            tokenizer: The tokenizer instance to save alongside the model.
            output_dir (str): Base output directory where models are stored.
            experiment_name (str): Name of the experiment, used as subdirectory name.
            
        Returns:
            str: Full path to the saved model directory.
            
        Side Effects:
            - Creates directory structure if it doesn't exist
            - Saves model weights, configuration, and tokenizer files
            - Logs the save location
        """
        import os
        from pathlib import Path

        save_path = Path(output_dir) / experiment_name
        save_path.mkdir(parents=True, exist_ok=True)

        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

        logger.info(f"Model saved to {save_path}")

        return str(save_path)

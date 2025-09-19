"""
Configuration management for the Portuguese Legal NER training framework.

This module provides comprehensive configuration management through dataclasses
for model, data, training, and experiment settings. The ConfigManager class
handles loading/saving configurations from/to YAML files and provides utilities
for managing experiment configurations.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """
    Configuration for model settings.

    This dataclass encapsulates all model-related parameters including
    the base model name, number of classification labels, and dropout rates
    for different components of the transformer architecture.

    Attributes:
        name (str): Hugging Face model name or local path to model.
        num_labels (int): Number of classification labels (default: 19 for Portuguese legal NER).
        dropout (float): General dropout rate for the model.
        attention_dropout (float): Dropout rate for attention layers.
        hidden_dropout (float): Dropout rate for hidden layers.
    """

    name: str = "eduagarcia/RoBERTaLexPT-base"
    num_labels: int = 19  # 9 entities * 2 (B-, I-) + O
    dropout: float = 0.1
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1


@dataclass
class InferenceConfig:
    """
    Configuration for inference settings.

    This dataclass contains all parameters related to inference execution,
    including model path, input/output files, and processing options.

    Attributes:
        model_path (str): Path to the trained model directory.
        input_file (str): Path to input text file with paragraphs.
        output_file (str): Path to output JSONL file with predictions.
        batch_size (int): Batch size for inference processing.
        max_length (int): Maximum sequence length for tokenization.
        confidence_threshold (float): Minimum confidence for entity predictions.
    """

    model_path: str = ""
    input_file: str = ""
    output_file: str = ""
    batch_size: int = 16
    max_length: int = 512
    confidence_threshold: float = 0.5


@dataclass
class DataConfig:
    """
    Configuration for data loading and preprocessing settings.

    This dataclass contains all parameters related to data loading,
    file paths, and preprocessing options for the NER training pipeline.

    Attributes:
        train_file (str): Path to training data file in CoNLL format.
        val_file (str): Path to validation data file in CoNLL format.
        test_file (str): Path to test data file in CoNLL format.
        max_length (int): Maximum sequence length for tokenization.
        preprocessing_num_workers (int): Number of workers for data preprocessing.
    """

    train_file: str = "data/train.conll"
    val_file: str = "data/val.conll"
    test_file: str = "data/test.conll"
    max_length: int = 512
    preprocessing_num_workers: int = 4


@dataclass
class TrainingConfig:
    """
    Configuration for training hyperparameters and settings.

    This dataclass encompasses all training-related parameters including
    learning rates, batch sizes, evaluation strategies, and hardware options.

    Attributes:
        output_dir (str): Directory to save model checkpoints and outputs.
        num_train_epochs (int): Number of training epochs.
        per_device_train_batch_size (int): Batch size per device during training.
        per_device_eval_batch_size (int): Batch size per device during evaluation.
        gradient_accumulation_steps (int): Number of steps to accumulate gradients.
        learning_rate (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay coefficient for regularization.
        warmup_steps (int): Number of warmup steps for learning rate scheduler.
        logging_steps (int): Frequency of logging training metrics.
        eval_steps (int): Frequency of evaluation during training.
        save_steps (int): Frequency of saving model checkpoints.
        save_total_limit (int): Maximum number of checkpoints to keep.
        evaluation_strategy (str): Strategy for evaluation ("steps" or "epoch").
        load_best_model_at_end (bool): Whether to load best model at training end.
        metric_for_best_model (str): Metric to use for best model selection.
        greater_is_better (bool): Whether higher metric values are better.
        report_to (str): Logging platform ("none", "wandb", "tensorboard").
        seed (int): Random seed for reproducibility.
        fp16 (bool): Whether to use 16-bit floating point precision.
        dataloader_num_workers (int): Number of workers for data loading.
        push_to_hub (bool): Whether to push model to Hugging Face Hub.
    """

    output_dir: str = "models"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 3
    evaluation_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_f1"
    greater_is_better: bool = True
    report_to: str = "none"
    seed: int = 42
    fp16: bool = True
    dataloader_num_workers: int = 4
    push_to_hub: bool = False


@dataclass
class ExperimentConfig:
    """
    Main experiment configuration that aggregates all sub-configurations.

    This dataclass serves as the root configuration object that contains
    all experiment metadata and references to model, data, and training
    configurations.

    Attributes:
        experiment_name (str): Unique name for the experiment.
        experiment_type (str): Type of experiment ("ner_finetuning" or "domain_pretraining").
        description (str): Human-readable description of the experiment.
        tags (list): List of tags for experiment categorization.
        model (ModelConfig): Model configuration settings.
        data (DataConfig): Data loading and preprocessing settings.
        training (TrainingConfig): Training hyperparameters and settings.
        pretraining_data (Optional[str]): Path to pretraining data (for domain adaptation).
        pretrained_model_path (Optional[str]): Path to domain-adapted model checkpoint.
    """

    experiment_name: str = "pt_legal_ner_base"
    experiment_type: str = "ner_finetuning"  # or "domain_pretraining"
    description: str = ""
    tags: list = field(default_factory=list)

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Domain pretraining specific
    pretraining_data: Optional[str] = None
    pretrained_model_path: Optional[str] = None


@dataclass
class InferenceExperimentConfig:
    """
    Configuration for inference experiments.

    This dataclass aggregates inference-specific configurations and
    provides a structured way to configure NER inference on legal documents.

    Attributes:
        experiment_name (str): Unique name for the inference experiment.
        experiment_type (str): Should be "inference" for inference tasks.
        description (str): Human-readable description of the inference task.
        model (ModelConfig): Model configuration (mainly for reference).
        inference (InferenceConfig): Inference-specific settings.
    """

    experiment_name: str = "pt_legal_ner_inference"
    experiment_type: str = "inference"
    description: str = ""

    model: ModelConfig = field(default_factory=ModelConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


class ConfigManager:
    """
    Configuration manager for experiment configurations.

    This class provides utilities for loading, saving, and managing
    experiment configurations from/to YAML files. It handles the conversion
    between YAML format and Python dataclass objects.
    """

    def __init__(self, config_dir: str = "experiments/configs"):
        """
        Initialize the configuration manager.

        Args:
            config_dir (str): Directory to store configuration files.
                Defaults to "experiments/configs".
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def load_config(self, config_path: str) -> ExperimentConfig:
        """
        Load configuration from a YAML file.

        Reads a YAML configuration file and converts it to an ExperimentConfig
        object with nested dataclass instances for model, data, and training
        configurations.

        Args:
            config_path (str): Path to the YAML configuration file.

        Returns:
            ExperimentConfig: Complete experiment configuration object.

        Raises:
            FileNotFoundError: If the configuration file doesn't exist.
            yaml.YAMLError: If the YAML file is malformed.
        """
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Convert nested dicts to dataclass instances
        if "model" in config_dict:
            config_dict["model"] = ModelConfig(**config_dict["model"])
        if "data" in config_dict:
            config_dict["data"] = DataConfig(**config_dict["data"])
        if "training" in config_dict:
            config_dict["training"] = TrainingConfig(**config_dict["training"])

        return ExperimentConfig(**config_dict)

    def load_inference_config(self, config_path: str) -> InferenceExperimentConfig:
        """
        Load inference configuration from a YAML file.

        Reads a YAML configuration file specifically for inference tasks and
        converts it to an InferenceExperimentConfig object with nested
        dataclass instances.

        Args:
            config_path (str): Path to the YAML inference configuration file.

        Returns:
            InferenceExperimentConfig: Complete inference configuration object.

        Raises:
            FileNotFoundError: If the configuration file doesn't exist.
            yaml.YAMLError: If the YAML file is malformed.
        """
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Convert nested dicts to dataclass instances
        if "model" in config_dict:
            config_dict["model"] = ModelConfig(**config_dict["model"])
        if "inference" in config_dict:
            config_dict["inference"] = InferenceConfig(**config_dict["inference"])

        return InferenceExperimentConfig(**config_dict)

    def save_config(self, config: ExperimentConfig, config_path: str):
        """
        Save configuration to a YAML file.

        Converts an ExperimentConfig object to a dictionary format and
        saves it as a YAML file with proper formatting and indentation.

        Args:
            config (ExperimentConfig): Complete experiment configuration to save.
            config_path (str): Path where the YAML file should be saved.

        Raises:
            OSError: If the file cannot be written to the specified path.
        """
        config_dict = {
            "experiment_name": config.experiment_name,
            "experiment_type": config.experiment_type,
            "description": config.description,
            "tags": config.tags,
            "model": {
                "name": config.model.name,
                "num_labels": config.model.num_labels,
                "dropout": config.model.dropout,
                "attention_dropout": config.model.attention_dropout,
                "hidden_dropout": config.model.hidden_dropout,
            },
            "data": {
                "train_file": config.data.train_file,
                "val_file": config.data.val_file,
                "test_file": config.data.test_file,
                "max_length": config.data.max_length,
                "preprocessing_num_workers": config.data.preprocessing_num_workers,
            },
            "training": {
                "output_dir": config.training.output_dir,
                "num_train_epochs": config.training.num_train_epochs,
                "per_device_train_batch_size": config.training.per_device_train_batch_size,
                "per_device_eval_batch_size": config.training.per_device_eval_batch_size,
                "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
                "learning_rate": config.training.learning_rate,
                "weight_decay": config.training.weight_decay,
                "warmup_steps": config.training.warmup_steps,
                "logging_steps": config.training.logging_steps,
                "eval_steps": config.training.eval_steps,
                "save_steps": config.training.save_steps,
                "save_total_limit": config.training.save_total_limit,
                "evaluation_strategy": config.training.evaluation_strategy,
                "load_best_model_at_end": config.training.load_best_model_at_end,
                "metric_for_best_model": config.training.metric_for_best_model,
                "greater_is_better": config.training.greater_is_better,
                "report_to": config.training.report_to,
                "seed": config.training.seed,
                "fp16": config.training.fp16,
                "dataloader_num_workers": config.training.dataloader_num_workers,
                "push_to_hub": config.training.push_to_hub,
            },
        }

        if config.pretraining_data:
            config_dict["pretraining_data"] = config.pretraining_data
        if config.pretrained_model_path:
            config_dict["pretrained_model_path"] = config.pretrained_model_path

        with open(config_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

    def list_configs(self) -> list:
        """
        List all available configuration files.

        Scans the configuration directory for YAML files and returns
        a list of Path objects pointing to configuration files.

        Returns:
            list: List of Path objects for available YAML configuration files.
        """
        return list(self.config_dir.glob("*.yaml"))

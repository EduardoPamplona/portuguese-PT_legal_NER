"""Configuration management for PT Legal NER training."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Configuration for model settings."""

    name: str = "eduagarcia/RoBERTaLexPT-base"
    num_labels: int = 19  # 9 entities * 2 (B-, I-) + O
    dropout: float = 0.1
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1


@dataclass
class DataConfig:
    """Configuration for data settings."""

    train_file: str = "data/train.conll"
    val_file: str = "data/val.conll"
    test_file: str = "data/test.conll"
    max_length: int = 512
    preprocessing_num_workers: int = 4


@dataclass
class TrainingConfig:
    """Configuration for training settings."""

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
    """Main experiment configuration."""

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


class ConfigManager:
    """Configuration manager for experiments."""

    def __init__(self, config_dir: str = "experiments/configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def load_config(self, config_path: str) -> ExperimentConfig:
        """Load configuration from YAML file."""
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

    def save_config(self, config: ExperimentConfig, config_path: str):
        """Save configuration to YAML file."""
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
        """List all available configuration files."""
        return list(self.config_dir.glob("*.yaml"))

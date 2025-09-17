"""
Command-line interface for the Portuguese Legal NER training framework.

This module provides a comprehensive CLI for training and managing Portuguese
Legal Named Entity Recognition models. It supports multiple training workflows
including NER fine-tuning and domain-adaptive pretraining.

Key features:
- NER model training with configurable hyperparameters
- Domain-adaptive pretraining for better legal domain understanding
- Experiment tracking and management
- Configuration file-based training setup
- Resume training from checkpoints
- Experiment listing and inspection

The CLI serves as the primary entry point for users to interact with the
training framework, providing a user-friendly interface for complex ML workflows.
"""

import argparse
import logging
import sys
from pathlib import Path
import os
import warnings

try:
    from .config import ConfigManager, ExperimentConfig
    from .data import DataLoader
    from .models import ModelFactory
    from .training import TrainingManager, compute_metrics
    from .tracking import ExperimentTracker
except ImportError:
    from config import ConfigManager, ExperimentConfig
    from data import DataLoader
    from models import ModelFactory
    from training import TrainingManager, compute_metrics
    from tracking import ExperimentTracker
from transformers import TrainingArguments, set_seed

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def train_ner(config_path: str, resume_from_checkpoint: str = None):
    """
    Train a Named Entity Recognition model for Portuguese legal texts.
    
    Executes a complete NER training pipeline including data loading, model
    initialization, training, evaluation, and experiment tracking. Supports
    resuming from checkpoints and comprehensive result logging.
    
    Args:
        config_path (str): Path to YAML configuration file containing all
            training parameters, model settings, and data paths.
        resume_from_checkpoint (str, optional): Path to checkpoint directory
            to resume training from. If None, starts training from scratch.
            
    Side Effects:
        - Creates experiment directory and tracking files
        - Saves model checkpoints during training  
        - Logs metrics and results to experiment tracker
        - Prints training progress and final results
        - May create sample data if none exists
        
    Raises:
        ValueError: If no training data is available.
        FileNotFoundError: If configuration file doesn't exist.
        Various training-related exceptions from underlying components.
    """
    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.load_config(config_path)

    # Set seed
    set_seed(config.training.seed)

    # Initialize experiment tracking
    tracker = ExperimentTracker()
    experiment_id = tracker.start_experiment(
        experiment_name=config.experiment_name,
        config=config.__dict__,
        experiment_type=config.experiment_type,
    )

    try:
        # Load data
        data_loader = DataLoader(
            tokenizer_name=config.model.name, max_length=config.data.max_length
        )

        datasets = data_loader.load_datasets(
            config.data.train_file, config.data.val_file, config.data.test_file
        )

        if "train" not in datasets:
            raise ValueError("No training data available")

        # Create model and tokenizer
        model, tokenizer = ModelFactory.create_ner_model(
            model_name=config.model.name,
            num_labels=config.model.num_labels,
            dropout=config.model.dropout,
            attention_dropout=config.model.attention_dropout,
            hidden_dropout=config.model.hidden_dropout,
            pretrained_model_path=config.pretrained_model_path,
        )

        # Get data collator
        data_collator = ModelFactory.get_data_collator("ner", tokenizer)

        # Create training arguments
        training_args = TrainingArguments(
            output_dir=f"{config.training.output_dir}/{experiment_id}",
            num_train_epochs=config.training.num_train_epochs,
            per_device_train_batch_size=config.training.per_device_train_batch_size,
            per_device_eval_batch_size=config.training.per_device_eval_batch_size,
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            warmup_steps=config.training.warmup_steps,
            logging_steps=config.training.logging_steps,
            eval_steps=config.training.eval_steps,
            save_steps=config.training.save_steps,
            save_total_limit=config.training.save_total_limit,
            eval_strategy=config.training.evaluation_strategy,
            load_best_model_at_end=config.training.load_best_model_at_end,
            metric_for_best_model=config.training.metric_for_best_model,
            greater_is_better=config.training.greater_is_better,
            report_to=config.training.report_to,
            seed=config.training.seed,
            fp16=config.training.fp16,
            dataloader_num_workers=config.training.dataloader_num_workers,
            push_to_hub=config.training.push_to_hub,
        )

        # Create trainer
        training_manager = TrainingManager(experiment_tracker=tracker)
        trainer = training_manager.create_trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=datasets["train"],
            eval_dataset=datasets.get("validation"),
            data_collator=data_collator,
            training_args=training_args,
            compute_metrics_fn=compute_metrics,
        )

        # Train model
        train_result = training_manager.train(
            trainer=trainer, resume_from_checkpoint=resume_from_checkpoint
        )

        # Log training results
        if hasattr(train_result, "metrics"):
            logger.info(f"Training completed!")
            logger.info(f"Training loss: {train_result.training_loss:.4f}")
            if "train_runtime" in train_result.metrics:
                logger.info(
                    f"Training time: {train_result.metrics['train_runtime']:.2f}s"
                )

        # Evaluate on test set if available
        if "test" in datasets:
            test_result = training_manager.evaluate(
                trainer=trainer, eval_dataset=datasets["test"], metric_key_prefix="test"
            )

            # Log test results
            logger.info(f"Test evaluation completed!")
            for metric, value in test_result.items():
                if isinstance(value, (int, float)):
                    logger.info(f"  {metric}: {value:.4f}")

        # Save model
        model_path = ModelFactory.save_model(
            model=trainer.model,
            tokenizer=tokenizer,
            output_dir=config.training.output_dir,
            experiment_name=experiment_id,
        )

        tracker.log_model_artifact(model_path)

        logger.info(f"Training completed successfully. Model saved to: {model_path}")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        tracker.log_error(str(e))
        raise

    finally:
        tracker.end_experiment()


def train_pretraining(config_path: str, resume_from_checkpoint: str = None):
    """
    Train a domain-adaptive pretraining model for Portuguese legal texts.
    
    Performs masked language model pretraining on domain-specific text to
    adapt a base language model to Portuguese legal vocabulary and patterns.
    This improves performance when subsequently fine-tuning on NER tasks.
    
    Args:
        config_path (str): Path to YAML configuration file. Must include
            pretraining_data field pointing to raw text files.
        resume_from_checkpoint (str, optional): Path to checkpoint directory
            to resume pretraining from. If None, starts from base model.
            
    Side Effects:
        - Creates experiment directory and tracking files
        - Saves model checkpoints during pretraining
        - Logs pretraining metrics and progress
        - Saves final domain-adapted model for later NER fine-tuning
        
    Raises:
        ValueError: If pretraining data is not specified in config.
        FileNotFoundError: If configuration or data files don't exist.
        Various training-related exceptions from underlying components.
    """
    from transformers import DataCollatorForLanguageModeling, Trainer

    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.load_config(config_path)

    # Set seed
    set_seed(config.training.seed)

    # Initialize experiment tracking
    tracker = ExperimentTracker()
    experiment_id = tracker.start_experiment(
        experiment_name=config.experiment_name,
        config=config.__dict__,
        experiment_type="domain_pretraining",
    )

    try:
        # Load pretraining data
        data_loader = DataLoader(
            tokenizer_name=config.model.name, max_length=config.data.max_length
        )

        if not config.pretraining_data:
            raise ValueError("No pretraining data specified in config")

        dataset = data_loader.load_pretraining_data(config.pretraining_data)

        # Split dataset for validation
        dataset = dataset.train_test_split(test_size=0.1, seed=config.training.seed)

        # Create model and tokenizer
        model, tokenizer = ModelFactory.create_pretraining_model(config.model.name)

        # Get data collator for MLM
        data_collator = ModelFactory.get_data_collator("pretraining", tokenizer)

        # Create training arguments
        training_args = TrainingArguments(
            output_dir=f"{config.training.output_dir}/{experiment_id}",
            num_train_epochs=config.training.num_train_epochs,
            per_device_train_batch_size=config.training.per_device_train_batch_size,
            per_device_eval_batch_size=config.training.per_device_eval_batch_size,
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            warmup_steps=config.training.warmup_steps,
            logging_steps=config.training.logging_steps,
            eval_steps=config.training.eval_steps,
            save_steps=config.training.save_steps,
            save_total_limit=config.training.save_total_limit,
            eval_strategy=config.training.evaluation_strategy,
            load_best_model_at_end=config.training.load_best_model_at_end,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=config.training.report_to,
            seed=config.training.seed,
            fp16=config.training.fp16,
            dataloader_num_workers=config.training.dataloader_num_workers,
            push_to_hub=config.training.push_to_hub,
        )

        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            data_collator=data_collator,
            tokenizer=tokenizer,
        )

        # Train model
        tracker.start_training()
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        tracker.log_training_results(train_result)

        # Save model
        model_path = ModelFactory.save_model(
            model=trainer.model,
            tokenizer=tokenizer,
            output_dir=config.training.output_dir,
            experiment_name=experiment_id,
        )

        tracker.log_model_artifact(model_path)
        tracker.end_training()

        logger.info(f"Domain pretraining completed. Model saved to: {model_path}")

    except Exception as e:
        logger.error(f"Domain pretraining failed: {e}")
        tracker.log_error(str(e))
        raise

    finally:
        tracker.end_experiment()


def list_experiments():
    """
    List all available experiments with summary information.
    
    Displays a formatted table of all experiments showing key information
    like experiment ID, name, type, status, and duration. Helps users
    track and compare different experimental runs.
    
    Side Effects:
        - Prints formatted experiment list to console
        - Shows "No experiments found" if none exist
        - Displays experiment metadata in readable format
    """
    tracker = ExperimentTracker()
    experiments = tracker.list_experiments()

    if not experiments:
        print("No experiments found.")
        return

    print(f"\nFound {len(experiments)} experiments:\n")
    print(f"{'ID':<30} {'Name':<25} {'Type':<20} {'Status':<12} {'Start Time'}")
    print("-" * 100)

    for exp in experiments:
        exp_id = (
            exp["experiment_id"][:28] + ".."
            if len(exp["experiment_id"]) > 30
            else exp["experiment_id"]
        )
        name = (
            exp["experiment_name"][:23] + ".."
            if len(exp["experiment_name"]) > 25
            else exp["experiment_name"]
        )
        exp_type = (
            exp["experiment_type"][:18] + ".."
            if len(exp["experiment_type"]) > 20
            else exp["experiment_type"]
        )
        status = exp["status"]
        start_time = exp["start_time"][:16] if exp["start_time"] else "Unknown"

        print(f"{exp_id:<30} {name:<25} {exp_type:<20} {status:<12} {start_time}")


def show_experiment(experiment_id: str):
    """
    Display detailed information about a specific experiment.
    
    Shows comprehensive details about an experiment including configuration,
    training results, evaluation metrics, and file artifacts. Useful for
    analyzing and comparing experimental results.
    
    Args:
        experiment_id (str): Unique identifier of the experiment to display.
            Should be in format "{experiment_name}_{timestamp}".
            
    Side Effects:
        - Prints detailed experiment information to console
        - Shows "Experiment not found" message if ID doesn't exist
        - Displays training results, evaluation metrics, and artifact paths
    """
    tracker = ExperimentTracker()
    exp = tracker.get_experiment(experiment_id)

    if not exp:
        print(f"Experiment {experiment_id} not found.")
        return

    print(f"\nExperiment Details:")
    print("=" * 50)
    print(f"ID: {exp['experiment_id']}")
    print(f"Name: {exp['experiment_name']}")
    print(f"Type: {exp['experiment_type']}")
    print(f"Status: {exp['status']}")
    print(f"Start Time: {exp.get('start_time', 'Unknown')}")
    print(f"End Time: {exp.get('end_time', 'Unknown')}")

    if "training_results" in exp:
        print(f"\nTraining Results:")
        for key, value in exp["training_results"].items():
            print(f"  {key}: {value}")

    if "evaluation_results" in exp:
        print(f"\nEvaluation Results:")
        for key, value in exp["evaluation_results"].items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    if "model_path" in exp:
        print(f"\nModel Path: {exp['model_path']}")

    if "artifacts" in exp and exp["artifacts"]:
        print(f"\nArtifacts:")
        for artifact in exp["artifacts"]:
            print(f"  - {artifact}")


def main():
    """
    Main CLI entry point and command dispatcher.
    
    Parses command-line arguments and dispatches to appropriate functions
    for training, listing experiments, or showing experiment details.
    Provides a user-friendly interface to the training framework.
    
    The CLI supports the following commands:
    - train: Train a NER model using a configuration file
    - pretrain: Perform domain-adaptive pretraining  
    - list: List all available experiments
    - show: Show detailed information about a specific experiment
    
    Side Effects:
        - Sets environment variables to suppress verbose output
        - Configures warning filters for cleaner console output
        - Prints help information if no valid command provided
    """

    # Suppress warnings for cleaner output
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.parallel")
    warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

    parser = argparse.ArgumentParser(
        description="Portuguese Legal NER Training Framework"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train NER command
    train_parser = subparsers.add_parser("train", help="Train a NER model")
    train_parser.add_argument("config", help="Path to configuration file")
    train_parser.add_argument("--resume", help="Resume from checkpoint")

    # Train pretraining command
    pretrain_parser = subparsers.add_parser(
        "pretrain", help="Domain-adaptive pretraining"
    )
    pretrain_parser.add_argument("config", help="Path to configuration file")
    pretrain_parser.add_argument("--resume", help="Resume from checkpoint")

    # List experiments command
    subparsers.add_parser("list", help="List all experiments")

    # Show experiment command
    show_parser = subparsers.add_parser("show", help="Show experiment details")
    show_parser.add_argument("experiment_id", help="Experiment ID to show")

    args = parser.parse_args()

    if args.command == "train":
        train_ner(args.config, args.resume)
    elif args.command == "pretrain":
        train_pretraining(args.config, args.resume)
    elif args.command == "list":
        list_experiments()
    elif args.command == "show":
        show_experiment(args.experiment_id)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

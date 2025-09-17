"""
Training utilities and custom trainer classes for Portuguese Legal NER.

This module provides comprehensive training management for Named Entity
Recognition models, including custom trainer classes with enhanced logging,
evaluation capabilities, and experiment tracking integration.

Key features:
- Custom trainer with experiment tracking integration
- Comprehensive metrics computation for NER evaluation
- Detailed evaluation reports with classification metrics
- Confusion matrix generation and visualization
- Training lifecycle management
- Early stopping and checkpoint handling
- Integration with HuggingFace Transformers training pipeline

The module extends the standard HuggingFace training workflow with additional
monitoring, logging, and evaluation capabilities specifically designed for
NER tasks in the Portuguese legal domain.
"""

import logging
import os
from typing import Dict, Any, Optional
import numpy as np
import torch
from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

try:
    from .data import ID_TO_LABEL
except ImportError:
    from data import ID_TO_LABEL

logger = logging.getLogger(__name__)


def compute_metrics(eval_pred):
    """
    Compute metrics for evaluation.

    Args:
        eval_pred: Predictions and labels from evaluation

    Returns:
        Dictionary of metrics
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [ID_TO_LABEL[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [ID_TO_LABEL[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # Flatten for sklearn metrics
    flat_true_labels = [label for sublist in true_labels for label in sublist]
    flat_predictions = [pred for sublist in true_predictions for pred in sublist]

    # Calculate metrics
    report = classification_report(
        flat_true_labels, flat_predictions, output_dict=True, zero_division=0
    )

    return {
        "precision": report["macro avg"]["precision"],
        "recall": report["macro avg"]["recall"],
        "f1": report["macro avg"]["f1-score"],
        "accuracy": report["accuracy"],
    }


class CustomTrainer(Trainer):
    """
    Enhanced trainer with experiment tracking and detailed evaluation.
    
    Extends HuggingFace Trainer with additional capabilities for experiment
    tracking, detailed evaluation reporting, and enhanced monitoring of
    training progress specifically for NER tasks.
    
    Attributes:
        experiment_tracker: Optional experiment tracker for logging metrics and results.
    """

    def __init__(self, *args, experiment_tracker=None, **kwargs):
        """
        Initialize the custom trainer.
        
        Args:
            *args: Positional arguments passed to parent Trainer class.
            experiment_tracker: Optional ExperimentTracker instance for logging.
            **kwargs: Keyword arguments passed to parent Trainer class.
        """
        super().__init__(*args, **kwargs)
        self.experiment_tracker = experiment_tracker

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Enhanced evaluation with detailed reporting and metrics logging.
        
        Performs standard evaluation and additionally generates detailed
        classification reports and confusion matrices when experiment
        tracker is available.
        
        Args:
            eval_dataset: Dataset to evaluate on. Uses self.eval_dataset if None.
            ignore_keys: Keys to ignore in evaluation results.
            metric_key_prefix (str): Prefix for metric names in results.
            
        Returns:
            dict: Evaluation results dictionary with metrics.
            
        Side Effects:
            - Generates detailed evaluation report if tracker available
            - Logs classification report and confusion matrix
        """
        eval_result = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        if self.experiment_tracker and hasattr(self, "eval_dataset"):
            # Generate detailed evaluation report
            self._generate_detailed_evaluation_report()

        return eval_result

    def _generate_detailed_evaluation_report(self):
        """
        Generate comprehensive evaluation report with classification metrics.
        
        Creates detailed classification report and confusion matrix for
        NER evaluation, providing per-class performance metrics and
        visualization of prediction patterns.
        
        Side Effects:
            - Logs classification report to experiment tracker
            - Generates and logs confusion matrix visualization
            - Logs warning if report generation fails
        """
        try:
            eval_dataloader = self.get_eval_dataloader()
            predictions = self.predict(self.eval_dataset)

            pred_logits = predictions.predictions
            pred_labels = np.argmax(pred_logits, axis=2)
            true_labels = predictions.label_ids

            # Convert to label names
            true_predictions = []
            true_reference = []

            for prediction, label in zip(pred_labels, true_labels):
                for p, l in zip(prediction, label):
                    if l != -100:
                        true_predictions.append(ID_TO_LABEL[p])
                        true_reference.append(ID_TO_LABEL[l])

            # Generate classification report
            report = classification_report(
                true_reference, true_predictions, output_dict=True, zero_division=0
            )

            # Log detailed metrics
            if self.experiment_tracker:
                self.experiment_tracker.log_classification_report(report)

                # Generate and save confusion matrix
                cm = confusion_matrix(true_reference, true_predictions)
                self.experiment_tracker.log_confusion_matrix(
                    cm, list(set(true_reference))
                )

        except Exception as e:
            logger.warning(f"Failed to generate detailed evaluation report: {e}")


class TrainingManager:
    """
    Manager for orchestrating training experiments and workflows.
    
    Provides high-level management of the training process including trainer
    creation, training execution, and evaluation coordination. Integrates
    with experiment tracking and handles training lifecycle management.
    
    Attributes:
        experiment_tracker: Optional experiment tracker for logging and monitoring.
    """

    def __init__(self, experiment_tracker=None):
        """
        Initialize the training manager.
        
        Args:
            experiment_tracker: Optional ExperimentTracker instance for logging
                training progress and results.
        """
        self.experiment_tracker = experiment_tracker

    def create_trainer(
        self,
        model,
        tokenizer,
        train_dataset,
        eval_dataset,
        data_collator,
        training_args: TrainingArguments,
        compute_metrics_fn=None,
        early_stopping_patience: int = 3,
    ) -> CustomTrainer:
        """
        Create and configure a trainer instance for model training.
        
        Sets up a CustomTrainer with all necessary components including
        model, datasets, training arguments, and optional early stopping.
        
        Args:
            model: The transformer model to train (AutoModelForTokenClassification).
            tokenizer: Tokenizer instance matching the model.
            train_dataset: HuggingFace Dataset for training.
            eval_dataset: HuggingFace Dataset for evaluation.
            data_collator: Data collator for batching (DataCollatorForTokenClassification).
            training_args (TrainingArguments): Complete training configuration.
            compute_metrics_fn (callable, optional): Function to compute evaluation metrics.
                Defaults to the standard compute_metrics function.
            early_stopping_patience (int, optional): Number of evaluations with no
                improvement after which training stops. Defaults to 3.
                
        Returns:
            CustomTrainer: Configured trainer instance ready for training.
        """
        callbacks = []
        if early_stopping_patience > 0:
            callbacks.append(
                EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)
            )

        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics_fn or compute_metrics,
            callbacks=callbacks,
            experiment_tracker=self.experiment_tracker,
        )

        return trainer

    def train(
        self, trainer: CustomTrainer, resume_from_checkpoint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute the training process with comprehensive monitoring.
        
        Runs model training with automatic progress tracking, error handling,
        and result logging. Supports resuming from checkpoints and provides
        detailed training statistics.
        
        Args:
            trainer (CustomTrainer): Configured trainer instance ready for training.
            resume_from_checkpoint (Optional[str]): Path to checkpoint directory
                to resume training from. If None, starts training from scratch.
                
        Returns:
            Dict[str, Any]: Training results containing loss curves, timing metrics,
                and other training statistics from the HuggingFace training loop.
                
        Side Effects:
            - Updates experiment tracker with training progress
            - Logs training start/end events
            - Logs detailed training results and metrics
            - Handles and logs training errors
            - Ensures training end is marked even if training fails
            
        Raises:
            Exception: Re-raises any training exceptions after logging them.
        """
        logger.info("Starting training...")

        if self.experiment_tracker:
            self.experiment_tracker.start_training()

        try:
            train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

            logger.info("Training completed successfully")

            if self.experiment_tracker:
                self.experiment_tracker.log_training_results(train_result)

            return train_result

        except Exception as e:
            logger.error(f"Training failed: {e}")
            if self.experiment_tracker:
                self.experiment_tracker.log_error(str(e))
            raise

        finally:
            if self.experiment_tracker:
                self.experiment_tracker.end_training()

    def evaluate(
        self, trainer: CustomTrainer, eval_dataset=None, metric_key_prefix="eval"
    ) -> Dict[str, Any]:
        """
        Perform comprehensive model evaluation with detailed reporting.
        
        Evaluates the trained model on the specified dataset and generates
        detailed evaluation metrics including per-class performance statistics.
        
        Args:
            trainer (CustomTrainer): Trained model instance ready for evaluation.
            eval_dataset: Dataset to evaluate on. If None, uses trainer's eval_dataset.
            metric_key_prefix (str, optional): Prefix for metric names in results.
                Defaults to "eval".
                
        Returns:
            Dict[str, Any]: Comprehensive evaluation results including:
                - Standard metrics (loss, accuracy, precision, recall, F1)
                - Per-class performance statistics
                - Timing and throughput metrics
                
        Side Effects:
            - Logs evaluation progress and completion
            - Updates experiment tracker with evaluation results
            - Generates detailed classification reports if tracker available
        """
        logger.info("Starting evaluation...")

        eval_result = trainer.evaluate(
            eval_dataset=eval_dataset, metric_key_prefix=metric_key_prefix
        )

        logger.info("Evaluation completed")

        if self.experiment_tracker:
            self.experiment_tracker.log_evaluation_results(eval_result)

        return eval_result

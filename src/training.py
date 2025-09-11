"""Training utilities and custom trainer classes."""

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
    """Custom trainer with additional logging and monitoring."""

    def __init__(self, *args, experiment_tracker=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.experiment_tracker = experiment_tracker

    def log(self, logs: Dict[str, float]) -> None:
        """Override log method to add experiment tracking."""
        super().log(logs)

        if self.experiment_tracker:
            self.experiment_tracker.log_metrics(logs, step=self.state.global_step)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Override evaluate to add detailed metrics."""
        eval_result = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        if self.experiment_tracker and hasattr(self, "eval_dataset"):
            # Generate detailed evaluation report
            self._generate_detailed_evaluation_report()

        return eval_result

    def _generate_detailed_evaluation_report(self):
        """Generate detailed evaluation report with confusion matrix."""
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
    """Manager for training experiments."""

    def __init__(self, experiment_tracker=None):
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
        Create a trainer instance.

        Args:
            model: The model to train
            tokenizer: The tokenizer
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            data_collator: Data collator
            training_args: Training arguments
            compute_metrics_fn: Function to compute metrics
            early_stopping_patience: Patience for early stopping

        Returns:
            Trainer instance
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
        Run training.

        Args:
            trainer: The trainer instance
            resume_from_checkpoint: Path to checkpoint to resume from

        Returns:
            Training results
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
        Run evaluation.

        Args:
            trainer: The trainer instance
            eval_dataset: Evaluation dataset (optional)
            metric_key_prefix: Prefix for metric names

        Returns:
            Evaluation results
        """
        logger.info("Starting evaluation...")

        eval_result = trainer.evaluate(
            eval_dataset=eval_dataset, metric_key_prefix=metric_key_prefix
        )

        logger.info("Evaluation completed")

        if self.experiment_tracker:
            self.experiment_tracker.log_evaluation_results(eval_result)

        return eval_result

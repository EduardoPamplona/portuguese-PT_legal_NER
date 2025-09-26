"""
Evaluation module for Portuguese Legal NER model assessment.

This module provides comprehensive evaluation capabilities for assessing
trained Named Entity Recognition models on Portuguese legal documents.
It handles loading trained models, processing test datasets, and generating
detailed evaluation metrics including per-entity performance statistics.

Key features:
- Loading trained NER models with tokenizers
- Processing test data in CoNLL format
- Computing precision, recall, and F1-score for each entity type
- Generating detailed classification reports
- Output generation in structured format
- Batch processing for efficient evaluation
- Error handling and logging for production use

The EvaluationEngine class serves as the main interface for performing
NER model evaluation, providing comprehensive metrics and reporting.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
)
from seqeval.metrics import classification_report as seqeval_classification_report

try:
    from .data import ID_TO_LABEL, DataLoader
    from .config import EvaluationExperimentConfig
    from .training import compute_metrics
except ImportError:
    from data import ID_TO_LABEL, DataLoader
    from config import EvaluationExperimentConfig
    from training import compute_metrics

logger = logging.getLogger(__name__)


class EvaluationEngine:
    """
    Engine for performing NER model evaluation on Portuguese legal documents.

    This class handles loading trained models and performing comprehensive
    evaluation on test datasets, outputting detailed metrics and reports
    suitable for model assessment and comparison.
    """

    def __init__(self, model_path: str, device: str = "auto"):
        """
        Initialize the evaluation engine with a trained model.

        Args:
            model_path (str): Path to the trained model directory containing
                model files, tokenizer, and configuration.
            device (str): Device to run evaluation on ("auto", "cpu", "cuda").
                Auto will use HuggingFace's automatic device detection.

        Raises:
            FileNotFoundError: If model path doesn't exist.
            ValueError: If model files are corrupted or incompatible.
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path {model_path} does not exist")

        self.device = device
        logger.info(f"Loading model from {model_path} for evaluation")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.eval()

        # Move model to appropriate device
        if device == "cuda" and torch.cuda.is_available():
            self.model = self.model.cuda()
        elif device != "auto" and device != "cpu":
            logger.warning(f"Device {device} not available, using CPU")

        logger.info(
            f"Model loaded successfully for evaluation with "
            f"{len(ID_TO_LABEL)} entity labels"
        )

    def evaluate_dataset(
        self,
        test_file: str,
        max_length: int = 512,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """
        Evaluate the model on a test dataset.

        Args:
            test_file (str): Path to test data file in CoNLL format.
            max_length (int): Maximum sequence length for tokenization.
            batch_size (int): Batch size for evaluation processing.

        Returns:
            Dict[str, Any]: Comprehensive evaluation results including:
                - Overall metrics (F1, precision, recall, accuracy)
                - Per-entity-type performance statistics
                - Detailed classification report

        Raises:
            FileNotFoundError: If test file doesn't exist.
            ValueError: If test data is malformed.
        """
        logger.info(f"Starting evaluation on {test_file}")

        # Load test data
        data_loader = DataLoader(
            tokenizer_name=str(self.model_path),
            max_length=max_length
        )

        # Use the existing DataLoader to load just the test dataset
        datasets = data_loader.load_datasets(
            train_file=None,
            val_file=None,
            test_file=test_file
        )

        if "test" not in datasets:
            raise ValueError(f"Could not load test data from {test_file}")

        test_dataset = datasets["test"]
        logger.info(
            f"Loaded test dataset with {len(test_dataset)} examples"
        )

        # Create data collator
        data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            padding=True
        )

        # Create temporary training arguments for evaluation
        eval_args = TrainingArguments(
            output_dir="/tmp/eval_output",
            per_device_eval_batch_size=batch_size,
            dataloader_num_workers=0,  # Avoid multiprocessing issues
            remove_unused_columns=False,
            fp16=torch.cuda.is_available(),
        )

        # Create trainer for evaluation
        trainer = Trainer(
            model=self.model,
            args=eval_args,
            eval_dataset=test_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        # Run evaluation
        logger.info("Running model evaluation...")
        eval_results = trainer.evaluate()

        # Generate additional detailed metrics
        predictions = trainer.predict(test_dataset)
        detailed_metrics = self._generate_detailed_metrics(predictions)

        # Combine results
        final_results = {**eval_results, **detailed_metrics}

        logger.info("Evaluation completed successfully")
        return final_results

    def _generate_detailed_metrics(self, predictions) -> Dict[str, Any]:
        """
        Generate detailed evaluation metrics from predictions.

        Args:
            predictions: Prediction results from trainer.predict()

        Returns:
            Dict[str, Any]: Detailed metrics including classification report
        """
        pred_logits = predictions.predictions
        pred_labels = np.argmax(pred_logits, axis=2)
        true_labels = predictions.label_ids

        # Convert to label names for entity-level evaluation
        true_predictions = [
            [ID_TO_LABEL[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(pred_labels, true_labels)
        ]
        true_reference = [
            [ID_TO_LABEL[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(pred_labels, true_labels)
        ]

        # Generate entity-level classification report
        entity_report = seqeval_classification_report(
            true_reference, true_predictions, output_dict=True, zero_division=0
        )

        return {
            "detailed_classification_report": entity_report,
            "num_test_examples": len(true_predictions),
            "avg_sequence_length": np.mean([len(seq) for seq in true_predictions]),
        }

    def print_evaluation_results(self, results: Dict[str, Any]) -> None:
        """
        Print evaluation results in a formatted way.

        Args:
            results (Dict[str, Any]): Evaluation results dictionary
        """
        print("\n" + "="*60)
        print("PORTUGUESE LEGAL NER MODEL EVALUATION RESULTS")
        print("="*60)

        # Overall metrics
        print("\n📊 OVERALL METRICS:")
        print(f"   Precision: {results.get('eval_precision', 0.0):.4f}")
        print(f"   Recall:    {results.get('eval_recall', 0.0):.4f}")
        print(f"   F1-Score:  {results.get('eval_f1', 0.0):.4f}")
        print(f"   Accuracy:  {results.get('eval_accuracy', 0.0):.4f}")

        # Dataset info
        if 'num_test_examples' in results:
            print("\n📝 DATASET INFO:")
            print(f"   Test Examples: {results['num_test_examples']}")
            avg_len = results.get('avg_sequence_length', 0.0)
            print(f"   Avg Sequence Length: {avg_len:.1f}")

        # Per-entity metrics
        print("\n🏷️  PER-ENTITY METRICS:")
        print(f"{'Entity':<15} {'Precision':<10} {'Recall':<10} "
              f"{'F1-Score':<10} {'Support':<10}")
        print("-" * 60)

        entity_types = []
        for key in results.keys():
            if key.endswith('_f1') and not key.startswith('eval_'):
                entity_type = key.replace('_f1', '')
                entity_types.append(entity_type)

        for entity in sorted(entity_types):
            precision = results.get(f'eval_{entity}_precision', 0.0)
            recall = results.get(f'eval_{entity}_recall', 0.0)
            f1 = results.get(f'eval_{entity}_f1', 0.0)
            support = results.get(f'eval_{entity}_support', 0.0)

            print(f"{entity:<15} {precision:<10.4f} {recall:<10.4f} "
                  f"{f1:<10.4f} {support:<10.0f}")

        print("="*60)

    def save_evaluation_results(
        self,
        results: Dict[str, Any],
        output_file: str,
        save_detailed_report: bool = True
    ) -> None:
        """
        Save evaluation results to file.

        Args:
            results (Dict[str, Any]): Evaluation results dictionary
            output_file (str): Path to output file
            save_detailed_report (bool): Whether to include detailed classification report
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare results for saving
        save_results = results.copy()

        # Remove detailed report if not requested (it can be large)
        if not save_detailed_report and 'detailed_classification_report' in save_results:
            del save_results['detailed_classification_report']

        # Convert numpy types to native Python types for JSON serialization
        for key, value in save_results.items():
            if isinstance(value, np.floating):
                save_results[key] = float(value)
            elif isinstance(value, np.integer):
                save_results[key] = int(value)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(save_results, f, indent=2, ensure_ascii=False)

        logger.info(f"Evaluation results saved to {output_file}")


def load_evaluation_engine(config: EvaluationExperimentConfig) -> EvaluationEngine:
    """
    Factory function to create an EvaluationEngine from configuration.

    Args:
        config (EvaluationExperimentConfig): Evaluation configuration object

    Returns:
        EvaluationEngine: Configured evaluation engine ready for use

    Raises:
        ValueError: If required configuration is missing
    """
    if not config.evaluation.model_path:
        raise ValueError("Model path is required for evaluation")

    return EvaluationEngine(
        model_path=config.evaluation.model_path,
        device="auto"  # Let the engine handle device selection
    )
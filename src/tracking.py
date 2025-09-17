"""
Experiment tracking and management for Portuguese Legal NER training.

This module provides comprehensive experiment tracking capabilities for machine
learning experiments, including metrics logging, artifact management, and 
experiment organization. The ExperimentTracker class serves as the central
component for managing the lifecycle of training experiments.

Key features:
- Experiment lifecycle management (start, track, end)
- Metrics logging with timestamped entries
- Training and evaluation results tracking
- Classification reports and confusion matrix visualization
- Model artifact management
- Experiment listing and retrieval
- Automatic summary generation
- Error logging and debugging support

The tracker creates structured directories for each experiment, maintaining
metadata, metrics, and artifacts in an organized format for easy analysis
and comparison of different experimental runs.
"""

import os
import json
import logging
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """
    Comprehensive experiment tracking and management system.
    
    This class provides a complete solution for tracking machine learning experiments
    including metrics logging, artifact management, and experiment organization.
    It maintains structured directories for each experiment with proper metadata
    and result storage.
    
    Attributes:
        experiments_dir (Path): Base directory for storing all experiments.
        current_experiment (Optional[dict]): Metadata for the currently active experiment.
        experiment_path (Optional[Path]): Path to the current experiment directory.
    """

    def __init__(self, experiments_dir: str = "experiments/runs"):
        """
        Initialize the experiment tracker.
        
        Args:
            experiments_dir (str, optional): Directory to store experiment data.
                Defaults to "experiments/runs".
        """
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.current_experiment = None
        self.experiment_path = None

    def start_experiment(
        self,
        experiment_name: str,
        config: Dict[str, Any],
        experiment_type: str = "ner_finetuning",
    ):
        """
        Start a new experiment and initialize tracking.
        
        Creates a new experiment with a unique timestamp-based identifier,
        sets up the directory structure, and initializes experiment metadata.
        
        Args:
            experiment_name (str): Human-readable name for the experiment.
            config (Dict[str, Any]): Complete experiment configuration dictionary.
            experiment_type (str, optional): Type of experiment. Defaults to "ner_finetuning".
                Valid types: "ner_finetuning", "domain_pretraining".
                
        Returns:
            str: Unique experiment identifier in format "{name}_{timestamp}".
            
        Side Effects:
            - Creates experiment directory
            - Initializes experiment metadata
            - Logs experiment start information
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_id = f"{experiment_name}_{timestamp}"

        self.experiment_path = self.experiments_dir / experiment_id
        self.experiment_path.mkdir(parents=True, exist_ok=True)

        self.current_experiment = {
            "experiment_id": experiment_id,
            "experiment_name": experiment_name,
            "experiment_type": experiment_type,
            "start_time": datetime.datetime.now().isoformat(),
            "config": config,
            "metrics": [],
            "status": "running",
            "artifacts": [],
        }

        # Save initial experiment metadata
        self._save_experiment_metadata()

        logger.info(f"Started experiment: {experiment_id}")

        return experiment_id

    def log_metrics(self, metrics: Dict[str, float], step: int):
        """
        Log metrics for the current training step.
        
        Records training/evaluation metrics with timestamps for tracking
        model performance over time. Metrics are stored both in memory
        and persisted to a JSONL file for later analysis.
        
        Args:
            metrics (Dict[str, float]): Dictionary of metric names and values
                (e.g., {"loss": 0.5, "accuracy": 0.85, "f1": 0.90}).
            step (int): Current training step number.
            
        Side Effects:
            - Appends metrics to current experiment metadata
            - Writes metrics entry to metrics.jsonl file
            - Logs warning if no active experiment
        """
        if not self.current_experiment:
            logger.warning("No active experiment to log metrics to")
            return

        metric_entry = {
            "step": step,
            "timestamp": datetime.datetime.now().isoformat(),
            **metrics,
        }

        self.current_experiment["metrics"].append(metric_entry)

        # Save metrics to file
        metrics_file = self.experiment_path / "metrics.jsonl"
        with open(metrics_file, "a") as f:
            f.write(json.dumps(metric_entry) + "\n")

    def log_training_results(self, train_result):
        """
        Log comprehensive training results and statistics.
        
        Extracts and stores key training metrics from the training results
        object, including loss, throughput metrics, and timing information.
        
        Args:
            train_result: Training result object from HuggingFace Trainer.train().
                Expected to contain training_loss and metrics dictionary with
                runtime statistics.
                
        Side Effects:
            - Updates current experiment metadata with training results
            - Saves updated metadata to disk
        """
        if not self.current_experiment:
            return

        self.current_experiment["training_results"] = {
            "train_loss": float(train_result.training_loss),
            "train_samples_per_second": float(
                train_result.metrics.get("train_samples_per_second", 0)
            ),
            "train_steps_per_second": float(
                train_result.metrics.get("train_steps_per_second", 0)
            ),
            "total_training_time": float(train_result.metrics.get("train_runtime", 0)),
        }

        self._save_experiment_metadata()

    def log_evaluation_results(self, eval_result: Dict[str, Any]):
        """
        Log evaluation results and metrics.
        
        Stores evaluation metrics and results from model evaluation,
        typically including validation/test performance metrics.
        
        Args:
            eval_result (Dict[str, Any]): Dictionary containing evaluation metrics
                and results (e.g., {"eval_loss": 0.3, "eval_f1": 0.92, "eval_accuracy": 0.89}).
                
        Side Effects:
            - Updates current experiment metadata with evaluation results
            - Saves updated metadata to disk
        """
        if not self.current_experiment:
            return

        self.current_experiment["evaluation_results"] = eval_result
        self._save_experiment_metadata()

    def log_classification_report(self, report: Dict[str, Any]):
        """
        Log detailed classification report for NER evaluation.
        
        Saves a comprehensive classification report containing per-class
        precision, recall, F1-score, and support metrics for each entity type.
        
        Args:
            report (Dict[str, Any]): Classification report dictionary from 
                sklearn.metrics.classification_report with output_dict=True.
                Contains per-class metrics and aggregate statistics.
                
        Side Effects:
            - Saves classification report as JSON file
            - Adds report filename to experiment artifacts list
            - Updates experiment metadata
        """
        if not self.current_experiment:
            return

        report_file = self.experiment_path / "classification_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        self.current_experiment["artifacts"].append("classification_report.json")
        self._save_experiment_metadata()

    def log_confusion_matrix(self, cm: np.ndarray, labels: List[str]):
        """
        Log and visualize confusion matrix for model evaluation.
        
        Creates a heatmap visualization of the confusion matrix showing
        prediction accuracy across different entity types and saves it
        as a high-resolution PNG image.
        
        Args:
            cm (np.ndarray): 2D confusion matrix array from sklearn.metrics.confusion_matrix.
                Shape should be (n_classes, n_classes).
            labels (List[str]): List of class label names for axis labeling.
                Should correspond to the order of classes in the confusion matrix.
                
        Side Effects:
            - Creates and saves confusion matrix visualization as PNG
            - Adds confusion matrix filename to experiment artifacts
            - Closes matplotlib figure to free memory
            - Updates experiment metadata
        """
        if not self.current_experiment:
            return

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
        )
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()

        cm_file = self.experiment_path / "confusion_matrix.png"
        plt.savefig(cm_file, dpi=300, bbox_inches="tight")
        plt.close()

        self.current_experiment["artifacts"].append("confusion_matrix.png")
        self._save_experiment_metadata()

    def log_model_artifact(self, model_path: str):
        """
        Log the path to saved model artifacts.
        
        Records the location where the trained model has been saved,
        making it easy to locate and load the model for inference or
        further experimentation.
        
        Args:
            model_path (str): Absolute path to the saved model directory.
                Should contain model weights, configuration, and tokenizer files.
                
        Side Effects:
            - Updates experiment metadata with model path
            - Adds model reference to artifacts list
            - Saves updated metadata
        """
        if not self.current_experiment:
            return

        self.current_experiment["model_path"] = model_path
        self.current_experiment["artifacts"].append(f"model: {model_path}")
        self._save_experiment_metadata()

    def log_error(self, error_msg: str):
        """
        Log error information for failed experiments.
        
        Records error messages when experiments fail, helping with
        debugging and tracking experiment success rates.
        
        Args:
            error_msg (str): Detailed error message describing what went wrong.
                Should include relevant stack trace or diagnostic information.
                
        Side Effects:
            - Updates experiment status to "failed"
            - Records error message in experiment metadata
            - Saves updated metadata for post-mortem analysis
        """
        if not self.current_experiment:
            return

        self.current_experiment["error"] = error_msg
        self.current_experiment["status"] = "failed"
        self._save_experiment_metadata()

    def start_training(self):
        """
        Mark the beginning of the training phase.
        
        Records the timestamp when model training begins, allowing
        for accurate tracking of training duration and phases.
        
        Side Effects:
            - Records training start timestamp in experiment metadata
            - Saves updated metadata to disk
        """
        if not self.current_experiment:
            return

        self.current_experiment["training_start_time"] = (
            datetime.datetime.now().isoformat()
        )
        self._save_experiment_metadata()

    def end_training(self):
        """
        Mark the completion of the training phase.
        
        Records when training finishes and updates the experiment status
        to completed, enabling duration calculations and status tracking.
        
        Side Effects:
            - Records training end timestamp in experiment metadata
            - Updates experiment status to "completed"
            - Saves updated metadata to disk
        """
        if not self.current_experiment:
            return

        self.current_experiment["training_end_time"] = (
            datetime.datetime.now().isoformat()
        )
        self.current_experiment["status"] = "completed"
        self._save_experiment_metadata()

    def end_experiment(self):
        """
        Finalize the current experiment and generate summary.
        
        Marks the experiment as complete, generates a comprehensive summary
        report, and performs final cleanup. This should be called when
        all experiment activities are finished.
        
        Side Effects:
            - Records experiment end timestamp
            - Updates status to "completed" if still running
            - Generates and saves experiment summary
            - Saves final metadata to disk
            - Resets current experiment reference
        """
        if not self.current_experiment:
            return

        self.current_experiment["end_time"] = datetime.datetime.now().isoformat()
        if self.current_experiment["status"] == "running":
            self.current_experiment["status"] = "completed"

        self._save_experiment_metadata()

        logger.info(f"Ended experiment: {self.current_experiment['experiment_id']}")

        # Generate summary report
        self._generate_experiment_summary()

        self.current_experiment = None
        self.experiment_path = None

    def _save_experiment_metadata(self):
        """
        Save current experiment metadata to JSON file.
        
        Persists the complete experiment metadata to a JSON file in the
        experiment directory for later retrieval and analysis.
        
        Side Effects:
            - Writes experiment.json file to experiment directory
            - Uses string conversion for non-serializable objects
        """
        if not self.experiment_path:
            return

        metadata_file = self.experiment_path / "experiment.json"
        with open(metadata_file, "w") as f:
            json.dump(self.current_experiment, f, indent=2, default=str)

    def _generate_experiment_summary(self):
        """
        Generate a comprehensive summary report for the experiment.
        
        Creates a human-readable summary of the experiment including
        key metrics, duration, status, and results. Saves both as JSON
        and prints to console for immediate review.
        
        Side Effects:
            - Creates summary.json file with key experiment metrics
            - Prints formatted summary to console
            - Extracts and formats training and evaluation results
        """
        if not self.current_experiment or not self.experiment_path:
            return

        summary = {
            "Experiment ID": self.current_experiment["experiment_id"],
            "Name": self.current_experiment["experiment_name"],
            "Type": self.current_experiment["experiment_type"],
            "Status": self.current_experiment["status"],
            "Duration": self._calculate_duration(),
        }

        # Add training results if available
        if "training_results" in self.current_experiment:
            summary["Training Loss"] = self.current_experiment["training_results"][
                "train_loss"
            ]
            summary["Training Time"] = (
                f"{self.current_experiment['training_results']['total_training_time']:.2f}s"
            )

        # Add evaluation results if available
        if "evaluation_results" in self.current_experiment:
            eval_results = self.current_experiment["evaluation_results"]
            for key, value in eval_results.items():
                if isinstance(value, (int, float)):
                    summary[f"Eval {key.replace('eval_', '').title()}"] = f"{value:.4f}"

        # Add model path if available
        if "model_path" in self.current_experiment:
            summary["Model Path"] = self.current_experiment["model_path"]

        # Save summary
        summary_file = self.experiment_path / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        # Print summary
        print("\n" + "=" * 50)
        print(f"EXPERIMENT SUMMARY")
        print("=" * 50)
        for key, value in summary.items():
            print(f"{key}: {value}")
        print("=" * 50)

    def _calculate_duration(self) -> str:
        """
        Calculate the duration of the current experiment.
        
        Computes the time elapsed between experiment start and end (or current time
        if still running). Returns a human-readable duration string.
        
        Returns:
            str: Duration in format "HH:MM:SS" or "Unknown" if timing data is missing.
                Microseconds are stripped for cleaner display.
        """
        if not self.current_experiment:
            return "Unknown"

        start_time = self.current_experiment.get("start_time")
        end_time = self.current_experiment.get("end_time")

        if not start_time:
            return "Unknown"

        start_dt = datetime.datetime.fromisoformat(start_time)
        end_dt = (
            datetime.datetime.fromisoformat(end_time)
            if end_time
            else datetime.datetime.now()
        )

        duration = end_dt - start_dt
        return str(duration).split(".")[0]  # Remove microseconds

    def list_experiments(self) -> List[Dict[str, Any]]:
        """
        List all available experiments with their metadata.
        
        Scans the experiments directory for all experiment folders and
        loads their metadata. Returns experiments sorted by start time
        with most recent first.
        
        Returns:
            List[Dict[str, Any]]: List of experiment metadata dictionaries.
                Each dictionary contains experiment configuration, results,
                and status information. Empty list if no experiments found.
                
        Side Effects:
            - Logs warnings for experiments that cannot be loaded
            - Ignores malformed or corrupted experiment directories
        """
        experiments = []

        for exp_dir in self.experiments_dir.iterdir():
            if exp_dir.is_dir():
                metadata_file = exp_dir / "experiment.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file) as f:
                            exp_data = json.load(f)
                            experiments.append(exp_data)
                    except Exception as e:
                        logger.warning(f"Failed to load experiment {exp_dir.name}: {e}")

        # Sort by start time
        experiments.sort(key=lambda x: x.get("start_time", ""), reverse=True)

        return experiments

    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific experiment by its unique identifier.
        
        Searches through all available experiments to find one matching
        the provided experiment ID.
        
        Args:
            experiment_id (str): Unique experiment identifier in format
                "{experiment_name}_{timestamp}".
                
        Returns:
            Optional[Dict[str, Any]]: Experiment metadata dictionary if found,
                None if no experiment with the given ID exists.
        """
        experiments = self.list_experiments()
        for exp in experiments:
            if exp["experiment_id"] == experiment_id:
                return exp
        return None

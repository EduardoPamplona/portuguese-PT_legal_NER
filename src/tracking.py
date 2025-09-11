"""Experiment tracking and management."""

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
    """Track experiments and results."""

    def __init__(self, experiments_dir: str = "experiments/runs"):
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
        Start a new experiment.

        Args:
            experiment_name: Name of the experiment
            config: Experiment configuration
            experiment_type: Type of experiment
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
        """Log metrics for the current step."""
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
        """Log training results."""
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
        """Log evaluation results."""
        if not self.current_experiment:
            return

        self.current_experiment["evaluation_results"] = eval_result
        self._save_experiment_metadata()

    def log_classification_report(self, report: Dict[str, Any]):
        """Log detailed classification report."""
        if not self.current_experiment:
            return

        report_file = self.experiment_path / "classification_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        self.current_experiment["artifacts"].append("classification_report.json")
        self._save_experiment_metadata()

    def log_confusion_matrix(self, cm: np.ndarray, labels: List[str]):
        """Log confusion matrix visualization."""
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
        """Log model artifact path."""
        if not self.current_experiment:
            return

        self.current_experiment["model_path"] = model_path
        self.current_experiment["artifacts"].append(f"model: {model_path}")
        self._save_experiment_metadata()

    def log_error(self, error_msg: str):
        """Log an error."""
        if not self.current_experiment:
            return

        self.current_experiment["error"] = error_msg
        self.current_experiment["status"] = "failed"
        self._save_experiment_metadata()

    def start_training(self):
        """Mark training start."""
        if not self.current_experiment:
            return

        self.current_experiment["training_start_time"] = (
            datetime.datetime.now().isoformat()
        )
        self._save_experiment_metadata()

    def end_training(self):
        """Mark training end."""
        if not self.current_experiment:
            return

        self.current_experiment["training_end_time"] = (
            datetime.datetime.now().isoformat()
        )
        self.current_experiment["status"] = "completed"
        self._save_experiment_metadata()

    def end_experiment(self):
        """End the current experiment."""
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
        """Save experiment metadata to file."""
        if not self.experiment_path:
            return

        metadata_file = self.experiment_path / "experiment.json"
        with open(metadata_file, "w") as f:
            json.dump(self.current_experiment, f, indent=2, default=str)

    def _generate_experiment_summary(self):
        """Generate a summary report for the experiment."""
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
        """Calculate experiment duration."""
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
        """List all experiments."""
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
        """Get a specific experiment by ID."""
        experiments = self.list_experiments()
        for exp in experiments:
            if exp["experiment_id"] == experiment_id:
                return exp
        return None

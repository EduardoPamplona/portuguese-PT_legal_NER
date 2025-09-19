"""Portuguese Legal NER Training Framework."""

__version__ = "0.1.0"
__author__ = "Eduardo"
__description__ = "Portuguese Legal Named Entity Recognition Training Framework"

from .config import (
    ConfigManager,
    ExperimentConfig,
    InferenceExperimentConfig,
    InferenceConfig,
)
from .data import DataLoader, ENTITY_LABELS, LABEL_TO_ID, ID_TO_LABEL
from .models import ModelFactory
from .training import TrainingManager, compute_metrics
from .tracking import ExperimentTracker
from .inference import InferenceEngine, load_inference_engine

__all__ = [
    "ConfigManager",
    "ExperimentConfig",
    "InferenceExperimentConfig",
    "InferenceConfig",
    "DataLoader",
    "ModelFactory",
    "TrainingManager",
    "ExperimentTracker",
    "InferenceEngine",
    "load_inference_engine",
    "compute_metrics",
    "ENTITY_LABELS",
    "LABEL_TO_ID",
    "ID_TO_LABEL",
]

"""
Inference module for Portuguese Legal NER model predictions.

This module provides comprehensive inference capabilities for performing
Named Entity Recognition on Portuguese legal documents. It handles loading
trained models, processing text documents, and generating structured predictions.

Key features:
- Loading trained NER models with tokenizers
- Processing legal documents paragraph by paragraph
- Entity extraction with confidence scoring
- Output generation in JSONL format with character-level spans
- Batch processing for efficient inference
- Error handling and logging for production use

The InferenceEngine class serves as the main interface for performing
NER inference on legal documents, providing both single document and
batch processing capabilities.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline,
)

try:
    from .data import ID_TO_LABEL
    from .config import InferenceExperimentConfig
except ImportError:
    from data import ID_TO_LABEL
    from config import InferenceExperimentConfig

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    Engine for performing NER inference on Portuguese legal documents.

    This class handles loading trained models and performing entity extraction
    on legal documents, outputting results in a structured format suitable
    for downstream processing.
    """

    def __init__(self, model_path: str, device: str = "auto"):
        """
        Initialize the inference engine with a trained model.

        Args:
            model_path (str): Path to the trained model directory containing
                model files, tokenizer, and configuration.
            device (str): Device to run inference on ("auto", "cpu", "cuda").
                Auto will use HuggingFace's automatic device detection.

        Raises:
            FileNotFoundError: If model path doesn't exist.
            ValueError: If model files are corrupted or incompatible.
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path {model_path} does not exist")

        # Use HuggingFace's automatic device detection (same as training)
        self.device = device
        logger.info(
            f"Loading model from {model_path} with device setting: {self.device}"
        )

        # Load tokenizer and model with HuggingFace device management
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_path,
            device_map=(
                device if device != "auto" else None
            ),  # Let HuggingFace handle auto
        )
        self.model.eval()

        # Create inference pipeline with HuggingFace device management
        self.ner_pipeline = pipeline(
            "ner",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map=(
                device if device != "auto" else None
            ),  # Consistent with model loading
            aggregation_strategy="simple",
        )

        logger.info(f"Model loaded successfully with {len(ID_TO_LABEL)} entity labels")

    def process_document(
        self, text: str, confidence_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Process a single document and extract entities.

        Args:
            text (str): Input text document containing legal content.
            confidence_threshold (float): Minimum confidence score for
                including entities in results. Default 0.5.

        Returns:
            List[Dict[str, Any]]: List of entity predictions with format:
                {
                    "text": "paragraph text",
                    "labels": [[start, end, "LABEL"], ...]
                }
        """
        paragraphs = self._split_into_paragraphs(text)
        results = []

        for paragraph in paragraphs:
            if not paragraph.strip():
                continue

            entities = self._extract_entities(paragraph, confidence_threshold)
            if (
                entities or paragraph.strip()
            ):  # Include paragraphs even without entities
                results.append({"text": paragraph, "labels": entities})

        return results

    def process_file(
        self, input_file: str, output_file: str, confidence_threshold: float = 0.5
    ) -> None:
        """
        Process a text file and save results to JSONL format.

        Args:
            input_file (str): Path to input text file with legal documents.
            output_file (str): Path to output JSONL file for predictions.
            confidence_threshold (float): Minimum confidence for entity inclusion.

        Raises:
            FileNotFoundError: If input file doesn't exist.
            IOError: If unable to write to output file.
        """
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file {input_file} does not exist")

        logger.info(f"Processing file: {input_file}")

        # Read input file
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Process document
        results = self.process_document(text, confidence_threshold)

        # Save results
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for result in results:
                json.dump(result, f, ensure_ascii=False)
                f.write("\n")

        logger.info(f"Results saved to: {output_file}")
        logger.info(f"Processed {len(results)} paragraphs with entities")

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs based on line breaks.

        Args:
            text (str): Input text document.

        Returns:
            List[str]: List of paragraph strings.
        """
        # Split by double newlines or single newlines, filter empty lines
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        return paragraphs

    def _extract_entities(self, text: str, confidence_threshold: float) -> List[List]:
        """
        Extract entities from a single text paragraph.

        Args:
            text (str): Input paragraph text.
            confidence_threshold (float): Minimum confidence for inclusion.

        Returns:
            List[List]: List of entity spans in format [start, end, "LABEL"].
        """
        try:
            # Use the NER pipeline to get predictions
            entities = self.ner_pipeline(text)

            # Convert to required format and filter by confidence
            result_entities = []
            for entity in entities:
                if entity["score"] >= confidence_threshold:
                    start = entity["start"]
                    end = entity["end"]
                    label = entity["entity_group"]

                    # Ensure we have valid character indices
                    if 0 <= start < end <= len(text):
                        result_entities.append([start, end, label])

            return result_entities

        except Exception as e:
            logger.warning(f"Error processing text: {e}")
            return []

    def batch_process_files(
        self, input_files: List[str], output_dir: str, confidence_threshold: float = 0.5
    ) -> None:
        """
        Process multiple files in batch and save individual outputs.

        Args:
            input_files (List[str]): List of input file paths.
            output_dir (str): Directory to save output JSONL files.
            confidence_threshold (float): Minimum confidence for inclusion.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for input_file in input_files:
            input_path = Path(input_file)
            output_file = output_path / f"{input_path.stem}_predictions.jsonl"

            try:
                self.process_file(
                    input_file=input_file,
                    output_file=str(output_file),
                    confidence_threshold=confidence_threshold,
                )
            except Exception as e:
                logger.error(f"Error processing {input_file}: {e}")
                continue

        logger.info(f"Batch processing completed. Results in {output_dir}")


def load_inference_engine(config: InferenceExperimentConfig) -> InferenceEngine:
    """
    Factory function to create an InferenceEngine from configuration.

    Args:
        config (InferenceExperimentConfig): Complete inference configuration.

    Returns:
        InferenceEngine: Configured inference engine ready for processing.
    """
    return InferenceEngine(model_path=config.inference.model_path, device="auto")

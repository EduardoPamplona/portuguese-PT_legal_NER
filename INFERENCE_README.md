# Portuguese Legal NER Inference

This document describes how to use the inference feature for performing Named Entity Recognition on Portuguese legal documents.

## Overview

The inference feature allows you to:
- Apply trained NER models to new legal documents
- Extract entities with confidence scores
- Generate structured output in JSONL format
- Process documents paragraph by paragraph

## Quick Start

### 1. Prepare Your Model

First, train a model or use an existing trained model:

```bash
# Train a new model
pt-legal-ner train experiments/configs/ner_base.yaml

# Or use an existing model directory
# Make sure the model directory contains: model.safetensors, config.json, tokenizer files
```

### 2. Create Inference Configuration

Create a YAML configuration file for inference:

```yaml
experiment_name: "pt_legal_ner_inference"
experiment_type: "inference"
description: "Portuguese Legal NER inference on legal documents"

model:
  name: "eduagarcia/RoBERTaLexPT-base"
  num_labels: 19

inference:
  model_path: "models/your_trained_model"  # Path to your trained model
  input_file: "data/input/legal_document.txt"  # Input text file
  output_file: "data/output/predictions.jsonl"  # Output predictions
  batch_size: 16
  max_length: 512
  confidence_threshold: 0.5  # Minimum confidence for entity predictions
```

### 3. Prepare Input Document

Create a text file with your legal document. The text should be clean and divided into paragraphs:

```text
O réu João Silva foi notificado em Lisboa, em 12/02/1990.

A empresa Tech Solutions Ltda. está sediada no Porto e foi fundada em janeiro de 2020.

O processo número 1234/2023 será julgado pelo Tribunal da Relação de Coimbra no dia 15 de março de 2024.
```

### 4. Run Inference

Execute the inference command:

```bash
pt-legal-ner infer experiments/configs/inference_base.yaml
```

### 5. Check Results

The output will be a JSONL file with predictions:

```json
{"text": "O réu João Silva foi notificado em Lisboa, em 12/02/1990.", "labels": [[6, 16, "PER"], [37, 43, "LOC"], [49, 59, "DAT"]]}
{"text": "A empresa Tech Solutions Ltda. está sediada no Porto e foi fundada em janeiro de 2020.", "labels": [[10, 31, "ORG"], [52, 57, "LOC"], [75, 91, "DAT"]]}
```

## Output Format

Each line in the output JSONL file contains:
- `text`: The original paragraph text
- `labels`: Array of entity predictions with format `[start, end, "LABEL"]`
  - `start`: Character index where entity begins
  - `end`: Character index where entity ends  
  - `"LABEL"`: Entity type (e.g., "PER", "LOC", "ORG", "DAT")

## Entity Types

The model recognizes the following Portuguese legal entities:
- **PER**: Pessoa (Person)
- **LOC**: Local (Location)
- **ORG**: Organização (Organization)
- **DAT**: Data (Date)
- **MISC**: Miscellaneous
- And other domain-specific legal entities

## Configuration Options

### Inference Parameters

- `model_path`: Path to the trained model directory
- `input_file`: Path to input text file with legal documents
- `output_file`: Path to output JSONL file for predictions
- `batch_size`: Number of documents to process in parallel (default: 16)
- `max_length`: Maximum sequence length for tokenization (default: 512)
- `confidence_threshold`: Minimum confidence score for including entities (default: 0.5)

### Performance Optimization

- **GPU Usage**: The system automatically uses GPU if available
- **Batch Processing**: Adjust `batch_size` based on your hardware
- **Memory**: Reduce `max_length` if you encounter memory issues

## Advanced Usage

### Batch Processing Multiple Files

You can process multiple files by calling the inference programmatically:

```python
from src.inference import InferenceEngine

engine = InferenceEngine("models/your_model")
engine.batch_process_files(
    input_files=["doc1.txt", "doc2.txt", "doc3.txt"],
    output_dir="output_directory/",
    confidence_threshold=0.6
)
```

### Custom Confidence Thresholds

Adjust the confidence threshold based on your needs:
- **Higher threshold (0.7-0.9)**: More precise, fewer false positives
- **Lower threshold (0.3-0.5)**: More recall, more potential false positives

## Troubleshooting

### Common Issues

1. **Model not found**: Ensure the model path points to a directory with model files
2. **Input file not found**: Check the input file path and permissions
3. **CUDA errors**: Ensure PyTorch CUDA is properly installed for GPU usage
4. **Memory errors**: Reduce batch_size or max_length

### Error Messages

- `FileNotFoundError`: Check file paths in configuration
- `ValueError`: Usually indicates model compatibility issues
- `RuntimeError`: Often memory-related, try reducing batch size

## Examples

See the included sample files:
- `experiments/configs/inference_base.yaml`: Sample configuration
- `data/input/sample_legal_document.txt`: Sample input document
- `test_inference.py`: Test script for verifying the installation
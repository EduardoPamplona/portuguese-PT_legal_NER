# Portuguese Legal NER Model Evaluation

This document describes how to use the evaluation feature for assessing the performance of trained Named Entity Recognition models on Portuguese legal documents.

## Overview

The evaluation feature allows you to:
- Assess trained NER models on test datasets
- Generate comprehensive performance metrics for each entity type
- Compute precision, recall, and F1-score statistics
- Create detailed classification reports
- Save evaluation results for comparison and analysis
- Process test data in CoNLL format efficiently

## Quick Start

### 1. Prepare Your Test Data

Ensure your test data is in CoNLL format with tab-separated tokens and BIO labels:

```
João	B-PER
Silva	I-PER
foi	O
processado	O
no	O
Tribunal	B-ORG
de	I-ORG
Lisboa	I-ORG
.	O

```

### 2. Create Evaluation Configuration

Create a YAML configuration file for evaluation:

```yaml
experiment_name: "pt_legal_ner_evaluation"
experiment_type: "evaluation"
description: "Portuguese Legal NER model evaluation on test data"

model:
  name: "eduagarcia/RoBERTaLexPT-base"
  num_labels: 19

evaluation:
  model_path: "models/your_trained_model"  # Path to your trained model
  test_file: "data/test.conll"             # Test data in CoNLL format
  output_file: "evaluation_results.json"   # Output file for results (optional)
  batch_size: 32
  max_length: 512
  save_predictions: false                   # Save individual predictions
  save_detailed_report: true               # Include detailed classification report
```

### 3. Run Evaluation

```bash
pt-legal-ner evaluate experiments/configs/evaluation_base.yaml
```

### 4. Check Results

The evaluation will display comprehensive results in the console:

```
============================================================
PORTUGUESE LEGAL NER MODEL EVALUATION RESULTS
============================================================

📊 OVERALL METRICS:
   Precision: 0.9156
   Recall:    0.9089
   F1-Score:  0.9122
   Accuracy:  0.9834

📝 DATASET INFO:
   Test Examples: 150
   Avg Sequence Length: 28.5

🏷️  PER-ENTITY METRICS:
Entity          Precision  Recall     F1-Score   Support   
------------------------------------------------------------
PER             0.9500     0.9268     0.9383     41        
ORG             0.8750     0.9333     0.9032     15        
LOC             0.9231     0.8571     0.8889     21        
DAT             0.8889     0.8000     0.8421     10        
IDP             1.0000     0.8571     0.9231     7         
TEL             0.9000     0.9000     0.9000     10        
E-MAIL          1.0000     1.0000     1.0000     3         
CEP             0.8333     1.0000     0.9091     5         
MAT             1.0000     0.7500     0.8571     4         
============================================================
```

## Configuration Options

### Evaluation Parameters

- `model_path`: Path to the trained model directory
- `test_file`: Path to test data file in CoNLL format
- `output_file`: Path to output JSON file for evaluation results (optional)
- `batch_size`: Number of examples to process in parallel (default: 32)
- `max_length`: Maximum sequence length for tokenization (default: 512)
- `save_predictions`: Whether to save individual model predictions (default: false)
- `save_detailed_report`: Whether to include detailed classification report in output (default: true)

### Performance Optimization

- **GPU Usage**: The system automatically uses GPU if available
- **Batch Processing**: Adjust `batch_size` based on your hardware and model size
- **Memory**: Reduce `max_length` if you encounter memory issues

## Output Format

### Console Output

The evaluation displays:
- **Overall Metrics**: Aggregated precision, recall, F1-score, and accuracy
- **Dataset Information**: Number of test examples and average sequence length
- **Per-Entity Metrics**: Detailed performance for each entity type including support counts

### JSON Output (Optional)

If `output_file` is specified, results are saved in JSON format:

```json
{
  "eval_precision": 0.9156,
  "eval_recall": 0.9089,
  "eval_f1": 0.9122,
  "eval_accuracy": 0.9834,
  "eval_PER_precision": 0.9500,
  "eval_PER_recall": 0.9268,
  "eval_PER_f1": 0.9383,
  "eval_PER_support": 41.0,
  "num_test_examples": 150,
  "avg_sequence_length": 28.5,
  "detailed_classification_report": {
    "PER": {
      "precision": 0.95,
      "recall": 0.9268,
      "f1-score": 0.9383,
      "support": 41
    }
  }
}
```

## Entity Types

The evaluation covers all supported Portuguese legal entity types:

| Entity | Description | Example |
|--------|-------------|---------|
| **PER** | Person names | João Silva, Maria Santos |
| **ORG** | Organizations | Tribunal de Justiça, Ministério Público |
| **LOC** | Locations | Lisboa, Porto, Avenida da Liberdade |
| **DAT** | Dates | 15 de março de 2023, 2023-03-15 |
| **IDP** | Identity documents | Processo nº 12345, CC 12345678 |
| **TEL** | Telephone numbers | +351 912 345 678 |
| **E-MAIL** | Email addresses | exemplo@tribunal.pt |
| **CEP** | Postal codes | 1000-001 Lisboa |
| **MAT** | License plates | 12-AB-34 |

## Advanced Usage

### Comparing Models

To compare multiple models, run evaluation on each and save results:

```bash
# Evaluate model A
pt-legal-ner evaluate config_model_a.yaml

# Evaluate model B  
pt-legal-ner evaluate config_model_b.yaml

# Compare results from saved JSON files
```

### Custom Test Sets

You can evaluate on different test sets by updating the configuration:

```yaml
evaluation:
  test_file: "data/custom_test.conll"  # Use custom test data
  output_file: "custom_evaluation.json"
```

### Batch Size Optimization

For optimal performance, adjust batch size based on your hardware:

```yaml
evaluation:
  batch_size: 64  # Increase for better GPU utilization
  # batch_size: 16  # Decrease if you encounter memory issues
```

## Troubleshooting

### Common Issues

1. **Model Not Found**
   ```
   FileNotFoundError: Model path models/your_model does not exist
   ```
   **Solution**: Ensure the model path in configuration points to a valid trained model directory.

2. **Test Data Format Error**
   ```
   ValueError: Could not load test data from data/test.conll
   ```
   **Solution**: Verify test data is in proper CoNLL format with tab-separated tokens and labels.

3. **Memory Issues**
   ```
   RuntimeError: CUDA out of memory
   ```
   **Solution**: Reduce `batch_size` or `max_length` in configuration.

4. **No Entity Labels Found**
   ```
   Warning: No entities found in test data
   ```
   **Solution**: Check that test data contains labeled entities (not all O labels).

### Performance Tips

1. **Use GPU**: Evaluation is much faster on GPU if available
2. **Optimize Batch Size**: Start with 32 and adjust based on memory
3. **Preprocess Data**: Ensure test data is clean and properly formatted
4. **Monitor Memory**: Watch memory usage for large test sets

## Examples

### Basic Evaluation

```bash
# Simple evaluation with default settings
pt-legal-ner evaluate experiments/configs/evaluation_base.yaml
```

### Detailed Evaluation with Output

```yaml
# evaluation_detailed.yaml
evaluation:
  model_path: "models/pt_legal_ner_20240315_143022"
  test_file: "data/test.conll"
  output_file: "results/detailed_evaluation.json"
  batch_size: 32
  save_detailed_report: true
```

```bash
pt-legal-ner evaluate evaluation_detailed.yaml
```

### Production Evaluation

```yaml
# evaluation_production.yaml  
evaluation:
  model_path: "models/production_model"
  test_file: "data/production_test.conll"
  output_file: "results/production_metrics.json"
  batch_size: 64
  max_length: 512
  save_predictions: true
  save_detailed_report: true
```

```bash
pt-legal-ner evaluate evaluation_production.yaml
```

## Integration with Training

The evaluation metrics computed here are consistent with those used during training:

- Same entity-level evaluation using seqeval
- Identical precision, recall, and F1-score calculations
- Compatible with experiment tracking and logging
- Consistent handling of BIO tag sequences

This ensures that evaluation results are directly comparable with training and validation metrics.
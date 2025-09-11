# Portuguese Legal NER Training Framework

A comprehensive, production-ready framework for training Named Entity Recognition (NER) models specifically designed for Portuguese legal documents. This framework supports both standard fine-tuning and domain-adaptive pretraining workflows with built-in experiment tracking and management.

## 🚀 Features

- **🎯 Legal Domain Specialized**: Optimized for Portuguese legal text processing and anonymization
- **🏷️ Comprehensive Entity Support**: Detects 9 entity types crucial for legal document anonymization
- **🔄 Two-Stage Training**: Domain-adaptive pretraining followed by NER fine-tuning for optimal performance
- **📊 Experiment Tracking**: Built-in experiment management with metrics tracking and visualization
- **🏗️ Production Ready**: Modular, configurable, and scalable architecture
- **⚡ Easy to Use**: Simple CLI interface for training and experiment management

## 🏷️ Supported Entity Types

The model recognizes the following entities in Portuguese legal documents:

| Entity | Description | Examples |
|--------|-------------|----------|
| **PER** | Person names | João Silva, Maria Santos |
| **ORG** | Organizations | Tribunal de Justiça, Ministério Público |
| **LOC** | Locations | Lisboa, Porto, Avenida da Liberdade |
| **DAT** | Dates | 15 de março de 2023, 2023-03-15 |
| **IDP** | Identity documents | Processo nº 12345, CC 12345678 |
| **TEL** | Telephone numbers | +351 912 345 678 |
| **E-MAIL** | Email addresses | exemplo@tribunal.pt |
| **CEP** | Postal codes | 1000-001 Lisboa |
| **MAT** | License plates | 12-AB-34 |

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for training)
- 8GB+ RAM
- Git

### Installation

1. **Clone the repository**:
```bash
git clone <your-repo-url>
cd s_train
```

2. **Set up Python environment**:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -e .
```

### Basic Usage

1. **Prepare your data** in CoNLL format:
```bash
# Place your files in data/
data/
├── train.conll
├── val.conll
└── test.conll
```

2. **Train a NER model**:
```bash
pt-legal-ner train experiments/configs/ner_base.yaml
```

3. **View experiment results**:
```bash
pt-legal-ner list
pt-legal-ner show <experiment_id>
```

## 🎯 Training Workflows

### Option 1: Direct NER Fine-tuning (Faster)

Best for quick experimentation or when you have limited computational resources:

```bash
pt-legal-ner train experiments/configs/ner_base.yaml
```

### Option 2: Two-Stage Training (Best Performance)

Recommended for production models:

1. **Domain-adaptive pretraining**:
```bash
# First, prepare raw legal text data
echo "Portuguese legal text corpus..." > data/legal_corpus.txt

# Run domain pretraining
pt-legal-ner pretrain experiments/configs/domain_pretraining.yaml
```

2. **Update configuration** with pretrained model path:
```bash
# Edit experiments/configs/ner_domain_adapted.yaml
# Update: pretrained_model_path: "models/pt_legal_domain_pretraining_YYYYMMDD_HHMMSS"
```

3. **NER fine-tuning**:
```bash
pt-legal-ner train experiments/configs/ner_domain_adapted.yaml
```

## ⚙️ Configuration

Experiments are configured using YAML files. Here's a complete example:

```yaml
# experiments/configs/my_experiment.yaml
experiment_name: "my_legal_ner"
experiment_type: "ner_finetuning"
description: "Custom NER model for legal documents"
tags:
  - "ner"
  - "portuguese"
  - "legal"

model:
  name: "eduagarcia/RoBERTaLexPT-base"
  num_labels: 19
  dropout: 0.1
  attention_dropout: 0.1
  hidden_dropout: 0.1

data:
  train_file: "data/train.conll"
  val_file: "data/val.conll"
  test_file: "data/test.conll"
  max_length: 512
  preprocessing_num_workers: 4

training:
  output_dir: "models"
  num_train_epochs: 5
  per_device_train_batch_size: 16
  per_device_eval_batch_size: 32
  learning_rate: 2e-5
  weight_decay: 0.01
  warmup_steps: 500
  evaluation_strategy: "steps"
  eval_steps: 500
  save_steps: 500
  load_best_model_at_end: true
  metric_for_best_model: "eval_f1"
  seed: 42
  fp16: true
```

## 📊 Experiment Tracking

The framework automatically tracks:

- **Training Metrics**: Loss curves, learning rate schedules
- **Evaluation Results**: Precision, recall, F1-score per entity
- **Confusion Matrix**: Visual analysis of model performance  
- **Model Artifacts**: Saved models and tokenizers
- **Experiment Metadata**: Complete configuration and runtime info

Results are organized in `experiments/runs/<experiment_id>/`:

```
experiments/runs/my_experiment_20240315_143022/
├── experiment.json           # Complete metadata
├── metrics.jsonl            # Training metrics timeline
├── classification_report.json  # Detailed evaluation
├── confusion_matrix.png     # Confusion matrix plot
└── summary.json            # Quick summary
```

## 📁 Data Format

Training data should be in CoNLL format with tab-separated tokens and BIO labels:

```
João	B-PER
Silva	I-PER
foi	O
processado	O
no	O
Tribunal	B-ORG
de	I-ORG
Lisboa	I-ORG
em	O
15	B-DAT
de	I-DAT
março	I-DAT
.	O

```

## 🏗️ Project Structure

```
s_train/
├── src/                          # Core framework
│   ├── __init__.py              # Package initialization
│   ├── cli.py                   # Command-line interface
│   ├── config.py                # Configuration management
│   ├── data.py                  # Data loading and preprocessing
│   ├── models.py                # Model factory and utilities
│   ├── training.py              # Training logic and metrics
│   └── tracking.py              # Experiment tracking
├── experiments/
│   ├── configs/                 # Training configurations
│   │   ├── ner_base.yaml       # Basic NER fine-tuning
│   │   ├── domain_pretraining.yaml  # Domain pretraining
│   │   └── ner_domain_adapted.yaml  # Two-stage training
│   └── runs/                    # Experiment results
├── data/                        # Training data
├── models/                      # Saved models
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## 🛠️ CLI Commands

```bash
# Train a NER model
pt-legal-ner train <config_path> [--resume <checkpoint>]

# Domain pretraining
pt-legal-ner pretrain <config_path> [--resume <checkpoint>]

# List all experiments
pt-legal-ner list

# Show experiment details
pt-legal-ner show <experiment_id>
```

## 🔧 Advanced Usage

### Custom Model Configuration

```yaml
model:
  name: "your-custom-model"
  num_labels: 19
  dropout: 0.15          # Increase for regularization
  attention_dropout: 0.1
  hidden_dropout: 0.1
```

### Training Optimization

```yaml
training:
  # For larger datasets
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 4  # Effective batch size: 32
  
  # For better convergence
  learning_rate: 1e-5
  warmup_steps: 1000
  weight_decay: 0.01
  
  # For faster training
  fp16: true
  dataloader_num_workers: 8
```

### Resuming Training

```bash
pt-legal-ner train config.yaml --resume models/checkpoint-1000
```

## 📈 Performance Tips

1. **Use domain pretraining** for best results on legal text
2. **Adjust batch size** based on GPU memory (8-32 typically works well)
3. **Enable mixed precision** (fp16) for faster training
4. **Use early stopping** to prevent overfitting
5. **Monitor validation metrics** during training

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on top of [Hugging Face Transformers](https://huggingface.co/transformers/)
- Uses [RoBERTaLexPT](https://huggingface.co/eduagarcia/RoBERTaLexPT-base) as the base model
- Inspired by legal NLP research and Portuguese language processing

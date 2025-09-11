#!/bin/bash

# Portuguese Legal NER Training Framework Setup and Execution Script

echo "========================================="
echo "Portuguese Legal NER Training Framework"
echo "========================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -e .

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="

echo ""
echo "Available training options:"
echo ""
echo "1. Direct NER Fine-tuning (Faster):"
echo "   pt-legal-ner train experiments/configs/ner_base.yaml"
echo ""
echo "2. Two-Stage Training (Best Performance):"
echo "   a) Domain pretraining:"
echo "      pt-legal-ner pretrain experiments/configs/domain_pretraining.yaml"
echo "   b) NER fine-tuning (update config with pretrained model path):"
echo "      pt-legal-ner train experiments/configs/ner_domain_adapted.yaml"
echo ""
echo "3. View experiments:"
echo "   pt-legal-ner list"
echo "   pt-legal-ner show <experiment_id>"
echo ""
echo "For more information, see README.md"

#!/bin/bash

# Portuguese Legal NER Training Framework Setup Script
# This script sets up the complete environment after cloning the repository

set -e  # Exit on any error

echo "========================================="
echo "Portuguese Legal NER Training Framework"
echo "Setup Script"
echo "========================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.8+ and try again"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "📍 Found Python version: $PYTHON_VERSION"

# Check if Python version is compatible (3.8+)
if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)" 2>/dev/null; then
    echo "❌ Error: Python 3.8+ is required, found Python $PYTHON_VERSION"
    exit 1
fi

echo "✅ Python version check passed"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "🔧 Creating Python virtual environment..."
    python3 -m venv .venv
    echo "✅ Virtual environment created"
else
    echo "📁 Virtual environment already exists"
fi

# Activate virtual environment
echo "🚀 Activating virtual environment..."
source .venv/bin/activate

# Verify virtual environment activation
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment activated: $VIRTUAL_ENV"
else
    echo "❌ Failed to activate virtual environment"
    exit 1
fi

# Upgrade pip to latest version
echo "📦 Upgrading pip..."
pip install --upgrade pip --quiet

# Install the package and all dependencies
echo "📚 Installing Portuguese Legal NER package and dependencies..."
echo "   This may take a few minutes..."
pip install -e . --quiet

# Verify installation
echo "🔍 Verifying installation..."
if python -c "import src.cli; print('CLI import successful')" 2>/dev/null; then
    echo "✅ Package installation verified"
else
    echo "❌ Package installation failed"
    exit 1
fi

# Test CLI command
if pt-legal-ner --help >/dev/null 2>&1; then
    echo "✅ CLI command 'pt-legal-ner' is available"
else
    echo "❌ CLI command installation failed"
    exit 1
fi

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p data models experiments/runs
echo "✅ Project directories created"

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

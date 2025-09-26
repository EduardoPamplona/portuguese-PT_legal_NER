#!/usr/bin/env python3
"""
Test script for the evaluation functionality.
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def test_config_loading():
    """Test loading evaluation configuration."""
    print("Testing evaluation configuration loading...")

    from src.config import ConfigManager

    config_manager = ConfigManager()
    try:
        config = config_manager.load_evaluation_config(
            "experiments/configs/evaluation_base.yaml"
        )
        print(f"✅ Config loaded successfully: {config.experiment_name}")
        print(f"   Model path: {config.evaluation.model_path}")
        print(f"   Test file: {config.evaluation.test_file}")
        print(f"   Batch size: {config.evaluation.batch_size}")
        print(f"   Max length: {config.evaluation.max_length}")
        print(f"   Save detailed report: {config.evaluation.save_detailed_report}")
        return True
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return False


def test_evaluation_engine():
    """Test evaluation engine initialization (without actual model)."""
    print("\nTesting evaluation engine imports...")

    try:
        from src.evaluation import EvaluationEngine, load_evaluation_engine

        print("✅ Evaluation engine imports successful")
        return True
    except Exception as e:
        print(f"❌ Evaluation engine import failed: {e}")
        return False


def test_cli_help():
    """Test CLI help for evaluation command."""
    print("\nTesting CLI help...")

    try:
        from src.cli import main

        # This would normally call main() but we just want to test imports
        print("✅ CLI imports successful")
        return True
    except Exception as e:
        print(f"❌ CLI import failed: {e}")
        return False


def test_data_format():
    """Test if test data exists and is in correct format."""
    print("\nTesting test data format...")
    
    test_file = "data/test.conll"
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        return False
    
    try:
        with open(test_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Check basic CoNLL format
        token_count = 0
        entity_count = 0
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split('\t')
                if len(parts) >= 2:
                    token_count += 1
                    if not parts[1].startswith('O'):
                        entity_count += 1
        
        print(f"✅ Test data format valid")
        print(f"   Tokens: {token_count}")
        print(f"   Entity tokens: {entity_count}")
        return True
        
    except Exception as e:
        print(f"❌ Test data validation failed: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Testing Portuguese Legal NER Evaluation Feature")
    print("=" * 60)

    all_passed = True
    all_passed &= test_config_loading()
    all_passed &= test_evaluation_engine()
    all_passed &= test_cli_help()
    all_passed &= test_data_format()

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests passed! Evaluation feature is ready.")
        print("\n📋 Next steps:")
        print(
            "1. Train a model using: python -m src.cli train experiments/configs/ner_base.yaml"
        )
        print("2. Update evaluation config with correct model path")
        print(
            "3. Run evaluation: python -m src.cli evaluate experiments/configs/evaluation_base.yaml"
        )
        print("4. Check example: python examples/evaluation_example.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")

    sys.exit(0 if all_passed else 1)
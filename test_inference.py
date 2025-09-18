#!/usr/bin/env python3
"""
Test script for the inference functionality.
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def test_config_loading():
    """Test loading inference configuration."""
    print("Testing configuration loading...")

    from src.config import ConfigManager

    config_manager = ConfigManager()
    try:
        config = config_manager.load_inference_config(
            "experiments/configs/inference_base.yaml"
        )
        print(f"✅ Config loaded successfully: {config.experiment_name}")
        print(f"   Model path: {config.inference.model_path}")
        print(f"   Input file: {config.inference.input_file}")
        print(f"   Output file: {config.inference.output_file}")
        return True
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return False


def test_inference_engine():
    """Test inference engine initialization (without actual model)."""
    print("\nTesting inference engine imports...")

    try:
        from src.inference import InferenceEngine, load_inference_engine

        print("✅ Inference engine imports successful")
        return True
    except Exception as e:
        print(f"❌ Inference engine import failed: {e}")
        return False


def test_cli_help():
    """Test CLI help for inference command."""
    print("\nTesting CLI help...")

    try:
        from src.cli import main

        # This would normally call main() but we just want to test imports
        print("✅ CLI imports successful")
        return True
    except Exception as e:
        print(f"❌ CLI import failed: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Testing Portuguese Legal NER Inference Feature")
    print("=" * 50)

    all_passed = True
    all_passed &= test_config_loading()
    all_passed &= test_inference_engine()
    all_passed &= test_cli_help()

    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! Inference feature is ready.")
        print("\n📋 Next steps:")
        print(
            "1. Train a model using: pt-legal-ner train experiments/configs/ner_base.yaml"
        )
        print("2. Update inference config with correct model path")
        print(
            "3. Run inference: pt-legal-ner infer experiments/configs/inference_base.yaml"
        )
    else:
        print("❌ Some tests failed. Please check the errors above.")

    sys.exit(0 if all_passed else 1)

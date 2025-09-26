"""
Example evaluation script for trained Portuguese Legal NER models.
This script demonstrates how to use a trained model for evaluation.
"""

import os
import sys

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluation import EvaluationEngine
from config import ConfigManager


def evaluate_model(model_path: str, test_file: str):
    """Evaluate a trained model on test data."""
    print(f"Evaluating model: {model_path}")
    print(f"Test data: {test_file}")
    
    try:
        # Initialize evaluation engine
        evaluation_engine = EvaluationEngine(model_path)
        
        # Run evaluation
        results = evaluation_engine.evaluate_dataset(
            test_file=test_file,
            max_length=512,
            batch_size=32
        )
        
        # Print results
        evaluation_engine.print_evaluation_results(results)
        
        # Save results
        output_file = f"evaluation_results_{os.path.basename(model_path)}.json"
        evaluation_engine.save_evaluation_results(
            results=results,
            output_file=output_file,
            save_detailed_report=True
        )
        
        print(f"\nResults saved to: {output_file}")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")


def evaluate_with_config(config_path: str):
    """Evaluate using configuration file."""
    print(f"Using configuration: {config_path}")
    
    try:
        # Load configuration
        config_manager = ConfigManager()
        config = config_manager.load_evaluation_config(config_path)
        
        # Initialize evaluation engine
        evaluation_engine = EvaluationEngine(config.evaluation.model_path)
        
        # Run evaluation
        results = evaluation_engine.evaluate_dataset(
            test_file=config.evaluation.test_file,
            max_length=config.evaluation.max_length,
            batch_size=config.evaluation.batch_size
        )
        
        # Print results
        evaluation_engine.print_evaluation_results(results)
        
        # Save results if specified
        if config.evaluation.output_file:
            evaluation_engine.save_evaluation_results(
                results=results,
                output_file=config.evaluation.output_file,
                save_detailed_report=config.evaluation.save_detailed_report
            )
            print(f"\nResults saved to: {config.evaluation.output_file}")
            
    except Exception as e:
        print(f"Evaluation failed: {e}")


def main():
    print("🔍 Portuguese Legal NER Model Evaluation Example")
    print("=" * 60)
    
    # Example 1: Direct evaluation (update paths as needed)
    model_path = "models/your_experiment_id_here"
    test_file = "data/test.conll"
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please train a model first or update the model_path.")
        print("Available models:")
        models_dir = "models"
        if os.path.exists(models_dir):
            for item in os.listdir(models_dir):
                if os.path.isdir(os.path.join(models_dir, item)):
                    print(f"  - {item}")
        
        print("\nAlternatively, use configuration-based evaluation:")
        print("python examples/evaluation_example.py config")
        return
    
    if not os.path.exists(test_file):
        print(f"Test file not found at {test_file}")
        print("Please prepare test data in CoNLL format.")
        return
    
    # Check if user wants to use config
    if len(sys.argv) > 1 and sys.argv[1] == "config":
        config_path = "experiments/configs/evaluation_base.yaml"
        print(f"\n🔧 Configuration-based evaluation")
        evaluate_with_config(config_path)
    else:
        print(f"\n🔧 Direct evaluation")
        evaluate_model(model_path, test_file)


if __name__ == "__main__":
    main()
"""
Example inference script for trained Portuguese Legal NER models.
This script demonstrates how to use a trained model for entity recognition.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

from src.data import ID_TO_LABEL


def load_model(model_path: str):
    """Load trained model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(model_path)

    # Create pipeline
    ner_pipeline = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=0 if torch.cuda.is_available() else -1,
    )

    return ner_pipeline


def predict_entities(text: str, ner_pipeline):
    """Predict entities in text."""
    results = ner_pipeline(text)

    print(f"\nText: {text}")
    print("\nDetected entities:")
    print("-" * 50)

    for entity in results:
        print(f"Entity: {entity['word']}")
        print(f"Label: {entity['entity_group']}")
        print(f"Score: {entity['score']:.4f}")
        print(f"Start: {entity['start']}, End: {entity['end']}")
        print("-" * 30)


def main():
    # Example usage - update model_path to your trained model
    model_path = "models/your_experiment_id_here"

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please train a model first or update the model_path.")
        print("Available models:")
        models_dir = "models"
        if os.path.exists(models_dir):
            for item in os.listdir(models_dir):
                if os.path.isdir(os.path.join(models_dir, item)):
                    print(f"  - {item}")
        return

    # Load model
    print("Loading model...")
    ner_pipeline = load_model(model_path)

    # Example Portuguese legal text
    test_texts = [
        "O processo número 12345/2023 do réu João Silva foi julgado em 15 de março de 2023.",
        "Maria Santos, residente na Rua da Liberdade 123, 1000-001 Lisboa, contactável através do email maria@exemplo.pt.",
        "O Tribunal de Justiça de Lisboa decidiu sobre o caso com matrícula AB-12-34.",
    ]

    # Predict entities
    for text in test_texts:
        predict_entities(text, ner_pipeline)
        print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()

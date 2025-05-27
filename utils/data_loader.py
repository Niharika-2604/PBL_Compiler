"""
data_loader.py

Loads and processes the JSON dataset (code_pairs.json) containing
original/optimized code pairs for optimization modeling and ML training.

Author: Your Name
"""

import json
import os
from typing import List, Dict, Any, Tuple, Optional

# Safely attempt to import in case this is run standalone
try:
    from utils.ast_utils import get_ast_features
except ImportError:
    def get_ast_features(code: str) -> Dict[str, Any]:
        return {}

# ===============================
# Configuration
# ===============================

DEFAULT_JSON_PATH = os.path.join("data", "code_pairs.json")

# ===============================
# Public Methods
# ===============================

def load_code_pairs(json_path: str = DEFAULT_JSON_PATH) -> List[Dict[str, Any]]:
    """
    Load code pairs dataset from JSON file.

    Args:
        json_path (str): Path to JSON dataset.

    Returns:
        List[Dict]: Loaded list of examples.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Dataset not found: {json_path}")
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Optional: validate keys
    required_keys = {"id", "original_code", "optimized_code", "transformation"}
    for i, item in enumerate(data):
        if not required_keys.issubset(item.keys()):
            raise ValueError(f"Sample {i} is missing required keys: {required_keys - set(item.keys())}")
    
    return data


def get_feature_label_dataset(samples: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], str]]:
    """
    Extract (features, label) tuples for ML from code samples.

    Args:
        samples (List): Code samples loaded from JSON.

    Returns:
        List[Tuple[Dict, str]]: List of (features, label) pairs.
    """
    dataset = []

    for item in samples:
        code = item.get("original_code", "")
        label = item.get("transformation")

        if not code or not label:
            continue

        features = get_ast_features(code)
        if features:
            dataset.append((features, label))

    return dataset


def preview_samples(samples: List[Dict[str, Any]], count: int = 3):
    """
    Print a few samples for debugging/inspection.

    Args:
        samples (List): Loaded data
        count (int): Number of samples to preview.
    """
    print("\n🔍 Previewing Samples:")
    print("=" * 50)
    for sample in samples[:count]:
        print(f"ID: {sample['id']}")
        print(f"Transformation: {sample['transformation']}")
        print("Original Code:\n", sample['original_code'])
        print("Optimized Code:\n", sample['optimized_code'])
        print("-" * 50)

# ===============================
# Standalone Test Block
# ===============================

if __name__ == "__main__":
    print("📥 Loading dataset...")
    try:
        samples = load_code_pairs()
        preview_samples(samples)

        print("\n🧠 Extracting features + labels...")
        dataset = get_feature_label_dataset(samples)

        for features, label in dataset[:3]:
            print(f"\nLabel: {label}")
            print("Features:", features)

        print(f"\n✅ Loaded {len(dataset)} samples with features.")
    except Exception as e:
        print("❌ Error:", e)
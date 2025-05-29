
from typing import List, Tuple, Dict
from utils.data_loader import load_code_pairs
from utils.ast_utils import get_ast_features


def extract_features_and_labels(data_path: str = "data/code_pairs.json") -> Tuple[List[Dict], List[str]]:
    """
    Load the dataset and extract features/labels for training.

    Args:
        data_path (str): Path to code_pairs.json

    Returns:
        X (List[Dict]): Feature vectors
        y (List[str]): Corresponding transformation labels
    """
    data = load_code_pairs(data_path)

    X = []  # Feature vectors
    y = []  # Labels

    for sample in data:
        code = sample.get("original_code", "")
        label = sample.get("transformation", None)
        
        if not code or not label:
            continue

        features = get_ast_features(code)
        if features:
            X.append(features)
            y.append(label)

    return X, y


def extract_with_metrics(data_path: str = "data/code_pairs.json") -> Tuple[List[Dict], List[str]]:
    """
    Extract features and include performance metrics (optional enrichment).

    Args:
        data_path (str): JSON dataset path

    Returns:
        X (List[Dict]): Feature + metrics per sample
        y (List[str]): Labels
    """
    data = load_code_pairs(data_path)
    X, y = [], []

    for sample in data:
        code = sample.get("original_code", "")
        label = sample.get("transformation", None)
        metrics = sample.get("metrics", {})

        if not code or not label:
            continue

        features = get_ast_features(code)

        # Optionally add performance metrics as features
        if "original_time_ms" in metrics:
            features["runtime_ms"] = metrics["original_time_ms"]
        if "speedup_percent" in metrics:
            features["speedup_pct"] = metrics["speedup_percent"]

        X.append(features)
        y.append(label)

    return X, y


def preview_feature_set(X, y, count: int = 3):
    """
    Print sample features for inspection.

    Args:
        X: List of feature dicts OR Pandas DataFrame
        y: List of labels
        count (int): Number of records to preview
    """
    print("🔍 Feature Preview:")
    print("=" * 40)

    for i in range(min(count, len(y))):
        print(f"Label: {y[i]}")

        # ✅ Support list of dicts or DataFrame
        if isinstance(X, list):
            print("Features:", X[i])
        else:
            print("Features:", X.iloc[i].to_dict())

        print("-" * 40)


# ===============================
# Test Run
# ===============================

if __name__ == "__main__":
    print("🔧 Extracting features and labels...")
    X, y = extract_features_and_labels()

    print(f"\n✅ Extracted {len(X)} samples.")
    preview_feature_set(X, y)
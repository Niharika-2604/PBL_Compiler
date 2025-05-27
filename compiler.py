# Main orchestrator
"""
compiler.py

Main orchestrator for the AI-assisted compiler.
Predicts and applies optimizations to Python code
based on learned ML models and transformation modules.

Author: Your Name
"""

import argparse
import os

from utils.ast_utils import get_ast_features
from utils.metrics import measure_runtime, compare_metrics
from models.optimization_model import load_model, predict_transformation
from transformer.code_transformer import apply_transformation


# ===============================
# Core Pipeline
# ===============================

def optimize_code(code: str, model_path="models/optimization_predictor.pkl") -> dict:
    """
    Run the AI-assisted optimization pipeline on input code.

    Args:
        code (str): Original Python code snippet
        model_path (str): Path to trained ML model

    Returns:
        dict: Contains original, optimized code and metrics
    """
    print("🔍 Extracting code features...")
    features = get_ast_features(code)
    if not features:
        raise ValueError("Failed to parse or extract features from code.")

    print("🤖 Loading model and predicting transformation...")
    model = load_model(model_path)
    predicted_transformation = predict_transformation(model, features)

    print(f"✅ Suggested Optimization: {predicted_transformation}")

    print("🔁 Applying transformation...")
    optimized_code = apply_transformation(code, predicted_transformation)

    print("⏱️ Measuring performance...")
    orig_time = measure_runtime(code)
    opt_time = measure_runtime(optimized_code)

    metrics = compare_metrics(orig_time, opt_time)

    return {
        "predicted_transformation": predicted_transformation,
        "original_time_ms": orig_time,
        "optimized_time_ms": opt_time,
        "speedup_percent": metrics.get("speedup_percent", 0),
        "original_code": code,
        "optimized_code": optimized_code
    }

# ===============================
# CLI Interface
# ===============================

def main():
    parser = argparse.ArgumentParser(description="AI-Assisted Code Compiler")
    parser.add_argument("filepath", help="Path to a Python file to optimize")
    parser.add_argument("--output", help="Output path for optimized code", default="optimized.py")
    parser.add_argument("--show", action="store_true", help="Print optimized code to console")

    args = parser.parse_args()

    with open(args.filepath, "r") as f:
        code = f.read()

    print("🚀 Starting AI-assisted compilation...\n")

    try:
        result = optimize_code(code)

        print("\n📈 Performance Summary:")
        print(f"Original Time: {result['original_time_ms']} ms")
        print(f"Optimized Time: {result['optimized_time_ms']} ms")
        print(f"Speedup: {result['speedup_percent']} %")

        with open(args.output, "w") as f:
            f.write(result['optimized_code'])
        print(f"\n💾 Optimized code saved to {args.output}")

        if args.show:
            print("\n🔍 Optimized Code:\n")
            print(result['optimized_code'])

    except Exception as e:
        print("❌ Error during optimization:", e)


# ===============================
# Run from CLI
# ===============================

if __name__ == "__main__":
    main()
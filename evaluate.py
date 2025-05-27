# Benchmark original + optimized
"""
evaluate.py

Evaluates the performance of the trained optimization model across the code_pairs.json dataset.
Compares the model's predictions with the actual transformations,
and collects accuracy + reports.

Author: Your Name
"""

from utils.data_loader import load_code_pairs
from utils.ast_utils import get_ast_features
from models.optimization_model import load_model, predict_transformation
from utils.metrics import measure_runtime, compare_metrics

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import argparse

# ====================================
# Evaluation Workflow
# ====================================

def evaluate_model(model_path: str = "models/optimization_predictor.pkl", show_wrong: bool = False):
    print("📥 Loading dataset and model...")
    data = load_code_pairs()
    model = load_model(model_path)

    y_true = []
    y_pred = []

    for sample in data:
        features = get_ast_features(sample["original_code"])
        true_label = sample['transformation']
        if not features:
            continue

        predicted_label = predict_transformation(model, features)

        y_true.append(true_label)
        y_pred.append(predicted_label)

        if show_wrong and predicted_label != true_label:
            print(f"\n❌ Incorrect prediction for {sample['id']}")
            print(f"   True: {true_label} | Predicted: {predicted_label}")

    print("\n✅ Evaluation Results:")
    print(classification_report(y_true, y_pred))
    acc = accuracy_score(y_true, y_pred)
    print(f"🎯 Accuracy: {round(acc * 100, 2)}%")

    # Optional: confusion matrix
    try:
        from sklearn.metrics import ConfusionMatrixDisplay
        import matplotlib.pyplot as plt
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title("Confusion Matrix")
        plt.show()
    except ImportError:
        pass

# ====================================
# Re-run Performance Benchmarks (Optional)
# ====================================

def evaluate_speedups():
    print("\n🚀 Re-evaluating runtime performance across dataset...")
    data = load_code_pairs()

    results = []

    for sample in data:
        id = sample['id']
        print(f"🔎 Evaluating {id}...")

        orig = sample['original_code']
        opt = sample['optimized_code']

        try:
            t_orig = measure_runtime(orig)
            t_opt = measure_runtime(opt)
            perf = compare_metrics(t_orig, t_opt)
            results.append((id, perf['speedup_percent']))
            print(f"🏁 Speedup: {perf['speedup_percent']}%")
        except Exception as e:
            print(f"⚠️ Skipped {id}: {e}")

    # Summary
    valid = [v for _, v in results]
    if valid:
        avg = sum(valid) / len(valid)
        print(f"\n📊 Avg Speedup Across {len(valid)} samples: {round(avg, 2)}%")
    else:
        print("❌ Could not compute speedups.")


# ====================================
# CLI Interface
# ====================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate optimization model and runtime performance.")
    parser.add_argument("--metrics", action="store_true", help="Evaluate runtime speedups")
    parser.add_argument("--show-wrong", action="store_true", help="Print each misprediction")
    args = parser.parse_args()

    evaluate_model(show_wrong=args.show_wrong)

    if args.metrics:
        evaluate_speedups()


if __name__ == "__main__":
    main()
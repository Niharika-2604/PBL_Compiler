

from models.feature_extractor import extract_features_and_labels, preview_feature_set
from models.optimization_model import train_model
import argparse
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os
import pandas as pd


# =========================
# Config
# =========================
MODEL_DIR = "models"
MODEL_FILENAME = "optimization_predictor.pkl"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

# =========================
# Train and Save Model
# =========================

def run_training(save_model: bool = True, test_accuracy: bool = True):
    print("📥 Loading dataset...")
    X_dicts, y = extract_features_and_labels()
    X = pd.DataFrame(X_dicts)

    if len(X) == 0:
        print("❌ No training data found.")
        return

    print(f"✅ Loaded {len(X)} samples.")

    preview_feature_set(X, y, count=2)

    print("\n🧠 Splitting into train/test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("🏋️ Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    if test_accuracy:
        print("\n📊 Evaluating model on test set...")
        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred))
        print("✅ Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

    if save_model:
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        joblib.dump(model, MODEL_PATH)
        print(f"💾 Model saved to {MODEL_PATH}")

    return model


# ===============================
# CLI Support
# ===============================

def main():
    parser = argparse.ArgumentParser(description="Train the optimization model")
    parser.add_argument("--nosave", action="store_true", help="Do not save the model after training")
    parser.add_argument("--notest", action="store_true", help="Do not show evaluation metrics")
    args = parser.parse_args()

    run_training(save_model=not args.nosave, test_accuracy=not args.notest)


if __name__ == "__main__":
    main()
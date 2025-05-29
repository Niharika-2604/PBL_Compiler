

from typing import List, Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

# Feature extraction
from models.feature_extractor import extract_features_and_labels



MODEL_DIR = "models"
MODEL_FILENAME = "optimization_predictor.pkl"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

# ===============================
# Train Model
# ===============================

def train_model(save: bool = True) -> RandomForestClassifier:
   
    X, y = extract_features_and_labels()

    if not X or not y:
        raise ValueError("❌ No data found for training.")

    # Prepare training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Use RandomForest (you can switch to others)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("\n🔎 Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save
    if save:
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        joblib.dump(model, MODEL_PATH)
        print(f"\n✅ Model saved to {MODEL_PATH}")

    return model

# ===============================
# Load Model
# ===============================

def load_model(path: str = MODEL_PATH) -> RandomForestClassifier:
   
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Model file not found: {path}")
    model = joblib.load(path)
    print(f"📦 Loaded model from {path}")
    return model

# ===============================
# Predict on New Feature Dict
# ===============================

def predict_transformation(model, features: Dict[str, float]) -> str:
    
    if not model or not features:
        return "unknown"

    try:
        import pandas as pd
        features_df = pd.DataFrame([features])
        prediction = model.predict(features_df)[0]
        return prediction
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return "error"

# ===============================
# Test Mode
# ===============================

def get_feature_importances(model):
    """
    Get feature importances from the trained RandomForest model.

    Args:
        model: Trained RandomForestClassifier

    Returns:
        dict: Mapping of feature names to importance scores
    """
    try:
        feature_names = []
        # Extract feature names from training data
        from models.feature_extractor import extract_features_and_labels
        X, _ = extract_features_and_labels()
        if not X:
            return {}
        feature_names = list(X[0].keys())
        importances = model.feature_importances_
        return dict(zip(feature_names, importances))
    except Exception as e:
        print(f"❌ Failed to get feature importances: {e}")
        return {}

if __name__ == "__main__":
    print("🚀 Training optimization prediction model...")
    model = train_model()

    # Test on one of the samples
    from models.feature_extractor import extract_features_and_labels
    X, y = extract_features_and_labels()
    print("\n🧠 Predicting on sample feature:")
    pred = predict_transformation(model, X[0])
    print(f"👉 Prediction: {pred} | True: {y[0]}")

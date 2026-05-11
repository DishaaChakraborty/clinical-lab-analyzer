import os
import pickle
import numpy as np
import pandas as pd
from ml.feature_engineering import AdvancedFeatureEngineer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

agent2_path = os.path.join(BASE_DIR, "ml", "models", "agent2.pkl")
agent4_path = os.path.join(BASE_DIR, "ml", "models", "agent4.pkl")

# Load Agent 2
with open(agent2_path, "rb") as f:
    data2 = pickle.load(f)
    agent2 = data2["model"]

# Load Agent 4
with open(agent4_path, "rb") as f:
    data4 = pickle.load(f)
    agent4 = data4["model"]

# Load scaler and feature names
scaler_path = os.path.join(BASE_DIR, "ml", "models", "scaler.pkl")
feature_names_path = os.path.join(BASE_DIR, "ml", "models", "feature_names.pkl")

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

with open(feature_names_path, "rb") as f:
    feature_names = pickle.load(f)

# =========================
# LABEL MAPPING
# =========================

DISEASE_CLASSES = {
    0: "Low",
    1: "Medium",
    2: "High"
}

# =========================
# MAIN PREDICT FUNCTION
# =========================

def predict(input_data: dict):
    """
    input_data: dictionary of lab values
    """

    # Create DataFrame from input
    df = pd.DataFrame([input_data])

    # Apply feature engineering
    engineer = AdvancedFeatureEngineer()
    df, _ = engineer.engineer_features(df)

    # Ensure feature order matches training
    features = scaler.transform(df[feature_names].fillna(0))

    # Agent 2 → disease classification
    pred_class = agent2.predict(features)[0]
    pred_label = DISEASE_CLASSES.get(pred_class, "Unknown")

    # Agent 4 → risk score
    risk_pred = agent4.predict(features)[0]

    return {
        "disease_class": pred_label,
        "risk_level": int(risk_pred)
    }


# =========================
# TEST RUN
# =========================

if __name__ == "__main__":
    sample_input = {
        "age": 45,
        "gender": 1,  # 0 = F, 1 = M (adjust if needed)
        "glucose": 150,
        "cholesterol": 220,
        "hdl": 35,
        "ldl": 160,
        "triglycerides": 200,
        "hemoglobin": 13.5,
        "creatinine": 1.2,
        "bmi": 28,
        "systolic_bp": 140,
        "diastolic_bp": 90,
        "uric_acid": 6.5,
        "wbc": 7.0,
        "potassium": 4.5
    }

    result = predict(sample_input)
    print("\nPrediction Result:")
    print(result)
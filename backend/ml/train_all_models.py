import pandas as pd
import numpy as np
import pickle
import os

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from feature_engineering import AdvancedFeatureEngineer as FeatureEngineer
from agent1_parser import Agent1Parser
from agent2_reasoning import Agent2PatternDetection
from agent3_trends import Agent3TrendForecasting
from agent4_risk import Agent4RiskAssessment


def main():

    print("\n" + "="*70)
    print("🚀 FULL ML PIPELINE STARTED")
    print("="*70)

    # =============================
    # STEP 1: LOAD DATA
    # =============================
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(BASE_DIR, "../data/synthetic_data.csv")

    if not os.path.exists(path):
        print("❌ Dataset not found!")
        return

    df = pd.read_csv(path)
    print(f"✓ Loaded data: {df.shape}")

    # =============================
    # STEP 2: FEATURE ENGINEERING
    # =============================
    engineer = FeatureEngineer()
    df, feature_names = engineer.engineer_features(df)

    X = df[feature_names].fillna(0)

    if 'risk_label' not in df.columns:
        print("❌ 'risk_label' column missing!")
        return

    y = df['risk_label'].astype('category').cat.codes

    # =============================
    # STEP 3: SCALING
    # =============================
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    os.makedirs("backend/ml/models", exist_ok=True)

    with open("backend/ml/models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open("backend/ml/models/feature_names.pkl", "wb") as f:
        pickle.dump(feature_names, f)

    # =============================
    # STEP 4: TRAIN TEST SPLIT
    # =============================
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    print(f"✓ Train: {X_train.shape}, Test: {X_test.shape}")

    # =============================
    # AGENT 1 (Parser)
    # =============================
    print("\nAGENT 1 READY (No training required)")

    # =============================
    # AGENT 2
    # =============================
    agent2 = Agent2PatternDetection()
    agent2.train(X_train, y_train, hyperparameter_tuning=False)  # Disable tuning for now
    metrics2 = agent2.evaluate(X_train, X_test, y_train, y_test)
    agent2.save("backend/ml/models/agent2.pkl")

    # =============================
    # AGENT 3
    # =============================
    if 'report_date' in df.columns:
        agent3 = Agent3TrendForecasting()
        agent3.train(df)
        agent3.save("backend/ml/models/agent3_models.pkl")
    else:
        print("⚠️ Skipping Agent 3 (no report_date column)")

    # =============================
    # AGENT 4
    # =============================
    agent4 = Agent4RiskAssessment()
    agent4.train(X_train, y_train, hyperparameter_tuning=False)  # Disable tuning
    metrics4 = agent4.evaluate(X_test, y_test)
    agent4.save("backend/ml/models/agent4.pkl")

    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE")
    print("="*70)

    print(f"Agent2 CV Accuracy: {metrics2['cv_accuracy']:.4f}")
    print(f"Agent2 Test Accuracy: {metrics2['test_accuracy']:.4f}")
    print(f"Agent2 ROC-AUC: {metrics2.get('rocauc', 'N/A')}")
    print(f"Agent2 Overfit Gap: {metrics2['overfit_gap']:.4f}")
    print(f"Agent4 CV Accuracy: {metrics4['cv_accuracy']:.4f}")
    print(f"Agent4 Test Accuracy: {metrics4['test_accuracy']:.4f}")


if __name__ == "__main__":
    main()
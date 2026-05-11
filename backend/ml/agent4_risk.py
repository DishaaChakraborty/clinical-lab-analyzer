import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report
)
import pickle
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class Agent4RiskAssessment:
    """
    Agent 4: RISK ASSESSMENT
    Input: All features
    Output: Risk level (Low, Medium, High)
    Model: Random Forest with proper CV
    Target: 80%+ accuracy with ROC validation
    """
    
    RISK_LEVELS = {
        'Low': 0,
        'Medium': 1,
        'High': 2
    }
    
    def __init__(self):
        self.model = None
        self.cv_scores = {}
        self.rocauc_scores = {}
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, hyperparameter_tuning: bool = False):
        """Train risk classification model with proper CV"""
        
        print("\n" + "="*70)
        print("AGENT 4: RISK ASSESSMENT")
        print("="*70)
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        if hyperparameter_tuning:
            from sklearn.model_selection import GridSearchCV
            print("\n✓ Hyperparameter tuning with 5-fold CV...")
            param_grid = {
                'n_estimators': [150, 200, 250],
                'max_depth': [15, 20, 25],
                'min_samples_split': [3, 5, 8],
                'min_samples_leaf': [1, 2, 3],
            }
            
            grid = GridSearchCV(
                RandomForestClassifier(random_state=42, n_jobs=-1),
                param_grid,
                cv=cv,
                scoring='f1_weighted',
                n_jobs=-1
            )
            grid.fit(X_train, y_train)
            self.model = grid.best_estimator_
            print(f"✓ Best CV score: {grid.best_score_:.4f}")
        else:
            print("\n✓ Training with GradientBoosting...")
            self.model = GradientBoostingClassifier(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.1,
                min_samples_split=10,
                min_samples_leaf=5,
                subsample=0.8,
                random_state=42
            )
            self.model.fit(X_train, y_train)
        
        # Proper CV scores
        print("\n--- Cross-Validation Results ---")
        cv_acc = cross_val_score(self.model, X_train, y_train, cv=cv, scoring='accuracy')
        cv_f1 = cross_val_score(self.model, X_train, y_train, cv=cv, scoring='f1_weighted')
        
        print(f"CV Accuracy: {cv_acc.mean():.4f} (+/- {cv_acc.std()*2:.4f})")
        print(f"CV F1-Score: {cv_f1.mean():.4f} (+/- {cv_f1.std()*2:.4f})")
        
        self.cv_scores = {
            'accuracy': cv_acc.mean(),
            'accuracy_std': cv_acc.std(),
            'f1': cv_f1.mean(),
            'f1_std': cv_f1.std()
        }
        
        print(f"✓ Agent 4 model trained!")
        return self
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate model with proper metrics"""
        
        print("\n" + "-"*70)
        print("AGENT 4: EVALUATION")
        print("-"*70)
        
        y_pred = self.model.predict(X_test)
        
        test_acc = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred, average='weighted')
        test_precision = precision_score(y_test, y_pred, average='weighted')
        test_recall = recall_score(y_test, y_pred, average='weighted')
        
        print(f"\n=== TEST SET METRICS ===")
        print(f"Accuracy:  {test_acc:.4f}")
        print(f"Precision: {test_precision:.4f}")
        print(f"Recall:    {test_recall:.4f}")
        print(f"F1-Score:  {test_f1:.4f}")
        
        # ROC-AUC
        try:
            y_proba = self.model.predict_proba(X_test)
            if len(np.unique(y_test)) > 2:
                rocauc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
            else:
                rocauc = roc_auc_score(y_test, y_proba[:, 1])
            print(f"ROC-AUC:   {rocauc:.4f}")
            self.rocauc_scores['test'] = rocauc
        except Exception as e:
            print(f"ROC-AUC:   N/A")
        
        # Use CV score for target check
        cv_acc = self.cv_scores.get('accuracy', test_acc)
        if cv_acc >= 0.80:
            print(f"\n✅ TARGET ACHIEVED (CV): {cv_acc:.4f} >= 0.80")
        else:
            print(f"\n⚠️  CV Target: {cv_acc:.4f} (aim for 0.80+)")
        
        print(f"\n=== CLASSIFICATION REPORT ===")
        print(classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High']))
        
        print(f"\n=== CONFUSION MATRIX ===")
        print(confusion_matrix(y_test, y_pred))
        
        return {
            'test_accuracy': test_acc,
            'cv_accuracy': cv_acc,
            'test_f1': test_f1,
            'test_precision': test_precision,
            'test_recall': test_recall
        }
    
    def save(self, filepath: str = 'backend/ml/models/agent4_model.pkl'):
        """Save model with metadata"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'cv_scores': self.cv_scores,
                'rocauc_scores': self.rocauc_scores
            }, f)
        print(f"✓ Agent 4 model saved to {filepath}")
    
    def load(self, filepath: str = 'backend/ml/models/agent4_model.pkl'):
        """Load model"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.cv_scores = data.get('cv_scores', {})
            self.rocauc_scores = data.get('rocauc_scores', {})
        return self
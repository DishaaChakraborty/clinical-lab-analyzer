import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve
)
from typing import Dict, List, Tuple
import pickle
import logging

logger = logging.getLogger(__name__)

class Agent2PatternDetection:
    """
    Agent 2: PATTERN DETECTION (Disease Classification)
    Input: Medical features
    Output: Disease type (Low, Medium, High risk)
    Model: Ensemble with proper CV
    Target: 85%+ accuracy with ROC validation
    """
    
    DISEASE_CLASSES = {
        0: 'Low',
        1: 'Medium', 
        2: 'High'
    }
    
    def __init__(self):
        self.model = None
        self.best_params = None
        self.feature_importance = None
        self.cv_scores = {}
        self.rocauc_scores = {}
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, hyperparameter_tuning: bool = False):
        """
        Train disease classification model with PROPER cross-validation
        """
        
        print("\n" + "="*70)
        print("AGENT 2: PATTERN DETECTION (Disease Classification)")
        print("="*70)
        print(f"\nTraining set shape: {X_train.shape}")
        print(f"Classes: {np.unique(y_train)}")
        
        # Use StratifiedKFold for proper CV
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        if hyperparameter_tuning:
            print("\n✓ Hyperparameter tuning with 5-fold CV...")
            param_grid = {
                'n_estimators': [100, 150, 200],
                'max_depth': [10, 15, 20],
                'min_samples_split': [5, 10],
                'min_samples_leaf': [2, 4],
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
            self.best_params = grid.best_params_
            print(f"✓ Best CV score: {grid.best_score_:.4f}")
        else:
            print("\n✓ Training with GradientBoosting (often better than RF)...")
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
        
        # Calculate proper CV scores (not training score!)
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
        
        self.feature_importance = self.model.feature_importances_
        print(f"\n✓ Model trained successfully!")
        
        return self
    
    def evaluate(self, X_train: np.ndarray, X_test: np.ndarray,
                 y_train: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate model with proper metrics and ROC
        """
        
        print("\n" + "-"*70)
        print("AGENT 2: EVALUATION")
        print("-"*70)
        
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred, average='weighted')
        test_precision = precision_score(y_test, y_test_pred, average='weighted')
        test_recall = recall_score(y_test, y_test_pred, average='weighted')
        
        print(f"\n=== TRAIN vs TEST (Overfitting Check) ===")
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy:  {test_acc:.4f}")
        print(f"Gap:            {abs(train_acc - test_acc):.4f}")
        
        if abs(train_acc - test_acc) > 0.1:
            print("⚠️  WARNING: Potential overfitting! (gap > 0.1)")
        
        print(f"\n=== TEST SET METRICS ===")
        print(f"Accuracy:  {test_acc:.4f}")
        print(f"Precision: {test_precision:.4f}")
        print(f"Recall:    {test_recall:.4f}")
        print(f"F1-Score:  {test_f1:.4f}")
        
        # ROC-AUC for multi-class
        try:
            y_test_proba = self.model.predict_proba(X_test)
            if len(np.unique(y_test)) > 2:
                rocauc = roc_auc_score(y_test, y_test_proba, multi_class='ovr', average='weighted')
            else:
                rocauc = roc_auc_score(y_test, y_test_proba[:, 1])
            print(f"ROC-AUC:   {rocauc:.4f}")
            self.rocauc_scores['test'] = rocauc
        except Exception as e:
            print(f"ROC-AUC:   N/A ({e})")
        
        # Use CV score for target check (more reliable!)
        cv_acc = self.cv_scores.get('accuracy', test_acc)
        if cv_acc >= 0.85:
            print(f"\n✅ TARGET ACHIEVED (CV): {cv_acc:.4f} >= 0.85")
        else:
            print(f"\n⚠️  CV Target: {cv_acc:.4f} (aim for 0.85+)")
        
        print(f"\n=== CLASSIFICATION REPORT ===")
        print(classification_report(y_test, y_test_pred, target_names=['Low', 'Medium', 'High']))
        
        print(f"\n=== CONFUSION MATRIX ===")
        cm = confusion_matrix(y_test, y_test_pred)
        print(cm)
        
        return {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'cv_accuracy': cv_acc,
            'test_f1': test_f1,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'overfit_gap': abs(train_acc - test_acc)
        }
    
    def predict(self, X: np.ndarray) -> List[str]:
        """Make predictions"""
        class_ids = self.model.predict(X)
        return [self.DISEASE_CLASSES[cid] for cid in class_ids]
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        return self.model.predict_proba(X)
    
    def save(self, filepath: str = 'backend/ml/models/agent2_model.pkl'):
        """Save model with metadata"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'cv_scores': self.cv_scores,
                'rocauc_scores': self.rocauc_scores,
                'feature_importance': self.feature_importance,
                'best_params': self.best_params
            }, f)
        print(f"✓ Agent 2 model saved to {filepath}")
    
    def load(self, filepath: str = 'backend/ml/models/agent2_model.pkl'):
        """Load model"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.cv_scores = data.get('cv_scores', {})
            self.rocauc_scores = data.get('rocauc_scores', {})
            self.feature_importance = data.get('feature_importance', None)
            self.best_params = data.get('best_params', None)
        return self
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple
import pickle
import logging

logger = logging.getLogger(__name__)

class DataPreparation:
    """Prepare data for ML model training"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        feature_names: list,
        target_col: str = 'risk_label',
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        Prepare data for training
        
        Returns:
            X_train, X_test, y_train, y_test, metadata
        """
        
        print("\n" + "="*70)
        print("DATA PREPARATION")
        print("="*70)
        
        # Extract features and target
        X = df[feature_names].copy()
        y = df[target_col].copy()
        
        print(f"\nDataset shape: {X.shape}")
        print(f"Target distribution:")
        print(y.value_counts())
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Normalize features
        print(f"\nNormalizing features...")
        X_scaled = self.scaler.fit_transform(X)
        
        # Train/test split
        print(f"Splitting data (80/20)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        metadata = {
            'n_features': len(feature_names),
            'feature_names': feature_names,
            'n_train': X_train.shape[0],
            'n_test': X_test.shape[0],
            'target_classes': sorted(y.unique().tolist()),
            'class_weights': self._compute_class_weights(y_train)
        }
        
        self.feature_names = feature_names
        
        return X_train, X_test, y_train, y_test, metadata
    
    def _compute_class_weights(self, y):
        """Compute class weights for imbalanced data"""
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(y)
        weights = compute_class_weight(
            'balanced',
            classes=classes,
            y=y
        )
        
        return dict(zip(classes, weights))
    
    def save_scaler(self, filepath: str = 'ml/models/feature_scaler.pkl'):
        """Save fitted scaler"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"✓ Saved scaler to {filepath}")
    
    def load_scaler(self, filepath: str = 'ml/models/feature_scaler.pkl'):
        """Load saved scaler"""
        with open(filepath, 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"✓ Loaded scaler from {filepath}")
        return self.scaler
    
    def transform_new_data(self, X: np.ndarray) -> np.ndarray:
        """Transform new data using fitted scaler"""
        return self.scaler.transform(X)
    
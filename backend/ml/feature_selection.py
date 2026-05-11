import pandas as pd
import numpy as np
from typing import Tuple, List
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif
from sklearn.preprocessing import StandardScaler
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedFeatureSelector:
    """Select best features and remove noisy/useless ones"""
    
    def __init__(self, method='variance_and_correlation'):
        self.method = method
        self.selected_features = []
        self.removed_features = []
    
    def select_features(self, X: pd.DataFrame, y: np.ndarray, n_features: int = None) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select top features
        method: 'variance_and_correlation', 'kbest_f', 'kbest_mutual'
        """
        
        print("\n" + "="*70)
        print("FEATURE SELECTION")
        print("="*70)
        
        if n_features is None:
            n_features = min(50, X.shape[1])  # Keep top 50 features
        
        if n_features <= 0:
            n_features = min(10, X.shape[1])
        
        print(f"\nSelecting top {n_features} features from {X.shape[1]}")
        
        try:
            if self.method == 'variance_and_correlation':
                # Remove low variance features
                variances = X.var()
                variance_threshold = variances.quantile(0.1)  # Remove bottom 10% variance
                high_var_features = variances[variances > variance_threshold].index.tolist()
                
                if len(high_var_features) == 0:
                    high_var_features = variances.nlargest(max(5, n_features)).index.tolist()
                
                # Remove highly correlated features
                corr_matrix = X[high_var_features].corr().abs()
                upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                
                cols_to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
                
                selected = [f for f in high_var_features if f not in cols_to_drop]
                selected = selected[:n_features]  # Keep top n
                
                if len(selected) == 0:
                    selected = high_var_features[:n_features]
            
            elif self.method == 'kbest_f':
                # Ensure we have valid data
                n_features = min(n_features, X.shape[1])
                selector = SelectKBest(f_classif, k=n_features)
                selector.fit(X, y)
                selected = X.columns[selector.get_support()].tolist()
            
            elif self.method == 'kbest_mutual':
                # Ensure we have valid data
                n_features = min(n_features, X.shape[1])
                selector = SelectKBest(mutual_info_classif, k=n_features)
                selector.fit(X, y)
                selected = X.columns[selector.get_support()].tolist()
            
            else:
                logger.warning(f"Unknown method: {self.method}. Using variance_and_correlation.")
                selected = X.columns.tolist()[:n_features]
        
        except Exception as e:
            logger.error(f"Error in feature selection: {str(e)}")
            selected = X.columns.tolist()[:n_features]
        
        self.selected_features = selected
        self.removed_features = [f for f in X.columns if f not in selected]
        
        print(f"\n[OK] Selected {len(selected)} features")
        print(f"[OK] Removed {len(self.removed_features)} noisy/useless features")
        
        if len(self.removed_features) > 0 and len(self.removed_features) <= 10:
            print(f"  Removed: {', '.join(self.removed_features[:10])}")
        
        print(f"{'='*70}\n")
        
        return X[selected], selected


if __name__ == '__main__':
    from feature_engineering import AdvancedFeatureEngineer
    
    # Load data
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / 'backend' / 'ml' / 'data' / 'synthetic_data.csv'
    
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        exit(1)
    
    df = pd.read_csv(data_path)
    
    # Engineer features first
    print("\n" + "="*70)
    print("STEP 1: FEATURE ENGINEERING")
    print("="*70)
    engineer = AdvancedFeatureEngineer()
    features_df, feature_names = engineer.engineer_features(df)
    
    # Prepare for selection
    X = features_df[feature_names]
    y = (df['risk_label'] == 'High').astype(int) if 'risk_label' in df.columns else np.zeros(len(df))
    
    # Test all three methods
    print("\n" + "="*70)
    print("STEP 2: FEATURE SELECTION")
    print("="*70)
    
    methods = ['variance_and_correlation', 'kbest_f', 'kbest_mutual']
    results = {}
    
    for method in methods:
        print(f"\n\nMETHOD: {method}")
        print("="*70)
        selector = AdvancedFeatureSelector(method=method)
        X_selected, selected_features = selector.select_features(X, y, n_features=50)
        results[method] = {
            'selected': selected_features,
            'removed': selector.removed_features,
            'n_selected': len(selected_features)
        }
    
    # Summary comparison
    print("\n" + "="*70)
    print("STEP 3: SUMMARY - SELECTED FEATURES BY METHOD")
    print("="*70)
    
    for method, result in results.items():
        print(f"\n{method}:")
        print(f"  Selected {result['n_selected']} features:")
        for i, feat in enumerate(result['selected'][:10], 1):
            print(f"    {i:2d}. {feat}")
        if len(result['selected']) > 10:
            print(f"    ... and {len(result['selected']) - 10} more")
        
        print(f"\n  Removed {len(result['removed'])} features:")
        for i, feat in enumerate(result['removed'][:10], 1):
            print(f"    {i:2d}. {feat}")
        if len(result['removed']) > 10:
            print(f"    ... and {len(result['removed']) - 10} more")
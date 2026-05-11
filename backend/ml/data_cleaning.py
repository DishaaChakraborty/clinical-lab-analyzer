import pandas as pd
import numpy as np
from typing import Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCleaner:
    """Production-grade data cleaning"""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.cleaning_report = {}
    
    def clean(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Complete cleaning pipeline"""
        
        if self.verbose:
            print("\n" + "="*70)
            print("DATA CLEANING")
            print("="*70)
        
        df_clean = df.copy()
        report = {}
        
        # 1. Remove exact duplicates
        before = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        report['duplicates_removed'] = before - len(df_clean)
        if self.verbose and report['duplicates_removed'] > 0:
            print(f"✓ Removed {report['duplicates_removed']} duplicate rows")
        
        # 2. Handle missing values
        missing_count = df_clean.isnull().sum().sum()
        if missing_count > 0:
            for col in df_clean.columns:
                if df_clean[col].isnull().any():
                    if df_clean[col].dtype in [np.float64, np.int64]:
                        df_clean[col].fillna(df_clean[col].median(), inplace=True)
                    else:
                        df_clean[col].fillna(df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 'Unknown', inplace=True)
            report['missing_values_fixed'] = missing_count
            if self.verbose:
                print(f"✓ Fixed {missing_count} missing values")
        
        # 3. Fix negative values in medical parameters
        medical_params = ['glucose', 'cholesterol', 'hemoglobin', 'age', 'bmi', 'hdl', 'ldl']
        negative_fixed = 0
        for col in medical_params:
            if col in df_clean.columns:
                mask = df_clean[col] < 0
                if mask.any():
                    df_clean.loc[mask, col] = df_clean[col].median()
                    negative_fixed += mask.sum()
        report['negative_fixed'] = negative_fixed
        if self.verbose and negative_fixed > 0:
            print(f"✓ Fixed {negative_fixed} negative medical values")
        
        # 4. Detect outliers (IQR method)
        outliers_detected = 0
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            
            outlier_mask = (df_clean[col] < lower) | (df_clean[col] > upper)
            outlier_count = outlier_mask.sum()
            
            if outlier_count > 0:
                # Cap outliers instead of removing
                df_clean.loc[outlier_mask & (df_clean[col] < lower), col] = lower
                df_clean.loc[outlier_mask & (df_clean[col] > upper), col] = upper
                outliers_detected += outlier_count
        
        report['outliers_capped'] = outliers_detected
        if self.verbose and outliers_detected > 0:
            print(f"✓ Detected and capped {outliers_detected} outliers")
        
        # 5. Handle extreme values
        extreme_fixed = 0
        if 'age' in df_clean.columns:
            mask = (df_clean['age'] < 0) | (df_clean['age'] > 120)
            extreme_fixed += mask.sum()
            df_clean.loc[mask, 'age'] = df_clean[~mask]['age'].median()
        
        if 'glucose' in df_clean.columns:
            mask = df_clean['glucose'] > 600
            extreme_fixed += mask.sum()
            df_clean.loc[mask, 'glucose'] = df_clean[~mask]['glucose'].quantile(0.99)
        
        report['extreme_values_fixed'] = extreme_fixed
        if self.verbose and extreme_fixed > 0:
            print(f"✓ Fixed {extreme_fixed} extreme values")
        
        # 6. Validate data consistency
        if 'cholesterol' in df_clean.columns and 'hdl' in df_clean.columns:
            invalid = (df_clean['hdl'] > df_clean['cholesterol']).sum()
            if invalid > 0:
                df_clean.loc[df_clean['hdl'] > df_clean['cholesterol'], 'hdl'] = \
                    df_clean.loc[df_clean['hdl'] > df_clean['cholesterol'], 'cholesterol'] * 0.3
                report['consistency_fixes'] = invalid
                if self.verbose:
                    print(f"✓ Fixed {invalid} cholesterol/HDL inconsistencies")
        
        if self.verbose:
            print(f"\n{'='*70}\n")
        
        self.cleaning_report = report
        return df_clean, report


class FeatureScaler:
    """Advanced feature scaling"""
    
    def __init__(self, method='standard'):
        """
        method: 'standard' (z-score), 'minmax', 'robust'
        """
        self.method = method
        self.scaler = None
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform"""
        if self.method == 'standard':
            self.scaler = (X.mean(axis=0), X.std(axis=0))
            return (X - self.scaler[0]) / (self.scaler[1] + 1e-8)
        elif self.method == 'minmax':
            self.scaler = (X.min(axis=0), X.max(axis=0) - X.min(axis=0))
            return (X - self.scaler[0]) / (self.scaler[1] + 1e-8)
        elif self.method == 'robust':
            q25 = np.percentile(X, 25, axis=0)
            q75 = np.percentile(X, 75, axis=0)
            self.scaler = (np.median(X, axis=0), q75 - q25)
            return (X - self.scaler[0]) / (self.scaler[1] + 1e-8)
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform only"""
        if self.method == 'standard':
            return (X - self.scaler[0]) / (self.scaler[1] + 1e-8)
        elif self.method == 'minmax':
            return (X - self.scaler[0]) / (self.scaler[1] + 1e-8)
        elif self.method == 'robust':
            return (X - self.scaler[0]) / (self.scaler[1] + 1e-8)
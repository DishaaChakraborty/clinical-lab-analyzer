import pandas as pd
import numpy as np
from typing import Tuple, List
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class AdvancedFeatureEngineer:
    """Engineer advanced features from raw lab data"""
    
    def __init__(self):
        pass
    
    def engineer_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Engineer features from raw data
        """
        
        print("\n" + "="*70)
        print("ADVANCED FEATURE ENGINEERING")
        print("="*70)
        print(f"\nDataFrame shape: {df.shape}")
        print(f"Column names: {df.columns.tolist()}\n")
        
        # Category 1: Direct lab values - use all numeric columns except target
        print("[OK] Category 1: Direct lab values")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Remove target column if present
        lab_cols = [col for col in numeric_cols if col not in ['risk_label', 'target', 'label']]
        print(f"  Found {len(lab_cols)} numeric columns")
        print(f"  Using columns: {lab_cols[:5]}..." if len(lab_cols) > 5 else f"  Using columns: {lab_cols}")
        
        # Category 2: Ratios and interactions
        print("\n[OK] Category 2: Ratios and interactions")
        ratio_features = []
        if len(lab_cols) > 1:
            for i in range(len(lab_cols)):
                for j in range(i+1, len(lab_cols)):
                    col1, col2 = lab_cols[i], lab_cols[j]
                    # Ratio
                    ratio_name = f"{col1}_over_{col2}"
                    df[ratio_name] = df[col1] / (df[col2] + 1e-6)  # Avoid division by zero
                    ratio_features.append(ratio_name)
                    # Product
                    product_name = f"{col1}_times_{col2}"
                    df[product_name] = df[col1] * df[col2]
                    ratio_features.append(product_name)
        
        print(f"  Created {len(ratio_features)} ratio/interaction features")
        
        # Category 3: Statistical aggregations
        print("\n[OK] Category 3: Statistical aggregations")
        stat_features = []
        if len(lab_cols) > 0:
            # Mean of all labs
            df['lab_mean'] = df[lab_cols].mean(axis=1)
            stat_features.append('lab_mean')
            # Std of all labs
            df['lab_std'] = df[lab_cols].std(axis=1)
            stat_features.append('lab_std')
            # Min of all labs
            df['lab_min'] = df[lab_cols].min(axis=1)
            stat_features.append('lab_min')
            # Max of all labs
            df['lab_max'] = df[lab_cols].max(axis=1)
            stat_features.append('lab_max')
        
        print(f"  Created {len(stat_features)} statistical features")
        
        # Category 4: Binned features
        print("\n[OK] Category 4: Binned features")
        binned_features = []
        for col in lab_cols[:10]:  # Limit to first 10 to avoid explosion
            try:
                # Quartile bins
                binned_name = f"{col}_quartile"
                df[binned_name] = pd.qcut(df[col], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
                # Convert to dummy
                dummies = pd.get_dummies(df[binned_name], prefix=binned_name)
                df = pd.concat([df, dummies], axis=1)
                binned_features.extend(dummies.columns.tolist())
                df.drop(binned_name, axis=1, inplace=True)
            except Exception as e:
                logger.warning(f"Could not bin column {col}: {e}")
                continue
        
        print(f"  Created {len(binned_features)} binned features")
        
        # Category 5: Polynomial features (degree 2)
        print("\n[OK] Category 5: Polynomial features")
        poly_features = []
        try:
            from sklearn.preprocessing import PolynomialFeatures
            poly_cols = [col for col in lab_cols[:5] if col in df.columns]  # Limit to first 5 to avoid explosion
            if len(poly_cols) > 0:
                poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
                poly_data = poly.fit_transform(df[poly_cols])
                poly_feature_names = poly.get_feature_names_out(poly_cols)
                poly_df = pd.DataFrame(poly_data, columns=poly_feature_names, index=df.index)
                df = pd.concat([df, poly_df], axis=1)
                poly_features.extend(poly_feature_names.tolist())
        except Exception as e:
            logger.warning(f"Could not create polynomial features: {e}")
        
        print(f"  Created {len(poly_features)} polynomial features")
        
        # Combine all feature names
        all_features = lab_cols + ratio_features + stat_features + binned_features + poly_features
        
        print(f"\n[OK] Total features engineered: {len(all_features)}")
        print(f"{'='*70}\n")
        
        return df, all_features


if __name__ == '__main__':
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / 'backend' / 'ml' / 'data' / 'synthetic_data.csv'
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        exit(1)
    df = pd.read_csv(data_path)
    
    engineer = AdvancedFeatureEngineer()
    features_df, feature_names = engineer.engineer_features(df)
    
    # Print sample values
    print("\n" + "="*70)
    print("SAMPLE ENGINEERED FEATURE VALUES (First 5 rows)")
    print("="*70 + "\n")
    
    # Show first few rows with feature columns
    if len(feature_names) > 0:
        print(features_df[feature_names].head(5).to_string())
    else:
        print("No engineered features found.")
    
    print("\n" + "="*70)
    print("FEATURE STATISTICS")
    print("="*70 + "\n")
    
    # Show statistics
    if len(feature_names) > 0:
        print(features_df[feature_names].describe().to_string())
    else:
        print("No engineered features found.")
    
    print("\n" + "="*70)
    print("FEATURE NAMES")
    print("="*70 + "\n")
    
    # Print all feature names
    for i, name in enumerate(feature_names, 1):
        print(f"{i:3d}. {name}")
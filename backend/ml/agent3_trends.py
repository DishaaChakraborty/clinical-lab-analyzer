import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
import pickle
import logging

logger = logging.getLogger(__name__)

class Agent3TrendForecasting:
    """Agent 3: TREND FORECASTING - Regression models for parameter prediction"""
    
    def __init__(self):
        self.models = {}
        self.r2_scores = {}
    
    def train(self, df: pd.DataFrame):
        """Train forecasting models"""
        
        print("\n" + "="*70)
        print("AGENT 3: TREND FORECASTING (Regression)")
        print("="*70)
        
        # Check for required columns
        required_cols = ['glucose', 'cholesterol', 'hdl', 'ldl', 'triglycerides', 'hemoglobin', 'creatinine']
        available_cols = [col for col in required_cols if col in df.columns]
        
        if not available_cols:
            raise ValueError(f"❌ Missing all required columns. Have: {df.columns.tolist()}")
        
        print(f"\nTraining {len(available_cols)} forecasting models")
        
        for param in available_cols:
            if param not in df.columns:
                continue
            
            # Simple time-based regression
            X = np.arange(len(df)).reshape(-1, 1)
            y = df[param].values
            
            # Skip if constant
            if np.std(y) < 0.01:
                self.models[param] = LinearRegression()
                self.models[param].fit(X, y)
                self.r2_scores[param] = 0.0
                continue
            
            try:
                # Linear Regression
                lr = LinearRegression()
                lr.fit(X, y)
                lr_r2 = lr.score(X, y)
                
                # Ridge Regression
                ridge = Ridge(alpha=1.0)
                ridge.fit(X, y)
                ridge_r2 = ridge.score(X, y)
                
                # Lasso Regression
                lasso = Lasso(alpha=0.1, max_iter=10000)
                lasso.fit(X, y)
                lasso_r2 = lasso.score(X, y)
                
                # Polynomial Regression
                poly = PolynomialFeatures(degree=2)
                X_poly = poly.fit_transform(X)
                poly_lr = LinearRegression()
                poly_lr.fit(X_poly, y)
                poly_r2 = poly_lr.score(X_poly, y)
                
                # Gradient Boosting
                gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
                gb.fit(X, y)
                gb_r2 = gb.score(X, y)
                
                # Select best model
                scores = {'linear': lr_r2, 'ridge': ridge_r2, 'lasso': lasso_r2, 'poly': poly_r2, 'gb': gb_r2}
                best_model_name = max(scores, key=scores.get)
                best_score = scores[best_model_name]
                
                if best_model_name == 'linear':
                    self.models[param] = lr
                elif best_model_name == 'ridge':
                    self.models[param] = ridge
                elif best_model_name == 'lasso':
                    self.models[param] = lasso
                elif best_model_name == 'poly':
                    self.models[param] = poly_lr
                else:
                    self.models[param] = gb
                
                self.r2_scores[param] = best_score
                
                status = "✓" if best_score > 0.75 else "⚠"
                print(f"{status} {param:15s} - R²: {best_score:.4f} ({best_model_name})")
                
            except Exception as e:
                print(f"⚠ {param}: {str(e)}")
                # Fallback to linear
                lr = LinearRegression()
                lr.fit(X, y)
                self.models[param] = lr
                self.r2_scores[param] = 0.0
        
        avg_r2 = np.mean(list(self.r2_scores.values())) if self.r2_scores else 0
        print(f"\n✓ Average R²: {avg_r2:.4f}")
        print(f"{'='*70}\n")
        
        return self
    
    def save(self, filepath='backend/ml/models/agent3_models.pkl'):
        """Save models"""
        data = {'models': self.models, 'r2_scores': self.r2_scores}
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"✓ Agent 3 models saved")
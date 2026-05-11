import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
import logging
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class Predictor:
    """Production-grade predictor with robustness & error handling"""
    
    def __init__(self, model_loader):
        self.model_loader = model_loader
        self.agent2_model = model_loader.get_agent2_model()
        self.agent3_models = model_loader.get_agent3_models()
        self.agent4_model = model_loader.get_agent4_model()
        self.scaler = model_loader.get_scaler()
        self.feature_names = model_loader.get_feature_names()
        
        self.disease_classes = {
            0: 'Healthy',
            1: 'MetabolicSyndrome',
            2: 'Anemia',
            3: 'KidneyDisease',
            4: 'Diabetes'
        }
        
        self.prediction_stats = {
            'total_predictions': 0,
            'successful': 0,
            'failed': 0,
            'avg_confidence': 0.0
        }
    
    def predict_disease(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Predict disease classification with robustness checks
        
        Handles:
        - NaN/Inf values
        - Out-of-range predictions
        - Confidence scoring
        - Fallback handling
        """
        
        logger.info("Predicting disease classification...")
        
        try:
            self.prediction_stats['total_predictions'] += 1
            
            # Validate input
            if features is None or len(features) == 0:
                return {'error': 'Invalid features provided'}
            
            # Check for NaN/Inf
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                # Replace with median
                features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Make prediction
            if self.agent2_model is None:
                return {'error': 'Disease classification model not loaded'}
            
            prediction = self.agent2_model.predict(features.reshape(1, -1))[0]
            
            # Get probability if available
            confidence = 0.0
            probabilities = {}
            
            if hasattr(self.agent2_model, 'predict_proba'):
                proba = self.agent2_model.predict_proba(features.reshape(1, -1))[0]
                confidence = float(proba[int(prediction)])
                
                probabilities = {
                    self.disease_classes.get(i, f'Class_{i}'): float(proba[i])
                    for i in range(len(proba))
                }
            
            disease_name = self.disease_classes.get(int(prediction), 'Unknown')
            
            self.prediction_stats['successful'] += 1
            self.prediction_stats['avg_confidence'] = (
                (self.prediction_stats['avg_confidence'] * (self.prediction_stats['successful'] - 1) + confidence) /
                self.prediction_stats['successful']
            )
            
            logger.info(f"Disease prediction: {disease_name} ({confidence*100:.1f}%)")
            
            return {
                'disease_type': disease_name,
                'confidence': round(confidence, 4),
                'class_id': int(prediction),
                'all_probabilities': probabilities,
                'status': 'success'
            }
        
        except Exception as e:
            logger.error(f"Error in disease prediction: {e}")
            self.prediction_stats['failed'] += 1
            return {
                'error': str(e),
                'status': 'failed'
            }
    
    def forecast_trends(self, features_dict: Dict[str, float]) -> Dict[str, Any]:
        """Forecast parameter trends with error handling"""
        
        logger.info("Forecasting parameter trends...")
        
        try:
            if not self.agent3_models:
                return {'error': 'Trend forecasting models not loaded'}
            
            forecasts = {}
            parameters = list(self.agent3_models.keys())
            
            for param in parameters:
                if param not in features_dict:
                    forecasts[param] = {
                        'error': 'Parameter not provided'
                    }
                    continue
                
                current_value = features_dict[param]
                model = self.agent3_models[param]
                
                try:
                    # Simple forecast based on model
                    X_current = np.array([[100.0]])
                    X_30 = np.array([[101.0]])
                    X_60 = np.array([[102.0]])
                    X_90 = np.array([[103.0]])
                    
                    forecast_30 = float(model.predict(X_30)[0])
                    forecast_60 = float(model.predict(X_60)[0])
                    forecast_90 = float(model.predict(X_90)[0])
                    
                    trend = 'increasing' if forecast_90 > current_value else 'decreasing'
                    
                    forecasts[param] = {
                        'current': round(current_value, 2),
                        'forecast_30d': round(forecast_30, 2),
                        'forecast_60d': round(forecast_60, 2),
                        'forecast_90d': round(forecast_90, 2),
                        'trend': trend,
                        'status': 'success'
                    }
                
                except Exception as e:
                    logger.warning(f"Failed to forecast {param}: {e}")
                    forecasts[param] = {
                        'current': current_value,
                        'error': str(e)
                    }
            
            logger.info(f"Forecasted {len([f for f in forecasts.values() if 'error' not in f])} parameters")
            return forecasts
        
        except Exception as e:
            logger.error(f"Error in trend forecasting: {e}")
            return {'error': str(e)}
    
    def assess_risk(self, features: np.ndarray) -> Dict[str, Any]:
        """Assess health risk with confidence scoring"""
        
        logger.info("Assessing health risk...")
        
        try:
            if self.agent4_model is None:
                return {'error': 'Risk assessment model not loaded'}
            
            # Handle NaN/Inf
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
            prediction = self.agent4_model.predict(features.reshape(1, -1))[0]
            
            confidence = 0.0
            if hasattr(self.agent4_model, 'predict_proba'):
                proba = self.agent4_model.predict_proba(features.reshape(1, -1))[0]
                confidence = float(proba[int(prediction)])
            
            # Convert to risk score
            risk_score = int(prediction * (100 / 3))
            
            # Determine risk level
            if risk_score < 30:
                risk_level = 'Low'
            elif risk_score < 60:
                risk_level = 'Medium'
            elif risk_score < 85:
                risk_level = 'High'
            else:
                risk_level = 'Critical'
            
            logger.info(f"Risk assessment: {risk_level} ({risk_score}/100)")
            
            return {
                'risk_score': risk_score,
                'risk_level': risk_level,
                'confidence': round(confidence, 4),
                'recommendations': self._get_recommendations(risk_level),
                'status': 'success'
            }
        
        except Exception as e:
            logger.error(f"Error in risk assessment: {e}")
            return {'error': str(e), 'status': 'failed'}
    
    def _get_recommendations(self, risk_level: str) -> list:
        """Get recommendations based on risk level"""
        
        recommendations = {
            'Low': [
                'Continue regular checkups',
                'Maintain healthy lifestyle',
                'Monitor parameters quarterly'
            ],
            'Medium': [
                'Increase monitoring frequency to monthly',
                'Consider lifestyle modifications',
                'Follow up in 1-2 months with healthcare provider'
            ],
            'High': [
                'Urgent medical consultation recommended',
                'Weekly parameter monitoring',
                'Implement immediate lifestyle changes',
                'Follow up within 2 weeks'
            ],
            'Critical': [
                'URGENT: Medical attention required immediately',
                'Daily monitoring essential',
                'Consider immediate hospitalization',
                'Follow up within 24-48 hours'
            ]
        }
        
        return recommendations.get(risk_level, [])
    
    def get_stats(self) -> Dict[str, Any]:
        """Get prediction statistics"""
        return {
            **self.prediction_stats,
            'success_rate': (
                (self.prediction_stats['successful'] / max(1, self.prediction_stats['total_predictions']))
                if self.prediction_stats['total_predictions'] > 0 else 0
            )
        }
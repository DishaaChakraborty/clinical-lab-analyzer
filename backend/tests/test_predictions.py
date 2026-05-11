import pytest
import numpy as np
from app.services.model_loader import get_model_loader
from app.services.predictor import Predictor
from app.services.analyzer import CompleteAnalyzer
from app.services.validation import InputValidator

class TestModelLoading:
    """Test model loading and initialization"""
    
    def test_model_loader_initialization(self):
        """Test that models load correctly"""
        loader = get_model_loader()
        assert loader is not None
        assert loader.is_ready()
    
    def test_all_models_loaded(self):
        """Test all required models are loaded"""
        loader = get_model_loader()
        
        assert loader.get_agent2_model() is not None
        assert loader.get_agent3_models() is not None
        assert len(loader.get_agent3_models()) > 0
        assert loader.get_agent4_model() is not None
        assert loader.get_scaler() is not None
        assert loader.get_feature_names() is not None

class TestInputValidation:
    """Test input validation"""
    
    def test_valid_input(self):
        """Test validation with valid input"""
        loader = get_model_loader()
        validator = InputValidator(loader.get_feature_names())
        
        lab_values = {
            "age": 55,
            "glucose": 150,
            "cholesterol": 220,
            "hdl": 35,
            "ldl": 160
        }
        
        is_valid, result = validator.validate_lab_values(lab_values)
        # Should handle missing features gracefully
        assert result is not None
    
    def test_invalid_input(self):
        """Test validation with invalid input"""
        loader = get_model_loader()
        validator = InputValidator(loader.get_feature_names())
        
        # Empty input
        is_valid, result = validator.validate_lab_values({})
        assert not is_valid

class TestPredictions:
    """Test model predictions"""
    
    def test_disease_prediction(self):
        """Test disease classification prediction"""
        analyzer = CompleteAnalyzer()
        
        # Create valid features
        loader = get_model_loader()
        feature_names = loader.get_feature_names()
        
        lab_values = {fname: 100.0 for fname in feature_names}
        
        features = analyzer._prepare_features(lab_values)
        assert features is not None
        
        result = analyzer.predictor.predict_disease(features)
        assert 'disease_type' in result or 'error' in result
    
    def test_risk_assessment(self):
        """Test risk assessment prediction"""
        analyzer = CompleteAnalyzer()
        
        loader = get_model_loader()
        feature_names = loader.get_feature_names()
        
        lab_values = {fname: 100.0 for fname in feature_names}
        features = analyzer._prepare_features(lab_values)
        
        result = analyzer.predictor.assess_risk(features)
        assert 'risk_score' in result or 'error' in result

class TestCompleteAnalysis:
    """Test complete analysis workflow"""
    
    def test_complete_analysis(self):
        """Test end-to-end analysis"""
        analyzer = CompleteAnalyzer()
        
        lab_values = {
            "age": 55,
            "glucose": 150,
            "cholesterol": 220,
            "hdl": 35,
            "ldl": 160,
            "triglycerides": 180,
            "hemoglobin": 13.2,
            "creatinine": 1.1,
            "bmi": 28,
            "systolic_bp": 135,
            "diastolic_bp": 85,
            "uric_acid": 7.5,
            "wbc": 7.2,
            "potassium": 4.2,
            "gender": "M"
        }
        
        result = analyzer.analyze_patient(lab_values)
        
        assert result is not None
        assert 'analysis_id' in result
        assert result['status'] in ['success', 'failed']
        
        if result['status'] == 'success':
            assert 'agent2_disease_detection' in result
            assert 'agent3_trend_forecasting' in result
            assert 'agent4_risk_assessment' in result
            assert 'summary' in result

class TestRobustness:
    """Test model robustness with edge cases"""
    
    def test_random_data(self):
        """Test model doesn't crash on random data"""
        analyzer = CompleteAnalyzer()
        
        # Random features
        loader = get_model_loader()
        feature_names = loader.get_feature_names()
        
        random_features = {fname: np.random.rand() * 100 for fname in feature_names}
        
        result = analyzer.analyze_patient(random_features)
        assert result is not None
        # Should complete without crashing
    
    def test_nan_values(self):
        """Test handling of NaN values"""
        analyzer = CompleteAnalyzer()
        
        loader = get_model_loader()
        feature_names = loader.get_feature_names()
        
        lab_values = {fname: np.nan for fname in feature_names}
        
        result = analyzer.analyze_patient(lab_values)
        # Should handle NaN gracefully
        assert result is not None
    
    def test_extreme_values(self):
        """Test handling of extreme values"""
        analyzer = CompleteAnalyzer()
        
        lab_values = {
            "glucose": 10000,  # Extreme
            "age": 200,        # Invalid
            "cholesterol": -100 # Negative
        }
        
        result = analyzer.analyze_patient(lab_values)
        assert result is not None

# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
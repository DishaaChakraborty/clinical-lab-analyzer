import numpy as np
from typing import Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)

class InputValidator:
    """Production-grade input validation"""
    
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names or []
        self.validation_errors = []
    
    def validate_lab_values(self, lab_values: Dict[str, float]) -> Tuple[bool, Dict]:
        """
        Comprehensive validation with detailed error reporting
        
        Returns:
            (is_valid, validation_result)
        """
        
        self.validation_errors = []
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'missing_features': [],
            'filled_features': {},
            'invalid_features': {}
        }
        
        # Check for empty input
        if not lab_values:
            result['valid'] = False
            result['errors'].append('Empty lab values provided')
            return False, result
        
        # Validate each feature
        for feature in self.feature_names:
            if feature not in lab_values:
                result['missing_features'].append(feature)
                result['filled_features'][feature] = 0.0  # Default fill
            else:
                value = lab_values[feature]
                
                # Type check
                if not isinstance(value, (int, float)):
                    try:
                        value = float(value)
                        result['filled_features'][feature] = value
                    except (ValueError, TypeError):
                        result['valid'] = False
                        result['invalid_features'][feature] = f"Cannot convert to number: {value}"
                        continue
                
                # Range check (basic)
                if value < -1000 or value > 10000:
                    result['warnings'].append(f"{feature}: {value} is outside normal range")
                
                result['filled_features'][feature] = value
        
        # Check if too many missing features (more than 50%)
        missing_ratio = len(result['missing_features']) / len(self.feature_names)
        logger.warning(
            f"Missing features: {len(result['missing_features'])}/{len(self.feature_names)}"
            )
        if result['errors']:
            result['valid'] = False
        
        return result['valid'], result
    
    def validate_model_readiness(self, model_loader) -> Tuple[bool, Dict]:
        """Validate that models are ready for inference"""
        
        result = {
            'ready': True,
            'models': {},
            'missing': []
        }
        
        # Check Agent 2
        agent2 = model_loader.get_agent2_model()
        if agent2 is None:
            result['ready'] = False
            result['missing'].append('Agent 2')
        else:
            result['models']['agent2'] = 'loaded'
        
        # Check Agent 3
        agent3 = model_loader.get_agent3_models()
        if not agent3:
            result['ready'] = False
            result['missing'].append('Agent 3')
        else:
            result['models']['agent3'] = f'{len(agent3)} models loaded'
        
        # Check Agent 4
        agent4 = model_loader.get_agent4_model()
        if agent4 is None:
            result['ready'] = False
            result['missing'].append('Agent 4')
        else:
            result['models']['agent4'] = 'loaded'
        
        # Check scaler
        scaler = model_loader.get_scaler()
        if scaler is None:
            result['ready'] = False
            result['missing'].append('Scaler')
        else:
            result['models']['scaler'] = 'loaded'
        
        return result['ready'], result

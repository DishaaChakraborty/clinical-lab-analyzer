import pickle
import joblib
import os
import numpy as np
from typing import Dict, Any, Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class ModelCache:
    """Cache for loaded models with TTL"""
    
    def __init__(self, ttl_seconds: int = 3600):
        self.cache = {}
        self.timestamps = {}
        self.ttl = ttl_seconds
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache if not expired"""
        if key not in self.cache:
            return None
        
        # Check if expired
        if datetime.now() - self.timestamps[key] > timedelta(seconds=self.ttl):
            del self.cache[key]
            del self.timestamps[key]
            return None
        
        return self.cache[key]
    
    def set(self, key: str, value: Any):
        """Set item in cache"""
        self.cache[key] = value
        self.timestamps[key] = datetime.now()
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self.timestamps.clear()

class ModelLoader:
    """Load and manage pre-trained ML models - Production Grade"""
    
    def __init__(self, model_path: str = "backend/ml/models", use_cache: bool = True):
        self.model_path = model_path
        self.models = {}
        self.scaler = None
        self.feature_names = None
        self.use_cache = use_cache
        self.cache = ModelCache(ttl_seconds=3600)
        self.load_status = {}
    
    def load_all_models(self) -> bool:
        """Load all trained models with error handling"""
        
        logger.info("="*70)
        logger.info("LOADING TRAINED MODELS")
        logger.info("="*70)
        
        all_loaded = True
        
        # Load Agent 2 (Disease Classification)
        if not self._load_agent2():
            all_loaded = False
        
        # Load Agent 3 (Trend Forecasting)
        if not self._load_agent3():
            all_loaded = False
        
        # Load Agent 4 (Risk Assessment)
        if not self._load_agent4():
            all_loaded = False
        
        # Load Scaler
        if not self._load_scaler():
            all_loaded = False
        
        # Load Feature Names
        if not self._load_feature_names():
            all_loaded = False
        
        logger.info("="*70)
        if all_loaded:
            logger.info("ALL MODELS LOADED SUCCESSFULLY")
        else:
            logger.warning("SOME MODELS FAILED TO LOAD")
        logger.info("="*70)
        
        return all_loaded
    
    def _load_agent2(self) -> bool:
        """Load Agent 2 with fallback"""
        try:
            # Try multiple possible names for Agent 2 model
            possible_names = ['agent2_best_model.pkl', 'agent2.pkl', 'agent2_model.pkl']
            agent2_path = None
            
            for name in possible_names:
                path = os.path.join(self.model_path, name)
                if os.path.exists(path):
                    agent2_path = path
                    break
            
            if not agent2_path:
                logger.warning(f"Agent 2 model not found. Tried: {possible_names}")
                self.load_status['agent2'] = 'NOT_FOUND'
                return False
            
            # Try joblib first (better for sklearn), then pickle
            try:
                with open(agent2_path, 'rb') as f:
                    self.models['agent2'] = joblib.load(f)
            except:
                with open(agent2_path, 'rb') as f:
                    self.models['agent2'] = pickle.load(f)
            
            logger.info("✓ Loaded Agent 2 (Disease Classifier)")
            self.load_status['agent2'] = 'LOADED'
            return True
        
        except Exception as e:
            logger.error(f"Error loading Agent 2: {e}")
            self.load_status['agent2'] = 'ERROR'
            return False
    
    def _load_agent3(self) -> bool:
        """Load Agent 3 with error handling"""
        try:
            agent3_path = os.path.join(self.model_path, 'agent3_models.pkl')
            
            if not os.path.exists(agent3_path):
                logger.warning(f"Agent 3 models not found: {agent3_path}")
                self.load_status['agent3'] = 'NOT_FOUND'
                return False
            
            try:
                with open(agent3_path, 'rb') as f:
                    agent3_data = joblib.load(f)
            except:
                with open(agent3_path, 'rb') as f:
                    agent3_data = pickle.load(f)
            
            # Handle both dict and object formats
            if isinstance(agent3_data, dict):
                self.models['agent3'] = agent3_data.get('models', agent3_data)
            else:
                self.models['agent3'] = agent3_data.models if hasattr(agent3_data, 'models') else {}
            
            logger.info(f"✓ Loaded Agent 3 ({len(self.models.get('agent3', {}))} trend models)")
            self.load_status['agent3'] = 'LOADED'
            return True
        
        except Exception as e:
            logger.error(f"Error loading Agent 3: {e}")
            self.load_status['agent3'] = 'ERROR'
            return False
    
    def _load_agent4(self) -> bool:
        """Load Agent 4 with error handling"""
        try:
            agent4_path = os.path.join(self.model_path, 'agent4_best_model.pkl')
            
            if not os.path.exists(agent4_path):
                logger.warning(f"Agent 4 model not found: {agent4_path}")
                self.load_status['agent4'] = 'NOT_FOUND'
                return False
            
            try:
                with open(agent4_path, 'rb') as f:
                    self.models['agent4'] = joblib.load(f)
            except:
                with open(agent4_path, 'rb') as f:
                    self.models['agent4'] = pickle.load(f)
            
            logger.info("✓ Loaded Agent 4 (Risk Assessment)")
            self.load_status['agent4'] = 'LOADED'
            return True
        
        except Exception as e:
            logger.error(f"Error loading Agent 4: {e}")
            self.load_status['agent4'] = 'ERROR'
            return False
    
    def _load_scaler(self) -> bool:
        """Load feature scaler"""
        try:
            scaler_path = os.path.join(self.model_path, 'scaler.pkl')
            
            if not os.path.exists(scaler_path):
                logger.warning(f"Scaler not found: {scaler_path}")
                self.load_status['scaler'] = 'NOT_FOUND'
                return False
            
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            logger.info("✓ Loaded Feature Scaler (StandardScaler)")
            self.load_status['scaler'] = 'LOADED'
            return True
        
        except Exception as e:
            logger.error(f"Error loading scaler: {e}")
            self.load_status['scaler'] = 'ERROR'
            return False
    
    def _load_feature_names(self) -> bool:
        """Load feature names"""
        try:
            features_path = os.path.join(self.model_path, 'feature_names.pkl')
            
            if not os.path.exists(features_path):
                logger.warning(f"Feature names not found: {features_path}")
                self.load_status['features'] = 'NOT_FOUND'
                return False
            
            with open(features_path, 'rb') as f:
                self.feature_names = pickle.load(f)
            
            logger.info(f"✓ Loaded {len(self.feature_names)} feature names")
            self.load_status['features'] = 'LOADED'
            return True
        
        except Exception as e:
            logger.error(f"Error loading feature names: {e}")
            self.load_status['features'] = 'ERROR'
            return False
    
    def get_agent2_model(self):
        """Get Agent 2 model with caching"""
        if self.use_cache:
            cached = self.cache.get('agent2')
            if cached:
                return cached
        
        model = self.models.get('agent2')
        if self.use_cache and model:
            self.cache.set('agent2', model)
        return model
    
    def get_agent3_models(self):
        """Get Agent 3 models"""
        return self.models.get('agent3', {})
    
    def get_agent4_model(self):
        """Get Agent 4 model"""
        return self.models.get('agent4')
    
    def get_scaler(self):
        """Get feature scaler"""
        return self.scaler
    
    def get_feature_names(self):
        """Get feature names"""
        return self.feature_names
    
    def get_load_status(self) -> Dict[str, str]:
        """Get model loading status"""
        return self.load_status
    
    def is_ready(self) -> bool:
        """Check if all models are loaded"""
        return all(
            status == 'LOADED' 
            for status in self.load_status.values()
        )

# Global model loader instance
_model_loader = None

def get_model_loader(model_path: str = "backend/ml/models", use_cache: bool = True) -> ModelLoader:
    """Get or create model loader (dependency injection)"""
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoader(model_path, use_cache)
        _model_loader.load_all_models()
    return _model_loader
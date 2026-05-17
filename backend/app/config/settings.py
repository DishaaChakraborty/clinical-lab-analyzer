from pydantic_settings import BaseSettings
from typing import Optional
import os
from functools import lru_cache

class Settings(BaseSettings):
    """Application configuration settings"""
    
    # API Settings
    API_TITLE: str = "Clinical Lab Analysis API"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "Production-grade AI-assisted clinical lab report analysis"
    
    # Server Settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    RELOAD: bool = os.getenv("RELOAD", "False").lower() == "true"
    
    # Database Settings
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", 
        "mysql+pymysql://root:ik Disha12@@localhost:3306/clinical_lab"
    )
    
    # Model Settings
    MODEL_PATH: str = os.getenv("MODEL_PATH", "backend/ml/models")
    ENABLE_MODEL_CACHE: bool = True
    MODEL_CACHE_TTL: int = 3600  # seconds
    
    # API Settings
    MAX_REQUEST_SIZE: int = 10 * 1024 * 1024  # 10MB
    REQUEST_TIMEOUT: int = 30  # seconds
    MAX_CONCURRENT_REQUESTS: int = 100
    
    # Logging Settings
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = "logs/api.log"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Feature Settings
    ENABLE_ROBUSTNESS_TESTING: bool = True
    ENABLE_INTERPRETABILITY: bool = True
    ENABLE_CACHING: bool = True
    ENABLE_METRICS: bool = True
    
    # Validation Settings
    MIN_GLUCOSE: float = 50.0
    MAX_GLUCOSE: float = 600.0
    MIN_AGE: int = 0
    MAX_AGE: int = 150
    
    # SHAP Settings
    SHAP_SAMPLE_SIZE: int = 100
    SHAP_ENABLE: bool = True
    
    # Monitoring Settings
    ENABLE_PROMETHEUS: bool = True
    METRICS_PORT: int = 9090
    
    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()

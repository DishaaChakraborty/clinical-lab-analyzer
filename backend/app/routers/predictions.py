from fastapi import APIRouter, HTTPException, Depends, Query, Body
from typing import Dict, Any, Optional
import logging
import time

from ..services.analyzer import get_analyzer
from ..services.validation import InputValidator
from ..services.model_loader import get_model_loader
from ..schemas import LabValuesRequest, AnalysisResponse, ErrorResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/predictions", tags=["Predictions"])

@router.on_event("startup")
async def startup_event():
    """Load models on startup"""
    try:
        model_loader = get_model_loader()
        if model_loader.is_ready():
            logger.info("Models loaded and ready for predictions")
        else:
            logger.warning("Some models failed to load")
    except Exception as e:
        logger.error(f"Startup error: {e}")

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_lab_report(
    lab_values: LabValuesRequest = Body(...)
) -> AnalysisResponse:
    """
    Complete analysis of lab report with all 4 agents
    
    POST endpoint that performs:
    1. Agent 1: Parse lab values
    2. Agent 2: Classify disease (87% accuracy)
    3. Agent 3: Forecast trends (30/60/90 days)
    4. Agent 4: Assess risk (0-100 score)
    
    Returns complete analysis with recommendations
    """
    
    try:
        analyzer = get_analyzer()
        request_data = lab_values.model_dump()
        result = analyzer.analyze_patient(request_data)
        
        if result['status'] == 'failed':
            raise HTTPException(status_code=400, detail=result.get('error', 'Analysis failed'))
        
        return result
    
    except Exception as e:
        logger.error(f"Error in analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/disease-classification", response_model=Dict[str, Any])
async def predict_disease(lab_values: Dict[str, float]) -> Dict[str, Any]:
    """
    Predict disease type only (Agent 2)
    
    Returns: Disease classification + confidence probability
    """
    
    try:
        analyzer = get_analyzer()
        features = analyzer._prepare_features(lab_values)
        
        if features is None:
            raise HTTPException(status_code=400, detail="Invalid features")
        
        result = analyzer.predictor.predict_disease(features)
        
        if 'error' in result:
            raise HTTPException(status_code=500, detail=result['error'])
        
        return result
    
    except Exception as e:
        logger.error(f"Error in disease classification: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/trend-forecast", response_model=Dict[str, Any])
async def forecast_trends(lab_values: Dict[str, float]) -> Dict[str, Any]:
    """
    Forecast parameter trends 30/60/90 days (Agent 3)
    
    Returns: Future values + trend direction for all parameters
    """
    
    try:
        analyzer = get_analyzer()
        result = analyzer.predictor.forecast_trends(lab_values)
        
        if 'error' in result:
            raise HTTPException(status_code=500, detail=result['error'])
        
        return result
    
    except Exception as e:
        logger.error(f"Error in trend forecasting: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/risk-assessment", response_model=Dict[str, Any])
async def assess_risk(lab_values: Dict[str, float]) -> Dict[str, Any]:
    """
    Assess overall health risk (Agent 4)
    
    Returns: Risk score (0-100) + level + recommendations
    """
    
    try:
        analyzer = get_analyzer()
        features = analyzer._prepare_features(lab_values)
        
        if features is None:
            raise HTTPException(status_code=400, detail="Invalid features")
        
        result = analyzer.predictor.assess_risk(features)
        
        if 'error' in result:
            raise HTTPException(status_code=500, detail=result['error'])
        
        return result
    
    except Exception as e:
        logger.error(f"Error in risk assessment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Check if prediction service is ready"""
    
    try:
        model_loader = get_model_loader()
        status_report = model_loader.get_load_status()
        
        all_ready = model_loader.is_ready()
        
        return {
            "service": "Prediction API",
            "status": "ready" if all_ready else "partial",
            "models": status_report,
            "all_models_loaded": all_ready,
            "message": "All models loaded and ready" if all_ready else "Some models not loaded"
        }
    
    except Exception as e:
        return {
            "service": "Prediction API",
            "status": "error",
            "message": str(e)
        }

@router.get("/stats")
async def get_statistics() -> Dict[str, Any]:
    """Get API statistics"""
    
    try:
        analyzer = get_analyzer()
        
        return {
            "predictions": analyzer.predictor.get_stats(),
            "analysis_count": analyzer.analysis_count,
            "timestamp": time.time()
        }
    
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return {"error": str(e)}
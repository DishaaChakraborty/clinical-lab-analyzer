from pydantic import BaseModel, EmailStr, Field, validator, ConfigDict
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

# Enums
class RiskLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"

class ParameterStatus(str, Enum):
    NORMAL = "Normal"
    LOW = "Low"
    HIGH = "High"
    CRITICAL = "Critical"

# Patient Schemas
class PatientBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    age: Optional[int] = Field(None, ge=0, le=150)
    gender: Optional[str] = Field(None, max_length=10)
    email: Optional[str] = None
    phone: Optional[str] = Field(None, max_length=20)
    medical_history: Optional[str] = None

class PatientCreate(PatientBase):
    pass

class PatientUpdate(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    medical_history: Optional[str] = None

class PatientResponse(PatientBase):
    id: int
    date_created: datetime
    date_updated: Optional[datetime]
    is_active: bool

    model_config = ConfigDict(from_attributes=True)

# Parameter Schemas
class ParsedParameterBase(BaseModel):
    parameter_name: str
    value: float
    normal_range_min: Optional[float] = None
    normal_range_max: Optional[float] = None
    unit: Optional[str] = None
    status: ParameterStatus = ParameterStatus.NORMAL

class ParsedParameterResponse(ParsedParameterBase):
    id: int
    report_id: int
    is_outlier: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

# Report Schemas
class ReportBase(BaseModel):
    report_date: datetime

class ReportCreate(ReportBase):
    patient_id: int

class ReportResponse(ReportBase):
    id: int
    patient_id: int
    file_name: Optional[str]
    file_type: Optional[str]
    is_parsed: bool
    parameters: List[ParsedParameterResponse] = []
    created_at: datetime
    updated_at: Optional[datetime]

    model_config = ConfigDict(from_attributes=True)

# Risk Assessment Schemas
class RiskAssessmentBase(BaseModel):
    risk_score: float = Field(..., ge=0, le=100)
    risk_level: RiskLevel
    confidence: float = Field(..., ge=0, le=1)
    pattern_detected: Optional[str] = None

class RiskAssessmentResponse(RiskAssessmentBase):
    id: int
    patient_id: int
    report_id: Optional[int]
    primary_risks: Optional[List[str]] = []
    secondary_risks: Optional[List[str]] = []
    recommendations: Optional[List[str]] = []
    follow_up_days: Optional[int]
    model_version: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

# Trend Analysis Schemas
class TrendAnalysisResponse(BaseModel):
    parameter_name: str
    current_value: float
    previous_value: Optional[float]
    trend_direction: str
    trend_magnitude: float
    trend_strength: float
    forecast_30days: Optional[float]
    forecast_60days: Optional[float]
    forecast_90days: Optional[float]
    forecast_confidence: float
    data_points_count: int
    analysis_start_date: datetime
    analysis_end_date: datetime

    model_config = ConfigDict(from_attributes=True)

# Error Response
class ErrorResponse(BaseModel):
    detail: str
    error_code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# List Response
class ListResponse(BaseModel):
    total: int
    items: List[Any]
    

class LabValuesRequest(BaseModel):
    """Lab values request schema"""
    
    age: Optional[float] = Field(None, description="Patient age in years")
    glucose: Optional[float] = Field(None, description="Glucose in mg/dL")
    cholesterol: Optional[float] = Field(None, description="Total cholesterol in mg/dL")
    hdl: Optional[float] = Field(None, description="HDL cholesterol in mg/dL")
    ldl: Optional[float] = Field(None, description="LDL cholesterol in mg/dL")
    triglycerides: Optional[float] = Field(None, description="Triglycerides in mg/dL")
    hemoglobin: Optional[float] = Field(None, description="Hemoglobin in g/dL")
    creatinine: Optional[float] = Field(None, description="Creatinine in mg/dL")
    bmi: Optional[float] = Field(None, description="BMI")
    systolic_bp: Optional[float] = Field(None, description="Systolic blood pressure")
    diastolic_bp: Optional[float] = Field(None, description="Diastolic blood pressure")
    uric_acid: Optional[float] = Field(None, description="Uric acid")
    wbc: Optional[float] = Field(None, description="White blood cells")
    potassium: Optional[float] = Field(None, description="Potassium")
    gender: Optional[str] = Field(None, description="Gender (M/F)")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
        }
    )

class AnalysisResponse(BaseModel):
    """Complete analysis response"""
    
    analysis_id: str
    timestamp: str
    status: str
    agent1_parsing: Dict[str, Any]
    agent2_disease_detection: Dict[str, Any]
    agent3_trend_forecasting: Dict[str, Any]
    agent4_risk_assessment: Dict[str, Any]
    summary: str
    processing_time_ms: float
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "analysis_id": "ANALYSIS_1_1234567890",
                "timestamp": "2024-03-20T10:30:45.123456",
                "status": "success",
                "agent1_parsing": {"status": "completed", "parameters_extracted": 50},
                "agent2_disease_detection": {"disease_type": "MetabolicSyndrome", "confidence": 0.92},
                "agent3_trend_forecasting": {"glucose": {"current": 150, "forecast_90d": 165, "trend": "increasing"}},
                "agent4_risk_assessment": {"risk_score": 70, "risk_level": "High", "confidence": 0.84},
                "summary": "CLINICAL ANALYSIS SUMMARY...",
                "processing_time_ms": 45.23
            }
        }
    )
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, Boolean, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime

from backend.app.routers import patients, predictions
from backend.app.database import engine,Base
class Patient(Base):
    """Patient model"""
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False, index=True)
    age = Column(Integer)
    gender = Column(String(10))
    email = Column(String(100), unique=True, index=True)
    phone = Column(String(20))
    medical_history = Column(Text, nullable=True)
    date_created = Column(DateTime(timezone=True), server_default=func.now())
    date_updated = Column(DateTime(timezone=True), onupdate=func.now())
    is_active = Column(Boolean, default=True)

    # Relationships
    reports = relationship("Report", back_populates="patient", cascade="all, delete-orphan")
    risk_assessments = relationship("RiskAssessment", back_populates="patient", cascade="all, delete-orphan")
    trend_analyses = relationship("TrendAnalysis", back_populates="patient", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Patient(id={self.id}, name={self.name}, age={self.age})>"


class Report(Base):
    """Lab report model"""
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False, index=True)
    report_date = Column(DateTime(timezone=True), nullable=False, index=True)
    raw_text = Column(Text, nullable=True)
    file_name = Column(String(200))
    file_type = Column(String(20))  # csv, txt, pdf
    is_parsed = Column(Boolean, default=False)
    parse_error = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    patient = relationship("Patient", back_populates="reports")
    parameters = relationship("ParsedParameter", back_populates="report", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Report(id={self.id}, patient_id={self.patient_id}, date={self.report_date})>"


class ParsedParameter(Base):
    """Parsed lab parameter model"""
    __tablename__ = "parsed_parameters"

    id = Column(Integer, primary_key=True, index=True)
    report_id = Column(Integer, ForeignKey("reports.id"), nullable=False, index=True)
    parameter_name = Column(String(100), nullable=False, index=True)
    value = Column(Float, nullable=False)
    normal_range_min = Column(Float, nullable=True)
    normal_range_max = Column(Float, nullable=True)
    unit = Column(String(50))
    status = Column(String(20))  # Normal, Low, High, Critical
    is_outlier = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    report = relationship("Report", back_populates="parameters")

    def __repr__(self):
        return f"<ParsedParameter(parameter={self.parameter_name}, value={self.value})>"


class RiskAssessment(Base):
    """Risk assessment model"""
    __tablename__ = "risk_assessments"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False, index=True)
    report_id = Column(Integer, ForeignKey("reports.id"), nullable=True)
    risk_score = Column(Float, nullable=False)
    risk_level = Column(String(20), nullable=False)  # Low, Medium, High, Critical
    confidence = Column(Float)
    
    primary_risks = Column(JSON)
    secondary_risks = Column(JSON)
    pattern_detected = Column(String(100))
    pattern_confidence = Column(Float)
    
    recommendations = Column(JSON)
    follow_up_days = Column(Integer)
    
    model_version = Column(String(20))
    model_algorithm = Column(String(50))
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    patient = relationship("Patient", back_populates="risk_assessments")

    def __repr__(self):
        return f"<RiskAssessment(patient_id={self.patient_id}, level={self.risk_level})>"


class TrendAnalysis(Base):
    """Trend analysis model"""
    __tablename__ = "trend_analyses"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False, index=True)
    parameter_name = Column(String(100), nullable=False, index=True)
    
    current_value = Column(Float)
    previous_value = Column(Float)
    trend_direction = Column(String(20))  # increasing, decreasing, stable
    trend_magnitude = Column(Float)
    trend_strength = Column(Float)
    
    forecast_30days = Column(Float)
    forecast_60days = Column(Float)
    forecast_90days = Column(Float)
    forecast_confidence = Column(Float)
    
    data_points_count = Column(Integer)
    analysis_start_date = Column(DateTime(timezone=True))
    analysis_end_date = Column(DateTime(timezone=True))
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    patient = relationship("Patient", back_populates="trend_analyses")

    def __repr__(self):
        return f"<TrendAnalysis(patient_id={self.patient_id}, parameter={self.parameter_name})>"


class MLModelMetadata(Base):
    """ML Model metadata and versioning"""
    __tablename__ = "ml_model_metadata"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(50), unique=True, index=True)
    algorithm = Column(String(100))
    version = Column(String(20))
    training_date = Column(DateTime(timezone=True))
    
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    roc_auc = Column(Float)
    r2_score = Column(Float)
    
    training_samples = Column(Integer)
    test_samples = Column(Integer)
    cross_val_mean = Column(Float)
    cross_val_std = Column(Float)
    
    model_path = Column(String(200))
    feature_columns = Column(JSON)
    
    is_active = Column(Boolean, default=True)
    is_production = Column(Boolean, default=False)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    def __repr__(self):
        return f"<MLModelMetadata(model={self.model_name}, v{self.version})>"


class AuditLog(Base):
    """Audit log for all operations"""
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    operation = Column(String(50))
    entity_type = Column(String(50))
    entity_id = Column(Integer)
    details = Column(JSON)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    ip_address = Column(String(50))
    user_agent = Column(String(200))

    def __repr__(self):
        return f"<AuditLog(operation={self.operation}, entity={self.entity_type})>"

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

from fastapi import APIRouter, HTTPException, Depends, status
from sqlalchemy.orm import Session
from typing import List, Optional
import logging

from backend.app.database import get_db
from backend.app.models.patient import Patient

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/patients", tags=["Patients"])

# Pydantic schemas
from pydantic import BaseModel, EmailStr, ConfigDict

class PatientCreate(BaseModel):
    first_name: str
    last_name: str
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    date_of_birth: Optional[str] = None

class PatientUpdate(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[EmailStr] = None
    phone: Optional[str] = None

class PatientResponse(BaseModel):
    id: int
    first_name: str
    last_name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    is_active: bool

    model_config = ConfigDict(from_attributes=True)

# GET all patients
@router.get("", response_model=List[PatientResponse])
async def list_patients(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """Get list of all patients"""
    try:
        patients = db.query(Patient).filter(Patient.is_active == True).offset(skip).limit(limit).all()
        return patients
    except Exception as e:
        logger.error(f"Error listing patients: {e}")
        raise HTTPException(status_code=500, detail="Error fetching patients")

# GET single patient
@router.get("/{patient_id}", response_model=PatientResponse)
async def get_patient(patient_id: int, db: Session = Depends(get_db)):
    """Get single patient by ID"""
    try:
        patient = db.query(Patient).filter(Patient.id == patient_id, Patient.is_active == True).first()
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")
        return patient
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting patient {patient_id}: {e}")
        raise HTTPException(status_code=500, detail="Error fetching patient")

# CREATE patient
@router.post("", response_model=PatientResponse, status_code=status.HTTP_201_CREATED)
async def create_patient(patient: PatientCreate, db: Session = Depends(get_db)):
    """Create new patient"""
    try:
        # Check if email already exists
        if patient.email:
            existing = db.query(Patient).filter(Patient.email == patient.email).first()
            if existing:
                raise HTTPException(status_code=400, detail="Email already registered")
        
        db_patient = Patient(**patient.dict())
        db.add(db_patient)
        db.commit()
        db.refresh(db_patient)
        
        logger.info(f"Created patient: {db_patient.id}")
        return db_patient
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating patient: {e}")
        raise HTTPException(status_code=500, detail="Error creating patient")

# UPDATE patient
@router.put("/{patient_id}", response_model=PatientResponse)
async def update_patient(patient_id: int, patient: PatientUpdate, db: Session = Depends(get_db)):
    """Update patient information"""
    try:
        db_patient = db.query(Patient).filter(Patient.id == patient_id).first()
        if not db_patient:
            raise HTTPException(status_code=404, detail="Patient not found")
        
        update_data = patient.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(db_patient, field, value)
        
        db.commit()
        db.refresh(db_patient)
        
        logger.info(f"Updated patient: {patient_id}")
        return db_patient
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating patient {patient_id}: {e}")
        raise HTTPException(status_code=500, detail="Error updating patient")

# DELETE patient
@router.delete("/{patient_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_patient(patient_id: int, db: Session = Depends(get_db)):
    """Delete patient (soft delete - sets is_active to False)"""
    try:
        db_patient = db.query(Patient).filter(Patient.id == patient_id).first()
        if not db_patient:
            raise HTTPException(status_code=404, detail="Patient not found")
        
        db_patient.is_active = False
        db.commit()
        
        logger.info(f"Deleted patient: {patient_id}")
        return None
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting patient {patient_id}: {e}")
        raise HTTPException(status_code=500, detail="Error deleting patient")
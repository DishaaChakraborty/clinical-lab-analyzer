import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class SyntheticDataGenerator:
    """Generate realistic synthetic patient data for ML model training"""
    
    def __init__(self, num_patients: int = 1000, seed: int = 42):
        self.num_patients = num_patients
        self.data = []
        random.seed(seed)
        np.random.seed(seed)
    
    def generate_healthy_patient(self, patient_id: int) -> List[Dict]:
        """Generate healthy patient with stable, normal values"""
        records = []
        
        base_glucose = np.random.normal(92, 5)  # Mean 92, SD 5
        base_chol = np.random.normal(180, 15)
        
        num_reports = np.random.randint(3, 7)
        
        for report_num in range(1, num_reports + 1):
            days_ago = 30 * (num_reports - report_num)
            report_date = (datetime.now() - timedelta(days=days_ago)).isoformat()
            
            record = {
                'patient_id': patient_id,
                'age': np.random.randint(20, 55),
                'gender': np.random.choice(['M', 'F']),
                'report_number': report_num,
                'report_date': report_date,
                
                # Stable, normal values
                'glucose': np.clip(base_glucose + np.random.normal(0, 2), 70, 110),
                'cholesterol': np.clip(base_chol + np.random.normal(0, 5), 150, 200),
                'hdl': np.random.uniform(45, 65),
                'ldl': np.random.uniform(80, 130),
                'triglycerides': np.random.uniform(80, 150),
                'hemoglobin': np.random.uniform(13.5, 16.5),
                'creatinine': np.random.uniform(0.7, 1.0),
                'bmi': np.random.uniform(19, 24),
                'systolic_bp': np.random.uniform(110, 120),
                'diastolic_bp': np.random.uniform(70, 80),
                'uric_acid': np.random.uniform(3.5, 5.5),
                'wbc': np.random.uniform(4.5, 7.0),
                'potassium': np.random.uniform(3.5, 5.0),
                
                'risk_label': 'Low'
            }
            records.append(record)
        
        return records
    
    def generate_metabolic_syndrome_patient(self, patient_id: int) -> List[Dict]:
        """Generate patient with progressive metabolic syndrome"""
        records = []
        
        glucose_slope = np.random.uniform(2, 5)  # Increasing 2-5 per report
        chol_slope = np.random.uniform(3, 8)
        
        num_reports = np.random.randint(3, 7)
        
        for report_num in range(1, num_reports + 1):
            days_ago = 30 * (num_reports - report_num)
            report_date = (datetime.now() - timedelta(days=days_ago)).isoformat()
            
            # Progressive increase
            glucose = 102 + (report_num * glucose_slope) + np.random.normal(0, 2)
            cholesterol = 200 + (report_num * chol_slope) + np.random.normal(0, 5)
            
            record = {
                'patient_id': patient_id,
                'age': np.random.randint(38, 72),
                'gender': np.random.choice(['M', 'F']),
                'report_number': report_num,
                'report_date': report_date,
                
                'glucose': np.clip(glucose, 90, 250),
                'cholesterol': np.clip(cholesterol, 180, 280),
                'hdl': np.random.uniform(25, 40),  # Low
                'ldl': np.random.uniform(140, 200),  # High
                'triglycerides': np.random.uniform(150, 300),  # High
                'hemoglobin': np.random.uniform(13.0, 15.5),
                'creatinine': np.random.uniform(0.9, 1.3),
                'bmi': np.random.uniform(26, 34),  # Overweight
                'systolic_bp': np.random.uniform(125, 145),
                'diastolic_bp': np.random.uniform(80, 95),
                'uric_acid': np.random.uniform(5.5, 8.0),
                'wbc': np.random.uniform(6.0, 8.5),
                'potassium': np.random.uniform(3.5, 5.2),
                
                'risk_label': 'Medium' if cholesterol < 240 else 'High'
            }
            records.append(record)
        
        return records
    
    def generate_anemia_patient(self, patient_id: int) -> List[Dict]:
        """Generate patient with iron deficiency anemia"""
        records = []
        
        num_reports = np.random.randint(3, 7)
        
        for report_num in range(1, num_reports + 1):
            days_ago = 30 * (num_reports - report_num)
            report_date = (datetime.now() - timedelta(days=days_ago)).isoformat()
            
            # Declining hemoglobin trend
            hemoglobin = 14.0 - (report_num * 0.4) + np.random.normal(0, 0.2)
            
            record = {
                'patient_id': patient_id,
                'age': np.random.randint(22, 65),
                'gender': np.random.choice(['F', 'F', 'M']),  # More females
                'report_number': report_num,
                'report_date': report_date,
                
                'glucose': np.random.uniform(85, 115),
                'cholesterol': np.random.uniform(170, 220),
                'hdl': np.random.uniform(40, 60),
                'ldl': np.random.uniform(100, 160),
                'triglycerides': np.random.uniform(80, 160),
                'hemoglobin': np.clip(hemoglobin, 10.5, 15),  # Declining
                'creatinine': np.random.uniform(0.7, 1.1),
                'bmi': np.random.uniform(19, 28),
                'systolic_bp': np.random.uniform(110, 130),
                'diastolic_bp': np.random.uniform(70, 85),
                'uric_acid': np.random.uniform(3.0, 6.5),
                'wbc': np.random.uniform(4.5, 8.0),
                'potassium': np.random.uniform(3.8, 4.8),
                
                'risk_label': 'High' if hemoglobin < 12.0 else 'Medium'
            }
            records.append(record)
        
        return records
    
    def generate_kidney_disease_patient(self, patient_id: int) -> List[Dict]:
        """Generate patient with progressive kidney dysfunction"""
        records = []
        
        creatinine_slope = np.random.uniform(0.08, 0.20)
        uric_acid_slope = np.random.uniform(0.3, 0.8)
        
        num_reports = np.random.randint(3, 7)
        
        for report_num in range(1, num_reports + 1):
            days_ago = 30 * (num_reports - report_num)
            report_date = (datetime.now() - timedelta(days=days_ago)).isoformat()
            
            creatinine = 1.0 + (report_num * creatinine_slope) + np.random.normal(0, 0.05)
            uric_acid = 5.5 + (report_num * uric_acid_slope) + np.random.normal(0, 0.2)
            
            record = {
                'patient_id': patient_id,
                'age': np.random.randint(48, 82),
                'gender': np.random.choice(['M', 'F']),
                'report_number': report_num,
                'report_date': report_date,
                
                'glucose': np.random.uniform(85, 145),
                'cholesterol': np.random.uniform(190, 250),
                'hdl': np.random.uniform(30, 50),
                'ldl': np.random.uniform(120, 180),
                'triglycerides': np.random.uniform(100, 220),
                'hemoglobin': np.random.uniform(12.0, 14.5),
                'creatinine': np.clip(creatinine, 0.8, 2.8),  # Increasing
                'bmi': np.random.uniform(22, 32),
                'systolic_bp': np.random.uniform(135, 160),
                'diastolic_bp': np.random.uniform(85, 100),
                'uric_acid': np.clip(uric_acid, 5.5, 9.0),  # High
                'wbc': np.random.uniform(5.0, 8.5),
                'potassium': np.random.uniform(4.5, 5.5),  # Can be elevated
                
                'risk_label': 'High'
            }
            records.append(record)
        
        return records
    
    def generate_diabetes_patient(self, patient_id: int) -> List[Dict]:
        """Generate patient with uncontrolled diabetes"""
        records = []
        
        glucose_slope = np.random.uniform(5, 12)
        
        num_reports = np.random.randint(3, 7)
        
        for report_num in range(1, num_reports + 1):
            days_ago = 30 * (num_reports - report_num)
            report_date = (datetime.now() - timedelta(days=days_ago)).isoformat()
            
            glucose = 160 + (report_num * glucose_slope) + np.random.normal(0, 5)
            
            record = {
                'patient_id': patient_id,
                'age': np.random.randint(45, 75),
                'gender': np.random.choice(['M', 'F']),
                'report_number': report_num,
                'report_date': report_date,
                
                'glucose': np.clip(glucose, 140, 300),  # High
                'cholesterol': np.random.uniform(200, 280),
                'hdl': np.random.uniform(25, 45),  # Low
                'ldl': np.random.uniform(140, 200),
                'triglycerides': np.random.uniform(150, 350),  # Very high
                'hemoglobin': np.random.uniform(12.5, 15.0),
                'creatinine': np.random.uniform(0.9, 1.4),
                'bmi': np.random.uniform(25, 36),  # Overweight
                'systolic_bp': np.random.uniform(125, 160),
                'diastolic_bp': np.random.uniform(80, 100),
                'uric_acid': np.random.uniform(5.0, 8.0),
                'wbc': np.random.uniform(5.5, 8.5),
                'potassium': np.random.uniform(3.8, 5.2),
                
                'risk_label': 'High'
            }
            records.append(record)
        
        return records
    
    def generate_all_patients(self) -> pd.DataFrame:
        """Generate all patient data"""
        print(f"\n{'='*70}")
        print(f"Generating {self.num_patients} synthetic patient records...")
        print(f"{'='*70}\n")
        
        patient_id = 1
        distribution = {
            'healthy': 300,
            'metabolic': 300,
            'anemia': 150,
            'kidney': 150,
            'diabetes': 100
        }
        
        # Healthy patients
        print("  ► Generating 300 healthy patients...")
        for _ in range(distribution['healthy']):
            self.data.extend(self.generate_healthy_patient(patient_id))
            patient_id += 1
        
        # Metabolic syndrome patients
        print("  ► Generating 300 metabolic syndrome patients...")
        for _ in range(distribution['metabolic']):
            self.data.extend(self.generate_metabolic_syndrome_patient(patient_id))
            patient_id += 1
        
        # Anemia patients
        print("  ► Generating 150 anemia patients...")
        for _ in range(distribution['anemia']):
            self.data.extend(self.generate_anemia_patient(patient_id))
            patient_id += 1
        
        # Kidney disease patients
        print("  ► Generating 150 kidney disease patients...")
        for _ in range(distribution['kidney']):
            self.data.extend(self.generate_kidney_disease_patient(patient_id))
            patient_id += 1
        
        # Diabetes patients
        print("  ► Generating 100 diabetes patients...")
        for _ in range(distribution['diabetes']):
            self.data.extend(self.generate_diabetes_patient(patient_id))
            patient_id += 1
        
        df = pd.DataFrame(self.data)
        
        print(f"\n{'='*70}")
        print(f"✓ Generated {len(df)} total records from {patient_id - 1} patients")
        print(f"{'='*70}\n")
        
        # Print statistics
        print("=== DATASET STATISTICS ===\n")
        print(f"Total records: {len(df)}")
        print(f"Total unique patients: {df['patient_id'].nunique()}")
        print(f"Date range: {df['report_date'].min()} to {df['report_date'].max()}")
        
        print(f"\n=== RISK DISTRIBUTION ===")
        print(df['risk_label'].value_counts())
        
        print(f"\n=== AGE STATISTICS ===")
        print(f"Min age: {df['age'].min()}")
        print(f"Max age: {df['age'].max()}")
        print(f"Mean age: {df['age'].mean():.1f}")
        
        print(f"\n=== PARAMETER RANGES ===")
        print(f"Glucose: {df['glucose'].min():.1f} - {df['glucose'].max():.1f}")
        print(f"Cholesterol: {df['cholesterol'].min():.1f} - {df['cholesterol'].max():.1f}")
        print(f"Hemoglobin: {df['hemoglobin'].min():.1f} - {df['hemoglobin'].max():.1f}")
        print(f"Creatinine: {df['creatinine'].min():.2f} - {df['creatinine'].max():.2f}")
        
        print(f"\n{'='*70}\n")
        
        return df
    
    def save_to_csv(self, filepath: str = 'backend/ml/data/synthetic_data.csv') -> pd.DataFrame:
        """Save to CSV"""
        if not self.data:
            raise ValueError("No data generated. Run generate_all_patients first.")
        
        df = pd.DataFrame(self.data)
        df.to_csv(filepath, index=False)
        print(f"✓ Saved to {filepath}")
        return df
    
    def get_dataframe(self) -> pd.DataFrame:
        """Get DataFrame"""
        if not self.data:
            raise ValueError("No data generated. Run generate_all_patients first.")
        return pd.DataFrame(self.data)


# Utility function
def generate_training_data(
    num_patients: int = 1000,
    save_path: str = 'backend/ml/data/synthetic_data.csv'
) -> pd.DataFrame:
    """Convenience function to generate and save training data"""
    generator = SyntheticDataGenerator(num_patients=num_patients)
    df = generator.generate_all_patients()
    generator.save_to_csv(save_path)
    return df


if __name__ == "__main__":
    # Run this script to generate data
    generate_training_data()